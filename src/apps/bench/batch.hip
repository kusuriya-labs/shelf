#include <shelf/core/memory/arena.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/continuous_batcher.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <chrono>

namespace {

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

uint32_t count_attention_layers(const shelf::qwen35::Qwen35TextConfig& tc) {
    return static_cast<uint32_t>(
        shelf::qwen_hybrid::count_layer_topology(tc).attention_layers);
}

void usage(FILE* f) {
    std::fprintf(f,
        "Usage: shelf-bench batch [options]\n"
        "  --model-dir PATH       (required)\n"
        "  --device N             (default 0)\n"
        "  --page-tokens N        (default 16)\n"
        "  --arena-gib N          (default auto = free VRAM minus 1 GiB)\n"
        "  --prefix-tokens N      shared prefix length (default 512)\n"
        "  --suffix-tokens N      per-request suffix length (default 8)\n"
        "  --max-new-tokens N     decode tokens per request (default 8)\n"
        "  --requests N           number of requests (default 4)\n"
        "  --warmup N             discarded requests before measuring (default 0)\n"
        "  --max-sched-tokens N   scheduler token budget per step (0 = prefix_tokens)\n"
        "  --same-suffix          identical suffix for every request\n");
}

struct BatchOptions {
    std::string model_dir;
    int device = 0;
    int page_tokens = 16;
    double arena_gib = 0.0;
    bool arena_auto = true;
    int prefix_tokens = 512;
    int suffix_tokens = 8;
    int max_new_tokens = 8;
    int requests = 4;
    int warmup = 0;
    int max_sched_tokens = 0;
    bool same_suffix = false;
};

bool parse_int(const char* s, int& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

bool parse_args(int argc, char** argv, BatchOptions& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model-dir") { const char* v = next("--model-dir"); if (!v) return false; o.model_dir = v; }
        else if (a == "--device") { const char* v = next("--device"); if (!v || !parse_int(v, o.device)) return false; }
        else if (a == "--page-tokens") { const char* v = next("--page-tokens"); if (!v || !parse_int(v, o.page_tokens)) return false; }
        else if (a == "--arena-gib") { const char* v = next("--arena-gib"); if (!v) return false; if (std::strcmp(v, "auto") == 0) { o.arena_auto = true; } else { o.arena_gib = std::strtod(v, nullptr); if (o.arena_gib <= 0.0) return false; o.arena_auto = false; } }
        else if (a == "--prefix-tokens") { const char* v = next("--prefix-tokens"); if (!v || !parse_int(v, o.prefix_tokens)) return false; }
        else if (a == "--suffix-tokens") { const char* v = next("--suffix-tokens"); if (!v || !parse_int(v, o.suffix_tokens)) return false; }
        else if (a == "--max-new-tokens") { const char* v = next("--max-new-tokens"); if (!v || !parse_int(v, o.max_new_tokens)) return false; }
        else if (a == "--requests") { const char* v = next("--requests"); if (!v || !parse_int(v, o.requests)) return false; }
        else if (a == "--warmup") { const char* v = next("--warmup"); if (!v || !parse_int(v, o.warmup)) return false; }
        else if (a == "--max-sched-tokens") { const char* v = next("--max-sched-tokens"); if (!v || !parse_int(v, o.max_sched_tokens)) return false; }
        else if (a == "--same-suffix") { o.same_suffix = true; }
        else if (a == "-h" || a == "--help") { usage(stdout); exit(0); }
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return false; }
    }
    return !o.model_dir.empty() && o.prefix_tokens > 0 && o.suffix_tokens > 0;
}

struct RunResult {
    uint64_t step_tokens = 0;
    uint64_t steps = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    std::vector<std::vector<int32_t>> generated;
    bool ok = false;
};

RunResult run_pass(shelf::qwen35::Executor& executor,
                   shelf::state::SequenceSlotPool& seq_pool,
                   shelf::state::GdnStatePool& gdn_pool,
                   shelf::state::PagedKVPool& kv_pool,
                   shelf::qwen35::runtime::ContinuousBatcher& batcher,
                   const BatchOptions& opts,
                   hipStream_t stream) {
    RunResult out;
    std::vector<int32_t> prefix(static_cast<std::size_t>(opts.prefix_tokens));
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        prefix[i] = static_cast<int32_t>((i * 131u + 17u) % 1000u);
    }
    std::vector<uint64_t> ids;

    auto drain = [&]() -> bool {
        while (batcher.has_pending()) {
            auto step = batcher.step();
            if (!step.ok()) {
                std::fprintf(stderr, "step failed: %s\n", step.status().message().c_str());
                return false;
            }
            if (!step.value().executed) {
                std::fprintf(stderr, "step not executed with pending requests\n");
                return false;
            }
            out.step_tokens += step.value().num_tokens;
            out.steps += 1;
        }
        return true;
    };

    auto wall_start = std::chrono::steady_clock::now();
    {
        auto seed = batcher.submit(std::vector<int32_t>(prefix), 0);
        if (!seed.ok()) return out;
        if (!drain()) return out;
    }

    for (int r = 0; r < opts.warmup + opts.requests; ++r) {
        std::vector<int32_t> prompt = prefix;
        prompt.reserve(prefix.size() + static_cast<std::size_t>(opts.suffix_tokens));
        const int suffix_base = opts.same_suffix ? 0 : r * 7;
        for (int s = 0; s < opts.suffix_tokens; ++s) {
            prompt.push_back(static_cast<int32_t>(100 + (suffix_base + s) % 1000));
        }
        const uint32_t new_tokens = (r < opts.warmup) ? 0 : static_cast<uint32_t>(opts.max_new_tokens);
        auto submit = batcher.submit(std::move(prompt), new_tokens);
        if (!submit.ok()) {
            std::fprintf(stderr, "submit failed: %s\n", submit.status().message().c_str());
            return out;
        }
        ids.push_back(submit.value());
        if (!drain()) return out;
    }
    auto wall_end = std::chrono::steady_clock::now();
    if (hipStreamSynchronize(stream) != hipSuccess) return out;
    out.prefill_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i < static_cast<std::size_t>(opts.warmup)) continue;
        const auto* req = batcher.request(ids[i]);
        if (req == nullptr) return out;
        out.generated.push_back(req->generated);
    }
    out.ok = true;
    return out;
}

}  // namespace

int run_batch(int argc, char** argv) {
    BatchOptions opts;
    if (!parse_args(argc, argv, opts)) {
        usage(stderr);
        return 2;
    }

    if (hipSetDevice(opts.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::size_t arena_bytes = 0;
    if (opts.arena_auto) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const hipError_t arena_err = hipMemGetInfo(&free_bytes, &total_bytes);
        if (arena_err != hipSuccess) {
            std::fprintf(stderr, "hipMemGetInfo failed: %s\n", hipGetErrorString(arena_err));
            return 1;
        }
        const std::size_t margin = 1ull * 1024ull * 1024ull * 1024ull;
        arena_bytes = free_bytes > margin ? free_bytes - margin : free_bytes / 2u;
        if (arena_bytes == 0) {
            std::fprintf(stderr, "no VRAM available for the arena\n");
            return 1;
        }
        std::fprintf(stderr, "batch arena auto: free=%zu capacity=%zu\n", free_bytes, arena_bytes);
    } else {
        arena_bytes = static_cast<std::size_t>(opts.arena_gib * 1024.0 * 1024.0 * 1024.0);
    }

    auto arena_result = shelf::gpu::GpuArena::create(opts.device, arena_bytes);
    if (!arena_result.ok()) {
        std::fprintf(stderr, "GpuArena::create failed: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    shelf::qwen35::Qwen35LoadOptions load_options;
    auto model_result = shelf::qwen35::Qwen35Model::load_from_safetensors(
        opts.model_dir, arena, stream, load_options);
    if (!model_result.ok()) {
        std::fprintf(stderr, "model load failed: %s\n", model_result.status().message().c_str());
        return 1;
    }
    auto model = model_result.release();
    if (hipStreamSynchronize(stream) != hipSuccess) return 1;

    const auto& tc = model.text_config();
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);

    const uint32_t seq_len = static_cast<uint32_t>(opts.prefix_tokens + opts.suffix_tokens +
                                                   opts.max_new_tokens + 8);
    const uint32_t num_pages = ceil_div(seq_len, static_cast<uint32_t>(opts.page_tokens)) + 2;

    auto seq_pool_result = shelf::state::SequenceSlotPool::create(arena, 2, num_pages);
    if (!seq_pool_result.ok()) return 1;
    auto seq_pool = seq_pool_result.release();

    auto gdn_pool_result = shelf::state::GdnStatePool::create(
        arena, 2, shelf::qwen35::make_qwen35_gdn_state_layout(tc));
    if (!gdn_pool_result.ok()) return 1;
    auto gdn_pool = gdn_pool_result.release();

    auto kv_pool_result = shelf::state::PagedKVPool::create(
        arena, num_pages, static_cast<uint32_t>(opts.page_tokens),
        num_attn_layers, kv_heads, head_dim, shelf::kernel::KVCacheDType::BF16);
    if (!kv_pool_result.ok()) return 1;
    auto kv_pool = kv_pool_result.release();

    std::fprintf(stderr,
        "[batch] layers=%zu attn_layers=%u gdn_states=%u conv_dim=%u conv_pad=%u conv_hist=%u v_heads=%u hk=%u hv=%u\n",
        tc.num_hidden_layers, num_attn_layers, gdn_pool.num_gdn_states(),
        gdn_pool.conv_dim(), gdn_pool.conv_dim_padded(), gdn_pool.conv_history(),
        gdn_pool.num_v_heads(), gdn_pool.head_k(), gdn_pool.head_v());

    const uint32_t sched_tokens = opts.max_sched_tokens > 0
        ? static_cast<uint32_t>(opts.max_sched_tokens)
        : static_cast<uint32_t>(opts.prefix_tokens);
    shelf::qwen35::ExecutorConfig exec_config;
    exec_config.max_batch_rows = sched_tokens;
    exec_config.max_scheduled_requests = 1;

    auto exec_result = shelf::qwen35::create_model_executor(
        model, seq_pool, gdn_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        std::fprintf(stderr, "create_model_executor failed: %s\n", exec_result.status().message().c_str());
        return 1;
    }
    auto executor = exec_result.release();

    shelf::qwen35::runtime::ContinuousBatcherConfig cfg;
    cfg.max_num_batched_tokens = sched_tokens;
    cfg.max_scheduled_requests = 1;
    cfg.max_seq_len = seq_len;
    cfg.page_tokens = static_cast<uint32_t>(opts.page_tokens);
    cfg.eos_token_ids = tc.stop_tokens;

    auto run_one = [&]() -> RunResult {
        shelf::qwen35::runtime::ContinuousBatcherConfig c = cfg;
        shelf::qwen35::runtime::ContinuousBatcher batcher(
            executor, seq_pool, gdn_pool, kv_pool, c, stream);
        return run_pass(executor, seq_pool, gdn_pool, kv_pool, batcher, opts, stream);
    };

    std::printf("prefix_tokens=%d suffix_tokens=%d requests=%d max_new=%d\n",
                opts.prefix_tokens, opts.suffix_tokens, opts.requests,
                opts.max_new_tokens);

    RunResult result = run_one();
    if (!result.ok) {
        std::fprintf(stderr, "batch pass failed\n");
        return 1;
    }
    std::printf("step_tokens=%llu steps=%llu wall_ms=%.1f\n",
                static_cast<unsigned long long>(result.step_tokens),
                static_cast<unsigned long long>(result.steps), result.prefill_ms);
    for (std::size_t i = 0; i < result.generated.size(); ++i) {
        std::printf("GEN[%zu]=", i);
        bool first = true;
        for (int32_t tok : result.generated[i]) {
            std::printf("%s%d", first ? "" : ",", tok);
            first = false;
        }
        std::printf(" (n=%zu)\n", result.generated[i].size());
    }

    (void)hipStreamDestroy(stream);
    return 0;
}
