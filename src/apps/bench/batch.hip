#include <shelf/core/memory/arena.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/model_weights_factory.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/continuous_batcher.h>
#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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

uint32_t count_attention_layers(const shelf::qwen_hybrid::QwenHybridConfig& tc) {
    return static_cast<uint32_t>(
        shelf::qwen_hybrid::count_layer_topology(tc).attention_layers);
}

void usage(FILE* f) {
    std::fprintf(f,
        "Usage: shelf-bench batch [options]\n"
        "  --model-dir PATH       (required)\n"
        "  --device LIST          comma separated GPUs, one per --tensor-parallel rank\n"
        "                         (default 0; e.g. --device 0,2)\n"
        "  --tensor-parallel N    tensor parallel degree 1 | 2 | 4 (default 1)\n"
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
    std::vector<int> devices = {0};
    int tp = 1;
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

bool parse_devices(const char* s, std::vector<int>& out) {
    out.clear();
    if (s == nullptr || *s == '\0') return false;
    const std::string value(s);
    size_t start = 0;
    while (start <= value.size()) {
        size_t comma = value.find(',', start);
        const std::string token = value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        int device = 0;
        if (!parse_int(token.c_str(), device) || device < 0) return false;
        out.push_back(device);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return !out.empty();
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
        else if (a == "--device") { const char* v = next("--device"); if (!v || !parse_devices(v, o.devices)) return false; }
        else if (a == "--tensor-parallel") { const char* v = next("--tensor-parallel"); if (!v || !parse_int(v, o.tp) || (o.tp != 1 && o.tp != 2 && o.tp != 4)) return false; }
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
    if (o.model_dir.empty() || o.prefix_tokens <= 0 || o.suffix_tokens <= 0) return false;
    if (o.devices.size() != static_cast<size_t>(o.tp)) {
        std::fprintf(stderr, "--device expects %d entries for --tensor-parallel %d, got %zu\n",
                     o.tp, o.tp, o.devices.size());
        return false;
    }
    std::vector<int> sorted_devices = o.devices;
    std::sort(sorted_devices.begin(), sorted_devices.end());
    if (std::adjacent_find(sorted_devices.begin(), sorted_devices.end()) !=
        sorted_devices.end()) {
        std::fprintf(stderr, "--device entries must be unique\n");
        return false;
    }
    return true;
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

    if (hipSetDevice(opts.devices[0]) != hipSuccess) return 1;

    std::size_t arena_bytes = 0;
    if (opts.arena_auto) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const hipError_t arena_err = hipMemGetInfo(&free_bytes, &total_bytes);
        if (arena_err != hipSuccess) {
            std::fprintf(stderr, "hipMemGetInfo failed: %s\n", hipGetErrorString(arena_err));
            return 1;
        }
        const std::size_t margin = 512ull * 1024ull * 1024ull;
        arena_bytes = free_bytes > margin ? free_bytes - margin : free_bytes / 2u;
        if (arena_bytes == 0) {
            std::fprintf(stderr, "no VRAM available for the arena\n");
            return 1;
        }
        std::fprintf(stderr, "batch arena auto: free=%zu capacity=%zu\n", free_bytes, arena_bytes);
    } else {
        arena_bytes = static_cast<std::size_t>(opts.arena_gib * 1024.0 * 1024.0 * 1024.0);
    }

    const uint32_t seq_len = static_cast<uint32_t>(opts.prefix_tokens + opts.suffix_tokens +
                                                   opts.max_new_tokens + 8);
    const uint32_t num_pages = ceil_div(seq_len, static_cast<uint32_t>(opts.page_tokens)) + 2;
    const uint32_t sched_tokens = opts.max_sched_tokens > 0
        ? static_cast<uint32_t>(opts.max_sched_tokens)
        : static_cast<uint32_t>(opts.prefix_tokens);

    shelf::qwen35::ExecutorConfig exec_config;
    exec_config.max_batch_rows = sched_tokens;
    exec_config.max_scheduled_requests = 1;

    hipStream_t stream = nullptr;
    shelf::qwen_hybrid::QwenHybridConfig tc;
    std::unique_ptr<shelf::gpu::GpuArena> tp1_arena;
    std::unique_ptr<shelf::models::LoadedModel> tp1_model;
    std::unique_ptr<shelf::state::SequenceSlotPool> tp1_seq_pool;
    std::unique_ptr<shelf::state::GdnStatePool> tp1_gdn_pool;
    std::unique_ptr<shelf::state::PagedKVPool> tp1_kv_pool;
    std::unique_ptr<shelf::qwen35::Executor> tp1_executor;
    std::unique_ptr<shelf::qwen35::runtime::TpCoordinator> tp_coordinator;
    shelf::state::SequenceSlotPool* seq_pool = nullptr;
    shelf::state::GdnStatePool* gdn_pool = nullptr;
    shelf::state::PagedKVPool* kv_pool = nullptr;
    shelf::qwen35::Executor* executor = nullptr;

    if (opts.tp == 1) {
        if (hipStreamCreate(&stream) != hipSuccess) return 1;

        auto arena_result = shelf::gpu::GpuArena::create(opts.devices[0], arena_bytes);
        if (!arena_result.ok()) {
            std::fprintf(stderr, "GpuArena::create failed: %s\n",
                         arena_result.status().message().c_str());
            return 1;
        }
        tp1_arena = std::make_unique<shelf::gpu::GpuArena>(arena_result.release());

        auto model_result =
            shelf::models::LoadedModel::load(opts.model_dir, *tp1_arena, stream);
        if (!model_result.ok()) {
            std::fprintf(stderr, "model load failed: %s\n",
                         model_result.status().message().c_str());
            return 1;
        }
        tp1_model = std::make_unique<shelf::models::LoadedModel>(model_result.release());
        if (hipStreamSynchronize(stream) != hipSuccess) return 1;

        tc = tp1_model->hybrid_config();
        const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
        const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
        const uint32_t num_attn_layers = count_attention_layers(tc);

        auto seq_pool_result =
            shelf::state::SequenceSlotPool::create(*tp1_arena, 2, num_pages);
        if (!seq_pool_result.ok()) return 1;
        tp1_seq_pool =
            std::make_unique<shelf::state::SequenceSlotPool>(seq_pool_result.release());
        seq_pool = tp1_seq_pool.get();

        auto gdn_pool_result = shelf::state::GdnStatePool::create(
            *tp1_arena, 2, shelf::qwen35::make_qwen35_gdn_state_layout(tc));
        if (!gdn_pool_result.ok()) return 1;
        tp1_gdn_pool =
            std::make_unique<shelf::state::GdnStatePool>(gdn_pool_result.release());
        gdn_pool = tp1_gdn_pool.get();

        auto kv_pool_result = shelf::state::PagedKVPool::create(
            *tp1_arena, num_pages, static_cast<uint32_t>(opts.page_tokens),
            num_attn_layers, kv_heads, head_dim, shelf::kernel::KVCacheDType::BF16);
        if (!kv_pool_result.ok()) return 1;
        tp1_kv_pool =
            std::make_unique<shelf::state::PagedKVPool>(kv_pool_result.release());
        kv_pool = tp1_kv_pool.get();

        auto exec_result = tp1_model->create_executor(
            *seq_pool, *gdn_pool, *kv_pool, *tp1_arena, exec_config);
        if (!exec_result.ok()) {
            std::fprintf(stderr, "create_executor failed: %s\n",
                         exec_result.status().message().c_str());
            return 1;
        }
        tp1_executor =
            std::make_unique<shelf::qwen35::Executor>(exec_result.release());
        executor = tp1_executor.get();
    } else {
        shelf::qwen35::runtime::TpCoordinatorConfig tp_cfg;
        tp_cfg.model_dir = opts.model_dir;
        tp_cfg.devices = opts.devices;
        tp_cfg.max_batch_rows = sched_tokens;
        tp_cfg.max_scheduled_requests = 1;
        tp_cfg.max_scheduled_output_rows = 1;
        tp_cfg.max_seq_len = seq_len;
        tp_cfg.page_tokens = static_cast<uint32_t>(opts.page_tokens);
        tp_cfg.kv_cache_dtype = shelf::kernel::KVCacheDType::BF16;
        tp_cfg.arena_bytes = arena_bytes;
        tp_cfg.working_set_reserve_bytes = 0;
        shelf::Status sel = shelf::models::select_tp_rank_model(tp_cfg);
        if (!sel.ok()) {
            std::fprintf(stderr, "tensor parallel rank model: %s\n", sel.message().c_str());
            return 1;
        }
        auto tp_result = shelf::qwen35::runtime::TpCoordinator::create(tp_cfg);
        if (!tp_result.ok()) {
            std::fprintf(stderr, "TpCoordinator::create failed: %s\n",
                         tp_result.status().message().c_str());
            return 1;
        }
        tp_coordinator = tp_result.release();
        stream = tp_coordinator->rank(0).stream();
        tc = tp_coordinator->rank(0).hybrid_config();
        seq_pool = &tp_coordinator->rank(0).sequence_pool();
        gdn_pool = &tp_coordinator->rank(0).gdn_pool();
        kv_pool = &tp_coordinator->rank(0).kv_pool();
        executor = &tp_coordinator->rank(0).executor();
    }

    const uint32_t num_attn_layers = count_attention_layers(tc);
    std::fprintf(stderr,
        "[batch] layers=%zu attn_layers=%u gdn_states=%u conv_dim=%u conv_pad=%u conv_hist=%u v_heads=%u hk=%u hv=%u\n",
        tc.num_hidden_layers, num_attn_layers, gdn_pool->num_gdn_states(),
        gdn_pool->conv_dim(), gdn_pool->conv_dim_padded(), gdn_pool->conv_history(),
        gdn_pool->num_v_heads(), gdn_pool->head_k(), gdn_pool->head_v());

    shelf::qwen35::runtime::ContinuousBatcherConfig cfg;
    cfg.max_num_batched_tokens = sched_tokens;
    cfg.max_scheduled_requests = 1;
    cfg.max_seq_len = seq_len;
    cfg.page_tokens = static_cast<uint32_t>(opts.page_tokens);
    cfg.eos_token_ids = tc.stop_tokens;

    auto run_one = [&]() -> RunResult {
        shelf::qwen35::runtime::ContinuousBatcherConfig c = cfg;
        shelf::qwen35::runtime::ContinuousBatcher batcher(
            *executor, *seq_pool, *gdn_pool, *kv_pool, c, stream);
        if (tp_coordinator != nullptr) batcher.set_tp_batch_hook(tp_coordinator.get());
        return run_pass(*executor, *seq_pool, *gdn_pool, *kv_pool, batcher, opts, stream);
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

    tp_coordinator.reset();
    if (tp1_arena != nullptr) {
        (void)tp1_arena->shutdown();
    }
    if (opts.tp == 1) {
        (void)hipStreamDestroy(stream);
    }
    return 0;
}
