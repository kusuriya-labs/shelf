#include <phaseshift/core/memory/arena.h>
#include <phaseshift/models/qwen35/runtime/decode_backend.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/scheduling_types.h>
#include <phaseshift/inference/execution/decode_perf_stats.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/models/qwen35/state/state_types.h>
#include <phaseshift/models/model_source.h>
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/model/gdn_state_layout.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <thread>

#ifdef PHASESHIFT_HAS_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

namespace {

struct BenchOptions {
    std::string mode = "forward";
    int context = 128;
    int tokens = 8;
    int warmup = 0;
    bool compute_logits = true;
    int device = 0;
    std::string model_dir;
    std::optional<std::string> model_host_socket;
    int page_tokens = 16;
    double arena_gib = 0.0;
    bool arena_auto = true;
    std::string output;
    int prefill_chunk = 128;
    int gap_keepalive = 0;
    int gap_keepalive_flags = hipEventDisableSystemFence;
    std::string perf_stats_output;
    std::string tokens_file;
    int tokens_offset = 0;
    ps::qwen35::runtime::DecodeBackend decode_backend =
        ps::qwen35::runtime::DecodeBackend::Hip;
};

void print_usage(FILE* out) {
    fprintf(out,
        "Usage: phaseshift-bench tg [options]\n"
        "  --mode forward|greedy        (default forward)\n"
        "  --context N                  (default 128)\n"
        "  --tokens N                   (default 8)\n"
        "  --prefill-chunk N            (default 128)\n"
        "  --warmup N                   (default 0)\n"
        "  --compute-logits 0|1         (default 1)\n"
        "  --device N                   (default 0)\n"
        "  --model-dir PATH             (required)\n"
        "  --model-host SPEC            resident model host socket spec\n"
        "  --page-tokens N              (default 16)\n"
        "  --arena-gib N                (default auto = free VRAM minus 1 GiB)\n"
        "  --output PATH                (default none)\n"
        "  --gap-keepalive US           (default 0 = off; keep the CP awake)\n"
        "  --gap-keepalive-flags N      (default hipEventDisableSystemFence;\n"
        "                                -1 = hipEventCreate)\n"
        "  --perf-stats-output PATH     (default none, diagnostic decode counters)\n"
        "  --tokens-file PATH           PSKLDTOK corpus file used as the context\n"
        "  --tokens-offset N            corpus position to start the context (default 0)\n"
        "  --decode-backend TYPE        host (default)\n");
}

bool parse_bool(const std::string& value, bool& out) {
    if (value == "1" || value == "true" || value == "yes") {
        out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "no") {
        out = false;
        return true;
    }
    return false;
}

bool parse_int(const std::string& value, int& out) {
    size_t pos = 0;
    try {
        out = std::stoi(value, &pos);
    } catch (...) {
        return false;
    }
    return pos == value.size();
}

bool parse_double(const std::string& value, double& out) {
    size_t pos = 0;
    try {
        out = std::stod(value, &pos);
    } catch (...) {
        return false;
    }
    return pos == value.size();
}

bool parse_args(int argc, char** argv, BenchOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key = arg;
        std::string value;
        bool has_value = false;
        size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            key = arg.substr(0, eq);
            value = arg.substr(eq + 1);
            has_value = true;
        }
        auto need_value = [&]() -> std::string {
            if (has_value) return value;
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", key.c_str());
                exit(2);
            }
            ++i;
            return std::string(argv[i]);
        };
        if (key == "--mode") {
            opts.mode = need_value();
        } else if (key == "--context") {
            if (!parse_int(need_value(), opts.context) || opts.context < 0) {
                fprintf(stderr, "invalid --context\n");
                return false;
            }
        } else if (key == "--tokens") {
            if (!parse_int(need_value(), opts.tokens) || opts.tokens < 1) {
                fprintf(stderr, "invalid --tokens\n");
                return false;
            }
        } else if (key == "--prefill-chunk") {
            if (!parse_int(need_value(), opts.prefill_chunk) || opts.prefill_chunk < 1) {
                fprintf(stderr, "invalid --prefill-chunk\n");
                return false;
            }
        } else if (key == "--gap-keepalive") {
            if (!parse_int(need_value(), opts.gap_keepalive) || opts.gap_keepalive < 0) {
                fprintf(stderr, "invalid --gap-keepalive\n");
                return false;
            }
        } else if (key == "--gap-keepalive-flags") {
            if (!parse_int(need_value(), opts.gap_keepalive_flags)) {
                fprintf(stderr, "invalid --gap-keepalive-flags\n");
                return false;
            }
        } else if (key == "--warmup") {
            if (!parse_int(need_value(), opts.warmup) || opts.warmup < 0) {
                fprintf(stderr, "invalid --warmup\n");
                return false;
            }
        } else if (key == "--compute-logits") {
            if (!parse_bool(need_value(), opts.compute_logits)) {
                fprintf(stderr, "invalid --compute-logits\n");
                return false;
            }
        } else if (key == "--device") {
            if (!parse_int(need_value(), opts.device)) {
                fprintf(stderr, "invalid --device\n");
                return false;
            }
        } else if (key == "--model-dir") {
            opts.model_dir = need_value();
        } else if (key == "--model-host") {
            opts.model_host_socket = need_value();
        } else if (key == "--page-tokens") {
            if (!parse_int(need_value(), opts.page_tokens) || opts.page_tokens < 1) {
                fprintf(stderr, "invalid --page-tokens\n");
                return false;
            }
        } else if (key == "--arena-gib") {
            const std::string text = need_value();
            if (text == "auto") {
                opts.arena_auto = true;
            } else if (!parse_double(text, opts.arena_gib) || opts.arena_gib <= 0.0) {
                fprintf(stderr, "invalid --arena-gib (expected N or auto)\n");
                return false;
            } else {
                opts.arena_auto = false;
            }
        } else if (key == "--decode-backend") {
            if (!ps::qwen35::runtime::parse_decode_backend(
                    need_value().c_str(), &opts.decode_backend)) {
                fprintf(stderr, "invalid --decode-backend\n");
                return false;
            }
        } else if (key == "--output") {
            opts.output = need_value();
        } else if (key == "--perf-stats-output") {
            opts.perf_stats_output = need_value();
        } else if (key == "--tokens-file") {
            opts.tokens_file = need_value();
        } else if (key == "--tokens-offset") {
            if (!parse_int(need_value(), opts.tokens_offset) || opts.tokens_offset < 0) {
                fprintf(stderr, "invalid --tokens-offset\n");
                return false;
            }
        } else if (key == "--help" || key == "-h") {
            print_usage(stdout);
            exit(0);
        } else {
            fprintf(stderr, "unknown option: %s\n", key.c_str());
            return false;
        }
    }
    if (opts.mode != "forward" && opts.mode != "greedy") {
        fprintf(stderr, "--mode must be forward or greedy\n");
        return false;
    }
    if (opts.mode == "greedy" && !opts.compute_logits) {
        fprintf(stderr, "greedy mode requires --compute-logits 1\n");
        return false;
    }
    if (opts.model_dir.empty()) {
        fprintf(stderr, "--model-dir is required\n");
        return false;
    }
    return true;
}

bool roctx_available() {
#ifdef PHASESHIFT_HAS_ROCTX
    return true;
#else
    return false;
#endif
}

void roctx_resume() {
#ifdef PHASESHIFT_HAS_ROCTX
    roctxProfilerResume(0);
#endif
}

void roctx_pause() {
#ifdef PHASESHIFT_HAS_ROCTX
    roctxProfilerPause(0);
#endif
}

void roctx_range_push(const char* name) {
#ifdef PHASESHIFT_HAS_ROCTX
    roctxRangePush(name);
#endif
}

void roctx_range_pop() {
#ifdef PHASESHIFT_HAS_ROCTX
    roctxRangePop();
#endif
}

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t count_attention_layers(const ps::qwen35::Qwen35TextConfig& tc) {
    uint32_t n = 0;
    for (int lt : tc.layer_types) {
        if (lt != 0) ++n;
    }
    return n;
}

std::vector<int32_t> read_token_corpus(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return {};
    char magic[8] = {};
    uint32_t version = 0;
    uint64_t count = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "PSKLDTOK", 8) == 0;
    ok = ok && std::fread(&version, 4, 1, f) == 1 && version == 1u;
    ok = ok && std::fread(&count, 8, 1, f) == 1 && count > 0u && count < (1u << 22);
    std::vector<int32_t> tokens;
    if (ok) {
        tokens.resize(static_cast<std::size_t>(count));
        ok = std::fread(tokens.data(), sizeof(int32_t), tokens.size(), f) == tokens.size();
    }
    std::fclose(f);
    if (!ok) return {};
    return tokens;
}

}  // anonymous namespace

int run_tg(int argc, char** argv) {
    BenchOptions opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(stderr);
        return 2;
    }

    roctx_pause();

    hipError_t hip_err = hipSetDevice(opts.device);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    hipStream_t stream = nullptr;
    hip_err = hipStreamCreate(&stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipStreamCreate failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    std::size_t arena_bytes = 0;
    if (opts.arena_auto) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        hip_err = hipMemGetInfo(&free_bytes, &total_bytes);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipMemGetInfo failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }
        const std::size_t margin = 1ull * 1024ull * 1024ull * 1024ull;
        arena_bytes = free_bytes > margin ? free_bytes - margin : free_bytes / 2u;
        if (arena_bytes == 0) {
            fprintf(stderr, "no VRAM available for the arena\n");
            return 1;
        }
        fprintf(stderr, "tg arena auto: free=%zu capacity=%zu\n", free_bytes, arena_bytes);
    } else {
        arena_bytes = static_cast<std::size_t>(opts.arena_gib * 1024.0 * 1024.0 * 1024.0);
    }

    ps::models::ModelLoadContext source_context;
    source_context.qwen_model_dir = opts.model_dir;
    source_context.device = opts.device;
    source_context.model_host_socket = opts.model_host_socket;
    auto source_result = ps::models::ModelSource::create(source_context);
    if (!source_result.ok()) {
        fprintf(stderr, "ModelSource::create failed: %s\n",
            source_result.status().message().c_str());
        return 1;
    }
    std::unique_ptr<ps::models::ModelSource> source(source_result.release());

    const std::size_t resident = source->persistent_bytes();
    if (resident > 0 && arena_bytes <= resident) {
        fprintf(stderr, "resident model weights exceed the --arena-gib budget\n");
        return 1;
    }

    auto arena_result = ps::gpu::GpuArena::create(
        opts.device, source->ephemeral_bytes(arena_bytes));
    if (!arena_result.ok()) {
        fprintf(stderr, "GpuArena::create failed: %s\n", arena_result.status().message().c_str());
        return 1;
    }
    auto arena = arena_result.release();

    auto model_result = source->load_qwen35(arena, stream, {});
    if (!model_result.ok()) {
        fprintf(stderr, "model load failed: %s\n", model_result.status().message().c_str());
        return 1;
    }
    auto model = model_result.release();

    hip_err = hipStreamSynchronize(stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "weight sync failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    const auto& tc = model.text_config();
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);
    const uint32_t vocab_size = static_cast<uint32_t>(tc.vocab_size);
    const uint32_t hidden_size = static_cast<uint32_t>(tc.hidden_size);

    const uint32_t required_seq_len =
        static_cast<uint32_t>(opts.context)
        + static_cast<uint32_t>(opts.warmup)
        + static_cast<uint32_t>(opts.tokens)
        + 1u;

    const uint32_t num_pages = ceil_div(required_seq_len, static_cast<uint32_t>(opts.page_tokens)) + 1;

    auto seq_pool_result =
        ps::state::SequenceSlotPool::create(arena, 1, num_pages);
    if (!seq_pool_result.ok()) {
        fprintf(stderr, "SequenceSlotPool::create failed: %s\n",
            seq_pool_result.status().message().c_str());
        return 1;
    }
    auto seq_pool = seq_pool_result.release();

    auto rec_pool_result = ps::state::GdnStatePool::create(
        arena, 1, ps::qwen35::make_qwen35_gdn_state_layout(model.text_config()));
    if (!rec_pool_result.ok()) {
        fprintf(stderr, "GdnStatePool::create failed: %s\n",
            rec_pool_result.status().message().c_str());
        return 1;
    }
    auto rec_pool = rec_pool_result.release();

    auto kv_pool_result = ps::state::PagedKVPool::create(
        arena, num_pages, static_cast<uint32_t>(opts.page_tokens),
        num_attn_layers, kv_heads, head_dim, ps::kernel::KVCacheDType::BF16);
    if (!kv_pool_result.ok()) {
        fprintf(stderr, "PagedKVPool::create failed: %s\n",
            kv_pool_result.status().message().c_str());
        return 1;
    }
    auto kv_pool = kv_pool_result.release();

    ps::qwen35::ExecutorConfig exec_config;
    exec_config.max_scheduled_tokens =
        std::max(opts.prefill_chunk, opts.tokens + opts.warmup + 16);
    exec_config.max_scheduled_requests = 1;
    exec_config.backend = opts.decode_backend;

    auto exec_result = ps::qwen35::create_model_executor(
        model, seq_pool, rec_pool, kv_pool, arena, exec_config);
    if (!exec_result.ok()) {
        fprintf(stderr, "create_model_executor failed: %s\n",
            exec_result.status().message().c_str());
        return 1;
    }
    auto executor = exec_result.release();


    auto seq_result = ps::state::create_paged_sequence_state(
        seq_pool, rec_pool, kv_pool, required_seq_len, stream);
    if (!seq_result.ok()) {
        fprintf(stderr, "create_paged_sequence_state failed: %s\n", seq_result.status().message().c_str());
        return 1;
    }
    ps::state::PagedSequenceState seq = seq_result.release();

    int32_t token_id = 1;
    if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab_size) {
        fprintf(stderr, "token_id out of vocab range: vocab_size=%zu\n",
                static_cast<std::size_t>(vocab_size));
        return 1;
    }

    std::vector<int32_t> context_tokens(static_cast<std::size_t>(opts.context), token_id);
    if (!opts.tokens_file.empty()) {
        context_tokens = read_token_corpus(opts.tokens_file);
        if (context_tokens.empty()) {
            fprintf(stderr, "failed to read token corpus: %s\n", opts.tokens_file.c_str());
            return 1;
        }
        if (static_cast<std::size_t>(opts.tokens_offset) >= context_tokens.size()) {
            fprintf(stderr, "tokens-offset %d >= corpus size %zu\n",
                    opts.tokens_offset, context_tokens.size());
            return 1;
        }
        if (static_cast<std::size_t>(opts.tokens_offset) + static_cast<std::size_t>(opts.context)
            > context_tokens.size()) {
            fprintf(stderr, "tokens-offset %d + context %d exceeds corpus size %zu\n",
                    opts.tokens_offset, opts.context, context_tokens.size());
            return 1;
        }
        context_tokens.erase(
            context_tokens.begin(),
            context_tokens.begin() + static_cast<std::size_t>(opts.tokens_offset));
        context_tokens.resize(static_cast<std::size_t>(opts.context));
        for (int32_t t : context_tokens) {
            if (t < 0 || static_cast<std::size_t>(t) >= vocab_size) {
                fprintf(stderr, "corpus token %d out of vocab range: vocab_size=%zu\n",
                        t, static_cast<std::size_t>(vocab_size));
                return 1;
            }
        }
    }

        const uint32_t chunk = std::min<uint32_t>(
            static_cast<uint32_t>(opts.prefill_chunk), exec_config.max_scheduled_tokens);
        uint32_t processed = 0;
        while (processed < static_cast<uint32_t>(opts.context)) {
            const uint32_t chunk_len = std::min(chunk, static_cast<uint32_t>(opts.context) - processed);
            ps::qwen35::ScheduledRequest req;
            req.sequence = &seq; req.handle = seq.request_handle();
            req.execution_class = ps::runtime::ExecutionClass::PREFILL;
            req.token_begin = 0;
            req.num_tokens = chunk_len;
            req.prefix_tokens = seq.position;
            req.compute_logits = false;
            req.sample = false;
            std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
            ps::qwen35::ScheduledBatch batch;
            batch.token_ids = context_tokens.data() + processed;
            batch.requests = reqs;
            batch.num_tokens = chunk_len;
            batch.num_requests = 1;
            batch.num_decode_requests = 0;
            batch.num_verify_requests = 0;
            batch.num_prefill_requests = 1;
            auto st = ps::qwen35::execute_batch(executor, batch, stream);
            if (!st.ok()) {
                fprintf(stderr, "context population failed at %u: %s\n", processed, st.status().message().c_str());
                return 1;
            }
            (void)st.release();
            processed += chunk_len;
        }

    hip_err = hipStreamSynchronize(stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "context sync failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    auto run_decode_step = [&](bool compute_logits, bool sample) -> ps::Status {
        std::vector<int32_t> token_ids = {token_id};
        ps::qwen35::ScheduledRequest req;
        req.sequence = &seq; req.handle = seq.request_handle();
        req.execution_class = ps::runtime::ExecutionClass::DECODE;
        req.token_begin = 0;
        req.num_tokens = 1;
        req.prefix_tokens = seq.position;
        req.compute_logits = compute_logits;
        req.sample = sample;
        std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
        ps::qwen35::ScheduledBatch batch;
        batch.token_ids = token_ids.data();
        batch.requests = reqs;
        batch.num_tokens = 1;
        batch.num_requests = 1;
        batch.num_decode_requests = 1;
        batch.num_verify_requests = 0;
        batch.num_prefill_requests = 0;
        auto res = ps::qwen35::execute_batch(executor, batch, stream);
        if (!res.ok()) return res.status();
        (void)res.release();
        return ps::Status::make_ok();
    };

    for (int i = 0; i < opts.warmup; ++i) {
        auto st = run_decode_step(false, false);
        if (!st.ok()) {
            fprintf(stderr, "warmup forward failed at %d: %s\n", i, st.message().c_str());
            return 1;
        }
    }

    hip_err = hipStreamSynchronize(stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "warmup sync failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    hipEvent_t start_event = nullptr;
    hipEvent_t stop_event = nullptr;
    hip_err = hipEventCreate(&start_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventCreate failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }
    hip_err = hipEventCreate(&stop_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventCreate failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    roctx_resume();

    ps::qwen35::runtime::perf_stats_reset();
    if (ps::qwen35::runtime::semantic_timing_enabled())
        ps::qwen35::runtime::semantic_timing_begin(stream, 8192);
    roctx_range_push("PHASESHIFT_TG_MEASURED_DECODE");

    std::atomic<bool> keepalive_stop{false};
    hipStream_t keepalive_stream = nullptr;
    hipEvent_t keepalive_event = nullptr;
    std::thread keepalive_thread;
    if (opts.gap_keepalive > 0) {
        if (hipStreamCreate(&keepalive_stream) != hipSuccess) {
            fprintf(stderr, "keepalive hipStreamCreate failed\n");
            return 1;
        }
        const hipError_t ev_err = (opts.gap_keepalive_flags >= 0)
            ? hipEventCreateWithFlags(&keepalive_event,
                                      static_cast<unsigned>(opts.gap_keepalive_flags))
            : hipEventCreate(&keepalive_event);
        if (ev_err != hipSuccess) {
            fprintf(stderr, "keepalive event create failed: %s\n", hipGetErrorString(ev_err));
            return 1;
        }
        const int interval = opts.gap_keepalive;
        keepalive_thread = std::thread([&] {
            while (!keepalive_stop.load(std::memory_order_relaxed)) {
                const hipError_t ka_err = hipEventRecord(keepalive_event, keepalive_stream);
                (void)ka_err;
                std::this_thread::sleep_for(std::chrono::microseconds(interval));
            }
        });
    }

    auto wall_start = std::chrono::high_resolution_clock::now();
    hip_err = hipEventRecord(start_event, stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventRecord failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    if (opts.mode == "forward") {
        ps::qwen35::BatchExecutionOutput last_out;
        for (int i = 0; i < opts.tokens; ++i) {
            std::vector<int32_t> token_ids = {token_id};
            ps::qwen35::ScheduledRequest req;
            req.sequence = &seq; req.handle = seq.request_handle();
            req.execution_class = ps::runtime::ExecutionClass::DECODE;
            req.token_begin = 0;
            req.num_tokens = 1;
            req.prefix_tokens = seq.position;
            req.compute_logits = opts.compute_logits;
            req.sample = false;
            std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
            ps::qwen35::ScheduledBatch batch;
            batch.token_ids = token_ids.data();
            batch.requests = reqs;
            batch.num_tokens = 1;
            batch.num_requests = 1;
            batch.num_decode_requests = 1;
            batch.num_verify_requests = 0;
            batch.num_prefill_requests = 0;
            auto res = ps::qwen35::execute_batch(executor, batch, stream);
            if (!res.ok()) {
                fprintf(stderr, "forward failed at %d: %s\n", i, res.status().message().c_str());
                return 1;
            }
            last_out = std::move(res.release());
        }
        hip_err = hipEventRecord(stop_event, stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventRecord failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }
        hip_err = hipEventSynchronize(stop_event);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventSynchronize failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }
        if (opts.compute_logits && last_out.num_outputs == 1) {
            std::vector<float> logits(vocab_size);
            hip_err = hipMemcpy(logits.data(), last_out.logits.data<float>(),
                                vocab_size * sizeof(float), hipMemcpyDeviceToHost);
            if (hip_err == hipSuccess) {
                int argmax = 0;
                float maxv = logits[0];
                float sum = 0.0f;
                for (std::size_t v = 0; v < vocab_size; ++v) {
                    sum += logits[v];
                    if (logits[v] > maxv) { maxv = logits[v]; argmax = static_cast<int>(v); }
                }
                printf("LOGITS argmax=%d max=%.6f sum=%.3f\n", argmax, maxv, sum);
                if (!opts.output.empty()) {
                    FILE* f = std::fopen(opts.output.c_str(), "wb");
                    if (f != nullptr) {
                        std::fwrite(logits.data(), sizeof(float), vocab_size, f);
                        std::fclose(f);
                    }
                }
            }
        }
    } else {
        int32_t token_sum = 0;
        std::vector<int32_t> sampled_tokens;
        sampled_tokens.reserve(opts.tokens);
        for (int i = 0; i < opts.tokens; ++i) {
            std::vector<int32_t> token_ids = {token_id};
            ps::qwen35::ScheduledRequest req;
            req.sequence = &seq; req.handle = seq.request_handle();
            req.execution_class = ps::runtime::ExecutionClass::DECODE;
            req.token_begin = 0;
            req.num_tokens = 1;
            req.prefix_tokens = seq.position;
            req.compute_logits = true;
            req.sample = true;
            std::vector<ps::qwen35::ScheduledRequest> reqs = {req};
            ps::qwen35::ScheduledBatch batch;
            batch.token_ids = token_ids.data();
            batch.requests = reqs;
            batch.num_tokens = 1;
            batch.num_requests = 1;
            batch.num_decode_requests = 1;
            batch.num_verify_requests = 0;
            batch.num_prefill_requests = 0;
            auto res = ps::qwen35::execute_batch(executor, batch, stream);
            if (!res.ok()) {
                fprintf(stderr, "greedy forward failed at %d: %s\n", i, res.status().message().c_str());
                return 1;
            }
            ps::qwen35::BatchExecutionOutput out = std::move(res.release());
            if (out.num_outputs != 1) {
                fprintf(stderr, "greedy: expected 1 output, got %u\n", out.num_outputs);
                return 1;
            }
            int32_t sampled = -1;
            hip_err = hipMemcpy(&sampled, out.sampled_tokens.data<int32_t>(),
                                sizeof(int32_t), hipMemcpyDeviceToHost);
            if (hip_err != hipSuccess) {
                fprintf(stderr, "greedy sample copy failed at %d: %s\n", i, hipGetErrorString(hip_err));
                return 1;
            }
            hip_err = hipStreamSynchronize(stream);
            if (hip_err != hipSuccess) {
                fprintf(stderr, "greedy sync failed at %d: %s\n", i, hipGetErrorString(hip_err));
                return 1;
            }
            token_id = sampled;
            sampled_tokens.push_back(sampled);
            token_sum += sampled;
        }
        printf("GREEDY_TOKEN_SUM=%d\n", token_sum);
        printf("GREEDY_FIRST_TOKENS=");
        for (std::size_t i = 0; i < sampled_tokens.size() && i < 8; ++i) {
            if (i > 0) printf(",");
            printf("%d", sampled_tokens[i]);
        }
        printf("\n");
        hip_err = hipEventRecord(stop_event, stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventRecord failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }
        hip_err = hipEventSynchronize(stop_event);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventSynchronize failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }
    }

    auto wall_stop = std::chrono::high_resolution_clock::now();

    if (!opts.perf_stats_output.empty())
        ps::qwen35::runtime::perf_stats_dump(opts.perf_stats_output.c_str(),
                                             static_cast<uint32_t>(opts.tokens));
    if (ps::qwen35::runtime::semantic_timing_enabled()) {
        std::string sem_path = opts.perf_stats_output;
        const std::string::size_type slash = sem_path.find_last_of('/');
        sem_path = (slash != std::string::npos) ? sem_path.substr(0, slash + 1) + "semantic_timing.csv"
                                                 : "semantic_timing.csv";
        ps::qwen35::runtime::semantic_timing_end(stream, sem_path.c_str());
    }
    if (opts.gap_keepalive > 0) {
        keepalive_stop.store(true, std::memory_order_relaxed);
        keepalive_thread.join();
        hipError_t ka_err = hipStreamSynchronize(keepalive_stream);
        ka_err = hipEventDestroy(keepalive_event);
        ka_err = hipStreamDestroy(keepalive_stream);
        (void)ka_err;
    }

    roctx_range_pop();
    roctx_pause();

    float gpu_ms = 0.0f;
    hip_err = hipEventElapsedTime(&gpu_ms, start_event, stop_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventElapsedTime failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();

    const uint32_t measured_tokens = static_cast<uint32_t>(opts.tokens);
    const double gpu_us_per_token = gpu_ms * 1000.0 / static_cast<double>(measured_tokens);
    const double wall_us_per_token = wall_ms * 1000.0 / static_cast<double>(measured_tokens);
    const double gpu_tokens_per_sec =
        static_cast<double>(measured_tokens) / (gpu_ms / 1000.0);
    const double wall_tokens_per_sec =
        static_cast<double>(measured_tokens) / (wall_ms / 1000.0);

    const int final_position = opts.context + static_cast<int>(measured_tokens);
    const char* mode_label = opts.mode.c_str();

    const uint32_t kv_used_pages = seq.kv_pool()->num_used_pages();
    const std::size_t kv_reserved_bytes = seq.kv_pool()->reserved_bytes();

    const int roctx_enabled = roctx_available() ? 1 : 0;

    printf("PhaseShift Decode Benchmark\n");
    printf("\n");
#ifdef PHASESHIFT_GIT_SHA
    printf("git_sha=%s\n", PHASESHIFT_GIT_SHA);
#endif
    printf("device=%d\n", opts.device);
    printf("mode=%s\n", mode_label);
    printf("context=%d\n", opts.context);
    printf("tokens=%d\n", opts.tokens);
    printf("warmup=%d\n", opts.warmup);
    printf("compute_logits=%d\n", opts.compute_logits ? 1 : 0);
    printf("page_tokens=%d\n", opts.page_tokens);
    printf("\n");
    printf("gpu_ms=%.4f\n", gpu_ms);
    printf("wall_ms=%.4f\n", wall_ms);
    printf("gpu_us_per_token=%.2f\n", gpu_us_per_token);
    printf("wall_us_per_token=%.2f\n", wall_us_per_token);
    printf("gpu_tokens_per_sec=%.2f\n", gpu_tokens_per_sec);
    printf("wall_tokens_per_sec=%.2f\n", wall_tokens_per_sec);
    printf("\n");
    printf("final_position=%d\n", final_position);
    printf("kv_used_pages=%u\n", kv_used_pages);
    printf("kv_reserved_bytes=%zu\n", kv_reserved_bytes);
    printf("roctx=%s\n", roctx_enabled ? "enabled" : "disabled");

    if (!opts.output.empty() && opts.mode == "greedy") {
        FILE* out = fopen(opts.output.c_str(), "a");
        if (out == nullptr) {
            fprintf(stderr, "cannot open output file: %s\n", opts.output.c_str());
            return 1;
        }
        if (ftell(out) == 0) {
            fprintf(out,
                "schema_version,device,mode,context_start,measured_tokens,"
                "compute_logits,page_tokens,gpu_elapsed_ms,wall_elapsed_ms,"
                "gpu_us_per_token,wall_us_per_token,gpu_tokens_per_sec,"
                "wall_tokens_per_sec,final_position,kv_used_pages,"
                "kv_reserved_bytes,roctx_enabled\n");
        }
        fprintf(out,
            "2,%d,%s,%d,%d,%d,%d,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%d,%u,%zu,%d\n",
            opts.device,
            mode_label,
            opts.context,
            static_cast<int>(measured_tokens),
            opts.compute_logits ? 1 : 0,
            opts.page_tokens,
            gpu_ms,
            wall_ms,
            gpu_us_per_token,
            wall_us_per_token,
            gpu_tokens_per_sec,
            wall_tokens_per_sec,
            final_position,
            kv_used_pages,
            kv_reserved_bytes,
            roctx_enabled);
        fclose(out);
    }

    hip_err = hipEventDestroy(start_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventDestroy failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }
    hip_err = hipEventDestroy(stop_event);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipEventDestroy failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    auto release_st = ps::state::release_paged_sequence_state(seq, stream);
    if (!release_st.ok()) {
        fprintf(stderr, "release_paged_sequence_state failed: %s\n", release_st.message().c_str());
        return 1;
    }

    auto shutdown_st = arena.shutdown();
    if (!shutdown_st.ok()) {
        fprintf(stderr, "arena shutdown failed: %s\n", shutdown_st.message().c_str());
        return 1;
    }

    return 0;
}
