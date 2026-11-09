
#include <shelf/core/memory/arena.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35/runtime/decode_backend.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <optional>
#include <thread>

#ifdef SHELF_HAS_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>
#endif

namespace {

struct BenchOptions {
    int prompt_tokens = 128;
    int prefill_chunk = 0;
    int gap_keepalive = 0;
    int gap_keepalive_flags = hipEventDisableSystemFence;
    std::string mode = "forward";
    std::vector<int> devices = {0};
    int tp = 1;
    std::string model_dir;
    int page_tokens = 16;
    std::string kv_cache_dtype = "bf16";
    double arena_gib = 0.0;
    bool arena_auto = true;
    int runs = 3;
    int warmup = 0;
    std::string tokens_file;
    int tokens_offset = 0;
    std::string output;
    std::string perf_stats_output;
    shelf::qwen35::runtime::DecodeBackend decode_backend =
        shelf::qwen35::runtime::DecodeBackend::Hip;
};

void print_usage(FILE* out) {
    fprintf(out,
        "Usage: shelf-bench pp [options]\n"
        "  --prompt-tokens N            (default 128)\n"
        "  --prefill-chunk N            (default 0 = up to 8192 tokens per batch)\n"
        "  --gap-keepalive US           (default 0 = off; keep the CP awake)\n"
        "  --gap-keepalive-flags N      (default hipEventDisableSystemFence;\n"
        "                                -1 = hipEventCreate)\n"
        "  --mode forward|greedy        (default forward)\n"
        "  --device LIST                comma separated GPUs, one per --tensor-parallel rank\n"
        "                                (default 0; e.g. --device 0,2)\n"
        "  --tensor-parallel N          tensor parallel degree 1 | 2 | 4 (default 1)\n"
        "  --model-dir PATH             (required)\n"
        "  --page-tokens N              (default 16)\n"
        "  --kv-cache-dtype bf16|fp8    (default bf16)\n"
        "  --arena-gib N                (default auto = free VRAM minus 0.5 GiB)\n"
        "  --runs N                     (default 3)\n"
        "  --warmup N                   (default 0)\n"
        "  --tokens-file PATH           PSKLDTOK corpus file used as the prompt\n"
        "  --tokens-offset N            corpus position to start the prompt (default 0)\n"
        "  --perf-stats-output PATH     (default none; per-kernel dispatch counters JSON,\n"
        "                                needs SHELF_QWEN35_PERF_STATS=1)\n"
        "  --decode-backend TYPE        hip (default)\n"
        "  --output PATH                (default none)\n");
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

bool parse_devices(const std::string& value, std::vector<int>& out) {
    out.clear();
    size_t start = 0;
    while (start <= value.size()) {
        size_t comma = value.find(',', start);
        std::string token = value.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        int device = 0;
        if (!parse_int(token, device) || device < 0) return false;
        out.push_back(device);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return !out.empty();
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
        if (key == "--prompt-tokens") {
            if (!parse_int(need_value(), opts.prompt_tokens) || opts.prompt_tokens < 1) {
                fprintf(stderr, "invalid --prompt-tokens\n");
                return false;
            }
        } else if (key == "--prefill-chunk") {
            if (!parse_int(need_value(), opts.prefill_chunk) || opts.prefill_chunk < 0) {
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
        } else if (key == "--mode") {
            opts.mode = need_value();
        } else if (key == "--device") {
            if (!parse_devices(need_value(), opts.devices)) {
                fprintf(stderr, "invalid --device\n");
                return false;
            }
        } else if (key == "--tensor-parallel") {
            if (!parse_int(need_value(), opts.tp) ||
                (opts.tp != 1 && opts.tp != 2 && opts.tp != 4)) {
                fprintf(stderr, "--tensor-parallel must be 1, 2 or 4\n");
                return false;
            }
        } else if (key == "--model-dir") {
            opts.model_dir = need_value();
        } else if (key == "--kv-cache-dtype") {
            opts.kv_cache_dtype = need_value();
            if (opts.kv_cache_dtype != "bf16" && opts.kv_cache_dtype != "fp8") {
                fprintf(stderr, "invalid --kv-cache-dtype\n");
                return 1;
            }
        } else if (key == "--page-tokens") {
            if (!parse_int(need_value(), opts.page_tokens) || opts.page_tokens < 1) {
                fprintf(stderr, "invalid --page-tokens\n");
                return false;
            }
        } else if (key == "--decode-backend") {
            if (!shelf::qwen35::runtime::parse_decode_backend(
                    need_value().c_str(), &opts.decode_backend)) {
                fprintf(stderr, "invalid --decode-backend\n");
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
        } else if (key == "--runs") {
            if (!parse_int(need_value(), opts.runs) || opts.runs < 1) {
                fprintf(stderr, "invalid --runs\n");
                return false;
            }
        } else if (key == "--warmup") {
            if (!parse_int(need_value(), opts.warmup) || opts.warmup < 0) {
                fprintf(stderr, "invalid --warmup\n");
                return false;
            }
        } else if (key == "--tokens-file") {
            opts.tokens_file = need_value();
        } else if (key == "--tokens-offset") {
            if (!parse_int(need_value(), opts.tokens_offset) || opts.tokens_offset < 0) {
                fprintf(stderr, "invalid --tokens-offset\n");
                return false;
            }
        } else if (key == "--output") {
            opts.output = need_value();
        } else if (key == "--perf-stats-output") {
            opts.perf_stats_output = need_value();
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
    if (opts.model_dir.empty()) {
        fprintf(stderr, "--model-dir is required\n");
        return false;
    }
    if (opts.devices.size() != static_cast<size_t>(opts.tp)) {
        fprintf(stderr, "--device expects %d entries for --tensor-parallel %d, got %zu\n",
                opts.tp, opts.tp, opts.devices.size());
        return false;
    }
    std::vector<int> sorted_devices = opts.devices;
    std::sort(sorted_devices.begin(), sorted_devices.end());
    if (std::adjacent_find(sorted_devices.begin(), sorted_devices.end()) !=
        sorted_devices.end()) {
        fprintf(stderr, "--device entries must be unique\n");
        return false;
    }
    return true;
}

bool roctx_available() {
#ifdef SHELF_HAS_ROCTX
    return true;
#else
    return false;
#endif
}

void roctx_resume() {
#ifdef SHELF_HAS_ROCTX
    roctxProfilerResume(0);
#endif
}

void roctx_pause() {
#ifdef SHELF_HAS_ROCTX
    roctxProfilerPause(0);
#endif
}

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1) / b;
}

uint32_t count_attention_layers(const shelf::qwen_hybrid::QwenHybridConfig& tc) {
    return static_cast<uint32_t>(
        shelf::qwen_hybrid::count_layer_topology(tc).attention_layers);
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

int run_pp(int argc, char** argv) {
    BenchOptions opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(stderr);
        return 2;
    }

    roctx_pause();

    hipError_t hip_err = hipSetDevice(opts.devices[0]);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
        fprintf(stderr, "no GPU available\n");
        return 1;
    }
    for (int device : opts.devices) {
        if (device >= dev_count) {
            fprintf(stderr, "device %d out of range (visible devices: %d)\n",
                    device, dev_count);
            return 1;
        }
    }

    std::size_t arena_bytes = 0;
    if (opts.arena_auto) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        const hipError_t arena_err = hipMemGetInfo(&free_bytes, &total_bytes);
        if (arena_err != hipSuccess) {
            fprintf(stderr, "hipMemGetInfo failed: %s\n", hipGetErrorString(arena_err));
            return 1;
        }
        const std::size_t margin = 512ull * 1024ull * 1024ull;
        arena_bytes = free_bytes > margin ? free_bytes - margin : free_bytes / 2u;
        if (arena_bytes == 0) {
            fprintf(stderr, "no VRAM available for the arena\n");
            return 1;
        }
        fprintf(stderr, "pp arena auto: free=%zu capacity=%zu\n", free_bytes, arena_bytes);
    } else {
        arena_bytes = static_cast<std::size_t>(opts.arena_gib * 1024.0 * 1024.0 * 1024.0);
    }

    const uint32_t required_seq_len =
        static_cast<uint32_t>(opts.prompt_tokens) + 16;
    const uint32_t prefill_row_limit =
        shelf::runtime::row_bucket_limit(shelf::runtime::RowBucket::R8192);
    const uint32_t prompt_tokens = static_cast<uint32_t>(opts.prompt_tokens);
    const uint32_t exec_max_tokens =
        std::min(std::max(prompt_tokens, 16u), prefill_row_limit);
    if (opts.prefill_chunk == 0 && prompt_tokens > prefill_row_limit) {
        opts.prefill_chunk = static_cast<int>(prefill_row_limit);
    }

    hipStream_t stream = nullptr;
    std::unique_ptr<shelf::qwen35::runtime::TpCoordinator> tp_coordinator;
    std::optional<shelf::gpu::GpuArena> tp1_arena;
    std::optional<shelf::models::LoadedModel> tp1_loaded;
    std::optional<shelf::state::SequenceSlotPool> tp1_seq_pool;
    std::optional<shelf::state::GdnStatePool> tp1_rec_pool;
    std::optional<shelf::state::PagedKVPool> tp1_kv_pool;
    std::optional<shelf::qwen35::Executor> tp1_executor;

    if (opts.tp == 1) {
        hip_err = hipStreamCreate(&stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipStreamCreate failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }

        auto arena_result = shelf::gpu::GpuArena::create(opts.devices[0], arena_bytes);
        if (!arena_result.ok()) {
            fprintf(stderr, "GpuArena::create failed: %s\n", arena_result.status().message().c_str());
            return 1;
        }
        tp1_arena.emplace(arena_result.release());

        auto model_result = shelf::models::LoadedModel::load(
            opts.model_dir, *tp1_arena, stream);
        if (!model_result.ok()) {
            fprintf(stderr, "model load failed: %s\n", model_result.status().message().c_str());
            return 1;
        }
        tp1_loaded.emplace(model_result.release());

        const auto& cfg = tp1_loaded->hybrid_config();
        const uint32_t kv_heads = static_cast<uint32_t>(cfg.num_key_value_heads);
        const uint32_t head_dim = static_cast<uint32_t>(cfg.attention_head_dim);
        const uint32_t num_attn_layers = count_attention_layers(cfg);

        hip_err = hipStreamSynchronize(stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "weight sync failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }

        const uint32_t num_pages = ceil_div(required_seq_len, static_cast<uint32_t>(opts.page_tokens)) + 1;

        auto seq_pool_result =
            shelf::state::SequenceSlotPool::create(*tp1_arena, 1, num_pages);
        if (!seq_pool_result.ok()) {
            fprintf(stderr, "SequenceSlotPool::create failed: %s\n",
                seq_pool_result.status().message().c_str());
            return 1;
        }
        tp1_seq_pool.emplace(seq_pool_result.release());

        auto rec_pool_result = shelf::state::GdnStatePool::create(
            *tp1_arena, 1, shelf::qwen35::make_qwen35_gdn_state_layout(cfg));
        if (!rec_pool_result.ok()) {
            fprintf(stderr, "GdnStatePool::create failed: %s\n",
                rec_pool_result.status().message().c_str());
            return 1;
        }
        tp1_rec_pool.emplace(rec_pool_result.release());

        auto kv_pool_result = shelf::state::PagedKVPool::create(
            *tp1_arena, num_pages, static_cast<uint32_t>(opts.page_tokens),
            num_attn_layers, kv_heads, head_dim,
            opts.kv_cache_dtype == "fp8" ? shelf::kernel::KVCacheDType::FP8_E4M3
                                         : shelf::kernel::KVCacheDType::BF16);
        if (!kv_pool_result.ok()) {
            fprintf(stderr, "PagedKVPool::create failed: %s\n",
                kv_pool_result.status().message().c_str());
            return 1;
        }
        tp1_kv_pool.emplace(kv_pool_result.release());

        shelf::qwen35::ExecutorConfig exec_config;
        exec_config.max_batch_rows = exec_max_tokens;
        exec_config.max_scheduled_requests = 1;
        exec_config.backend = opts.decode_backend;

        auto exec_result = tp1_loaded->create_executor(
            *tp1_seq_pool, *tp1_rec_pool, *tp1_kv_pool, *tp1_arena, exec_config);
        if (!exec_result.ok()) {
            fprintf(stderr, "create_executor failed: %s\n", exec_result.status().message().c_str());
            return 1;
        }
        tp1_executor.emplace(exec_result.release());
    } else {
        shelf::qwen35::runtime::TpCoordinatorConfig tp_cfg;
        tp_cfg.model_dir = opts.model_dir;
        tp_cfg.devices = opts.devices;
        tp_cfg.max_batch_rows = exec_max_tokens;
        tp_cfg.max_scheduled_requests = 1;
        tp_cfg.max_seq_len = required_seq_len;
        tp_cfg.page_tokens = static_cast<uint32_t>(opts.page_tokens);
        tp_cfg.kv_cache_dtype = opts.kv_cache_dtype == "fp8"
                                    ? shelf::kernel::KVCacheDType::FP8_E4M3
                                    : shelf::kernel::KVCacheDType::BF16;
        tp_cfg.arena_bytes = arena_bytes;
        tp_cfg.working_set_reserve_bytes = 0;
        shelf::Status sel = shelf::models::select_tp_rank_model(tp_cfg);
        if (!sel.ok()) {
            fprintf(stderr, "tensor parallel rank model: %s\n", sel.message().c_str());
            return 1;
        }
        auto tp_result = shelf::qwen35::runtime::TpCoordinator::create(tp_cfg);
        if (!tp_result.ok()) {
            fprintf(stderr, "TpCoordinator::create failed: %s\n",
                tp_result.status().message().c_str());
            return 1;
        }
        tp_coordinator = tp_result.release();
        stream = tp_coordinator->rank(0).stream();
    }

    const std::size_t vocab_size =
        tp1_loaded
            ? tp1_loaded->hybrid_config().vocab_size
            : tp_coordinator->rank(0).hybrid_config().vocab_size;
    auto& seq_pool = tp_coordinator ? tp_coordinator->rank(0).sequence_pool() : *tp1_seq_pool;
    auto& rec_pool = tp_coordinator ? tp_coordinator->rank(0).gdn_pool() : *tp1_rec_pool;
    auto& kv_pool = tp_coordinator ? tp_coordinator->rank(0).kv_pool() : *tp1_kv_pool;
    auto& executor = tp_coordinator ? tp_coordinator->rank(0).executor() : *tp1_executor;

    const int32_t token_id = 1;
    if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab_size) {
        fprintf(stderr, "token_id out of vocab range: vocab_size=%zu\n", vocab_size);
        return 1;
    }

    int32_t* prompt_tokens_host = nullptr;
    hip_err = hipHostAlloc(&prompt_tokens_host, sizeof(int32_t) * required_seq_len, hipHostMallocMapped);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipHostAlloc failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }
    for (int i = 0; i < static_cast<int>(required_seq_len); ++i) {
        prompt_tokens_host[i] = token_id;
    }
    if (!opts.tokens_file.empty()) {
        const std::vector<int32_t> corpus = read_token_corpus(opts.tokens_file);
        if (corpus.empty()) {
            fprintf(stderr, "failed to read token corpus: %s\n", opts.tokens_file.c_str());
            return 1;
        }
        if (static_cast<std::size_t>(opts.tokens_offset) + prompt_tokens > corpus.size()) {
            fprintf(stderr, "tokens-offset %d + prompt %u exceeds corpus size %zu\n",
                    opts.tokens_offset, prompt_tokens, corpus.size());
            return 1;
        }
        for (uint32_t i = 0; i < prompt_tokens; ++i) {
            const int32_t token = corpus[static_cast<std::size_t>(opts.tokens_offset) + i];
            if (token < 0 || static_cast<std::size_t>(token) >= vocab_size) {
                fprintf(stderr, "corpus token %d out of vocab range: vocab_size=%zu\n",
                        token, vocab_size);
                return 1;
            }
            prompt_tokens_host[i] = token;
        }
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

    std::vector<double> gpu_times_ms;
    std::vector<double> wall_times_ms;
    gpu_times_ms.reserve(static_cast<size_t>(opts.runs));
    wall_times_ms.reserve(static_cast<size_t>(opts.runs));

    auto run_once = [&](int run, bool measured) -> int {
        auto seq_result = shelf::state::create_paged_sequence_state(
            seq_pool, rec_pool, kv_pool, required_seq_len, stream);
        if (!seq_result.ok()) {
            fprintf(stderr, "run %d create_paged_sequence_state failed: %s\n", run, seq_result.status().message().c_str());
            return 1;
        }
        shelf::state::PagedSequenceState seq = seq_result.release();

        if (tp_coordinator != nullptr) {
            auto mirror_st = tp_coordinator->on_sequence_created(seq);
            if (!mirror_st.ok()) {
                fprintf(stderr, "run %d tp sequence mirror create failed: %s\n", run,
                        mirror_st.message().c_str());
                return 1;
            }
        }

        const uint32_t total_tokens = static_cast<uint32_t>(opts.prompt_tokens);
        const uint32_t chunk = (opts.prefill_chunk > 0)
            ? std::min<uint32_t>(static_cast<uint32_t>(opts.prefill_chunk), total_tokens)
            : total_tokens;

        hip_err = hipStreamSynchronize(stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "run %d setup sync failed: %s\n", run, hipGetErrorString(hip_err));
            return 1;
        }

        roctx_resume();

        auto wall_start = std::chrono::high_resolution_clock::now();
        hip_err = hipEventRecord(start_event, stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventRecord failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }

        for (uint32_t processed = 0; processed < total_tokens;) {
            const uint32_t chunk_len = std::min(chunk, total_tokens - processed);
            const bool last = processed + chunk_len == total_tokens;
            shelf::qwen35::ScheduledRequest req;
            req.sequence = &seq; req.handle = seq.request_handle();
            req.execution_class = shelf::runtime::ExecutionClass::PREFILL;
            req.token_begin = 0;
            req.num_tokens = chunk_len;
            req.prefix_tokens = seq.position;
            req.compute_logits = last;
            req.sample = last && opts.mode == "greedy";
            std::vector<shelf::qwen35::ScheduledRequest> reqs = {req};
            shelf::qwen35::ScheduledBatch batch;
            batch.token_ids = prompt_tokens_host + processed;
            batch.requests = reqs;
            batch.num_tokens = chunk_len;
            batch.num_requests = 1;
            batch.num_decode_requests = 0;
            batch.num_verify_requests = 0;
            batch.num_prefill_requests = 1;
            auto st = tp_coordinator != nullptr
                          ? tp_coordinator->on_execute(batch)
                          : shelf::qwen35::execute_batch(executor, batch, stream);
            if (!st.ok()) {
                fprintf(stderr, "execute_batch failed at %u: %s\n", processed,
                        st.status().message().c_str());
                return 1;
            }
            shelf::qwen35::BatchExecutionOutput out = std::move(st.release());
            (void)out;
            processed += chunk_len;
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

        auto wall_stop = std::chrono::high_resolution_clock::now();

        roctx_pause();

        float gpu_ms = 0.0f;
        hip_err = hipEventElapsedTime(&gpu_ms, start_event, stop_event);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "hipEventElapsedTime failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }

        if (measured) {
            const double wall_ms =
                std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
            gpu_times_ms.push_back(static_cast<double>(gpu_ms));
            wall_times_ms.push_back(wall_ms);
        }

        if (tp_coordinator != nullptr) {
            auto mirror_st = tp_coordinator->on_sequence_released(seq);
            if (!mirror_st.ok()) {
                fprintf(stderr, "run %d tp sequence mirror release failed: %s\n", run,
                        mirror_st.message().c_str());
                return 1;
            }
        }
        auto release_st = shelf::state::release_paged_sequence_state(seq, stream);
        if (!release_st.ok()) {
            fprintf(stderr, "run %d release_paged_sequence_state failed: %s\n", run, release_st.message().c_str());
            return 1;
        }
        return 0;
    };

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

    for (int run = 0; run < opts.warmup; ++run) {
        if (run_once(run, false) != 0) return 1;
    }

    shelf::qwen35::runtime::perf_stats_reset();
    for (int run = 0; run < opts.runs; ++run) {
        if (run_once(run, true) != 0) return 1;
    }
    if (!opts.perf_stats_output.empty()) {
        shelf::qwen35::runtime::perf_stats_dump(opts.perf_stats_output.c_str(), 0u);
    }

    if (opts.gap_keepalive > 0) {
        keepalive_stop.store(true, std::memory_order_relaxed);
        keepalive_thread.join();
        hipError_t ka_err = hipStreamSynchronize(keepalive_stream);
        ka_err = hipEventDestroy(keepalive_event);
        ka_err = hipStreamDestroy(keepalive_stream);
        (void)ka_err;
    }

    hip_err = hipHostFree(prompt_tokens_host);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "hipHostFree failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }

    if (gpu_times_ms.empty()) {
        fprintf(stderr, "no valid measurements collected\n");
        return 1;
    }

    auto median = [](std::vector<double> v) -> double {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        if (n == 0) return 0.0;
        if (n % 2 == 1) return v[n / 2];
        return (v[n / 2 - 1] + v[n / 2]) * 0.5;
    };

    auto percentile = [](std::vector<double> v, double p) -> double {
        std::sort(v.begin(), v.end());
        const size_t n = v.size();
        if (n == 0) return 0.0;
        double idx = p * static_cast<double>(n - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = lo + 1 < n ? lo + 1 : lo;
        double frac = idx - static_cast<double>(lo);
        return v[lo] * (1.0 - frac) + v[hi] * frac;
    };

    auto stddev = [](const std::vector<double>& v, double mean) -> double {
        if (v.size() < 2) return 0.0;
        double acc = 0.0;
        for (double x : v) {
            double d = x - mean;
            acc += d * d;
        }
        return std::sqrt(acc / static_cast<double>(v.size() - 1));
    };

    const double median_gpu_ms = median(gpu_times_ms);
    const double median_wall_ms = median(wall_times_ms);
    const double min_gpu_ms = gpu_times_ms.empty() ? 0.0 : *std::min_element(gpu_times_ms.begin(), gpu_times_ms.end());
    const double p10_gpu_ms = percentile(gpu_times_ms, 0.10);
    const double p90_gpu_ms = percentile(gpu_times_ms, 0.90);
    const double stddev_gpu_ms = stddev(gpu_times_ms, median_gpu_ms);

    const double median_gpu_us_per_token = median_gpu_ms * 1000.0 / static_cast<double>(opts.prompt_tokens);
    const double median_wall_us_per_token = median_wall_ms * 1000.0 / static_cast<double>(opts.prompt_tokens);
    const double median_gpu_tokens_per_sec =
        static_cast<double>(opts.prompt_tokens) / (median_gpu_ms / 1000.0);
    const double median_wall_tokens_per_sec =
        static_cast<double>(opts.prompt_tokens) / (median_wall_ms / 1000.0);

    const std::size_t kv_reserved_bytes = kv_pool.reserved_bytes();
    const auto& rank0_arena = tp_coordinator ? tp_coordinator->rank(0).arena() : *tp1_arena;

    const int roctx_enabled = roctx_available() ? 1 : 0;

    printf("Shelf Prefill Benchmark\n");
    printf("\n");
    {
        std::string device_list;
        for (size_t i = 0; i < opts.devices.size(); ++i) {
            if (i > 0) device_list += ",";
            device_list += std::to_string(opts.devices[i]);
        }
        printf("device=%s\n", device_list.c_str());
    }
    printf("tp=%d\n", opts.tp);
    printf("mode=%s\n", opts.mode.c_str());
    printf("prompt_tokens=%d\n", opts.prompt_tokens);
    printf("page_tokens=%d\n", opts.page_tokens);
    printf("runs=%d\n", opts.runs);
    printf("warmup=%d\n", opts.warmup);
    printf("\n");
    printf("gpu_ms_median=%.4f\n", median_gpu_ms);
    printf("gpu_ms_min=%.4f\n", min_gpu_ms);
    printf("gpu_ms_p10=%.4f\n", p10_gpu_ms);
    printf("gpu_ms_p90=%.4f\n", p90_gpu_ms);
    printf("gpu_ms_stddev=%.4f\n", stddev_gpu_ms);
    printf("wall_ms_median=%.4f\n", median_wall_ms);
    printf("gpu_us_per_token=%.2f\n", median_gpu_us_per_token);
    printf("wall_us_per_token=%.2f\n", median_wall_us_per_token);
    printf("gpu_tokens_per_sec=%.2f\n", median_gpu_tokens_per_sec);
    printf("wall_tokens_per_sec=%.2f\n", median_wall_tokens_per_sec);
    printf("\n");
    printf("kv_reserved_bytes=%zu\n", kv_reserved_bytes);
    printf("execution_workspace_bytes=%zu\n", executor.execution_workspace.allocation_bytes());
    printf("rank0_arena_used_bytes=%zu\n", rank0_arena.used());
    printf("rank0_arena_committed_bytes=%zu\n", rank0_arena.committed());
    printf("roctx=%s\n", roctx_enabled ? "enabled" : "disabled");
    printf("\n");
    printf("run_detail:\n");
    for (int i = 0; i < opts.runs; ++i) {
        const double run_gpu_us_per_token =
            gpu_times_ms[i] * 1000.0 / static_cast<double>(opts.prompt_tokens);
        const double run_wall_us_per_token =
            wall_times_ms[i] * 1000.0 / static_cast<double>(opts.prompt_tokens);
        const double run_gpu_tokens_per_sec =
            static_cast<double>(opts.prompt_tokens) / (gpu_times_ms[i] / 1000.0);
        const double run_wall_tokens_per_sec =
            static_cast<double>(opts.prompt_tokens) / (wall_times_ms[i] / 1000.0);
        printf("  run=%d gpu_ms=%.4f wall_ms=%.4f gpu_us_per_token=%.2f wall_us_per_token=%.2f "
               "gpu_tokens_per_sec=%.2f wall_tokens_per_sec=%.2f\n",
               i, gpu_times_ms[i], wall_times_ms[i], run_gpu_us_per_token, run_wall_us_per_token,
               run_gpu_tokens_per_sec, run_wall_tokens_per_sec);
    }

    if (!opts.output.empty()) {
        FILE* out = fopen(opts.output.c_str(), "a");
        if (out == nullptr) {
            fprintf(stderr, "cannot open output file: %s\n", opts.output.c_str());
            return 1;
        }
        if (ftell(out) == 0) {
            fprintf(out,
                "schema_version,device,mode,prompt_tokens,page_tokens,runs,warmup,"
                "gpu_elapsed_ms_median,wall_elapsed_ms_median,"
                "gpu_ms_min,gpu_ms_p10,gpu_ms_p90,gpu_ms_stddev,"
                "gpu_us_per_token,wall_us_per_token,gpu_tokens_per_sec,"
                "wall_tokens_per_sec,kv_reserved_bytes,roctx_enabled,tp\n");
        }
        fprintf(out,
            "6,%d,%s,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%zu,%d,%d\n",
            opts.devices[0],
            opts.mode.c_str(),
            opts.prompt_tokens,
            opts.page_tokens,
            opts.runs,
            opts.warmup,
            median_gpu_ms,
            median_wall_ms,
            min_gpu_ms,
            p10_gpu_ms,
            p90_gpu_ms,
            stddev_gpu_ms,
            median_gpu_us_per_token,
            median_wall_us_per_token,
            median_gpu_tokens_per_sec,
            median_wall_tokens_per_sec,
            kv_reserved_bytes,
            roctx_enabled,
            opts.tp);
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

    if (tp_coordinator != nullptr) {
        for (int r = 0; r < tp_coordinator->tp_size(); ++r) {
            auto rank_shutdown = tp_coordinator->rank(r).arena().shutdown();
            if (!rank_shutdown.ok()) {
                fprintf(stderr, "rank %d arena shutdown failed: %s\n", r,
                        rank_shutdown.message().c_str());
                return 1;
            }
        }
    } else {
        auto shutdown_st = tp1_arena->shutdown();
        if (!shutdown_st.ok()) {
            fprintf(stderr, "arena shutdown failed: %s\n", shutdown_st.message().c_str());
            return 1;
        }
    }

    return 0;
}
