#include "quality_core.h"

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/weights/model_weights.h>
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

namespace {

struct QualityOptions {
    std::string model_dir;
    std::string tokens_file;
    int context = 128;
    int score_start = 0;
    int positions = 0;
    int prefill_chunk = 2048;
    int page_tokens = 16;
    std::vector<int> devices = {0};
    int tp = 1;
    double arena_gib = 0.0;
    bool arena_auto = true;
    shelf::kernel::KVCacheDType kv_dtype = shelf::kernel::KVCacheDType::BF16;
    std::string dump_path;
    std::string compare_path;
};

void print_usage(FILE* out) {
    fprintf(out,
        "Usage: shelf-bench quality [options]\n"
        "  --model-dir PATH             (required)\n"
        "  --tokens-file PATH           PSKLDTOK corpus (required)\n"
        "  --context N                  tokens pre-filled before scoring (default 128)\n"
        "  --score-start-position N     first scored token position, 1-based absolute\n"
        "                                (default context+1; must be > context)\n"
        "  --positions N                scored positions (default 0 = to end, max 2048)\n"
        "  --prefill-chunk N            prefill chunk size (default 2048)\n"
        "  --kv-cache-dtype bf16|fp8    (default bf16)\n"
        "  --page-tokens N              (default 16)\n"
        "  --arena-gib N|auto           (default auto = free VRAM minus 1 GiB)\n"
        "  --device LIST                comma separated GPUs, one per --tensor-parallel rank\n"
        "                                (default 0; e.g. --device 0,2)\n"
        "  --tensor-parallel N          1 | 2 | 4 (default 1)\n"
        "  --dump PATH                  write teacher-forced logits rows\n"
        "  --compare PATH               compare against a baseline dump\n");
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

bool parse_args(int argc, char** argv, QualityOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key = arg;
        std::string value;
        bool has_value = false;
        const size_t eq = arg.find('=');
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
            return std::string(argv[++i]);
        };
        if (key == "--model-dir") {
            opts.model_dir = need_value();
        } else if (key == "--tokens-file") {
            opts.tokens_file = need_value();
        } else if (key == "--context") {
            if (!parse_int(need_value(), opts.context) || opts.context < 1) {
                fprintf(stderr, "invalid --context\n");
                return false;
            }
        } else if (key == "--score-start-position") {
            if (!parse_int(need_value(), opts.score_start) || opts.score_start < 0) {
                fprintf(stderr, "invalid --score-start-position\n");
                return false;
            }
        } else if (key == "--positions") {
            if (!parse_int(need_value(), opts.positions) || opts.positions < 0) {
                fprintf(stderr, "invalid --positions\n");
                return false;
            }
        } else if (key == "--prefill-chunk") {
            if (!parse_int(need_value(), opts.prefill_chunk) || opts.prefill_chunk < 1) {
                fprintf(stderr, "invalid --prefill-chunk\n");
                return false;
            }
        } else if (key == "--page-tokens") {
            if (!parse_int(need_value(), opts.page_tokens) || opts.page_tokens < 1) {
                fprintf(stderr, "invalid --page-tokens\n");
                return false;
            }
        } else if (key == "--kv-cache-dtype") {
            const std::string dtype = need_value();
            if (dtype == "bf16") {
                opts.kv_dtype = shelf::kernel::KVCacheDType::BF16;
            } else if (dtype == "fp8") {
                opts.kv_dtype = shelf::kernel::KVCacheDType::FP8_E4M3;
            } else {
                fprintf(stderr, "invalid --kv-cache-dtype: %s\n", dtype.c_str());
                return false;
            }
        } else if (key == "--device") {
            if (!parse_devices(need_value(), opts.devices)) {
                fprintf(stderr, "invalid --device\n");
                return false;
            }
        } else if (key == "--tensor-parallel") {
            if (!parse_int(need_value(), opts.tp) ||
                (opts.tp != 1 && opts.tp != 2 && opts.tp != 4)) {
                fprintf(stderr, "invalid --tensor-parallel (expected 1, 2 or 4)\n");
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
        } else if (key == "--dump") {
            opts.dump_path = need_value();
        } else if (key == "--compare") {
            opts.compare_path = need_value();
        } else if (key == "--help" || key == "-h") {
            print_usage(stdout);
            exit(0);
        } else {
            fprintf(stderr, "unknown argument: %s\n", key.c_str());
            return false;
        }
    }
    if (opts.model_dir.empty()) {
        fprintf(stderr, "model-dir required\n");
        return false;
    }
    if (opts.tokens_file.empty()) {
        fprintf(stderr, "tokens-file required\n");
        return false;
    }
    if (opts.dump_path.empty() && opts.compare_path.empty()) {
        fprintf(stderr, "either --dump or --compare is required\n");
        return false;
    }
    if (opts.score_start != 0 && opts.score_start <= opts.context) {
        fprintf(stderr, "--score-start-position must be greater than --context\n");
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

uint32_t ceil_div(uint32_t a, uint32_t b) {
    return (a + b - 1u) / b;
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

shelf::bench::quality::KvDtype to_quality_dtype(shelf::kernel::KVCacheDType dtype) {
    return dtype == shelf::kernel::KVCacheDType::FP8_E4M3
               ? shelf::bench::quality::KvDtype::Fp8E4M3
               : shelf::bench::quality::KvDtype::Bf16;
}

}

int run_quality(int argc, char** argv) {
    QualityOptions opts;
    if (!parse_args(argc, argv, opts)) {
        print_usage(stderr);
        return 2;
    }

    hipError_t hip_err = hipSetDevice(opts.devices[0]);
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
        fprintf(stderr, "quality arena auto: free=%zu capacity=%zu\n", free_bytes, arena_bytes);
    } else {
        arena_bytes = static_cast<std::size_t>(opts.arena_gib * 1024.0 * 1024.0 * 1024.0);
    }

    std::vector<int32_t> corpus = read_token_corpus(opts.tokens_file);
    if (corpus.empty()) {
        fprintf(stderr, "failed to read token corpus: %s\n", opts.tokens_file.c_str());
        return 1;
    }

    const uint32_t context = static_cast<uint32_t>(opts.context);
    const uint32_t score_start =
        opts.score_start == 0 ? context + 1u : static_cast<uint32_t>(opts.score_start);
    if (score_start <= context) {
        fprintf(stderr, "score start %u must be greater than context %u\n", score_start, context);
        return 1;
    }
    const uint32_t max_scorable =
        corpus.size() > static_cast<std::size_t>(score_start)
            ? static_cast<uint32_t>(corpus.size() - static_cast<std::size_t>(score_start))
            : 0u;
    if (max_scorable == 0u) {
        fprintf(stderr, "corpus too short for --context %d and --score-start-position %u\n",
                opts.context, score_start);
        return 1;
    }
    uint32_t positions = opts.positions == 0 ? max_scorable : std::min<uint32_t>(
        static_cast<uint32_t>(opts.positions), max_scorable);
    positions = std::min<uint32_t>(positions, 2048u);
    if (static_cast<uint64_t>(score_start) + positions >
        static_cast<uint64_t>(corpus.size())) {
        fprintf(stderr, "corpus too short for score start %u and %u positions\n",
                score_start, positions);
        return 1;
    }
    const uint32_t required_seq_len = score_start + positions;
    const uint32_t num_pages =
        ceil_div(required_seq_len, static_cast<uint32_t>(opts.page_tokens)) + 1u;

    shelf::qwen35::ExecutorConfig exec_config;
    exec_config.max_batch_rows = static_cast<uint32_t>(
        std::max(opts.prefill_chunk, 2));
    exec_config.max_scheduled_requests = 1u;
    exec_config.max_scheduled_output_rows = 1u;

    std::unique_ptr<shelf::qwen35::runtime::TpCoordinator> tp_coord;
    std::unique_ptr<shelf::gpu::GpuArena> tp1_arena;
    std::unique_ptr<shelf::models::LoadedModel> tp1_loaded;
    std::unique_ptr<shelf::state::SequenceSlotPool> tp1_seq_pool;
    std::unique_ptr<shelf::state::GdnStatePool> tp1_gdn_pool;
    std::unique_ptr<shelf::state::PagedKVPool> tp1_kv_pool;
    std::unique_ptr<shelf::qwen35::Executor> tp1_executor;
    shelf::state::SequenceSlotPool* seq_pool = nullptr;
    shelf::state::GdnStatePool* gdn_pool = nullptr;
    shelf::state::PagedKVPool* kv_pool = nullptr;
    shelf::qwen35::Executor* executor = nullptr;

    if (opts.tp == 1) {
        auto arena_result = shelf::gpu::GpuArena::create(opts.devices[0], arena_bytes);
        if (!arena_result.ok()) {
            fprintf(stderr, "GpuArena::create failed: %s\n",
                    arena_result.status().message().c_str());
            return 1;
        }
        tp1_arena = std::make_unique<shelf::gpu::GpuArena>(arena_result.release());

        auto model_result =
            shelf::models::LoadedModel::load(opts.model_dir, *tp1_arena, stream);
        if (!model_result.ok()) {
            fprintf(stderr, "model load failed: %s\n", model_result.status().message().c_str());
            return 1;
        }
        tp1_loaded = std::make_unique<shelf::models::LoadedModel>(model_result.release());

        hip_err = hipStreamSynchronize(stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "weight sync failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }
    } else {
        shelf::qwen35::runtime::TpCoordinatorConfig tp_cfg;
        tp_cfg.model_dir = opts.model_dir;
        tp_cfg.devices = opts.devices;
        tp_cfg.max_batch_rows = exec_config.max_batch_rows;
        tp_cfg.max_scheduled_requests = exec_config.max_scheduled_requests;
        tp_cfg.max_scheduled_output_rows = exec_config.max_scheduled_output_rows;
        tp_cfg.max_seq_len = required_seq_len;
        tp_cfg.page_tokens = static_cast<uint32_t>(opts.page_tokens);
        tp_cfg.kv_cache_dtype = opts.kv_dtype;
        tp_cfg.arena_bytes = arena_bytes;
        tp_cfg.working_set_reserve_bytes = 1ull * 1024ull * 1024ull * 1024ull;
        shelf::Status sel = shelf::models::select_tp_rank_model(tp_cfg);
        if (!sel.ok()) {
            fprintf(stderr, "tensor parallel rank model: %s\n", sel.message().c_str());
            return 1;
        }
        auto coord_result = shelf::qwen35::runtime::TpCoordinator::create(tp_cfg);
        if (!coord_result.ok()) {
            fprintf(stderr, "TpCoordinator::create failed: %s\n",
                    coord_result.status().message().c_str());
            return 1;
        }
        tp_coord = coord_result.release();
        seq_pool = &tp_coord->rank(0).sequence_pool();
        gdn_pool = &tp_coord->rank(0).gdn_pool();
        kv_pool = &tp_coord->rank(0).kv_pool();
        executor = &tp_coord->rank(0).executor();
        fprintf(stderr, "quality tp coordinator: ranks=%d transport=%s\n",
                tp_coord->tp_size(), tp_coord->transport_name());
    }

    shelf::qwen_hybrid::QwenHybridConfig tc;
    if (tp1_loaded != nullptr)
        tc = tp1_loaded->hybrid_config();
    else
        tc = tp_coord->rank(0).hybrid_config();
    const uint32_t vocab = static_cast<uint32_t>(tc.vocab_size);
    const uint32_t kv_heads = static_cast<uint32_t>(tc.num_key_value_heads);
    const uint32_t head_dim = static_cast<uint32_t>(tc.attention_head_dim);
    const uint32_t num_attn_layers = count_attention_layers(tc);

    for (int32_t token : corpus) {
        if (token < 0 || static_cast<uint32_t>(token) >= vocab) {
            fprintf(stderr, "corpus token %d out of vocab range: vocab=%u\n", token, vocab);
            return 1;
        }
    }

    if (opts.tp == 1) {
        auto seq_pool_result = shelf::state::SequenceSlotPool::create(*tp1_arena, 1u, num_pages);
        if (!seq_pool_result.ok()) {
            fprintf(stderr, "SequenceSlotPool::create failed: %s\n",
                    seq_pool_result.status().message().c_str());
            return 1;
        }
        tp1_seq_pool = std::make_unique<shelf::state::SequenceSlotPool>(seq_pool_result.release());
        seq_pool = tp1_seq_pool.get();

        auto gdn_pool_result = shelf::state::GdnStatePool::create(
            *tp1_arena, 1u, shelf::qwen35::make_qwen35_gdn_state_layout(tc));
        if (!gdn_pool_result.ok()) {
            fprintf(stderr, "GdnStatePool::create failed: %s\n",
                    gdn_pool_result.status().message().c_str());
            return 1;
        }
        tp1_gdn_pool = std::make_unique<shelf::state::GdnStatePool>(gdn_pool_result.release());
        gdn_pool = tp1_gdn_pool.get();

        auto kv_pool_result = shelf::state::PagedKVPool::create(
            *tp1_arena, num_pages, static_cast<uint32_t>(opts.page_tokens), num_attn_layers,
            kv_heads, head_dim, opts.kv_dtype);
        if (!kv_pool_result.ok()) {
            fprintf(stderr, "PagedKVPool::create failed: %s\n",
                    kv_pool_result.status().message().c_str());
            return 1;
        }
        tp1_kv_pool = std::make_unique<shelf::state::PagedKVPool>(kv_pool_result.release());
        kv_pool = tp1_kv_pool.get();

        auto exec_result = tp1_loaded->create_executor(
            *seq_pool, *gdn_pool, *kv_pool, *tp1_arena, exec_config);
        if (!exec_result.ok()) {
            fprintf(stderr, "create_model_executor failed: %s\n",
                    exec_result.status().message().c_str());
            return 1;
        }
        tp1_executor = std::make_unique<shelf::qwen35::Executor>(exec_result.release());
        executor = tp1_executor.get();
    }

    auto seq_result = shelf::state::create_paged_sequence_state(
        *seq_pool, *gdn_pool, *kv_pool, required_seq_len, stream);
    if (!seq_result.ok()) {
        fprintf(stderr, "create_paged_sequence_state failed: %s\n",
                seq_result.status().message().c_str());
        return 1;
    }
    shelf::state::PagedSequenceState seq = seq_result.release();
    if (tp_coord != nullptr) {
        shelf::Status mirror_status = tp_coord->on_sequence_created(seq);
        if (!mirror_status.ok()) {
            fprintf(stderr, "tp sequence mirror failed: %s\n",
                    mirror_status.message().c_str());
            return 1;
        }
    }

    const uint32_t chunk = std::min<uint32_t>(
        static_cast<uint32_t>(opts.prefill_chunk), exec_config.max_batch_rows);
    uint32_t processed = 0u;
    while (processed < context) {
        const uint32_t chunk_len = std::min(chunk, context - processed);
        shelf::qwen35::ScheduledRequest req;
        req.sequence = &seq;
        req.handle = seq.request_handle();
        req.execution_class = shelf::runtime::ExecutionClass::PREFILL;
        req.token_begin = 0;
        req.num_tokens = chunk_len;
        req.prefix_tokens = seq.position;
        req.compute_logits = tp_coord != nullptr;
        req.sample = false;
        std::vector<shelf::qwen35::ScheduledRequest> reqs = {req};
        shelf::qwen35::ScheduledBatch batch;
        batch.token_ids = corpus.data() + processed;
        batch.requests = reqs;
        batch.num_tokens = chunk_len;
        batch.num_requests = 1u;
        batch.num_decode_requests = 0u;
        batch.num_verify_requests = 0u;
        batch.num_prefill_requests = 1u;
        shelf::Result<shelf::qwen35::BatchExecutionOutput> exec_out{shelf::Status::invalid_state(
            "unbound batch execution", __FILE__, __LINE__)};
        if (tp_coord != nullptr) {
            exec_out = tp_coord->on_execute(batch);
        } else {
            exec_out = shelf::qwen35::execute_batch(*executor, batch, stream);
        }
        if (!exec_out.ok()) {
            fprintf(stderr, "context prefill failed at %u: %s\n", processed,
                    exec_out.status().message().c_str());
            return 1;
        }
        (void)exec_out.release();
        processed += chunk_len;
    }
    hip_err = hipStreamSynchronize(stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "context sync failed: %s\n", hipGetErrorString(hip_err));
        return 1;
    }
    if (tp_coord != nullptr) {
        hip_err = hipStreamSynchronize(tp_coord->rank(0).stream());
        if (hip_err != hipSuccess) {
            fprintf(stderr, "tp prefill sync failed: %s\n", hipGetErrorString(hip_err));
            return 1;
        }
    }

    shelf::bench::quality::DumpWriter writer;
    shelf::bench::quality::DumpReader reader;
    shelf::bench::quality::Metrics metrics;
    if (!opts.dump_path.empty()) {
        if (!writer.open(opts.dump_path, vocab, to_quality_dtype(opts.kv_dtype), score_start)) {
            fprintf(stderr, "cannot open --dump path: %s\n", opts.dump_path.c_str());
            return 1;
        }
    }
    if (!opts.compare_path.empty()) {
        std::string error;
        if (!reader.open(opts.compare_path, error)) {
            fprintf(stderr, "cannot open --compare path: %s (%s)\n",
                    opts.compare_path.c_str(), error.c_str());
            return 1;
        }
        if (reader.header().vocab != vocab) {
            fprintf(stderr, "baseline vocab %u does not match model vocab %u\n",
                    reader.header().vocab, vocab);
            return 1;
        }
        if (reader.header().positions != positions) {
            fprintf(stderr, "baseline positions %llu does not match requested %u\n",
                    static_cast<unsigned long long>(reader.header().positions), positions);
            return 1;
        }
        if (reader.header().reserved != 0u && reader.header().reserved != score_start) {
            fprintf(stderr, "baseline score start %u does not match requested %u\n",
                    reader.header().reserved, score_start);
            return 1;
        }
        if (reader.header().reserved == 0u) {
            fprintf(stderr, "baseline dump has no score start metadata; assuming %u\n",
                    score_start);
        }
    }

    std::vector<float> host_logits(vocab);
    std::vector<float> baseline_logits;
    const uint32_t feed_begin = context;
    const uint32_t feed_end = score_start + positions - 1u;
    for (uint32_t feed_index = feed_begin; feed_index < feed_end; ++feed_index) {
        const bool scoring = (feed_index + 1u) >= score_start;
        const int32_t feed_token = corpus[feed_index];
        const int32_t target = corpus[feed_index + 1u];
        std::vector<int32_t> token_ids = {feed_token};
        shelf::qwen35::ScheduledRequest req;
        req.sequence = &seq;
        req.handle = seq.request_handle();
        req.execution_class = shelf::runtime::ExecutionClass::DECODE;
        req.token_begin = 0;
        req.num_tokens = 1u;
        req.prefix_tokens = seq.position;
        req.compute_logits = true;
        req.sample = false;
        std::vector<shelf::qwen35::ScheduledRequest> reqs = {req};
        shelf::qwen35::ScheduledBatch batch;
        batch.token_ids = token_ids.data();
        batch.requests = reqs;
        batch.num_tokens = 1u;
        batch.num_requests = 1u;
        batch.num_decode_requests = 1u;
        batch.num_verify_requests = 0u;
        batch.num_prefill_requests = 0u;
        shelf::Result<shelf::qwen35::BatchExecutionOutput> result{shelf::Status::invalid_state(
            "unbound batch execution", __FILE__, __LINE__)};
        if (tp_coord != nullptr) {
            result = tp_coord->on_execute(batch);
        } else {
            result = shelf::qwen35::execute_batch(*executor, batch, stream);
        }
        if (!result.ok()) {
            fprintf(stderr, "teacher-forced step %u failed: %s\n", feed_index,
                    result.status().message().c_str());
            return 1;
        }
        shelf::qwen35::BatchExecutionOutput out = std::move(result.release());
        if (out.num_outputs != 1u) {
            fprintf(stderr, "teacher-forced step %u produced %u outputs\n", feed_index,
                    out.num_outputs);
            return 1;
        }
        const auto& logits = out.logits;
        if (logits.ndim() != 2u || logits.dim(1) != vocab ||
            logits.element_size() != sizeof(float)) {
            fprintf(stderr,
                    "unexpected logits shape ndim=%zu dim0=%zu dim1=%zu elem=%zu\n",
                    logits.ndim(), logits.ndim() > 0u ? logits.dim(0) : 0u,
                    logits.ndim() > 1u ? logits.dim(1) : 0u, logits.element_size());
            return 1;
        }
        if (!scoring) continue;
        hip_err = hipStreamSynchronize(stream);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "logits sync failed at %u: %s\n", feed_index,
                    hipGetErrorString(hip_err));
            return 1;
        }
        if (tp_coord != nullptr) {
            hip_err = hipStreamSynchronize(tp_coord->rank(0).stream());
            if (hip_err != hipSuccess) {
                fprintf(stderr, "tp logits sync failed at %u: %s\n", feed_index,
                        hipGetErrorString(hip_err));
                return 1;
            }
        }
        hip_err = hipMemcpy(host_logits.data(), logits.data<float>(), vocab * sizeof(float),
                            hipMemcpyDeviceToHost);
        if (hip_err != hipSuccess) {
            fprintf(stderr, "logits copy failed at %u: %s\n", feed_index,
                    hipGetErrorString(hip_err));
            return 1;
        }
        const uint32_t position = feed_index + 1u;
        if (!opts.dump_path.empty() && !writer.write(target, position, host_logits.data())) {
            fprintf(stderr, "dump write failed at %u\n", feed_index);
            return 1;
        }
        if (!opts.compare_path.empty()) {
            int32_t baseline_target = 0;
            uint32_t baseline_position = 0u;
            std::string error;
            if (!reader.read(baseline_logits, baseline_target, baseline_position, error)) {
                fprintf(stderr, "baseline read failed at %u: %s\n", feed_index, error.c_str());
                return 1;
            }
            if (baseline_target != target || baseline_position != position) {
                fprintf(stderr,
                        "baseline row %u mismatch target=%d/%d position=%u/%u\n",
                        feed_index, baseline_target, target, baseline_position, position);
                return 1;
            }
            shelf::bench::quality::accumulate_row(
                baseline_logits.data(), host_logits.data(), vocab, target, metrics);
        }
    }

    if (!opts.dump_path.empty() && !writer.close()) {
        fprintf(stderr, "dump close failed\n");
        return 1;
    }
    if (!opts.compare_path.empty() && !reader.close()) {
        fprintf(stderr, "baseline close failed\n");
        return 1;
    }

    printf("QUALITY_POSITIONS=%u\n", positions);
    printf("QUALITY_CONTEXT=%u\n", context);
    printf("QUALITY_SCORE_START=%u\n", score_start);
    printf("QUALITY_VOCAB=%u\n", vocab);
    printf("QUALITY_KV_DTYPE=%s\n",
           opts.kv_dtype == shelf::kernel::KVCacheDType::FP8_E4M3 ? "fp8" : "bf16");
    if (!opts.dump_path.empty()) {
        printf("QUALITY_DUMP=%s\n", opts.dump_path.c_str());
        printf("QUALITY_DUMP_POSITIONS=%llu\n",
               static_cast<unsigned long long>(writer.positions()));
    }
    if (!opts.compare_path.empty()) {
        const double base_ppl = shelf::bench::quality::perplexity(
            metrics.nll_base, metrics.positions);
        const double candidate_ppl = shelf::bench::quality::perplexity(
            metrics.nll_candidate, metrics.positions);
        const double delta_pct = base_ppl == 0.0
            ? 0.0
            : (candidate_ppl - base_ppl) / base_ppl * 100.0;
        printf("QUALITY_BASELINE=%s\n", opts.compare_path.c_str());
        printf("QUALITY_PPL_BASELINE=%.9g\n", base_ppl);
        printf("QUALITY_PPL_CANDIDATE=%.9g\n", candidate_ppl);
        printf("QUALITY_PPL_DELTA_PCT=%.9g\n", delta_pct);
        printf("QUALITY_NLL_DELTA_MEAN=%.9g\n",
               metrics.positions == 0u ? 0.0
                   : (metrics.nll_candidate - metrics.nll_base) /
                         static_cast<double>(metrics.positions));
        printf("QUALITY_KL_MEAN=%.9g\n", shelf::bench::quality::mean_kl(metrics));
        printf("QUALITY_TOP1_AGREEMENT=%.9g\n",
               shelf::bench::quality::top1_agreement(metrics));
        printf("QUALITY_TOP5_OVERLAP=%.9g\n",
               shelf::bench::quality::top5_overlap(metrics));
        printf("QUALITY_LOGIT_MAX_ABS_DELTA=%.9g\n", metrics.max_abs_delta);
    }
    fflush(stdout);
    return 0;
}
