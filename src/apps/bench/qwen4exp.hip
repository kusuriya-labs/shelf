#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/models/qwen4exp/model/lower_to_primitives.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/model/qwen4exp_weights.h>
#include <shelf/models/qwen4exp/ple/ple_forward.h>
#include <shelf/models/qwen4exp/ple/ple_row_gather.h>
#include <shelf/models/qwen4exp/runtime/qwen4exp_state.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/program/program.h>
#include <shelf/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace q4e = ps::qwen4exp;
namespace rt = ps::runtime;
namespace state = ps::state;

int g_fail = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        std::printf("FAIL: %s\n", what);
        ++g_fail;
    }
}

struct Buf {
    void* dev = nullptr;
    std::size_t size = 0;
    bool alloc(std::size_t bytes) {
        size = bytes;
        return hipMalloc(&dev, bytes) == hipSuccess;
    }
    bool fill(const void* src, std::size_t bytes) {
        return dev != nullptr && hipMemcpy(dev, src, bytes, hipMemcpyHostToDevice) == hipSuccess;
    }
    ~Buf() {
        if (dev) (void)hipFree(dev);
    }
};

struct StageState {
    bool setup_done = false;
    bool pools_ready = false;
    bool seq_ready = false;
    int device = -1;
    hipStream_t stream = nullptr;
    ps::gpu::GpuArena* arena = nullptr;
    q4e::Qwen4ExpModelWeights weights;
    q4e::Qwen4ExpLoweredPrimitives lowered;
    rt::Program plan;
    rt::Program plan_decode;
    bool has_decode_plan = false;
    rt::Program plan_prefix;
    rt::Program plan_prefix_decode;
    bool has_prefix_decode_plan = false;
    std::optional<ps::weights::SafetensorsCollection> row_collection;
    std::optional<q4e::ple::PleBranch> ple_branch;
    std::optional<q4e::runtime::PleNgramContext> ngram;
    std::optional<q4e::runtime::PleConvStatePool> conv_pool;
    bool ple_ready = false;
    std::optional<state::SequenceSlotPool> seq_pool;
    std::optional<state::GdnStatePool> gdn_pool;
    std::optional<state::PagedKVPool> kv_pool;
    std::optional<state::PagedSequenceState> seq;
    std::uint32_t reserved_tokens = 0;

    void shutdown() {
        if (seq_ready && seq.has_value()) {
            if (ple_ready) {
                (void)ngram->release(seq->slot);
                (void)conv_pool->release(seq->slot);
            }
            (void)state::release_paged_sequence_state(*seq, stream);
            seq.reset();
        }
        seq_ready = false;
        if (ple_ready) {
            (void)ple_branch->shutdown();
            (void)ngram->shutdown();
            (void)conv_pool->shutdown();
            ple_ready = false;
        }
        row_collection.reset();
        if (stream != nullptr) {
            (void)hipStreamDestroy(stream);
            stream = nullptr;
        }
        pools_ready = false;
        seq_pool.reset();
        gdn_pool.reset();
        kv_pool.reset();
        setup_done = false;
        if (arena != nullptr) {
            arena->shutdown();
            delete arena;
            arena = nullptr;
        }
        device = -1;
        reserved_tokens = 0;
    }
};

void patch_external(rt::Program& plan, rt::ValueId id, void* dev_ptr) {
    for (auto& v : plan.values) {
        if (v.logical_value.id == id.id) v.offset = reinterpret_cast<uint64_t>(dev_ptr);
    }
}

std::uint32_t ceil_div(std::uint32_t a, std::uint32_t b) {
    return (a + b - 1u) / b;
}

struct Options {
    std::string model_dir;
    int device = 0;
    std::size_t arena_bytes = 0;
    std::uint32_t prompt_tokens = 32;
    std::uint32_t max_tokens = 128;
    std::uint32_t page_tokens = 16;
    bool skip_experts = false;
    bool connect_moe = true;
    bool connect_ple = true;
    std::uint32_t pp = 1;
    std::uint32_t decode_steps = 0;
};

void usage() {
    std::printf("usage: shelf-bench qwen4exp [options]\n");
    std::printf("  --model-dir PATH       model directory (required)\n");
    std::printf("  --device N             device index (default 0)\n");
    std::printf("  --arena-gib N          arena capacity in GiB (default 30)\n");
    std::printf("  --prompt-tokens N      prefill token count (default 32)\n");
    std::printf("  --max-tokens N         program max token rows (default 128)\n");
    std::printf("  --page-tokens N        KV page tokens (default 16)\n");
    std::printf("  --skip-experts         expert weights は読み込まない\n");
    std::printf("  --no-connect-moe       lowering を connect_moe=false で行う\n");
    std::printf("  --no-connect-ple       lowering を connect_ple=false で行う\n");
    std::printf("  --pp N                 pipeline parallel stages (1 or 4)\n");
    std::printf("  --decode-steps N       decode steps after prefill (default 0)\n");
    std::printf("  --help\n");
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto need = [&](int& idx) -> const char* {
            if (idx + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", k.c_str());
                return nullptr;
            }
            return argv[++idx];
        };
        if (k == "--help" || k == "-h") {
            usage();
            return false;
        } else if (k == "--model-dir") {
            const char* v = need(i);
            if (!v) return false;
            o.model_dir = v;
        } else if (k == "--device") {
            const char* v = need(i);
            if (!v) return false;
            o.device = std::atoi(v);
        } else if (k == "--arena-gib") {
            const char* v = need(i);
            if (!v) return false;
            char* end = nullptr;
            const double g = std::strtod(v, &end);
            if (end == v || g <= 0.0) {
                std::fprintf(stderr, "bad --arena-gib value: %s\n", v);
                return false;
            }
            o.arena_bytes = static_cast<std::size_t>(g * (1ull << 30));
        } else if (k == "--prompt-tokens") {
            const char* v = need(i);
            if (!v) return false;
            o.prompt_tokens = static_cast<std::uint32_t>(std::atoi(v));
        } else if (k == "--max-tokens") {
            const char* v = need(i);
            if (!v) return false;
            o.max_tokens = static_cast<std::uint32_t>(std::atoi(v));
        } else if (k == "--page-tokens") {
            const char* v = need(i);
            if (!v) return false;
            o.page_tokens = static_cast<std::uint32_t>(std::atoi(v));
        } else if (k == "--skip-experts") {
            o.skip_experts = true;
        } else if (k == "--no-connect-moe") {
            o.connect_moe = false;
        } else if (k == "--no-connect-ple") {
            o.connect_ple = false;
        } else if (k == "--pp") {
            const char* v = need(i);
            if (!v) return false;
            o.pp = static_cast<std::uint32_t>(std::atoi(v));
        } else if (k == "--decode-steps") {
            const char* v = need(i);
            if (!v) return false;
            o.decode_steps = static_cast<std::uint32_t>(std::atoi(v));
        } else {
            std::fprintf(stderr, "unknown option: %s\n", k.c_str());
            usage();
            return false;
        }
    }
    if (o.model_dir.empty()) {
        std::fprintf(stderr, "--model-dir is required\n");
        return false;
    }
    if (o.prompt_tokens == 0 || o.prompt_tokens > o.max_tokens) {
        std::fprintf(stderr, "--prompt-tokens must be in [1, --max-tokens]\n");
        return false;
    }
    if (o.arena_bytes == 0) o.arena_bytes = 30ull << 30;
    if (o.pp != 1u && o.pp != 4u) {
        std::fprintf(stderr, "--pp must be 1 or 4\n");
        return false;
    }
    return true;
}

std::uint32_t count_layer_type(const q4e::Qwen4ExpTextConfig& cfg, bool qsa,
                               std::uint32_t layer_begin = 0,
                               std::uint32_t layer_end = 0) {
    const std::size_t begin = layer_begin;
    const std::size_t end = layer_end == 0
                                ? cfg.num_hidden_layers
                                : (layer_end < cfg.num_hidden_layers
                                       ? layer_end
                                       : cfg.num_hidden_layers);
    std::uint32_t n = 0;
    for (std::size_t l = begin; l < end; ++l) {
        if (cfg.layer_is_qsa(l) == qsa) ++n;
    }
    return n;
}

bool read_f32(const void* dev, std::size_t n, std::vector<float>& out) {
    out.resize(n);
    return hipMemcpy(out.data(), dev, n * sizeof(float), hipMemcpyDeviceToHost) ==
           hipSuccess;
}

struct ProgramDevice {
    void* values = nullptr;
    void* weights = nullptr;
    void* params = nullptr;
    void* workspace = nullptr;
    void* states = nullptr;
    void* ranges = nullptr;
    std::vector<rt::DeviceWorkspaceRange> ws_ranges;
    rt::DeviceProgramView view{};

    ProgramDevice() = default;
    ProgramDevice(const ProgramDevice&) = delete;
    ProgramDevice& operator=(const ProgramDevice&) = delete;
    ~ProgramDevice() { free(); }

    void free() {
        void* bufs[] = {values, weights, params, workspace, states, ranges};
        for (void* q : bufs) {
            if (q != nullptr) (void)hipFree(q);
        }
        values = nullptr;
        weights = nullptr;
        params = nullptr;
        workspace = nullptr;
        states = nullptr;
        ranges = nullptr;
    }
};

bool upload_program_device(const rt::Program& plan,
                           const std::vector<rt::WeightSlot>& weights,
                           const std::vector<rt::StaticParameterSlot>& params,
                           ProgramDevice& out) {
    out.ws_ranges.resize(plan.workspace.ranges.size());
    for (std::size_t i = 0; i < plan.workspace.ranges.size(); ++i) {
        out.ws_ranges[i].offset = plan.workspace.ranges[i].offset;
        out.ws_ranges[i].bytes = plan.workspace.ranges[i].bytes;
    }
    std::vector<rt::DeviceValueBinding> vals(plan.values.size());
    for (std::size_t i = 0; i < plan.values.size(); ++i) {
        const auto& s = plan.values[i];
        auto& d = vals[i];
        d.logical_value = s.logical_value.id;
        d.storage = s.storage;
        d.dtype = s.dtype;
        d.row_domain = s.row_domain;
        d.slot = s.slot;
        d.feature_count = s.feature_count;
        d.row_stride = s.row_stride;
        d.offset = s.offset;
        d.bytes = s.bytes;
    }
    std::vector<rt::DeviceStateBinding> hs(plan.states.size());
    for (std::size_t i = 0; i < plan.states.size(); ++i) {
        hs[i].slot = plan.states[i].slot;
        hs[i].kind = static_cast<std::uint8_t>(plan.states[i].kind);
        hs[i].state_index = plan.states[i].state_index;
    }

    const auto alloc_upload = [&](void*& dst, const void* src, std::size_t bytes,
                                  const char* what) {
        if (bytes == 0) return true;
        if (hipMalloc(&dst, bytes) != hipSuccess) {
            std::fprintf(stderr, "device buffer allocation failed (%s)\n", what);
            return false;
        }
        if (hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            std::fprintf(stderr, "device binding upload failed (%s)\n", what);
            return false;
        }
        return true;
    };

    if (!alloc_upload(out.values, vals.data(), vals.size() * sizeof(rt::DeviceValueBinding),
                      "values") ||
        !alloc_upload(out.weights, weights.data(),
                      weights.size() * sizeof(rt::WeightSlot), "weights") ||
        !alloc_upload(out.params, params.data(),
                      params.size() * sizeof(rt::StaticParameterSlot), "params") ||
        !alloc_upload(out.states, hs.data(),
                      hs.size() * sizeof(rt::DeviceStateBinding), "states") ||
        !alloc_upload(out.ranges, out.ws_ranges.data(),
                      out.ws_ranges.size() * sizeof(rt::DeviceWorkspaceRange), "ranges")) {
        out.free();
        return false;
    }
    if (plan.workspace.total_bytes != 0) {
        if (hipMalloc(&out.workspace, plan.workspace.total_bytes) != hipSuccess) {
            std::fprintf(stderr, "device buffer allocation failed (workspace)\n");
            out.free();
            return false;
        }
    }

    out.view.values = static_cast<const rt::DeviceValueBinding*>(out.values);
    out.view.value_count = static_cast<std::uint32_t>(plan.values.size());
    out.view.weight_table = out.weights;
    out.view.weight_slot_count = static_cast<std::uint32_t>(weights.size());
    out.view.parameter_table = out.params;
    out.view.parameter_slot_count = static_cast<std::uint32_t>(params.size());
    out.view.workspace_base = reinterpret_cast<std::uint64_t>(out.workspace);
    out.view.workspace_ranges = static_cast<const rt::DeviceWorkspaceRange*>(out.ranges);
    out.view.workspace_range_count = static_cast<std::uint32_t>(out.ws_ranges.size());
    out.view.states = static_cast<const rt::DeviceStateBinding*>(out.states);
    out.view.state_count = static_cast<std::uint32_t>(plan.states.size());
    return true;
}

bool dispatch_plan(const rt::Program& plan, const rt::DeviceProgramView& view,
                   void* ctx_dev,
                   const ::ps::kernel::ModelDispatchStateView& state_view, void* scratch,
                   void* err, hipStream_t stream, const char* tag) {
    for (std::size_t di = 0; di < plan.dispatches.size(); ++di) {
        if (std::getenv("SHELF_Q4E_TRACE") != nullptr) {
            std::printf("    %s dispatch %zu %s\n", tag, di,
                        rt::kernel_to_name(plan.dispatches[di].kernel_id));
        }
        (void)hipMemsetAsync(err, 0, sizeof(std::uint32_t), stream);
        const ps::Status st = ::ps::kernel::launch_model_dispatch_correctness(
            view, plan.dispatches[di], reinterpret_cast<rt::DeviceBatchContext*>(ctx_dev),
            state_view, static_cast<float*>(scratch), static_cast<std::uint32_t*>(err),
            stream);
        if (!st.ok()) {
            std::fprintf(stderr, "%s dispatch %zu (%s) launch failed: %s\n", tag, di,
                         rt::kernel_to_name(plan.dispatches[di].kernel_id),
                         st.message().c_str());
            return false;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) {
            std::fprintf(stderr, "%s dispatch %zu sync failed (%s)\n", tag, di,
                         rt::kernel_to_name(plan.dispatches[di].kernel_id));
            return false;
        }
        std::uint32_t e = 0;
        if (hipMemcpy(&e, err, sizeof(e), hipMemcpyDeviceToHost) != hipSuccess) {
            return false;
        }
        if (e != 0u) {
            std::fprintf(stderr, "%s dispatch %zu (%s) status=%u\n", tag, di,
                         rt::kernel_to_name(plan.dispatches[di].kernel_id), e);
            return false;
        }
    }
    return true;
}

void check_state_bounds(const rt::Program& plan, std::uint32_t gdn_layers,
                        std::uint32_t qsa_layers, const char* tag) {
    for (const auto& s : plan.states) {
        const std::uint32_t limit = s.kind == rt::PrimitiveStateKind::KV_CACHE
                                        ? qsa_layers
                                        : gdn_layers;
        if (s.state_index >= limit) {
            std::printf("FAIL: %s state_index %u kind %s exceeds pool %u\n", tag,
                        s.state_index, rt::to_string(s.kind), limit);
            ++g_fail;
        }
    }
}

bool setup_ple(const Options& opts, const q4e::Qwen4ExpTextConfig& cfg, StageState& st) {
    const std::uint32_t ple_layer = static_cast<std::uint32_t>(cfg.ple_layer_index);
    if (ple_layer >= st.weights.layers.size() ||
        !st.weights.layers[ple_layer].ple.present) {
        std::fprintf(stderr, "PLE weights are not loaded\n");
        return false;
    }
    if (hipStreamSynchronize(st.stream) != hipSuccess) {
        std::fprintf(stderr, "weight load sync failed\n");
        return false;
    }
    auto collection = ps::weights::SafetensorsCollection::open(opts.model_dir);
    if (!collection.ok()) {
        std::fprintf(stderr, "SafetensorsCollection::open failed: %s\n",
                     collection.status().message().c_str());
        return false;
    }
    st.row_collection.emplace(collection.release());
    const std::string ple_prefix = "model.language_model.layers." +
                                   std::to_string(cfg.ple_layer_index) + ".ple.";
    q4e::Qwen4ExpPleWeights& ple_weights = st.weights.layers[ple_layer].ple;
    const auto load_matrix = [&](const char* name, ps::weights::MatrixWeight& dst) {
        auto r = ps::weights::load_bf16_matrix(*st.row_collection, ple_prefix + name,
                                               *st.arena, st.stream,
                                               ps::weights::WeightLoadOptions{});
        if (!r.ok()) {
            std::fprintf(stderr, "PLE %s load failed: %s\n", name,
                         r.status().message().c_str());
            return false;
        }
        dst = r.release();
        return true;
    };
    const auto load_vector = [&](const char* name, ps::gpu::Tensor& dst) {
        auto r = ps::weights::load_bf16_tensor(*st.row_collection, ple_prefix + name,
                                               *st.arena, st.stream);
        if (!r.ok()) {
            std::fprintf(stderr, "PLE %s load failed: %s\n", name,
                         r.status().message().c_str());
            return false;
        }
        dst = r.release();
        return true;
    };
    if (!load_matrix("key_proj.weight", ple_weights.key_proj) ||
        !load_matrix("value_proj.weight", ple_weights.value_proj) ||
        !load_vector("conv1d.weight", ple_weights.conv1d) ||
        !load_vector("norm_key.weight", ple_weights.norm_key) ||
        !load_vector("norm_query.weight", ple_weights.norm_query) ||
        !load_vector("norm_conv.weight", ple_weights.norm_conv)) {
        return false;
    }
    const std::string shard_prefix = ple_prefix +
                                     "ple_embedding.ngram_embedding.shard_";
    auto layout = q4e::ple::table_layout(*st.row_collection, shard_prefix);
    if (!layout.ok()) {
        std::fprintf(stderr, "PLE table_layout failed: %s\n",
                     layout.status().message().c_str());
        return false;
    }
    auto scale_layout =
        q4e::ple::table_layout(*st.row_collection, shard_prefix, ".weight_scale");
    if (!scale_layout.ok()) {
        std::fprintf(stderr, "PLE scale table_layout failed: %s\n",
                     scale_layout.status().message().c_str());
        return false;
    }
    q4e::ple::PleRowSource source;
    source.kind = q4e::ple::PleRowSource::Kind::ram;
    source.layout = layout.release();
    source.scale_layout = scale_layout.release();
    source.collection = &*st.row_collection;
    auto branch = q4e::ple::PleBranch::create(
        cfg, st.weights.layers[ple_layer].ple, opts.max_tokens, source, st.stream);
    if (!branch.ok()) {
        std::fprintf(stderr, "PleBranch::create failed: %s\n",
                     branch.status().message().c_str());
        return false;
    }
    st.ple_branch.emplace(branch.release());
    const std::uint32_t hc_hidden = static_cast<std::uint32_t>(cfg.hc_count) *
                                    static_cast<std::uint32_t>(cfg.hidden_size);
    auto ngram = q4e::runtime::PleNgramContext::create(*st.arena, 1,
                                                        st.ple_branch->eos_token());
    if (!ngram.ok()) {
        std::fprintf(stderr, "PleNgramContext::create failed: %s\n",
                     ngram.status().message().c_str());
        return false;
    }
    st.ngram.emplace(ngram.release());
    auto conv = q4e::runtime::PleConvStatePool::create(1, hc_hidden);
    if (!conv.ok()) {
        std::fprintf(stderr, "PleConvStatePool::create failed: %s\n",
                     conv.status().message().c_str());
        return false;
    }
    st.conv_pool.emplace(conv.release());
    st.ple_ready = true;
    std::printf("PLE ok: rows=%llu cols=%llu shards=%u rows_per_shard=%llu\n",
                static_cast<unsigned long long>(source.layout.shard_count) *
                    source.layout.rows_per_shard,
                static_cast<unsigned long long>(source.layout.row_cols),
                source.scale_layout.shard_count,
                static_cast<unsigned long long>(source.layout.rows_per_shard));
    return true;
}

}

int run_qwen4exp_stage(const Options& opts, const q4e::Qwen4ExpTextConfig& cfg,
                       std::uint32_t stage_begin, std::uint32_t stage_end,
                       void* hidden_in_dev, void* hidden_out_dev,
                       std::size_t* out_bytes, StageState& st,
                       rt::ExecutionClass cls, std::uint32_t row_count,
                       std::uint32_t pos_begin,
                       const std::vector<std::int32_t>* tokens_in,
                       std::int32_t* sampled_out) {
    const int device =
        std::getenv("SHELF_Q4E_SINGLE_DEVICE") != nullptr
            ? opts.device
            : (opts.pp == 1u ? opts.device
                             : static_cast<int>(stage_begin /
                                                (cfg.num_hidden_layers / opts.pp)));
    if (hipSetDevice(device) != hipSuccess) {
        std::fprintf(stderr, "hipSetDevice(%d) failed\n", device);
        return 1;
    }

    const std::uint32_t gdn_layers = count_layer_type(cfg, false, stage_begin, stage_end);
    const std::uint32_t qsa_layers = count_layer_type(cfg, true, stage_begin, stage_end);
    const std::uint32_t conv_kernel = static_cast<std::uint32_t>(cfg.linear_conv_kernel_dim);
    const std::uint32_t conv_dim = static_cast<std::uint32_t>(
        2u * cfg.linear_num_key_heads * cfg.linear_key_head_dim +
        cfg.linear_num_value_heads * cfg.linear_value_head_dim);
    const std::uint32_t conv_history = conv_kernel > 0 ? conv_kernel - 1u : 0u;
    const std::uint32_t val_heads = static_cast<std::uint32_t>(cfg.linear_num_value_heads);
    const std::uint32_t val_dim = static_cast<std::uint32_t>(cfg.linear_value_head_dim);
    const std::uint32_t kv_heads = static_cast<std::uint32_t>(cfg.num_key_value_heads);
    const std::uint32_t head_dim = static_cast<std::uint32_t>(cfg.head_dim);
    const std::uint32_t hidden = static_cast<std::uint32_t>(cfg.hidden_size);
    const std::uint32_t vocab = static_cast<std::uint32_t>(cfg.vocab_size);
    const std::uint32_t hc_hidden =
        static_cast<std::uint32_t>(cfg.hc_count) * hidden;

    std::printf("layers=%zu gdn=%u qsa=%u experts=%zu hidden=%u vocab=%u\n",
                cfg.num_hidden_layers, gdn_layers, qsa_layers, cfg.num_experts, hidden, vocab);
    std::printf("conv_dim=%u conv_history=%u val_heads=%u val_dim=%u kv_heads=%u head_dim=%u\n",
                conv_dim, conv_history, val_heads, val_dim, kv_heads, head_dim);

    if (st.stream == nullptr) {
        if (hipStreamCreate(&st.stream) != hipSuccess) return 1;
        st.device = device;
    }
    hipStream_t& stream = st.stream;

    if (st.arena == nullptr) {
        auto arena_result = ps::gpu::GpuArena::create(device, opts.arena_bytes);
        if (!arena_result.ok()) {
            std::fprintf(stderr, "GpuArena::create failed: %s\n",
                         arena_result.status().message().c_str());
            return 1;
        }
        st.arena = new ps::gpu::GpuArena(arena_result.release());
    }
    ps::gpu::GpuArena& arena = *st.arena;
    if (st.setup_done) {
        std::printf("  [stage] setup reused\n");
    }

    if (!st.setup_done) {
    const auto setup_start = std::chrono::steady_clock::now();
    q4e::Qwen4ExpLoadOptions load_opts;
    load_opts.load_expert_weights = !opts.skip_experts;
    load_opts.layer_begin = stage_begin;
    load_opts.layer_end = stage_end;
    const std::uint64_t used_before = [&]() {
        std::size_t f = 0, t = 0;
        (void)hipMemGetInfo(&f, &t);
        return static_cast<std::uint64_t>(t - f);
    }();
    auto weights_result =
        q4e::load_qwen4exp_weights_quantized(opts.model_dir, arena, stream, load_opts);
    if (!weights_result.ok()) {
        std::fprintf(stderr, "load_qwen4exp_weights_quantized failed: %s\n",
                     weights_result.status().message().c_str());
        return 1;
    }
    st.weights = weights_result.release();
    std::uint64_t used_after = 0;
    {
        std::size_t f = 0, t = 0;
        (void)hipMemGetInfo(&f, &t);
        used_after = static_cast<std::uint64_t>(t - f);
    }
    std::printf("arena used=%.3f GiB  device delta=%.3f GiB\n",
                static_cast<double>(arena.used()) / (1ull << 30),
                static_cast<double>(used_after - used_before) / (1ull << 30));
    std::printf("experts loaded=%d\n", load_opts.load_expert_weights ? 1 : 0);

    const bool effective_connect_moe = opts.connect_moe && !opts.skip_experts;
    q4e::Qwen4ExpLowerOptions lower_opts;
    lower_opts.connect_moe = effective_connect_moe;
    lower_opts.connect_ple = opts.connect_ple && stage_begin == 0u;
    lower_opts.stage_begin = stage_begin;
    lower_opts.stage_end = stage_end;
    auto lower_result = q4e::lower_qwen4exp_to_primitives(cfg, st.weights, lower_opts);
    if (!lower_result.ok()) {
        std::fprintf(stderr, "lowering failed: %s\n", lower_result.status().message().c_str());
        return 1;
    }
    st.lowered = lower_result.release();
    std::printf("lowering ok: connect_moe=%d connect_ple=%d nodes=%zu weights=%zu\n",
                effective_connect_moe ? 1 : 0, opts.connect_ple ? 1 : 0,
                st.lowered.graph.nodes.size(), st.lowered.weights.size());

    rt::ProgramBuildOptions build_opts;
    build_opts.validate_workspace = true;
    build_opts.no_workspace_reuse = true;
    build_opts.max_token_rows = row_count;
    build_opts.max_output_rows = 1;
    const rt::WeightTableView weight_view{st.lowered.weights.data(),
                                           st.lowered.weights.size()};
    auto bucket_result = rt::resolve_row_bucket(opts.max_tokens);
    if (!bucket_result.ok()) {
        std::fprintf(stderr, "resolve_row_bucket failed: %s\n",
                     bucket_result.status().message().c_str());
        return 1;
    }
    const rt::RowBucket bucket = bucket_result.release();
    auto built = rt::build_program(st.lowered.graph, weight_view, bucket,
                                   rt::ExecutionClass::PREFILL, build_opts);
    if (!built.ok()) {
        std::fprintf(stderr, "build_program failed: %s\n",
                     built.status().message().c_str());
        return 1;
    }
    st.plan = built.release();
    st.plan.parameter_slot_count = static_cast<uint32_t>(st.lowered.parameters.size());
    std::printf("build_program ok: values=%zu dispatches=%zu states=%zu workspace=%zu\n",
                st.plan.values.size(), st.plan.dispatches.size(), st.plan.states.size(),
                st.plan.workspace.total_bytes);

    if (st.lowered.has_ple_prefix) {
        const rt::WeightTableView pw{st.lowered.ple_prefix_weights.data(),
                                     st.lowered.ple_prefix_weights.size()};
        auto prefix_built = rt::build_program(st.lowered.ple_prefix_graph, pw, bucket,
                                              rt::ExecutionClass::PREFILL, build_opts);
        if (!prefix_built.ok()) {
            std::fprintf(stderr, "prefix build_program failed: %s\n",
                         prefix_built.status().message().c_str());
            return 1;
        }
        st.plan_prefix = prefix_built.release();
        st.plan_prefix.parameter_slot_count =
            static_cast<uint32_t>(st.lowered.ple_prefix_parameters.size());
        std::printf(
            "prefix build ok: values=%zu dispatches=%zu states=%zu workspace=%zu\n",
            st.plan_prefix.values.size(), st.plan_prefix.dispatches.size(),
            st.plan_prefix.states.size(), st.plan_prefix.workspace.total_bytes);
        if (!setup_ple(opts, cfg, st)) return 1;
    }
    st.setup_done = true;
    std::printf("  [stage] setup wall_ms=%.1f\n",
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - setup_start).count());
    }

    if (cls == rt::ExecutionClass::DECODE && !st.has_decode_plan) {
        const rt::WeightTableView dw{st.lowered.weights.data(),
                                     st.lowered.weights.size()};
        auto dbucket = rt::resolve_row_bucket(opts.max_tokens);
        if (!dbucket.ok()) {
            std::fprintf(stderr, "decode resolve_row_bucket failed: %s\n",
                         dbucket.status().message().c_str());
            return 1;
        }
        const rt::RowBucket bucket2 = dbucket.release();
        rt::ProgramBuildOptions db;
        db.validate_workspace = true;
        db.no_workspace_reuse = true;
        db.max_token_rows = row_count;
        db.max_output_rows = 1;
        auto dbuilt = rt::build_program(st.lowered.graph, dw, bucket2,
                                        rt::ExecutionClass::DECODE, db);
        if (!dbuilt.ok()) {
            std::fprintf(stderr, "decode build_program failed: %s\n",
                         dbuilt.status().message().c_str());
            return 1;
        }
        st.plan_decode = dbuilt.release();
        st.plan_decode.parameter_slot_count =
            static_cast<uint32_t>(st.lowered.parameters.size());
        st.has_decode_plan = true;
        std::printf("decode program ok: dispatches=%zu\n",
                    st.plan_decode.dispatches.size());
        if (st.lowered.has_ple_prefix) {
            const rt::WeightTableView pw{st.lowered.ple_prefix_weights.data(),
                                         st.lowered.ple_prefix_weights.size()};
            auto pbuilt = rt::build_program(st.lowered.ple_prefix_graph, pw, bucket2,
                                            rt::ExecutionClass::DECODE, db);
            if (!pbuilt.ok()) {
                std::fprintf(stderr, "decode prefix build_program failed: %s\n",
                             pbuilt.status().message().c_str());
                return 1;
            }
            st.plan_prefix_decode = pbuilt.release();
            st.plan_prefix_decode.parameter_slot_count =
                static_cast<uint32_t>(st.lowered.ple_prefix_parameters.size());
            st.has_prefix_decode_plan = true;
            std::printf("decode prefix program ok: dispatches=%zu\n",
                        st.plan_prefix_decode.dispatches.size());
        }
    }

    const q4e::Qwen4ExpModelWeights& weights = st.weights;
    const q4e::Qwen4ExpLoweredPrimitives& lowered = st.lowered;
    rt::Program& plan =
        (cls == rt::ExecutionClass::DECODE && st.has_decode_plan) ? st.plan_decode
                                                                  : st.plan;
    const bool use_prefix = lowered.has_ple_prefix;
    rt::Program& prefix_plan =
        (use_prefix && cls == rt::ExecutionClass::DECODE && st.has_prefix_decode_plan)
            ? st.plan_prefix_decode
            : st.plan_prefix;
    check(use_prefix == !st.plan_prefix.dispatches.empty(),
          "prefix program built iff has_ple_prefix");
    (void)weights;

    const std::uint32_t num_pages =
        ceil_div(opts.max_tokens * 2u + opts.prompt_tokens + 4u, opts.page_tokens) + 1u;
    if (!st.pools_ready) {
        auto seq_pool_result = state::SequenceSlotPool::create(arena, 1, num_pages);
        if (!seq_pool_result.ok()) {
            std::fprintf(stderr, "SequenceSlotPool::create failed: %s\n",
                         seq_pool_result.status().message().c_str());
            return 1;
        }
        st.seq_pool.emplace(seq_pool_result.release());

        const state::GdnStatePoolLayout gdn_layout = state::GdnStatePoolLayout::make(
            gdn_layers, conv_dim, conv_history, val_heads, val_dim, val_dim);
        auto gdn_result = state::GdnStatePool::create(arena, 1, gdn_layout);
        if (!gdn_result.ok()) {
            std::fprintf(stderr, "GdnStatePool::create failed: %s\n",
                         gdn_result.status().message().c_str());
            return 1;
        }
        st.gdn_pool.emplace(gdn_result.release());

        auto kv_result = state::PagedKVPool::create(
            arena, num_pages, opts.page_tokens, qsa_layers, kv_heads, head_dim,
            ps::kernel::KVCacheDType::BF16);
        if (!kv_result.ok()) {
            std::fprintf(stderr, "PagedKVPool::create failed: %s\n",
                         kv_result.status().message().c_str());
            return 1;
        }
        st.kv_pool.emplace(kv_result.release());
        st.pools_ready = true;
        std::printf("pools ok: pages=%u gdn_layers=%u qsa_layers=%u\n", num_pages,
                    gdn_layers, qsa_layers);
    }

    ::ps::kernel::ModelDispatchStateView state_view{};
    state_view.block_table = st.seq_pool->device_view();
    state_view.gdn = st.gdn_pool->device_view();
    state_view.kv_bf16 = state::make_paged_kv_bf16_view(*st.kv_pool);
    state_view.kv_dtype = 0;
    if (use_prefix) {
        check_state_bounds(prefix_plan, gdn_layers, qsa_layers, "prefix");
    }
    check_state_bounds(plan, gdn_layers, qsa_layers, "main");

    std::printf("  [stage] external ids resolved\n");
    const bool is_first = stage_begin == 0;
    const bool is_final = stage_end == cfg.num_hidden_layers;
    const rt::ValueId in_id = lowered.graph.external_inputs[0];
    const rt::ValueId ple_in_id =
        use_prefix ? lowered.graph.external_inputs[1] : rt::ValueId{UINT32_MAX};
    const rt::ValueId prefix_in_id =
        use_prefix ? lowered.ple_prefix_graph.external_inputs[0]
                   : rt::ValueId{UINT32_MAX};
    const rt::ValueId prefix_out_id =
        use_prefix ? lowered.ple_prefix_graph.external_outputs[0]
                   : rt::ValueId{UINT32_MAX};
    const rt::ValueId out0_id = lowered.graph.external_outputs[0];
    const rt::ValueId logits_id =
        is_final ? lowered.graph.external_outputs[1] : rt::ValueId{UINT32_MAX};
    const rt::ValueId sampled_id =
        is_final ? lowered.graph.external_outputs[2] : rt::ValueId{UINT32_MAX};
    const rt::ValueId final_hidden_id =
        is_final && lowered.graph.external_outputs.size() > 3
            ? lowered.graph.external_outputs[3]
            : rt::ValueId{UINT32_MAX};

    const rt::ValueBinding* out_spec = plan.find_value(out0_id);
    if (out_spec == nullptr) {
        std::fprintf(stderr, "stage output value not found\n");
        return 1;
    }
    const std::size_t out_feats = out_spec->feature_count;
    const std::size_t out_rows = out_spec->row_domain == rt::ValueRowDomain::OUTPUT_ROWS
                                     ? 1u
                                     : static_cast<std::size_t>(row_count);
    const bool out_rows_are_tokens =
        out_spec->row_domain != rt::ValueRowDomain::OUTPUT_ROWS;
    const std::size_t out_elements = out_rows * out_feats;

    Buf tok_dev;
    Buf hidden_dev;
    Buf logits_dev;
    Buf sampled_dev;
    Buf final_dev;
    Buf x4_dev;
    Buf ple_dev;
    if (!tok_dev.alloc(static_cast<std::size_t>(opts.max_tokens) * sizeof(std::int32_t)) ||
        (is_final && !hidden_dev.alloc(static_cast<std::size_t>(opts.max_tokens) * hidden *
                                       sizeof(float))) ||
        (is_final && !logits_dev.alloc(static_cast<std::size_t>(opts.max_tokens) * vocab *
                                       sizeof(float))) ||
        (is_final && !sampled_dev.alloc(static_cast<std::size_t>(opts.max_tokens) *
                                        sizeof(std::int32_t))) ||
        (is_final && !final_dev.alloc(static_cast<std::size_t>(opts.max_tokens) * hidden *
                                      sizeof(float))) ||
        (use_prefix && (!x4_dev.alloc(static_cast<std::size_t>(opts.max_tokens) *
                                      hc_hidden * sizeof(float)) ||
                        !ple_dev.alloc(static_cast<std::size_t>(opts.max_tokens) *
                                       hc_hidden * sizeof(float))))) {
        std::fprintf(stderr, "output buffer allocation failed\n");
        return 1;
    }
    std::printf("  [stage] buffers allocated\n");
    if (use_prefix) {
        patch_external(prefix_plan, prefix_in_id, tok_dev.dev);
        patch_external(prefix_plan, prefix_out_id, x4_dev.dev);
        patch_external(plan, in_id, x4_dev.dev);
        patch_external(plan, ple_in_id, ple_dev.dev);
    } else if (is_first) {
        patch_external(plan, in_id, tok_dev.dev);
    } else {
        patch_external(plan, in_id, hidden_in_dev);
    }
    if (is_final) {
        patch_external(plan, out0_id, hidden_dev.dev);
        patch_external(plan, logits_id, logits_dev.dev);
        patch_external(plan, sampled_id, sampled_dev.dev);
        patch_external(plan, final_hidden_id, final_dev.dev);
    } else {
        if (hidden_out_dev == nullptr) {
            std::fprintf(stderr, "hidden out buffer missing\n");
            return 1;
        }
        patch_external(plan, out0_id, hidden_out_dev);
        *out_bytes = static_cast<std::size_t>(opts.max_tokens) * out_feats *
                     sizeof(float);
    }

    Buf req_dev;
    Buf slots_dev;
    Buf pos_dev;
    Buf out_rows_dev;
    Buf samp_dev;
    if (!req_dev.alloc(sizeof(rt::DeviceRequestDescriptor)) ||
        !slots_dev.alloc(static_cast<std::size_t>(opts.max_tokens) * sizeof(std::uint32_t)) ||
        !pos_dev.alloc(static_cast<std::size_t>(opts.max_tokens) * sizeof(std::uint32_t)) ||
        !out_rows_dev.alloc(sizeof(std::uint32_t)) ||
        !samp_dev.alloc(sizeof(rt::DeviceSamplingParams))) {
        std::fprintf(stderr, "context buffer allocation failed\n");
        return 1;
    }
    rt::DeviceSamplingParams sampling{};
    sampling.temperature = 1.0f;
    sampling.top_p = 1.0f;
    sampling.top_k = 0;
    sampling.mode = 1;
    if (!samp_dev.fill(&sampling, sizeof(sampling))) {
        std::fprintf(stderr, "sampling params upload failed\n");
        return 1;
    }

    std::printf("  [stage] bindings built\n");
    if (!st.seq_ready) {
        auto seq_result = state::create_paged_sequence_state(
            *st.seq_pool, *st.gdn_pool, *st.kv_pool,
            opts.max_tokens * 2u + opts.prompt_tokens, stream);
        if (!seq_result.ok()) {
            std::fprintf(stderr, "sequence create failed: %s\n",
                         seq_result.status().message().c_str());
            return 1;
        }
        st.seq.emplace(seq_result.release());
        st.seq_ready = true;
        if (use_prefix) {
            if (!st.ngram->initialize(st.seq->slot).ok() ||
                !st.conv_pool->initialize(st.seq->slot).ok()) {
                std::fprintf(stderr, "PLE slot initialize failed\n");
                return 1;
            }
        }
    }
    state::PagedSequenceState& seq = *st.seq;
    const std::uint32_t needed_tokens = pos_begin + row_count;
    seq.position = st.reserved_tokens;
    if (needed_tokens > st.reserved_tokens) {
    const ps::Status reserve_st = state::reserve_sequence_append(
        seq, needed_tokens - st.reserved_tokens, stream);
    if (!reserve_st.ok()) {
        std::fprintf(stderr, "reserve_sequence_append failed: %s\n",
                     reserve_st.message().c_str());
        return 1;
    }
    st.reserved_tokens = needed_tokens;
    seq.position = needed_tokens;
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::fprintf(stderr, "reserve sync failed\n");
        return 1;
    }
    }

    std::vector<std::int32_t> tokens(
        tokens_in != nullptr ? tokens_in->size()
                             : static_cast<std::size_t>(row_count),
        tokens_in != nullptr ? 0 : 1);
    if (tokens_in != nullptr && tokens_in->size() == row_count) {
        tokens = *tokens_in;
    }
    if (tokens.size() < static_cast<std::size_t>(row_count)) {
        std::fprintf(stderr, "token count mismatch\n");
        return 1;
    }
    tokens[0] = 2;
    std::vector<std::uint32_t> slots(row_count, seq.slot);
    std::vector<std::uint32_t> positions(row_count, 0);
    for (std::uint32_t i = 0; i < row_count; ++i) positions[i] = pos_begin + i;

    if (!tok_dev.fill(tokens.data(), tokens.size() * sizeof(std::int32_t)) ||
        !slots_dev.fill(slots.data(), slots.size() * sizeof(std::uint32_t)) ||
        !pos_dev.fill(positions.data(), positions.size() * sizeof(std::uint32_t))) {
        std::fprintf(stderr, "context upload failed\n");
        return 1;
    }
    const std::uint32_t output_row =
        out_spec->row_domain == rt::ValueRowDomain::OUTPUT_ROWS ? 0u
                                                                : row_count - 1u;
    if (!out_rows_dev.fill(&output_row, sizeof(output_row))) {
        std::fprintf(stderr, "output row upload failed\n");
        return 1;
    }

    rt::DeviceRequestDescriptor req{};
    std::memset(&req, 0, sizeof(req));
    req.request_handle = seq.request_handle();
    req.row_begin = 0;
    req.row_count = row_count;
    req.prefix_length = pos_begin;
    req.sequence_length = pos_begin + row_count;
    req.execution_class = cls;
    req.compute_logits = 1;
    req.output_count = 1;
    if (!req_dev.fill(&req, sizeof(req))) {
        std::fprintf(stderr, "request upload failed\n");
        return 1;
    }

    rt::DeviceBatchContext ctx{};
    ctx.actual_rows = row_count;
    ctx.num_requests = 1;
    ctx.num_outputs = 1;
    ctx.num_prefill_requests = cls == rt::ExecutionClass::PREFILL ? 1u : 0u;
    ctx.num_decode_requests = cls == rt::ExecutionClass::DECODE ? 1u : 0u;
    ctx.token_ids = reinterpret_cast<const std::int32_t*>(tok_dev.dev);
    ctx.requests = reinterpret_cast<const rt::DeviceRequestDescriptor*>(req_dev.dev);
    ctx.row_sequence_slots = reinterpret_cast<std::uint32_t*>(slots_dev.dev);
    ctx.row_positions = reinterpret_cast<std::uint32_t*>(pos_dev.dev);
    ctx.output_rows = reinterpret_cast<std::uint32_t*>(out_rows_dev.dev);
    ctx.output_sampling_params = reinterpret_cast<rt::DeviceSamplingParams*>(samp_dev.dev);

    Buf ctx_buf;
    Buf scratch_buf;
    Buf err_buf;
    if (!ctx_buf.alloc(sizeof(ctx)) ||
        !scratch_buf.alloc(static_cast<std::size_t>(256) * 4096 * sizeof(float)) ||
        !err_buf.alloc(sizeof(std::uint32_t)) ||
        hipMemcpy(ctx_buf.dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) {
        std::fprintf(stderr, "device buffer allocation failed\n");
        return 1;
    }
    void* ctx_dev = ctx_buf.dev;
    void* scratch = scratch_buf.dev;
    void* err = err_buf.dev;

    ProgramDevice main_device;
    ProgramDevice prefix_device;
    if (!upload_program_device(plan, lowered.weights, lowered.parameters, main_device) ||
        (use_prefix && !upload_program_device(prefix_plan, lowered.ple_prefix_weights,
                                              lowered.ple_prefix_parameters,
                                              prefix_device))) {
        return 1;
    }

    std::printf("  [stage] sequence reserved\n");
    bool all_ok = true;
    if (use_prefix) {
        std::printf("  [stage] prefix dispatching %zu\n", prefix_plan.dispatches.size());
        all_ok = dispatch_plan(prefix_plan, prefix_device.view, ctx_dev, state_view,
                               scratch, err, stream, "prefix");
    }
    if (all_ok && use_prefix) {
        std::vector<float> x4_host;
        std::vector<std::uint32_t> tokens_u(row_count);
        if (!read_f32(x4_dev.dev, static_cast<std::size_t>(row_count) * hc_hidden,
                      x4_host)) {
            std::fprintf(stderr, "x4 readback failed\n");
            all_ok = false;
        } else {
            for (std::uint32_t i = 0; i < row_count; ++i) {
                tokens_u[i] = static_cast<std::uint32_t>(tokens[i]);
            }
            std::vector<float> ple_host(static_cast<std::size_t>(row_count) * hc_hidden);
            const ps::Status ple_st = st.ple_branch->run(
                tokens_u.data(), row_count, x4_host.data(), *st.ngram, *st.conv_pool,
                seq.slot, ple_host.data(), stream);
            if (!ple_st.ok()) {
                std::fprintf(stderr, "PLE branch failed: %s\n",
                             ple_st.message().c_str());
                all_ok = false;
            } else if (!ple_dev.fill(ple_host.data(),
                                     ple_host.size() * sizeof(float))) {
                std::fprintf(stderr, "PLE output upload failed\n");
                all_ok = false;
            } else {
                float x4_min = x4_host[0];
                float x4_max = x4_host[0];
                float ple_min = ple_host[0];
                float ple_max = ple_host[0];
                for (float v : x4_host) {
                    if (v < x4_min) x4_min = v;
                    if (v > x4_max) x4_max = v;
                }
                for (float v : ple_host) {
                    if (v < ple_min) ple_min = v;
                    if (v > ple_max) ple_max = v;
                }
                std::printf(
                    "  [stage] ple ok rows=%u x4[%.4g,%.4g] ple[%.4g,%.4g]\n", row_count,
                    x4_min, x4_max, ple_min, ple_max);
            }
        }
    }
    if (all_ok) {
        std::printf("  [stage] dispatching %zu\n", plan.dispatches.size());
        all_ok = dispatch_plan(plan, main_device.view, ctx_dev, state_view, scratch,
                               err, stream, "main");
    }

    if (all_ok && is_final && sampled_out != nullptr && sampled_dev.dev != nullptr) {
        if (hipMemcpy(sampled_out,
                      static_cast<const char*>(sampled_dev.dev) +
                          sizeof(std::int32_t) * output_row,
                      sizeof(std::int32_t), hipMemcpyDeviceToHost) != hipSuccess) {
            std::fprintf(stderr, "sampled readback failed\n");
            all_ok = false;
        }
    }
    if (all_ok && is_final) {
        std::vector<float> logits;
        const std::size_t last = static_cast<std::size_t>(output_row);
        if (read_f32(static_cast<const char*>(logits_dev.dev) + last * vocab * sizeof(float),
                     vocab, logits)) {
            std::int32_t argmax = 0;
            for (std::size_t i = 1; i < logits.size(); ++i) {
                if (logits[i] > logits[argmax]) argmax = static_cast<std::int32_t>(i);
            }
            float nz = 0.0f;
            float mn = logits.empty() ? 0.0f : logits[0];
            float mx = mn;
            for (float v : logits) {
                if (v != 0.0f) nz += 1.0f;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            const std::int32_t sampled = sampled_out != nullptr ? *sampled_out : -1;
            const float at_sampled =
                (sampled >= 0 && static_cast<std::size_t>(sampled) < logits.size())
                    ? logits[static_cast<std::size_t>(sampled)]
                    : -1.0f;
            std::printf(
                "STAGE_OK stage=[%u,%u) dispatches=%zu argmax=%d sampled=%d "
                "logits[nz=%d min=%.4g max=%.4g at_max=%.4g at_sampled=%.4g]\n",
                stage_begin, stage_end, plan.dispatches.size(), argmax, sampled,
                static_cast<int>(nz), mn, mx, logits.empty() ? 0.0f : logits[argmax],
                at_sampled);
        } else {
            std::fprintf(stderr, "logits readback failed\n");
            all_ok = false;
        }
    } else if (all_ok) {
        std::vector<float> probe;
        if (!read_f32(hidden_out_dev, out_elements, probe)) {
            std::fprintf(stderr, "hidden readback failed\n");
            all_ok = false;
        } else {
            float nz = 0.0f;
            for (float v : probe) nz += v != 0.0f ? 1.0f : 0.0f;
            std::printf("STAGE_OK stage=[%u,%u) dispatches=%zu hidden_nonzero=%d\n",
                        stage_begin, stage_end, plan.dispatches.size(),
                        nz > 0.0f ? 1 : 0);
        }
    }

    if (!all_ok || g_fail != 0) return 1;
    return 0;
}

int run_qwen4exp(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Options opts;
    if (!parse_args(argc, argv, opts)) return 2;

    std::printf("qwen4exp bench\n");
    std::printf("model: %s\n", opts.model_dir.c_str());

    auto config_result = q4e::load_qwen4exp_config(opts.model_dir);
    if (!config_result.ok()) {
        std::fprintf(stderr, "load config failed: %s\n",
                     config_result.status().message().c_str());
        return 1;
    }
    const q4e::Qwen4ExpTextConfig cfg = config_result.value();
    const std::uint32_t total = static_cast<std::uint32_t>(cfg.num_hidden_layers);
    std::printf("pp=%u layers=%zu gdn=%u qsa=%u experts=%zu hidden=%u vocab=%u\n",
                opts.pp, cfg.num_hidden_layers, count_layer_type(cfg, false),
                count_layer_type(cfg, true), cfg.num_experts,
                static_cast<std::uint32_t>(cfg.hidden_size),
                static_cast<std::uint32_t>(cfg.vocab_size));

    const std::uint32_t per_stage = total / opts.pp;
    const bool single = std::getenv("SHELF_Q4E_SINGLE_DEVICE") != nullptr;
    const auto stage_device = [&](std::uint32_t begin) {
        if (single || opts.pp == 1u) return opts.device;
        return static_cast<int>(begin / (total / opts.pp));
    };

    const std::size_t hidden_cap = static_cast<std::size_t>(opts.max_tokens) * 65536u *
                                   sizeof(float);
    void* pinned[2] = {nullptr, nullptr};
    for (int b = 0; b < 2; ++b) {
        if (hipHostMalloc(&pinned[b], hidden_cap,
                          hipHostMallocMapped | hipHostMallocPortable) != hipSuccess) {
            std::fprintf(stderr, "hipHostMalloc failed\n");
            return 1;
        }
    }
    std::printf("hidden hand-off: %zu x %zu bytes (pinned, mapped, no copy)\n", 2u,
                hidden_cap);

    StageState stages[4];
    int rc = 0;
    auto run_stages = [&](rt::ExecutionClass cls, std::uint32_t row_count,
                          std::uint32_t pos_begin,
                          const std::vector<std::int32_t>* tokens_in,
                          std::int32_t* sampled_out) -> int {
    for (std::uint32_t si = 0; si < opts.pp; ++si) {
        const std::uint32_t stage_begin = si * per_stage;
        const std::uint32_t stage_end =
            (si + 1u == opts.pp) ? total : stage_begin + per_stage;
        const int device = stage_device(stage_begin);
        if (hipSetDevice(device) != hipSuccess) {
            std::fprintf(stderr, "hipSetDevice(%d) failed\n", device);
            rc = 1;
            break;
        }
        void* out_view = nullptr;
        void* in_view = nullptr;
        if (si + 1u < opts.pp) {
            if (hipHostGetDevicePointer(&out_view, pinned[si % 2], 0) != hipSuccess) {
                std::fprintf(stderr, "hipHostGetDevicePointer(out) failed\n");
                rc = 1;
                break;
            }
        }
        if (si > 0) {
            if (hipHostGetDevicePointer(&in_view, pinned[(si - 1u) % 2], 0) !=
                hipSuccess) {
                std::fprintf(stderr, "hipHostGetDevicePointer(in) failed\n");
                rc = 1;
                break;
            }
        }
        std::printf("--- stage %u [%u,%u) device %d ---\n", si, stage_begin, stage_end,
                    device);
        std::size_t out_bytes = 0;
        const int stage_rc =
            run_qwen4exp_stage(opts, cfg, stage_begin, stage_end, in_view, out_view,
                               &out_bytes, stages[si], cls, row_count, pos_begin,
                               tokens_in, sampled_out);
        if (stage_rc != 0) {
            rc = 1;
            break;
        }
        if (si + 1u < opts.pp) {
            std::printf("hidden hand-off %u -> %u: %zu bytes in-place\n", si, si + 1u,
                        out_bytes);
        }
    }
    return rc;
    };

    std::vector<std::int32_t> prompt_tokens(opts.prompt_tokens, 1);
    prompt_tokens[0] = 2;
    std::int32_t sampled = -1;
    const auto ms_since = [](const std::chrono::steady_clock::time_point& t0) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
            .count();
    };
    std::printf("=== prefill: %u tokens ===\n", opts.prompt_tokens);
    const auto prefill_start = std::chrono::steady_clock::now();
    rc = run_stages(rt::ExecutionClass::PREFILL, opts.prompt_tokens, 0, &prompt_tokens,
                    &sampled);
    const double prefill_ms = ms_since(prefill_start);
    if (rc == 0) {
        std::printf("prefill sampled=%d wall_ms=%.1f tok/s=%.3f (includes setup)\n", sampled,
                    prefill_ms,
                    prefill_ms > 0.0 ? opts.prompt_tokens * 1000.0 / prefill_ms : 0.0);
        double decode_total_ms = 0.0;
        for (std::uint32_t step = 0; step < opts.decode_steps; ++step) {
            const std::vector<std::int32_t> next{sampled};
            std::printf("=== decode %u/%u ===\n", step + 1u, opts.decode_steps);
            const auto step_start = std::chrono::steady_clock::now();
            rc = run_stages(rt::ExecutionClass::DECODE, 1u,
                            opts.prompt_tokens + step, &next, &sampled);
            const double step_ms = ms_since(step_start);
            if (rc != 0) break;
            decode_total_ms += step_ms;
            std::printf("decode step %u sampled=%d wall_ms=%.1f tok/s=%.3f\n", step, sampled,
                        step_ms, step_ms > 0.0 ? 1000.0 / step_ms : 0.0);
        }
        if (rc == 0 && opts.decode_steps > 0 && decode_total_ms > 0.0) {
            std::printf("decode summary steps=%u avg_wall_ms=%.1f tok/s=%.3f\n",
                        opts.decode_steps, decode_total_ms / opts.decode_steps,
                        opts.decode_steps * 1000.0 / decode_total_ms);
        }
    }

    for (std::uint32_t si = 0; si < opts.pp; ++si) {
        (void)hipSetDevice(stages[si].device);
        stages[si].shutdown();
    }
    for (int b = 0; b < 2; ++b) {
        if (pinned[b] != nullptr) (void)hipHostFree(pinned[b]);
    }

    if (rc != 0 || g_fail != 0) {
        std::printf("=== qwen4exp bench: FAIL ===\n");
        return 1;
    }
    std::printf("=== qwen4exp bench: PASS ===\n");
    return 0;
}

