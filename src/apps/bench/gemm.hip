#include <shelf/core/gpu/cleanup.h>
#include <shelf/kernels/correctness/standalone/gemm_bf16_correctness.h>
#include <shelf/kernels/optimized/linear/fp8.h>
#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/models/qwen35/kernels/dflash2/swiglu.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/program/linear_quant_contract.h>
#include <shelf/runtime/program/activation_workspace_layout.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/runtime/selector_types.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>

namespace {

struct Options {
    std::string dtype = "bf16";
    uint32_t mxfp4_split = 4u;
    uint32_t mxfp4_unroll = 0u;
    uint32_t mxfp4_fold = 1u;
    uint32_t mxfp4_a_rowref = 0u;
    uint32_t mxfp4_a_fold = 0u;

    uint32_t mxfp4_spread = 24u;
    std::string variant = "auto";
    std::vector<uint32_t> rows;
    uint32_t n = 5120;
    uint32_t k = 5120;
    int warmup = 100;
    int samples = 50;
    int launches = 20;
    uint32_t seed = 1234;
    std::string out;
    int device = 0;
    bool check = false;
    std::string model_dir;
    bool actual_qwen_shapes = false;
    std::string impl = "auto";
    std::string shapes_spec;
    bool verify_small_m = false;
    uint32_t split = 1;
};

void usage(FILE* f) {
    fprintf(f,
        "shelf-bench gemm [options]\n"
        "\n"
        "  --dtype bf16|fp8              (logical weight/compute family, default bf16)\n"
        "  --variant correctness|gemm|auto\n"
        "      correctness: production correctness path (default auto)\n"
        "      gemm:        force Big-M GEMM compute kind\n"
        "      auto:        production linear selector\n"
        "  --rows 1,2,4,...               (comma list, default 1,16,32,64,128)\n"
        "  --n N                          (default 5120)\n"
        "  --k N                          (default 5120)\n"
        "  --shapes N:K,N:K,...           (explicit shape list)\n"
        "  --impl auto|wmma|wide|splitk|exact   (bf16 forced implementation)\n"
        "  --impl auto|splitk|rowblock1|2|4      (mxfp4 forced implementation)\n"
        "  --mxfp4-split N                  mxfp4 split-K ways (default 8)\n"
        "  --mxfp4-fold 0|1                fold the row E8M0 reference into the operand\n"
        "  --mxfp4-spread N                mxfp4 synthetic E8M0 exponent spread (default 24)\n"
        "  --mxfp4-a-rowref 0|1            A plane in the row-ref folded form (MXFP4 W4A8)\n"
        "  --mxfp4-a-fold 0|1              time the A-plane E8M0 fold pass (arm C/D cost)\n"
        "  --mxfp4-a-fold uses one E8M0 byte per 32 K (arm C plane; arm D replicates)\n"


        "  --model-dir PATH               (with --actual-qwen-shapes: read Qwen config)\n"
        "  --actual-qwen-shapes           (benchmark all deduped Qwen linear shapes)\n"
        "  --warmup N                     (default 100)\n"
        "  --samples N                    (default 50)\n"
        "  --launches N                   (default 20)\n"
        "  --seed N                       (default 1234)\n"
        "  --device N                     (default 0)\n"
        "  --verify-small-m               benchmark the speculative verify rows 1,2,4,8\n"
        "  --check                        verify selected path against the production correctness path\n"
        "  --split N                      split rows into N launches (default 1)\n"
        "  --out PATH                     (append CSV rows)\n");
}

bool parse_rows(const std::string& s, std::vector<uint32_t>& out) {
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (cur.empty()) return false;
            uint32_t v = (uint32_t)std::stoul(cur);
            if (v == 0) return false;
            out.push_back(v);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        uint32_t v = (uint32_t)std::stoul(cur);
        if (v == 0) return false;
        out.push_back(v);
    }
    return !out.empty();
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = p * (double)(v.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

float bf16_bits_to_f32(uint16_t bits) {
    uint32_t ui = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &ui, 4);
    return f;
}

}  // namespace

struct Shape {
    std::string name;
    uint32_t n = 0;
    uint32_t k = 0;
};

std::vector<Shape> build_qwen_shapes(const shelf::qwen35::Qwen35TextConfig& c) {
    const uint32_t hidden = (uint32_t)c.hidden_size;
    const uint32_t inter = (uint32_t)c.intermediate_size;
    const uint32_t qa = (uint32_t)c.num_attention_heads * (uint32_t)c.attention_head_dim;
    const uint32_t kva = (uint32_t)c.num_key_value_heads * (uint32_t)c.attention_head_dim;
    const uint32_t gqk = (uint32_t)c.linear_num_key_heads * (uint32_t)c.linear_key_head_dim;
    const uint32_t gv = (uint32_t)c.linear_num_value_heads * (uint32_t)c.linear_value_head_dim;
    const uint32_t nv = (uint32_t)c.linear_num_value_heads;
    const std::vector<Shape> raw = {
        {"mlp_gate_up", inter, hidden},
        {"mlp_down", hidden, inter},
        {"attn_q", qa, hidden},
        {"attn_kv", kva, hidden},
        {"attn_o", hidden, qa},
        {"gdn_qkv", 2u * gqk + gv, hidden},
        {"gdn_z", gv, hidden},
        {"gdn_ba", nv, hidden},
        {"gdn_out", hidden, gv},
    };
    std::vector<Shape> dedup;
    for (const auto& s : raw) {
        if (s.n == 0 || s.k == 0) continue;
        bool seen = false;
        for (const auto& d : dedup)
            if (d.n == s.n && d.k == s.k) {
                seen = true;
                break;
            }
        if (!seen) dedup.push_back(s);
    }
    return dedup;
}

struct CorrectnessEnv {
    void* weight_slot_dev = nullptr;
    void* values_dev = nullptr;
    void* ranges_dev = nullptr;
    void* ctx_dev = nullptr;
    void* workspace = nullptr;
    float* scratch = nullptr;
    uint32_t* error_word = nullptr;
    ::shelf::runtime::DeviceProgramView program;
    ~CorrectnessEnv() {
        shelf::gpu::discard_cleanup_result(hipFree(weight_slot_dev));
        shelf::gpu::discard_cleanup_result(hipFree(values_dev));
        shelf::gpu::discard_cleanup_result(hipFree(ranges_dev));
        shelf::gpu::discard_cleanup_result(hipFree(ctx_dev));
        shelf::gpu::discard_cleanup_result(hipFree(workspace));
        shelf::gpu::discard_cleanup_result(hipFree(scratch));
        shelf::gpu::discard_cleanup_result(hipFree(error_word));
    }
};

bool gen_bf16_input(uint32_t rows, uint32_t k, uint32_t seed, shelf::bf16_t** d_input) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<shelf::bf16_t> h((size_t)rows * k);
    for (auto& v : h) {
        const float fv = dist(rng);
        uint32_t ui;
        std::memcpy(&ui, &fv, 4);
        ui = (ui + 0x8000u) & 0xFFFF0000u;
        v = shelf::bf16_t{(uint16_t)(ui >> 16)};
    }
    if (hipMalloc((void**)d_input, (size_t)rows * k * 2) != hipSuccess) return false;
    if (hipMemcpy(*d_input, h.data(), (size_t)rows * k * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    return true;
}

struct Bf16Weights {
    shelf::bf16_t* d_weight = nullptr;
    ~Bf16Weights() {
        shelf::gpu::discard_cleanup_result(hipFree(d_weight));
    }
};

bool gen_bf16_weights(uint32_t n, uint32_t k, uint32_t seed, Bf16Weights& w) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wd(-0.02f, 0.02f);
    std::vector<uint16_t> h((size_t)n * k);
    auto f32b = [](float f) {
        uint32_t ui;
        std::memcpy(&ui, &f, 4);
        ui = (ui + 0x8000u) & 0xFFFF0000u;
        return (uint16_t)(ui >> 16);
    };
    for (auto& v : h) v = f32b(wd(rng));
    if (hipMalloc((void**)&w.d_weight, h.size() * 2) != hipSuccess) return false;
    if (hipMemcpy(w.d_weight, h.data(), h.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    return true;
}

bool setup_correctness_bf16(const Bf16Weights& w, const shelf::bf16_t* d_input, shelf::bf16_t* d_output,
                            uint32_t rows, uint32_t n, uint32_t k, CorrectnessEnv& env) {
    ::shelf::runtime::WeightSlot slot{};
    slot.rows = n;
    slot.cols = k;
    slot.k_padded = k;
    slot.device_ptr = w.d_weight;
    slot.encoding = 0u;
    slot.compute_spec = 0u;

    if (hipMalloc(&env.weight_slot_dev, sizeof(::shelf::runtime::WeightSlot)) != hipSuccess) return false;
    if (hipMemcpy(env.weight_slot_dev, &slot, sizeof(::shelf::runtime::WeightSlot), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    ::shelf::runtime::DeviceValueBinding values[2];
    values[0] = {};
    values[0].logical_value = 0;
    values[0].storage = ::shelf::runtime::ValueStorage::EXTERNAL_INPUT;
    values[0].dtype = ::shelf::runtime::ValueDType::BF16;
    values[0].row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
    values[0].slot = 0;
    values[0].feature_count = k;
    values[0].row_stride = k;
    values[0].offset = reinterpret_cast<uint64_t>(d_input);
    values[0].bytes = (uint64_t)rows * k * sizeof(shelf::bf16_t);
    values[1] = {};
    values[1].logical_value = 1;
    values[1].storage = ::shelf::runtime::ValueStorage::EXTERNAL_OUTPUT;
    values[1].dtype = ::shelf::runtime::ValueDType::BF16;
    values[1].row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
    values[1].slot = 0;
    values[1].feature_count = n;
    values[1].row_stride = n;
    values[1].offset = reinterpret_cast<uint64_t>(d_output);
    values[1].bytes = (uint64_t)rows * n * sizeof(shelf::bf16_t);

    if (hipMalloc(&env.values_dev, 2 * sizeof(::shelf::runtime::DeviceValueBinding)) != hipSuccess) return false;
    if (hipMemcpy(env.values_dev, values, 2 * sizeof(::shelf::runtime::DeviceValueBinding), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    env.program = {};
    env.program.values = static_cast<const ::shelf::runtime::DeviceValueBinding*>(env.values_dev);
    env.program.value_count = 2;
    env.program.weight_table = env.weight_slot_dev;
    env.program.weight_slot_count = 1;

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_outputs = 0;
    if (hipMalloc(&env.ctx_dev, sizeof(::shelf::runtime::DeviceBatchContext)) != hipSuccess) return false;
    if (hipMemcpy(env.ctx_dev, &ctx, sizeof(::shelf::runtime::DeviceBatchContext), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    if (hipMalloc(&env.scratch, (size_t)64 * 2048 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&env.error_word, sizeof(uint32_t)) != hipSuccess) return false;
    if (hipMemset(env.error_word, 0, sizeof(uint32_t)) != hipSuccess) return false;
    return true;
}

shelf::Status launch_correctness_linear_bf16(const CorrectnessEnv& env, uint32_t rows, hipStream_t stream) {
    ::shelf::runtime::DispatchBinding lbind{};
    lbind.kernel_id = ::shelf::runtime::KernelId::LINEAR_BF16;
    lbind.weight_index = 0;
    lbind.input_slots[0] = 0;
    lbind.input_count = 1;
    lbind.output_slots[0] = 1;
    lbind.output_count = 1;
    shelf::gpu::discard_cleanup_result(hipMemset(env.error_word, 0, sizeof(uint32_t)));
    return shelf::kernel::launch_model_dispatch_correctness(
        env.program, lbind, static_cast<::shelf::runtime::DeviceBatchContext*>(env.ctx_dev),
        shelf::kernel::ModelDispatchStateView{}, env.scratch, env.error_word, stream);
}

bool setup_correctness_fp8(const void* d_codes, const void* d_scales,
                           const shelf::bf16_t* d_input, shelf::bf16_t* d_output,
                           uint32_t rows, uint32_t n, uint32_t k, uint32_t kp,
                           CorrectnessEnv& env, bool mxfp4 = false) {
    ::shelf::runtime::WeightSlot slot{};
    slot.rows = n;
    slot.cols = k;
    slot.k_padded = kp;
    slot.encoding = 1u;
    slot.compute_spec = 0u;
    slot.codes = d_codes;
    slot.scales = d_scales;
    slot.storage_scale_stride_bytes = (kp / 128u) * 4u;
    slot.weight_scale_group = 128u;
    if (mxfp4) {
        slot.encoding = 3u;
        slot.compute_spec = 2u;
        slot.weight_scale_group = 32u;
        slot.codes_row_stride_bytes = (kp >> 5) * 256u;
        slot.scale_row_stride_bytes = kp >> 5;
        slot.storage_scale_stride_bytes = kp >> 5;
        slot.preshuffled = true;
    }

    if (hipMalloc(&env.weight_slot_dev, sizeof(::shelf::runtime::WeightSlot)) != hipSuccess) return false;
    if (hipMemcpy(env.weight_slot_dev, &slot, sizeof(::shelf::runtime::WeightSlot), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    ::shelf::runtime::Int8ActivationWorkspaceLayout layout;
    const shelf::Status layout_st = ::shelf::runtime::make_activation_workspace_layout(
        ::shelf::quantization::kFp8FinegrainedActivationQuantSpec, kp, rows, layout);
    if (!layout_st.ok() || layout.total_bytes == 0u) return false;
    if (hipMalloc(&env.workspace, layout.total_bytes) != hipSuccess) return false;
    ::shelf::runtime::DeviceWorkspaceRange range{};
    range.offset = 0u;
    range.bytes = layout.total_bytes;
    if (hipMalloc(&env.ranges_dev, sizeof(::shelf::runtime::DeviceWorkspaceRange)) != hipSuccess) return false;
    if (hipMemcpy(env.ranges_dev, &range, sizeof(range), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    ::shelf::runtime::DeviceValueBinding values[2];
    values[0] = {};
    values[0].logical_value = 0;
    values[0].storage = ::shelf::runtime::ValueStorage::EXTERNAL_INPUT;
    values[0].dtype = ::shelf::runtime::ValueDType::BF16;
    values[0].row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
    values[0].slot = 0;
    values[0].feature_count = k;
    values[0].row_stride = k;
    values[0].offset = reinterpret_cast<uint64_t>(d_input);
    values[0].bytes = (uint64_t)rows * k * sizeof(shelf::bf16_t);
    values[1] = {};
    values[1].logical_value = 1;
    values[1].storage = ::shelf::runtime::ValueStorage::EXTERNAL_OUTPUT;
    values[1].dtype = ::shelf::runtime::ValueDType::BF16;
    values[1].row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
    values[1].slot = 0;
    values[1].feature_count = n;
    values[1].row_stride = n;
    values[1].offset = reinterpret_cast<uint64_t>(d_output);
    values[1].bytes = (uint64_t)rows * n * sizeof(shelf::bf16_t);

    if (hipMalloc(&env.values_dev, 2 * sizeof(::shelf::runtime::DeviceValueBinding)) != hipSuccess) return false;
    if (hipMemcpy(env.values_dev, values, 2 * sizeof(::shelf::runtime::DeviceValueBinding), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    env.program = {};
    env.program.values = static_cast<const ::shelf::runtime::DeviceValueBinding*>(env.values_dev);
    env.program.value_count = 2;
    env.program.weight_table = env.weight_slot_dev;
    env.program.weight_slot_count = 1;
    env.program.workspace_ranges =
        static_cast<const ::shelf::runtime::DeviceWorkspaceRange*>(env.ranges_dev);
    env.program.workspace_range_count = 1;
    env.program.workspace_base = reinterpret_cast<uint64_t>(env.workspace);

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_outputs = 0;
    if (hipMalloc(&env.ctx_dev, sizeof(::shelf::runtime::DeviceBatchContext)) != hipSuccess) return false;
    if (hipMemcpy(env.ctx_dev, &ctx, sizeof(::shelf::runtime::DeviceBatchContext), hipMemcpyHostToDevice) != hipSuccess)
        return false;

    if (hipMalloc(&env.scratch, (size_t)64 * 2048 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&env.error_word, sizeof(uint32_t)) != hipSuccess) return false;
    if (hipMemset(env.error_word, 0, sizeof(uint32_t)) != hipSuccess) return false;
    return true;
}

shelf::Status launch_correctness_linear_fp8(const CorrectnessEnv& env, uint32_t rows,
                                         uint32_t kp, hipStream_t stream, bool mxfp4 = false,
                                         bool act_rowref = false) {
    ::shelf::runtime::Int8ActivationWorkspaceLayout layout;
    const shelf::Status status = ::shelf::runtime::make_activation_workspace_layout(
        ::shelf::quantization::kFp8FinegrainedActivationQuantSpec, kp, rows, layout);
    if (!status.ok()) return status;

    ::shelf::runtime::DispatchBinding qbind{};
    qbind.kernel_id = act_rowref
                          ? ::shelf::runtime::KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF
                          : ::shelf::runtime::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED;
    qbind.workspace_slot = 0u;
    qbind.group_size = 32u;
    qbind.activation_kp = kp;
    qbind.activation_max_rows = rows;
    qbind.activation_code_stride = layout.code_row_stride_bytes;
    qbind.activation_scale_stride = layout.scale_row_stride_bytes;
    qbind.input_slots[0] = 0;
    qbind.input_count = 1;
    qbind.output_count = 0;

    ::shelf::runtime::DispatchBinding lbind{};
    lbind.kernel_id = mxfp4 ? ::shelf::runtime::KernelId::LINEAR_MXFP4_W4A8
                            : ::shelf::runtime::KernelId::LINEAR_FP8_FINEGRAINED;
    lbind.workspace_slot = 0u;
    lbind.weight_index = 0;
    lbind.activation_kp = kp;
    lbind.activation_max_rows = rows;
    lbind.activation_code_stride = layout.code_row_stride_bytes;
    lbind.activation_scale_stride = layout.scale_row_stride_bytes;
    lbind.input_count = 0;
    lbind.output_slots[0] = 1;
    lbind.output_count = 1;
    if (act_rowref) lbind.flags = ::shelf::runtime::kActFlagRowRefFolded;

    shelf::gpu::discard_cleanup_result(hipMemset(env.error_word, 0, sizeof(uint32_t)));
    shelf::Status st = shelf::kernel::launch_model_dispatch_correctness(
        env.program, qbind, static_cast<::shelf::runtime::DeviceBatchContext*>(env.ctx_dev),
        shelf::kernel::ModelDispatchStateView{}, env.scratch, env.error_word, stream);
    if (!st.ok()) return st;
    return shelf::kernel::launch_model_dispatch_correctness(
        env.program, lbind, static_cast<::shelf::runtime::DeviceBatchContext*>(env.ctx_dev),
        shelf::kernel::ModelDispatchStateView{}, env.scratch, env.error_word, stream);
}

shelf::kernel::Bf16GemmConfig bf16_config_for(uint32_t rows, uint32_t n, uint32_t k) {
    using Id = shelf::kernel::Bf16GemmConfigId;
    if (rows == 1u && (k & 7u) == 0u)
        return shelf::kernel::Bf16GemmConfig{Id::ExactRows, 1u};
    if (n <= shelf::kernel::kBf16GemmSplitKMaxOutFeatures)
        return shelf::kernel::Bf16GemmConfig{Id::WmmaKPartition, 0u};
    if (rows >= shelf::kernel::kBf16GemmWideMinRows)
        return shelf::kernel::Bf16GemmConfig{Id::WmmaWide, 0u};
    return shelf::kernel::Bf16GemmConfig{Id::Wmma, 0u};
}

shelf::kernel::Fp8FinegrainedGemmConfig fp8_gemm_config_for(uint32_t rows, uint32_t n) {
    using Id = shelf::kernel::Fp8FinegrainedGemmConfigId;
    if (rows >= 4u * shelf::kernel::kFp8FinegrainedGemmRowTile &&
        n >= shelf::kernel::kBlockScaledGemmRowBlockMinOutFeatures)
        return {Id::RowBlock4};
    if (rows > shelf::kernel::kFp8FinegrainedGemmRowTile &&
        n >= shelf::kernel::kBlockScaledGemmRowBlockMinOutFeatures)
        return {Id::RowBlock2};
    return {Id::RowBlock1};
}

int parse_shapes_spec(const std::string& spec, std::vector<Shape>& shapes) {
    std::string cur;
    std::vector<std::string> parts;
    for (char c : spec) {
        if (c == ',') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    parts.push_back(cur);
    for (const std::string& p : parts) {
        const size_t colon = p.find(':');
        if (colon == std::string::npos) {
            std::fprintf(stderr, "bad --shapes entry '%s'\n", p.c_str());
            return 2;
        }
        Shape s;
        s.name = p;
        s.n = (uint32_t)std::stoul(p.substr(0, colon));
        s.k = (uint32_t)std::stoul(p.substr(colon + 1));
        if (s.n == 0u || s.k == 0u) {
            std::fprintf(stderr, "bad --shapes entry '%s'\n", p.c_str());
            return 2;
        }
        shapes.push_back(s);
    }
    return 0;
}

int run_gemm_bf16(Options& o) {
    if (o.variant != "correctness" && o.variant != "gemm" && o.variant != "auto") {
        std::fprintf(stderr, "bf16 --variant must be correctness|gemm|auto\n");
        return 2;
    }
    std::vector<Shape> shapes;
    if (o.actual_qwen_shapes) {
        if (o.model_dir.empty()) {
            std::fprintf(stderr, "--actual-qwen-shapes requires --model-dir\n");
            return 2;
        }
        auto cfg_res = shelf::qwen35::read_qwen35_text_config(o.model_dir);
        if (!cfg_res.ok()) {
            std::fprintf(stderr, "read_qwen35_text_config: %s\n", cfg_res.status().message().c_str());
            return 2;
        }
        shapes = build_qwen_shapes(cfg_res.release());
    } else if (!o.shapes_spec.empty()) {
        const int pr = parse_shapes_spec(o.shapes_spec, shapes);
        if (pr != 0) return pr;
    } else {
        shapes.push_back({"custom", o.n, o.k});
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "no shapes to benchmark\n");
        return 2;
    }

    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "dtype,requested_variant,selected_variant,shape_name,rows,n,k,warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (const auto& shape : shapes) {
        Bf16Weights w;
        if (!gen_bf16_weights(shape.n, shape.k, o.seed, w)) {
            std::fprintf(stderr, "gen weights failed %s\n", shape.name.c_str());
            rc = 1;
            break;
        }
        for (uint32_t rows : o.rows) {
            shelf::bf16_t* d_input = nullptr;
            shelf::bf16_t* d_output = nullptr;
            bool ok = gen_bf16_input(rows, shape.k, o.seed + 7777u, &d_input);
            ok = ok && hipMalloc((void**)&d_output, (size_t)rows * shape.n * 2) == hipSuccess;
            if (!ok) {
                std::fprintf(stderr, "alloc failed %s rows=%u\n", shape.name.c_str(), rows);
                rc = 1;
                break;
            }
            CorrectnessEnv env;
            if (!setup_correctness_bf16(w, d_input, d_output, rows, shape.n, shape.k, env)) {
                std::fprintf(stderr, "setup correctness failed %s rows=%u\n", shape.name.c_str(), rows);
                rc = 1;
                break;
            }
            std::string eff = o.variant;
            shelf::kernel::Bf16GemmConfig bf16_cfg = bf16_config_for(rows, shape.n, shape.k);
            if (o.impl != "auto") {
                using Id = shelf::kernel::Bf16GemmConfigId;
                if (o.impl == "wmma")
                    bf16_cfg = shelf::kernel::Bf16GemmConfig{Id::Wmma, 0u};
                else if (o.impl == "wide")
                    bf16_cfg = shelf::kernel::Bf16GemmConfig{Id::WmmaWide, 0u};
                else if (o.impl == "splitk")
                    bf16_cfg = shelf::kernel::Bf16GemmConfig{Id::WmmaKPartition, 0u};
                else
                    bf16_cfg = shelf::kernel::Bf16GemmConfig{Id::ExactRows,
                                                          static_cast<uint8_t>(rows)};
                eff = "gemm";
            } else if (eff == "auto") {
                const bool supported = shelf::execution::linear_shape_supported(
                    shelf::execution::LinearComputeFamily::Bf16, shape.n, shape.k);
                eff = supported ? "gemm" : "correctness";
            }
            auto launch_one = [&]() -> hipError_t {
                if (eff == "gemm")
                    return shelf::kernel::launch_gemm_bf16(
                        bf16_cfg, w.d_weight, d_input, d_output,
                        shelf::kernel::Bf16GemmOutputDType::BF16, rows, shape.n, shape.k,
                        shape.k, shape.n, stream);
                shelf::Status st = launch_correctness_linear_bf16(env, rows, stream);
                return st.ok() ? hipSuccess : hipErrorUnknown;
            };
            if (o.check) {
                const size_t outn = (size_t)rows * shape.n;
                std::vector<uint16_t> h_c(outn), h_g(outn);
                shelf::Status stc = launch_correctness_linear_bf16(env, rows, stream);
                uint32_t ew = 0;
                if (!stc.ok() || hipStreamSynchronize(stream) != hipSuccess ||
                    hipMemcpy(&ew, env.error_word, sizeof(ew), hipMemcpyDeviceToHost) != hipSuccess || ew != 0) {
                    std::fprintf(stderr, "CHECK FAIL %s rows=%u correctness (err=%u)\n", shape.name.c_str(), rows, ew);
                    rc = 1;
                    break;
                }
                shelf::gpu::discard_cleanup_result(hipMemcpy(h_c.data(), d_output, outn * 2, hipMemcpyDeviceToHost));
                if (launch_one() != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
                    std::fprintf(stderr, "CHECK FAIL %s rows=%u %s launch\n", shape.name.c_str(), rows, o.variant.c_str());
                    rc = 1;
                    break;
                }
                shelf::gpu::discard_cleanup_result(hipMemcpy(h_g.data(), d_output, outn * 2, hipMemcpyDeviceToHost));
                uint64_t mismatch = 0, nan = 0, inf = 0;
                float max_rel = 0.f;
                for (size_t i = 0; i < outn; ++i) {
                    if (h_c[i] != h_g[i]) ++mismatch;
                    const float fc = bf16_bits_to_f32(h_c[i]);
                    const float fg = bf16_bits_to_f32(h_g[i]);
                    if (std::isnan(fc) || std::isnan(fg)) ++nan;
                    if (std::isinf(fc) || std::isinf(fg)) ++inf;
                    const float rel = std::fabs(fc - fg) / std::max(1.0f, std::fabs(fc));
                    if (rel > max_rel) max_rel = rel;
                }
                const bool okc = nan == 0 && inf == 0 && max_rel <= 2e-2f;
                std::printf("CHECK %-12s rows=%-4u n=%-6u k=%-6u mismatch=%llu max_rel=%.4g nan=%llu inf=%llu %s\n",
                            shape.name.c_str(), rows, shape.n, shape.k, (unsigned long long)mismatch,
                            max_rel, (unsigned long long)nan, (unsigned long long)inf, okc ? "OK" : "FAIL");
                if (!okc) rc = 1;
            }
            for (int i = 0; i < o.warmup; ++i) {
                if (launch_one() != hipSuccess) {
                    std::fprintf(stderr, "warmup launch failed %s rows=%u\n", shape.name.c_str(), rows);
                    rc = 1;
                    break;
                }
            }
            if (rc) break;
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            hipEvent_t ev0 = nullptr, ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; break; }
            std::vector<double> per_launch_us;
            per_launch_us.reserve(o.samples);
            for (int s = 0; s < o.samples; ++s) {
                if (hipEventRecord(ev0, stream) != hipSuccess) { rc = 1; break; }
                for (int l = 0; l < o.launches; ++l) {
                    if (launch_one() != hipSuccess) { rc = 1; break; }
                }
                if (rc) break;
                if (hipEventRecord(ev1, stream) != hipSuccess) { rc = 1; break; }
                if (hipEventSynchronize(ev1) != hipSuccess) { rc = 1; break; }
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) { rc = 1; break; }
                per_launch_us.push_back(ms * 1000.0 / (double)o.launches);
            }
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            if (rc) break;
            if (!per_launch_us.empty()) {
                const double mn = percentile(per_launch_us, 0.0);
                const double p50 = percentile(per_launch_us, 0.5);
                const double p95 = percentile(per_launch_us, 0.95);
                double mean = 0.0;
                for (double v : per_launch_us) mean += v;
                mean /= (double)per_launch_us.size();
                std::printf("bf16 %-12s rows=%-4u n=%-6u k=%-6u requested=%s selected=%s min=%.3f p50=%.3f p95=%.3f mean=%.3f us\n",
                            shape.name.c_str(), rows, shape.n, shape.k, o.variant.c_str(), eff.c_str(), mn, p50, p95, mean);
                if (!o.out.empty()) {
                    FILE* f = std::fopen(o.out.c_str(), "a");
                    if (f) {
                        std::fprintf(f, "%s,%s,%s,%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                     o.dtype.c_str(), o.variant.c_str(), eff.c_str(), shape.name.c_str(),
                                     rows, shape.n, shape.k,
                                     o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                        std::fclose(f);
                    }
                }
            }
            shelf::gpu::discard_cleanup_result(hipFree(d_input));
            shelf::gpu::discard_cleanup_result(hipFree(d_output));
            if (rc) break;
        }
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}



shelf::kernel::Mxfp4GemmConfig mxfp4_config_for(const std::string& impl, uint32_t split,
                                                uint32_t unroll) {
    using Id = shelf::kernel::Mxfp4GemmConfigId;
    Id id = Id::RowBlock1;
    if (impl == "splitk") id = Id::SplitK;
    else if (impl == "prefill2d") id = Id::Prefill2D;
    else if (impl == "rowblock2") id = Id::RowBlock2;
    else if (impl == "rowblock4") id = Id::RowBlock4;
    (void)unroll;
    return shelf::kernel::Mxfp4GemmConfig{id, static_cast<uint8_t>(split)};
}

const char* mxfp4_config_name(const shelf::kernel::Mxfp4GemmConfig& cfg) {
    switch (cfg.id) {
    case shelf::kernel::Mxfp4GemmConfigId::SplitK: return "splitk";
    case shelf::kernel::Mxfp4GemmConfigId::Prefill2D: return "prefill2d";
    case shelf::kernel::Mxfp4GemmConfigId::RowBlock1: return "rowblock1";
    case shelf::kernel::Mxfp4GemmConfigId::RowBlock2: return "rowblock2";
    case shelf::kernel::Mxfp4GemmConfigId::RowBlock4: return "rowblock4";
    }
    return "rowblock1";
}


// ---------------------------------------------------------------------------
// Experiment only (docs/research/20261103/mxfp4_gate_a_prime.md sec.11): the cost
// of folding the activation E8M0 scale into the FP8 A codes as a plane pass, which
// is what would let arms C and D run the GEMM without any A-side rescale.
// The A plane is the fragment-order FP8 workspace (1 byte per code). Within one
// 64 K fragment block, lane slot L holds 32 bytes: the 16-byte half u<4 is the
// even 32 K block and u>=4 the odd one, matching the GEMM's (ib >> 1) / (ib & 1)
// addressing. The row owning those bytes is (L & 15) of the 16-row tile.
// ---------------------------------------------------------------------------
__global__ void a_row_ref_kernel(const uint8_t* __restrict__ e8, uint8_t* __restrict__ ref,
                                 const uint32_t nb) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t m = 0u;
    const uint8_t* p = e8 + static_cast<size_t>(row) * nb;
    for (uint32_t i = 0; i < nb; ++i)
        m = m > p[i] ? m : p[i];
    ref[row] = static_cast<uint8_t>(m);
}

__global__ void a_plane_fold_kernel(const uint8_t* __restrict__ codes,
                                    const uint8_t* __restrict__ e8,
                                    const uint8_t* __restrict__ ref,
                                    uint8_t* __restrict__ out,
                                    const uint32_t nb) {
    const uint32_t f = blockIdx.x;                       // 64 K fragment block
    const uint32_t tile = blockIdx.y;                    // 16-row tile
    const uint32_t tid = threadIdx.x;                    // 256 threads = 32 lanes x 8 dwords
    const uint32_t lane = tid >> 3u;
    const uint32_t u = tid & 7u;
    const uint32_t ib = (f << 1u) + (u >> 2u);
    if (ib >= nb)
        return;
    const uint32_t row = tile * 16u + (lane & 15u);
    const uint32_t d = ref[row] - e8[static_cast<size_t>(row) * nb + ib];
    const uint32_t sh = (d > 15u) ? 15u << 3u : d << 3u;
    const size_t off = (static_cast<size_t>(tile) * (static_cast<size_t>(gridDim.x) << 10u)) +
                       (static_cast<size_t>(f) << 10u) + static_cast<size_t>(lane) * 32u +
                       (u << 2u);
    const uint32_t x = *reinterpret_cast<const uint32_t*>(codes + off);
    uint32_t res = 0u;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t b = (x >> (8u * j)) & 0x7Fu;
        const uint32_t o = (b > sh) ? (b - sh) : 0u;   // subnormals collapse to zero
        res |= o << (8u * j);
    }
    *reinterpret_cast<uint32_t*>(out + off) = res | (x & 0x80808080u);
}

int run_gemm_mxfp4(Options& o) {
    std::vector<Shape> shapes;
    if (o.actual_qwen_shapes) {
        if (o.model_dir.empty()) {
            std::fprintf(stderr, "--actual-qwen-shapes requires --model-dir\n");
            return 2;
        }
        auto cfg_res = shelf::qwen35::read_qwen35_text_config(o.model_dir);
        if (!cfg_res.ok()) {
            std::fprintf(stderr, "read_qwen35_text_config: %s\n", cfg_res.status().message().c_str());
            return 2;
        }
        shapes = build_qwen_shapes(cfg_res.release());
    } else {
        shapes.push_back({"custom", o.n, o.k});
    }
    if (!o.shapes_spec.empty() && !o.actual_qwen_shapes) {
        shapes.clear();
        if (parse_shapes_spec(o.shapes_spec, shapes) != 0) return 2;
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "no shapes to benchmark\n");
        return 2;
    }
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "dtype,requested_variant,selected_variant,shape_name,rows,n,k,warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (const auto& shape : shapes) {
        const uint32_t kp = (shape.k + 127u) & ~127u;
        const uint32_t nb = kp >> 5;
        const uint32_t n = shape.n;
        const uint32_t n_tiled = (n + 15u) & ~15u;
        const size_t codes_bytes = static_cast<size_t>(n_tiled / 16u) * nb * 256u;
        const size_t scales_bytes = static_cast<size_t>(n_tiled) * nb;
        uint8_t* d_codes = nullptr;
        uint8_t* d_scales = nullptr;
        uint8_t* d_ref = nullptr;
        if (hipMalloc((void**)&d_codes, codes_bytes) != hipSuccess ||
            hipMalloc((void**)&d_scales, scales_bytes) != hipSuccess ||
            hipMalloc((void**)&d_ref, scales_bytes / nb) != hipSuccess) {
            return 1;
        }
        {
            std::mt19937 rng(o.seed + 41u);
            std::vector<uint8_t> hc(codes_bytes);
            for (auto& v : hc) v = static_cast<uint8_t>(rng() & 0xFFu);
            std::vector<uint8_t> hs(scales_bytes);
            for (auto& v : hs) v = static_cast<uint8_t>(112u + (rng() % o.mxfp4_spread));
            std::vector<uint8_t> hr(scales_bytes / nb, 0u);
            for (uint32_t r = 0; r < n_tiled; ++r) {
                uint8_t ref = 0u;
                for (uint32_t ib = 0; ib < nb; ++ib)
                    ref = std::max(ref, hs[static_cast<size_t>(r) * nb + ib]);
                hr[r] = ref;
            }
            if (hipMemcpy(d_codes, hc.data(), hc.size(), hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(d_scales, hs.data(), hs.size(), hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(d_ref, hr.data(), hr.size(), hipMemcpyHostToDevice) != hipSuccess) {
                return 1;
            }
        }
        for (uint32_t rows : o.rows) {
            shelf::bf16_t* d_input = nullptr;
            shelf::bf16_t* d_output = nullptr;
            uint8_t* d_acodes = nullptr;
            float* d_ascales = nullptr;
            const size_t acodes_bytes = static_cast<size_t>((rows + 15u) / 16u) * kp * 16u;
            const uint32_t act_groups = (kp + 127u) / 128u;
            const size_t act_scale_stride =
                ((static_cast<size_t>(act_groups) * 4u) + 15u) & ~15u;
            const size_t ascales_bytes = static_cast<size_t>(rows) * act_scale_stride;
            bool ok = gen_bf16_input(rows, shape.k, o.seed + 7777u, &d_input);
            ok = ok && hipMalloc((void**)&d_acodes, acodes_bytes) == hipSuccess;
            ok = ok && hipMalloc((void**)&d_ascales, ascales_bytes) == hipSuccess;
            ok = ok && hipMalloc((void**)&d_output, static_cast<size_t>(rows) * n * 2) == hipSuccess;
            if (!ok) { rc = 1; break; }
            CorrectnessEnv cenv;
            if (o.check && !setup_correctness_fp8(
                    d_codes, d_scales, d_input, d_output, rows, n, shape.k, kp, cenv, true)) {
                std::fprintf(stderr, "CHECK FAIL mxfp4 %-12s rows=%u correctness setup\n",
                             shape.name.c_str(), rows);
                rc = 1;
                break;
            }
            hipError_t af = o.mxfp4_a_rowref
                ? shelf::kernel::launch_activation_quantize_mxfp4_rowref(
                      d_input, d_acodes, d_ascales, rows, shape.k, kp, shape.k, kp,
                      static_cast<uint32_t>(act_scale_stride), stream)
                : shelf::kernel::launch_activation_quantize_fp8_finegrained(
                      d_input, d_acodes, d_ascales, rows, shape.k, kp, shape.k, kp,
                      static_cast<uint32_t>(act_scale_stride), stream);
            if (af != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            const int8_t* a8 = reinterpret_cast<const int8_t*>(d_acodes);

            if (o.mxfp4_a_rowref) {
                auto time_launch = [&](const char* name, auto&& one) {
                    for (int i = 0; i < o.warmup; ++i)
                        if (one() != hipSuccess)
                            return;
                    if (hipStreamSynchronize(stream) != hipSuccess)
                        return;
                    hipEvent_t q0 = nullptr, q1 = nullptr;
                    if (hipEventCreate(&q0) != hipSuccess || hipEventCreate(&q1) != hipSuccess)
                        return;
                    std::vector<double> us;
                    for (int sample = 0; sample < o.samples; ++sample) {
                        if (hipEventRecord(q0, stream) != hipSuccess)
                            break;
                        for (int l = 0; l < o.launches; ++l)
                            if (one() != hipSuccess)
                                break;
                        if (hipEventRecord(q1, stream) != hipSuccess)
                            break;
                        if (hipEventSynchronize(q1) != hipSuccess)
                            break;
                        float ms = 0.0f;
                        if (hipEventElapsedTime(&ms, q0, q1) != hipSuccess)
                            break;
                        us.push_back(ms * 1000.0 / (double)o.launches);
                    }
                    shelf::gpu::discard_cleanup_result(hipEventDestroy(q0));
                    shelf::gpu::discard_cleanup_result(hipEventDestroy(q1));
                    if (us.empty())
                        return;
                    std::sort(us.begin(), us.end());
                    const double p50 = us[us.size() / 2];
                    const double traffic = (double)rows * shape.k * 2.0 + (double)acodes_bytes +
                                           (double)ascales_bytes;
                    std::printf("mxfp4 %-12s rows=%-4u n=%-6u k=%-6u requested=quantize"
                                " selected=%-8s      p50=%.3f us  %.0f GB/s\n",
                                shape.name.c_str(), rows, shape.n, shape.k, name, p50,
                                traffic / (p50 * 1e-6) / 1e9);
                };
                time_launch("group", [&]() {
                    return shelf::kernel::launch_activation_quantize_fp8_finegrained(
                        d_input, d_acodes, d_ascales, rows, shape.k, kp, shape.k, kp,
                        static_cast<uint32_t>(act_scale_stride), stream);
                });
                time_launch("rowref", [&]() {
                    return shelf::kernel::launch_activation_quantize_mxfp4_rowref(
                        d_input, d_acodes, d_ascales, rows, shape.k, kp, shape.k, kp,
                        static_cast<uint32_t>(act_scale_stride), stream);
                });
                if (hipGetLastError() != hipSuccess) { rc = 1; break; }
                if (o.mxfp4_a_rowref == 2u)
                    break;
            }

            const auto sel = shelf::execution::select_mxfp4_gemm_config(
                shelf::execution::Mxfp4GemmSelectorInput{rows, n, shape.k, kp});
            std::string selected = "gemm";
            shelf::kernel::Mxfp4GemmConfig cfg{shelf::kernel::Mxfp4GemmConfigId::RowBlock1,
                                               static_cast<uint8_t>(o.mxfp4_split)};
            if (o.variant == "correctness") {
                selected = "correctness";
            } else if (o.impl == "auto") {
                if (sel) cfg = *sel;
                selected = sel ? mxfp4_config_name(cfg) : "rowblock1";
            } else {
                cfg = mxfp4_config_for(o.impl, o.mxfp4_split, o.mxfp4_unroll);
                selected = mxfp4_config_name(cfg);
            }

            auto launch_one = [&]() -> hipError_t {
                if (selected == "correctness") {
                    shelf::Status st = launch_correctness_linear_fp8(
                        cenv, rows, kp, stream, true, o.mxfp4_a_rowref != 0u);
                    return st.ok() ? hipSuccess : hipErrorUnknown;
                }
                return shelf::kernel::launch_gemm_mxfp4_w4a8_auto(
                    cfg, d_codes, d_scales, o.mxfp4_fold ? d_ref : nullptr,
                    a8, d_ascales,
                    o.mxfp4_a_rowref ? shelf::kernel::Mxfp4ActivationForm::RowRefFolded
                                     : shelf::kernel::Mxfp4ActivationForm::GroupF32,
                    d_output,
                    shelf::kernel::Mxfp4GemmOutputDType::BF16, rows, n, kp, nb, kp,
                    static_cast<uint32_t>(act_scale_stride), n, stream);
            };
            if (o.check) {
                const size_t outn = (size_t)rows * n;
                std::vector<uint16_t> h_c(outn), h_g(outn);
                shelf::Status stc = launch_correctness_linear_fp8(
                    cenv, rows, kp, stream, true, o.mxfp4_a_rowref != 0u);
                uint32_t ew = 0;
                if (!stc.ok() || hipStreamSynchronize(stream) != hipSuccess ||
                    hipMemcpy(&ew, cenv.error_word, sizeof(ew), hipMemcpyDeviceToHost) != hipSuccess || ew != 0) {
                    std::fprintf(stderr, "CHECK FAIL mxfp4 %-12s rows=%u correctness (err=%u)\n",
                                 shape.name.c_str(), rows, ew);
                    rc = 1;
                    break;
                }
                shelf::gpu::discard_cleanup_result(
                    hipMemcpy(h_c.data(), d_output, outn * 2, hipMemcpyDeviceToHost));
                if (launch_one() != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
                    std::fprintf(stderr, "CHECK FAIL mxfp4 %-12s rows=%u optimized launch\n",
                                 shape.name.c_str(), rows);
                    rc = 1;
                    break;
                }
                shelf::gpu::discard_cleanup_result(
                    hipMemcpy(h_g.data(), d_output, outn * 2, hipMemcpyDeviceToHost));
                uint64_t mismatch = 0, nan = 0, inf = 0;
                float max_rel = 0.f;
                for (size_t i = 0; i < outn; ++i) {
                    if (h_c[i] != h_g[i]) ++mismatch;
                    const float fc = bf16_bits_to_f32(h_c[i]);
                    const float fg = bf16_bits_to_f32(h_g[i]);
                    if (std::isnan(fc) || std::isnan(fg)) ++nan;
                    if (std::isinf(fc) || std::isinf(fg)) ++inf;
                    const float rel = std::fabs(fc - fg) / std::max(1.0f, std::fabs(fc));
                    if (rel > max_rel) max_rel = rel;
                }
                const bool okc = nan == 0 && inf == 0 && max_rel <= 2e-2f;
                std::printf("CHECK mxfp4 %-12s rows=%-4u n=%-6u k=%-6u selected=%s mismatch=%llu max_rel=%.4g nan=%llu inf=%llu %s\n",
                            shape.name.c_str(), rows, shape.n, shape.k, selected.c_str(),
                            (unsigned long long)mismatch, max_rel,
                            (unsigned long long)nan, (unsigned long long)inf,
                            okc ? "OK" : "FAIL");
                if (!okc) rc = 1;
            }
            for (int i = 0; i < o.warmup; ++i)
                if (launch_one() != hipSuccess) { rc = 1; break; }
            if (rc) break;
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            hipEvent_t ev0 = nullptr, ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; break; }
            std::vector<double> per_launch_us;
            per_launch_us.reserve(o.samples);
            for (int s = 0; s < o.samples; ++s) {
                if (hipEventRecord(ev0, stream) != hipSuccess) { rc = 1; break; }
                for (int l = 0; l < o.launches; ++l)
                    if (launch_one() != hipSuccess) { rc = 1; break; }
                if (rc) break;
                if (hipEventRecord(ev1, stream) != hipSuccess) { rc = 1; break; }
                if (hipEventSynchronize(ev1) != hipSuccess) { rc = 1; break; }
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) { rc = 1; break; }
                per_launch_us.push_back(ms * 1000.0 / (double)o.launches);
            }
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            if (o.mxfp4_a_fold && !rc) {
                const uint32_t nb32 = kp >> 5;
                const uint32_t rows_pad = (rows + 15u) & ~15u;
                const size_t aplane = (size_t)(rows_pad >> 4) * (size_t)(kp >> 6) * 1024u;
                uint8_t* d_folded = nullptr;
                uint8_t* d_e8 = nullptr;
                uint8_t* d_ref = nullptr;
                bool okf = hipMalloc((void**)&d_folded, aplane) == hipSuccess &&
                           hipMalloc((void**)&d_e8, (size_t)rows_pad * nb32) == hipSuccess &&
                           hipMalloc((void**)&d_ref, rows_pad) == hipSuccess;
                if (okf) {
                    const uint32_t as_row_floats = (uint32_t)(act_scale_stride / 4u);
                    std::vector<float> hs((size_t)rows * as_row_floats, 0.0f);
                    okf = hipMemcpy(hs.data(), d_ascales, hs.size() * sizeof(float),
                                    hipMemcpyDeviceToHost) == hipSuccess;
                    std::vector<uint8_t> he((size_t)rows_pad * nb32, 0u);
                    for (uint32_t r = 0; r < rows && okf; ++r) {
                        for (uint32_t ib = 0; ib < nb32; ++ib) {
                            const uint32_t g = std::min(ib >> 2, act_groups - 1u);
                            const float sc = hs[(size_t)r * as_row_floats + g];
                            const int ex = (sc > 0.0f && std::isfinite(sc))
                                               ? (int)std::ceil(std::log2(sc)) : 0;
                            he[(size_t)r * nb32 + ib] = (uint8_t)(ex + 127);
                        }
                    }
                    okf = hipMemcpy(d_e8, he.data(), he.size(), hipMemcpyHostToDevice) == hipSuccess;
                }
                if (okf) {
                    auto fold_launch = [&]() -> hipError_t {
                        a_row_ref_kernel<<<rows_pad / 64u, 64u, 0, stream>>>(d_e8, d_ref, nb32);
                        a_plane_fold_kernel<<<dim3(kp >> 6, rows_pad >> 4, 1), 256, 0, stream>>>(
                            d_acodes, d_e8, d_ref, d_folded, nb32);
                        return hipGetLastError();
                    };
                    for (int i = 0; i < o.warmup; ++i)
                        if (fold_launch() != hipSuccess) { rc = 1; break; }
                    if (hipStreamSynchronize(stream) != hipSuccess) rc = 1;
                    hipEvent_t f0 = nullptr, f1 = nullptr;
                    if (hipEventCreate(&f0) != hipSuccess || hipEventCreate(&f1) != hipSuccess)
                        rc = 1;
                    std::vector<double> fold_us;
                    if (!rc) {
                        fold_us.reserve(o.samples);
                        for (int s = 0; s < o.samples && !rc; ++s) {
                            if (hipEventRecord(f0, stream) != hipSuccess) { rc = 1; break; }
                            for (int l = 0; l < o.launches; ++l)
                                if (fold_launch() != hipSuccess) { rc = 1; break; }
                            if (rc) break;
                            if (hipEventRecord(f1, stream) != hipSuccess) { rc = 1; break; }
                            if (hipEventSynchronize(f1) != hipSuccess) { rc = 1; break; }
                            float ms = 0.0f;
                            if (hipEventElapsedTime(&ms, f0, f1) != hipSuccess) { rc = 1; break; }
                            fold_us.push_back(ms * 1000.0 / (double)o.launches);
                        }
                    }
                    shelf::gpu::discard_cleanup_result(hipEventDestroy(f0));
                    shelf::gpu::discard_cleanup_result(hipEventDestroy(f1));
                    if (!fold_us.empty()) {
                        const double mn = percentile(fold_us, 0.0);
                        const double p50 = percentile(fold_us, 0.5);
                        const double p95 = percentile(fold_us, 0.95);
                        const double traffic = (double)aplane * 2.0 + (double)rows_pad * nb32;
                        std::printf("mxfp4 %-12s rows=%-4u n=%-6u k=%-6u requested=a_fold selected=a_fold      min=%.3f p50=%.3f p95=%.3f us  %.0f GB/s (A plane %.1f MiB)\n",
                                    shape.name.c_str(), rows, shape.n, shape.k, mn, p50, p95,
                                    traffic / (p50 * 1e-6) / 1e9, (double)aplane / 1048576.0);
                    }
                }
                shelf::gpu::discard_cleanup_result(hipFree(d_folded));
                shelf::gpu::discard_cleanup_result(hipFree(d_e8));
                shelf::gpu::discard_cleanup_result(hipFree(d_ref));
            }
            shelf::gpu::discard_cleanup_result(hipFree(d_input));
            shelf::gpu::discard_cleanup_result(hipFree(d_acodes));
            shelf::gpu::discard_cleanup_result(hipFree(d_ascales));
            shelf::gpu::discard_cleanup_result(hipFree(d_output));
            if (rc) break;
            if (!per_launch_us.empty()) {
                const double mn = percentile(per_launch_us, 0.0);
                const double p50 = percentile(per_launch_us, 0.5);
                const double p95 = percentile(per_launch_us, 0.95);
                double mean = 0.0;
                for (double v : per_launch_us) mean += v;
                mean /= (double)per_launch_us.size();
                const double weight_mib = (double)codes_bytes / (1024.0 * 1024.0);
                std::printf("mxfp4 %-12s rows=%-4u n=%-6u k=%-6u requested=%s selected=%-10s min=%.3f p50=%.3f p95=%.3f mean=%.3f us  %.0f GB/s\n",
                            shape.name.c_str(), rows, shape.n, shape.k, o.impl.c_str(),
                            selected.c_str(), mn, p50, p95, mean,
                            weight_mib * 1048576.0 / (p50 * 1e-6) / 1e9);
                if (!o.out.empty()) {
                    FILE* f = std::fopen(o.out.c_str(), "a");
                    if (f) {
                        std::fprintf(f, "mxfp4,%s,%s,%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                     o.impl.c_str(), selected.c_str(), shape.name.c_str(),
                                     rows, shape.n, shape.k,
                                     o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                        std::fclose(f);
                    }
                }
            }
        }
        shelf::gpu::discard_cleanup_result(hipFree(d_codes));
        shelf::gpu::discard_cleanup_result(hipFree(d_scales));
        if (rc) break;
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}

int run_gemm_fp8(Options& o) {
    std::vector<Shape> shapes;
    if (o.actual_qwen_shapes) {
        if (o.model_dir.empty()) {
            std::fprintf(stderr, "--actual-qwen-shapes requires --model-dir\n");
            return 2;
        }
        auto cfg_res = shelf::qwen35::read_qwen35_text_config(o.model_dir);
        if (!cfg_res.ok()) {
            std::fprintf(stderr, "read_qwen35_text_config: %s\n", cfg_res.status().message().c_str());
            return 2;
        }
        shapes = build_qwen_shapes(cfg_res.release());
    } else {
        shapes.push_back({"custom", o.n, o.k});
    }
    if (!o.shapes_spec.empty() && !o.actual_qwen_shapes) {
        shapes.clear();
        if (parse_shapes_spec(o.shapes_spec, shapes) != 0) return 2;
    }
    if (shapes.empty()) {
        std::fprintf(stderr, "no shapes to benchmark\n");
        return 2;
    }
    if (hipSetDevice(o.device) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "dtype,requested_variant,selected_variant,shape_name,rows,n,k,warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (const auto& shape : shapes) {
        const uint32_t kp = (shape.k + 127u) & ~127u;
        const uint32_t nb = kp >> 5;
        const uint32_t n = shape.n;
        const uint32_t n_tiled = (n + 15u) & ~15u;
        const size_t codes_bytes = static_cast<size_t>(n_tiled) * nb * 512u;
        const size_t scales_bytes = static_cast<size_t>(((n + 127u) / 128u)) * (kp / 128u) * 4u;
        uint8_t* d_codes = nullptr;
        uint8_t* d_scales = nullptr;
        if (hipMalloc((void**)&d_codes, codes_bytes) != hipSuccess ||
            hipMalloc((void**)&d_scales, scales_bytes) != hipSuccess) {
            return 1;
        }
        {
            std::mt19937 rng(o.seed + 33u);
            std::vector<uint8_t> hc(codes_bytes);
            for (auto& v : hc) {
                uint8_t c = static_cast<uint8_t>(rng() & 0xFFu);
                if ((c & 0x7Fu) == 0x7Fu) c &= 0xFEu;
                v = c;
            }
            std::vector<uint8_t> hs(scales_bytes, 0);
            float* hsf = reinterpret_cast<float*>(hs.data());
            for (size_t i = 0; i < hs.size() / 4u; ++i)
                hsf[i] = 0.01f + static_cast<float>(rng() % 1000u) / 100000.0f;
            if (hipMemcpy(d_codes, hc.data(), hc.size(), hipMemcpyHostToDevice) != hipSuccess ||
                hipMemcpy(d_scales, hs.data(), hs.size(), hipMemcpyHostToDevice) != hipSuccess) {
                return 1;
            }
        }
        for (uint32_t rows : o.rows) {
            shelf::bf16_t* d_input = nullptr;
            shelf::bf16_t* d_output = nullptr;
            uint8_t* d_acodes = nullptr;
            float* d_ascales = nullptr;
            const size_t acodes_bytes = static_cast<size_t>((rows + 15u) / 16u) * kp * 16u;
            const uint32_t act_groups = (kp + 127u) / 128u;
            const size_t act_scale_stride =
                ((static_cast<size_t>(act_groups) * 4u) + 15u) & ~15u;
            const size_t ascales_bytes = static_cast<size_t>(rows) * act_scale_stride;
            bool ok = gen_bf16_input(rows, shape.k, o.seed + 7777u, &d_input);
            ok = ok && hipMalloc((void**)&d_acodes, acodes_bytes) == hipSuccess;
            ok = ok && hipMalloc((void**)&d_ascales, ascales_bytes) == hipSuccess;
            ok = ok && hipMalloc((void**)&d_output, static_cast<size_t>(rows) * n * 2) == hipSuccess;
            if (!ok) { rc = 1; break; }
            CorrectnessEnv cenv;
            if (o.check && !setup_correctness_fp8(
                    d_codes, d_scales, d_input, d_output, rows, n, shape.k, kp, cenv)) {
                std::fprintf(stderr, "CHECK FAIL %s rows=%u correctness setup\n",
                             shape.name.c_str(), rows);
                rc = 1;
                break;
            }
            hipError_t af = shelf::kernel::launch_activation_quantize_fp8_finegrained(
                d_input, d_acodes, d_ascales, rows, shape.k, kp, shape.k, kp,
                static_cast<uint32_t>(act_scale_stride), stream);
            if (af != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            const int8_t* a8 = reinterpret_cast<const int8_t*>(d_acodes);
            const auto fp8_sel = shelf::execution::select_fp8_finegrained_gemm_config(
                shelf::execution::Fp8FinegrainedGemmSelectorInput{rows, n, shape.k, kp});
            const shelf::kernel::Fp8FinegrainedGemmConfig fp8_cfg =
                fp8_sel ? *fp8_sel : fp8_gemm_config_for(rows, n);
            const uint32_t split_parts = (o.split > 1u) ? o.split : 1u;
            const uint32_t split_sub = rows / split_parts;
            auto launch_one = [&]() -> hipError_t {
                for (uint32_t s = 0u; s < split_parts; ++s) {
                    const uint32_t sub_rows =
                        (s + 1u == split_parts) ? (rows - s * split_sub) : split_sub;
                    const int8_t* a8_off = a8 + static_cast<size_t>(s * split_sub) * kp;
                    const float* as_off = d_ascales +
                        static_cast<size_t>(s * split_sub) * (act_scale_stride / 4u);
                    void* out_off = reinterpret_cast<uint8_t*>(d_output) +
                        static_cast<size_t>(s * split_sub) * n * 2u;
                    hipError_t e = shelf::kernel::launch_gemm_fp8_finegrained_w8a8_wmma_auto(
                        fp8_cfg, d_codes,
                        reinterpret_cast<const float*>(d_scales), a8_off, as_off, out_off,
                        shelf::kernel::Fp8FinegrainedGemmOutputDType::BF16, sub_rows, n, kp,
                        (kp / 128u) * 4u, kp, static_cast<uint32_t>(act_scale_stride), n,
                        stream);
                    if (e != hipSuccess) return e;
                }
                return hipSuccess;
            };
            if (o.check) {
                const size_t outn = (size_t)rows * n;
                std::vector<uint16_t> h_c(outn), h_g(outn);
                shelf::Status stc = launch_correctness_linear_fp8(cenv, rows, kp, stream);
                uint32_t ew = 0;
                if (!stc.ok() || hipStreamSynchronize(stream) != hipSuccess ||
                    hipMemcpy(&ew, cenv.error_word, sizeof(ew), hipMemcpyDeviceToHost) != hipSuccess || ew != 0) {
                    std::fprintf(stderr, "CHECK FAIL %s rows=%u correctness (err=%u)\n",
                                 shape.name.c_str(), rows, ew);
                    rc = 1;
                    break;
                }
                shelf::gpu::discard_cleanup_result(
                    hipMemcpy(h_c.data(), d_output, outn * 2, hipMemcpyDeviceToHost));
                if (launch_one() != hipSuccess || hipStreamSynchronize(stream) != hipSuccess) {
                    std::fprintf(stderr, "CHECK FAIL %s rows=%u optimized launch\n",
                                 shape.name.c_str(), rows);
                    rc = 1;
                    break;
                }
                shelf::gpu::discard_cleanup_result(
                    hipMemcpy(h_g.data(), d_output, outn * 2, hipMemcpyDeviceToHost));
                uint64_t mismatch = 0, nan = 0, inf = 0, nan_c = 0, nan_g = 0;
                float max_rel = 0.f;
                for (size_t i = 0; i < outn; ++i) {
                    if (h_c[i] != h_g[i]) ++mismatch;
                    const float fc = bf16_bits_to_f32(h_c[i]);
                    const float fg = bf16_bits_to_f32(h_g[i]);
                    if (std::isnan(fc)) ++nan_c;
                    if (std::isnan(fg)) ++nan_g;
                    if (std::isnan(fc) || std::isnan(fg)) ++nan;
                    if (std::isinf(fc) || std::isinf(fg)) ++inf;
                    const float rel = std::fabs(fc - fg) / std::max(1.0f, std::fabs(fc));
                    if (rel > max_rel) max_rel = rel;
                }
                const bool okc = nan == 0 && inf == 0 && max_rel <= 2e-2f;
                std::printf("CHECK fp8 %-12s rows=%-4u n=%-6u k=%-6u mismatch=%llu max_rel=%.4g nan=%llu inf=%llu nan_c=%llu nan_g=%llu %s\n",
                            shape.name.c_str(), rows, shape.n, shape.k,
                            (unsigned long long)mismatch, max_rel,
                            (unsigned long long)nan, (unsigned long long)inf,
                            (unsigned long long)nan_c, (unsigned long long)nan_g,
                            okc ? "OK" : "FAIL");
                if (!okc) rc = 1;
            }
            for (int i = 0; i < o.warmup; ++i)
                if (launch_one() != hipSuccess) { rc = 1; break; }
            if (rc) break;
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; break; }
            hipEvent_t ev0 = nullptr, ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; break; }
            std::vector<double> per_launch_us;
            per_launch_us.reserve(o.samples);
            for (int s = 0; s < o.samples; ++s) {
                if (hipEventRecord(ev0, stream) != hipSuccess) { rc = 1; break; }
                for (int l = 0; l < o.launches; ++l)
                    if (launch_one() != hipSuccess) { rc = 1; break; }
                if (rc) break;
                if (hipEventRecord(ev1, stream) != hipSuccess) { rc = 1; break; }
                if (hipEventSynchronize(ev1) != hipSuccess) { rc = 1; break; }
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) { rc = 1; break; }
                per_launch_us.push_back(ms * 1000.0 / (double)o.launches);
            }
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            shelf::gpu::discard_cleanup_result(hipFree(d_input));
            shelf::gpu::discard_cleanup_result(hipFree(d_acodes));
            shelf::gpu::discard_cleanup_result(hipFree(d_ascales));
            shelf::gpu::discard_cleanup_result(hipFree(d_output));
            if (rc) break;
            if (!per_launch_us.empty()) {
                const double mn = percentile(per_launch_us, 0.0);
                const double p50 = percentile(per_launch_us, 0.5);
                const double p95 = percentile(per_launch_us, 0.95);
                double mean = 0.0;
                for (double v : per_launch_us) mean += v;
                mean /= (double)per_launch_us.size();
                std::printf("fp8 %-12s rows=%-4u n=%-6u k=%-6u requested=%s selected=gemm min=%.3f p50=%.3f p95=%.3f mean=%.3f us\n",
                            shape.name.c_str(), rows, shape.n, shape.k,
                            o.variant.c_str(), mn, p50, p95, mean);
                if (!o.out.empty()) {
                    FILE* f = std::fopen(o.out.c_str(), "a");
                    if (f) {
                        std::fprintf(f, "fp8,%s,gemm,%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                     o.variant.c_str(), shape.name.c_str(),
                                     rows, shape.n, shape.k,
                                     o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                        std::fclose(f);
                    }
                }
            }
        }
        shelf::gpu::discard_cleanup_result(hipFree(d_codes));
        shelf::gpu::discard_cleanup_result(hipFree(d_scales));
        if (rc) break;
    }
    shelf::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}


int run_gemm(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--dtype") o.dtype = next();
        else if (a == "--variant") o.variant = next();
        else if (a == "--rows") { if (!parse_rows(next(), o.rows)) { fprintf(stderr, "bad --rows\n"); return 2; } }
        else if (a == "--n") o.n = (uint32_t)std::stoul(next());
        else if (a == "--k") o.k = (uint32_t)std::stoul(next());
        else if (a == "--impl") o.impl = next();
        else if (a == "--shapes") o.shapes_spec = next();
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--samples") o.samples = std::stoi(next());
        else if (a == "--launches") o.launches = std::stoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::stoi(next());
        else if (a == "--out") o.out = next();
        else if (a == "--verify-small-m") o.verify_small_m = true;
        else if (a == "--split") o.split = static_cast<uint32_t>(std::stoi(next()));
        else if (a == "--mxfp4-split") o.mxfp4_split = static_cast<uint32_t>(std::stoi(next()));
        else if (a == "--mxfp4-unroll") o.mxfp4_unroll = static_cast<uint32_t>(std::stoi(next()));
        else if (a == "--mxfp4-fold") o.mxfp4_fold = static_cast<uint32_t>(std::stoi(next()));
        else if (a == "--mxfp4-spread") o.mxfp4_spread = static_cast<uint32_t>(std::stoi(next()));
        else if (a == "--mxfp4-a-rowref") o.mxfp4_a_rowref = static_cast<uint32_t>(std::stoi(next()));
        else if (a == "--mxfp4-a-fold") o.mxfp4_a_fold = static_cast<uint32_t>(std::stoi(next()));

        else if (a == "--check") o.check = true;
        else if (a == "--model-dir") o.model_dir = next();
        else if (a == "--actual-qwen-shapes") o.actual_qwen_shapes = true;
        else if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(stderr); return 2; }
    }
    if (o.verify_small_m) o.rows = {1, 2, 4, 8};
    if (o.rows.empty()) { o.rows = {1, 16, 32, 64, 128}; }
    if (o.warmup < 0 || o.samples <= 0 || o.launches <= 0) {
        fprintf(stderr, "bad --warmup/--samples/--launches\n");
        return 2;
    }
    if (o.dtype != "bf16" && o.dtype != "fp8" && o.dtype != "mxfp4") {
        fprintf(stderr, "valid --dtype: bf16|fp8|mxfp4 (got %s)\n", o.dtype.c_str());
        return 2;
    }
    if (o.variant != "correctness" && o.variant != "gemm" && o.variant != "auto") {
        fprintf(stderr, "valid --variant: correctness|gemm|auto (got %s)\n", o.variant.c_str());
        return 2;
    }
    if (o.dtype == "fp8") return run_gemm_fp8(o);
    if (o.dtype == "mxfp4") return run_gemm_mxfp4(o);
    return run_gemm_bf16(o);
}
