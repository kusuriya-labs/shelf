#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/kernels/optimized/rmsnorm.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/runtime/program/program.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <shelf/models/qwen35/runtime/selector_types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include <set>
#include <algorithm>
#include <functional>

namespace {

using VD = ::ps::runtime::ValueDType;
// use ::ps::kernel:: prefix

struct Profile {
    VD input_dtype = VD::BF16;
    VD output_dtype = VD::BF16;
    VD weight_dtype = VD::BF16;
    ::ps::execution::RmsNormSelectorWeightLayout weight_layout =
        ::ps::execution::RmsNormSelectorWeightLayout::None;
    uint32_t weight_mode = 0;  // 0 ONE_PLUS, 1 DIRECT
    uint32_t features = 0;
    uint32_t group_size = 0;
};

struct Options {
    std::string variant = "optimized";
    std::vector<uint32_t> rows = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::string input_dtype = "bf16";
    std::string output_dtype = "bf16";
    std::string weight_dtype = "bf16";
    std::string mode = "one_plus";
    uint32_t features = 2560;
    uint32_t group_size = 2560;
    std::string weight_layout = "per_feature";
    float eps = 1e-6f;
    int warmup = 100;
    int samples = 50;
    int launches = 20;
    uint32_t seed = 1234;
    int device = 0;
    bool check = false;
    std::string out;
    std::string model_dir;
    bool actual_qwen_profiles = false;
    double arena_gib = 8.0;
};

void usage(FILE* f) {
    fprintf(f,
        "shelf-bench rmsnorm [options]\n"
        "  --variant correctness|optimized|auto   (default optimized)\n"
        "  --rows 1,2,4,...                       (default 1..2048)\n"
        "  --input-dtype bf16|f32\n"
        "  --output-dtype bf16|f32\n"
        "  --weight-dtype bf16|f32|none\n"
        "  --mode one_plus|direct\n"
        "  --features N\n"
        "  --group-size N\n"
        "  --weight-layout per_feature|per_group|none\n"
        "  --eps FLOAT\n"
        "  --warmup N\n"
        "  --samples N\n"
        "  --launches N\n"
        "  --seed N\n"
        "  --device N\n"
        "  --check\n"
        "  --out PATH\n"
        "  --model-dir PATH\n"
        "  --actual-qwen-profiles\n"
        "  --arena-gib N\n");
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

bool parse_dtype(const std::string& s, VD& out) {
    if (s == "bf16") { out = VD::BF16; return true; }
    if (s == "f32") { out = VD::F32; return true; }
    if (s == "none") { return false; }
    return false;
}

bool parse_layout(const std::string& s, ::ps::execution::RmsNormSelectorWeightLayout& out) {
    using L = ::ps::execution::RmsNormSelectorWeightLayout;
    if (s == "per_feature") { out = L::PerFeature; return true; }
    if (s == "per_group") { out = L::PerGroup; return true; }
    if (s == "none") { out = L::None; return true; }
    return false;
}

const char* vd_name(VD v) {
    return v == VD::BF16 ? "bf16" : (v == VD::F32 ? "f32" : "i32");
}
const char* layout_name(::ps::execution::RmsNormSelectorWeightLayout l) {
    using L = ::ps::execution::RmsNormSelectorWeightLayout;
    return l == L::PerFeature ? "per_feature" : (l == L::PerGroup ? "per_group" : "none");
}
const char* mode_name(uint32_t m) { return m == 0 ? "one_plus" : "direct"; }

::ps::kernel::RmsNormDataType to_kd(VD v) { return v == VD::F32 ? ::ps::kernel::RmsNormDataType::F32 : ::ps::kernel::RmsNormDataType::BF16; }
::ps::kernel::RmsNormWeightLayout to_kl(::ps::execution::RmsNormSelectorWeightLayout l) {
    using L = ::ps::execution::RmsNormSelectorWeightLayout;
    if (l == L::PerFeature) return ::ps::kernel::RmsNormWeightLayout::PER_FEATURE;
    if (l == L::PerGroup) return ::ps::kernel::RmsNormWeightLayout::PER_GROUP;
    return ::ps::kernel::RmsNormWeightLayout::NONE;
}
::ps::kernel::RmsNormWeightMode to_km(uint32_t m) {
    return m == 0 ? ::ps::kernel::RmsNormWeightMode::ONE_PLUS : ::ps::kernel::RmsNormWeightMode::DIRECT;
}
uint32_t weight_elements(const Profile& p) {
    if (p.weight_layout == ::ps::execution::RmsNormSelectorWeightLayout::PerFeature) return p.features;
    if (p.weight_layout == ::ps::execution::RmsNormSelectorWeightLayout::PerGroup) return p.group_size;
    return 0;
}
uint32_t dtype_bytes(VD v) { return v == VD::F32 ? 4u : 2u; }

std::vector<Profile> extract_actual_profiles(const std::string& model_dir, double arena_gib,
                                             int device, hipStream_t stream) {
    std::size_t arena_bytes = (std::size_t)(arena_gib * 1024.0 * 1024.0 * 1024.0);
    auto arena_res = ps::gpu::GpuArena::create(device, arena_bytes);
    if (!arena_res.ok()) return {};
    auto arena = arena_res.release();
    ps::qwen35::Qwen35LoadOptions lo;
    auto model_res = ps::qwen35::Qwen35Model::load_from_safetensors(model_dir, arena, stream, lo);
    if (!model_res.ok()) return {};
    auto model = model_res.release();
    auto lp = ps::qwen35::lower_qwen35_to_primitives(model.text_config(), model.weights());
    if (!lp.ok()) return {};
    auto lowered = lp.release();
    const auto& g = lowered.graph;
    std::set<std::string> seen;
    std::vector<Profile> out;
    for (const auto& gn : g.nodes) {
        auto* rn = std::get_if<ps::runtime::RmsNormNode>(&gn.node);
        if (!rn) continue;
        Profile p;
        p.input_dtype = g.value_dtype(gn.inputs[0]);
        p.output_dtype = g.value_dtype(gn.outputs[0]);
        p.features = g.value_features(gn.inputs[0]);
        uint32_t grp = rn->group_size;
        p.group_size = (grp != 0 && grp <= p.features) ? grp : p.features;
        p.weight_mode = (rn->weight_mode == ps::runtime::RmsNormWeightMode::ONE_PLUS) ? 0u : 1u;
        p.weight_dtype = VD::BF16;
        uint32_t w_elems = 0;
        if (rn->parameter_index != UINT32_MAX && rn->parameter_index < lowered.parameters.size()) {
            w_elems = (uint32_t)lowered.parameters[rn->parameter_index].elements;
            p.weight_dtype = lowered.parameters[rn->parameter_index].dtype;
        }
        if (rn->parameter_index == UINT32_MAX) {
            p.weight_layout = ::ps::execution::RmsNormSelectorWeightLayout::None;
        } else if (w_elems == p.features) {
            p.weight_layout = ::ps::execution::RmsNormSelectorWeightLayout::PerFeature;
        } else if (w_elems == p.group_size) {
            p.weight_layout = ::ps::execution::RmsNormSelectorWeightLayout::PerGroup;
        } else {
            p.weight_layout = ::ps::execution::RmsNormSelectorWeightLayout::None;
        }
        char key[128];
        std::snprintf(key, sizeof key, "%d|%d|%d|%s|%u|%u|%u", (int)p.input_dtype, (int)p.output_dtype,
                      (int)p.weight_dtype, layout_name(p.weight_layout), p.weight_mode, p.features, p.group_size);
        if (seen.insert(key).second) out.push_back(p);
    }
    return out;
}

struct RmsEnv {
    Profile p;
    uint32_t rows = 0;
    void* in_dev = nullptr;
    void* out_dev = nullptr;
    void* wt_dev = nullptr;
    // correctness program
    void* values_dev = nullptr;
    void* params_dev = nullptr;
    void* ctx_dev = nullptr;
    float* scratch = nullptr;
    uint32_t* error_word = nullptr;
    ::ps::runtime::DeviceProgramView prog{};
    ::ps::runtime::DispatchBinding binding{};
    std::vector<double> h_in;
    std::vector<double> h_wt;

    ~RmsEnv() { teardown(); }
    void teardown() {
        ps::gpu::discard_cleanup_result(hipFree(in_dev)); in_dev = nullptr;
        ps::gpu::discard_cleanup_result(hipFree(out_dev)); out_dev = nullptr;
        ps::gpu::discard_cleanup_result(hipFree(wt_dev)); wt_dev = nullptr;
        ps::gpu::discard_cleanup_result(hipFree(values_dev)); values_dev = nullptr;
        ps::gpu::discard_cleanup_result(hipFree(params_dev)); params_dev = nullptr;
        ps::gpu::discard_cleanup_result(hipFree(ctx_dev)); ctx_dev = nullptr;
        ps::gpu::discard_cleanup_result(hipFree(scratch)); scratch = nullptr;
        ps::gpu::discard_cleanup_result(hipFree(error_word)); error_word = nullptr;
    }
};

void gen_data(RmsEnv& e, uint32_t seed) {
    const uint32_t F = e.p.features;
    const size_t n = (size_t)e.rows * F;
    e.h_in.resize(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 2000);
    for (size_t i = 0; i < n; ++i) {
        uint32_t h = std::hash<uint64_t>{}((uint64_t)i * 7919ull + seed);
        e.h_in[i] = ((double)(h % 2001u) - 1000.0) / 1000.0;
    }
    const uint32_t w = weight_elements(e.p);
    e.h_wt.resize(w);
    for (uint32_t i = 0; i < w; ++i) {
        uint32_t h = std::hash<uint64_t>{}((uint64_t)i * 6151ull + seed + 1);
        e.h_wt[i] = ((double)(h % 1001u) - 500.0) / 1000.0;
    }
    (void)dist; (void)F;
}

bool upload(RmsEnv& e) {
    const uint32_t F = e.p.features;
    const size_t n = (size_t)e.rows * F;
    const uint32_t in_b = dtype_bytes(e.p.input_dtype);
    const uint32_t out_b = dtype_bytes(e.p.output_dtype);
    const uint32_t wt_b = e.p.weight_layout == ::ps::execution::RmsNormSelectorWeightLayout::None
                              ? 0 : dtype_bytes(e.p.weight_dtype);
    const uint32_t w = weight_elements(e.p);
    if (hipMalloc(&e.in_dev, n * in_b) != hipSuccess) return false;
    if (hipMalloc(&e.out_dev, n * out_b) != hipSuccess) { e.teardown(); return false; }
    if (w > 0 && hipMalloc(&e.wt_dev, (size_t)w * wt_b) != hipSuccess) { e.teardown(); return false; }

    std::vector<uint8_t> hbuf(n * in_b);
    for (size_t i = 0; i < n; ++i) {
        if (e.p.input_dtype == VD::F32) {
            float v = (float)e.h_in[i];
            std::memcpy(&hbuf[i * 4], &v, 4);
        } else {
            uint16_t b = ps::f32_to_bf16_rne((float)e.h_in[i]);
            std::memcpy(&hbuf[i * 2], &b, 2);
        }
    }
    if (hipMemcpy(e.in_dev, hbuf.data(), hbuf.size(), hipMemcpyHostToDevice) != hipSuccess) { e.teardown(); return false; }
    std::vector<uint8_t> wbuf((size_t)w * wt_b);
    for (uint32_t i = 0; i < w; ++i) {
        if (e.p.weight_dtype == VD::F32) {
            float v = (float)e.h_wt[i];
            std::memcpy(&wbuf[(size_t)i * 4], &v, 4);
        } else {
            uint16_t b = ps::f32_to_bf16_rne((float)e.h_wt[i]);
            std::memcpy(&wbuf[(size_t)i * 2], &b, 2);
        }
    }
    if (w > 0 && hipMemcpy(e.wt_dev, wbuf.data(), wbuf.size(), hipMemcpyHostToDevice) != hipSuccess) {
        e.teardown(); return false;
    }
    return true;
}

bool build_correctness(RmsEnv& e) {
    const uint32_t F = e.p.features;
    const uint32_t grp = e.p.group_size;
    const bool has_w = e.p.weight_layout != ::ps::execution::RmsNormSelectorWeightLayout::None;
    ::ps::runtime::DeviceValueBinding values[2];
    values[0] = {};
    values[0].logical_value = 0;
    values[0].storage = ::ps::runtime::ValueStorage::EXTERNAL_INPUT;
    values[0].dtype = e.p.input_dtype;
    values[0].row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    values[0].slot = 0;
    values[0].feature_count = F;
    values[0].row_stride = F;
    values[0].offset = reinterpret_cast<uint64_t>(e.in_dev);
    values[0].bytes = (size_t)e.rows * F * dtype_bytes(e.p.input_dtype);
    values[1] = {};
    values[1].logical_value = 1;
    values[1].storage = ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT;
    values[1].dtype = e.p.output_dtype;
    values[1].row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
    values[1].slot = 0;
    values[1].feature_count = F;
    values[1].row_stride = F;
    values[1].offset = reinterpret_cast<uint64_t>(e.out_dev);
    values[1].bytes = (size_t)e.rows * F * dtype_bytes(e.p.output_dtype);
    if (hipMalloc(&e.values_dev, sizeof(values)) != hipSuccess) return false;
    if (hipMemcpy(e.values_dev, values, sizeof(values), hipMemcpyHostToDevice) != hipSuccess) return false;
    ::ps::runtime::StaticParameterSlot param{};
    param.device_ptr = e.wt_dev;
    param.elements = weight_elements(e.p);
    param.dtype = e.p.weight_dtype;
    if (has_w) {
        if (hipMalloc(&e.params_dev, sizeof(param)) != hipSuccess) return false;
        if (hipMemcpy(e.params_dev, &param, sizeof(param), hipMemcpyHostToDevice) != hipSuccess) return false;
    }
    ::ps::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = e.rows;
    ctx.num_outputs = 0;
    if (hipMalloc(&e.ctx_dev, sizeof(ctx)) != hipSuccess) return false;
    if (hipMemcpy(e.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) return false;
    const uint32_t kBlocks = 64;
    const uint32_t kCap = 2048;
    if (hipMalloc(&e.scratch, (size_t)kBlocks * kCap * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&e.error_word, sizeof(uint32_t)) != hipSuccess) return false;
    e.prog = {};
    e.prog.values = static_cast<const ::ps::runtime::DeviceValueBinding*>(e.values_dev);
    e.prog.value_count = 2;
    e.prog.parameter_table = has_w ? e.params_dev : nullptr;
    e.prog.parameter_slot_count = has_w ? 1u : 0u;
    e.prog.workspace_base = 0;
    e.binding = {};
    e.binding.kernel_id = ::ps::runtime::KernelId::RMS_NORM;
    e.binding.input_slots[0] = 0;
    e.binding.input_count = 1;
    e.binding.output_slots[0] = 1;
    e.binding.output_count = 1;
    e.binding.parameter_index0 = has_w ? 0u : ::ps::runtime::kNoParameter;
    e.binding.group_size = grp;
    e.binding.flags = e.p.weight_mode;
    e.binding.scalar_a = 1e-6f;
    return true;
}

hipError_t launch_correctness(const RmsEnv& e, hipStream_t stream) {
    if (hipMemsetAsync(e.error_word, 0, sizeof(uint32_t), stream) != hipSuccess) return hipGetLastError();
    auto st = ps::kernel::launch_model_dispatch_correctness(
        e.prog, e.binding, static_cast<::ps::runtime::DeviceBatchContext*>(e.ctx_dev),
        ps::kernel::ModelDispatchStateView{}, e.scratch, e.error_word, stream);
    if (!st.ok()) return hipErrorUnknown;
    return hipGetLastError();
}

hipError_t launch_optimized(const RmsEnv& e, hipStream_t stream) {
    const uint32_t F = e.p.features;
    return ps::kernel::launch_rmsnorm(
        e.in_dev, to_kd(e.p.input_dtype),
        e.wt_dev, to_kd(e.p.weight_dtype),
        to_kl(e.p.weight_layout), to_km(e.p.weight_mode),
        e.out_dev, to_kd(e.p.output_dtype),
        e.rows, F, e.p.group_size, F, F, 1e-6f, stream);
}

float bf16_rne_f32(float v) {
    uint32_t r = ps::f32_to_bf16_rne(v);
    uint32_t u = ((uint32_t)r) << 16;
    float o; std::memcpy(&o, &u, 4);
    return o;
}

double ref_value(const RmsEnv& e, uint32_t m, uint32_t g0, uint32_t h) {
    const uint32_t F = e.p.features;
    const uint32_t grp = e.p.group_size;
    auto xat = [&](uint32_t row, uint32_t col) -> double {
        double x = e.h_in[(size_t)row * F + col];
        if (e.p.input_dtype == VD::BF16) x = (double)bf16_rne_f32((float)x);
        return x;
    };
    double sum = 0.0;
    for (uint32_t hh = 0; hh < grp; ++hh) {
        double x = xat(m, g0 + hh);
        sum += x * x;
    }
    double inv = 1.0 / std::sqrt(sum / (double)grp + 1e-6);
    double x = xat(m, g0 + h);
    double y = x * inv;
    if (e.p.input_dtype == VD::F32) y = (double)bf16_rne_f32((float)y);
    if (e.p.weight_layout != ::ps::execution::RmsNormSelectorWeightLayout::None) {
        uint32_t widx =
            (e.p.weight_layout == ::ps::execution::RmsNormSelectorWeightLayout::PerFeature) ? (g0 + h) : h;
        double w = (double)bf16_rne_f32((float)e.h_wt[widx]);
        y *= (e.p.weight_mode == 0) ? (1.0 + w) : w;
    }
    return y;
}

bool check_output(const RmsEnv& e) {
    const uint32_t F = e.p.features;
    const size_t n = (size_t)e.rows * F;
    const uint32_t out_b = dtype_bytes(e.p.output_dtype);
    std::vector<uint8_t> buf(n * out_b);
    if (hipMemcpy(buf.data(), e.out_dev, buf.size(), hipMemcpyDeviceToHost) != hipSuccess) return false;
    const double bf16_eps = 0.00390625;
    for (uint32_t m = 0; m < e.rows; ++m) {
        for (uint32_t g0 = 0; g0 < F; g0 += e.p.group_size) {
            for (uint32_t h = 0; h < e.p.group_size; ++h) {
                const size_t c = (size_t)m * F + g0 + h;
                double got = 0.0;
                if (e.p.output_dtype == VD::F32) { float f; std::memcpy(&f, &buf[c * 4], 4); got = f; }
                else { uint16_t b; std::memcpy(&b, &buf[c * 2], 2); float f; uint32_t u = ((uint32_t)b) << 16; std::memcpy(&f, &u, 4); got = f; }
                const double ref = ref_value(e, m, g0, h);
                const double da = std::fabs(got - ref);
                const double tol = (e.p.output_dtype == VD::F32 && e.p.input_dtype != VD::F32)
                    ? std::max(2e-4, 5e-4 * std::max(1.0, std::fabs(ref)))
                    : 2.0 * bf16_eps * std::max(1.0, std::fabs(ref));
                if (da > tol) return false;
            }
        }
    }
    return true;
}

}  // namespace

int run_rmsnorm(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return argv[++i];
        };
        if (a == "--variant") o.variant = next();
        else if (a == "--rows") { o.rows.clear(); if (!parse_rows(next(), o.rows)) { fprintf(stderr, "bad --rows\n"); return 2; } }
        else if (a == "--input-dtype") o.input_dtype = next();
        else if (a == "--output-dtype") o.output_dtype = next();
        else if (a == "--weight-dtype") o.weight_dtype = next();
        else if (a == "--mode") o.mode = next();
        else if (a == "--features") o.features = (uint32_t)std::stoul(next());
        else if (a == "--group-size") o.group_size = (uint32_t)std::stoul(next());
        else if (a == "--weight-layout") o.weight_layout = next();
        else if (a == "--eps") o.eps = std::atof(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--samples") o.samples = std::stoi(next());
        else if (a == "--launches") o.launches = std::stoi(next());
        else if (a == "--seed") o.seed = (uint32_t)std::stoul(next());
        else if (a == "--device") o.device = std::stoi(next());
        else if (a == "--check") o.check = true;
        else if (a == "--out") o.out = next();
        else if (a == "--model-dir") o.model_dir = next();
        else if (a == "--actual-qwen-profiles") o.actual_qwen_profiles = true;
        else if (a == "--arena-gib") o.arena_gib = std::atof(next());
        else if (a == "--help" || a == "-h") { usage(stdout); return 0; }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(stderr); return 2; }
    }
    if (o.variant != "correctness" && o.variant != "optimized" && o.variant != "auto") {
        fprintf(stderr, "bad --variant: %s\n", o.variant.c_str()); return 2;
    }
    if (o.warmup < 0 || o.samples <= 0 || o.launches <= 0) { fprintf(stderr, "bad timing options\n"); return 2; }

    if (hipSetDevice(o.device) != hipSuccess) { fprintf(stderr, "hipSetDevice failed\n"); return 1; }
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::vector<Profile> profiles;
    if (o.actual_qwen_profiles) {
        if (o.model_dir.empty()) { fprintf(stderr, "--actual-qwen-profiles requires --model-dir\n"); return 2; }
        profiles = extract_actual_profiles(o.model_dir, o.arena_gib, o.device, stream);
        if (profiles.empty()) { fprintf(stderr, "profile extraction failed\n"); return 1; }
        std::printf("extracted %zu actual profiles\n", profiles.size());
    } else {
        Profile p;
        if (!parse_dtype(o.input_dtype, p.input_dtype) || !parse_dtype(o.output_dtype, p.output_dtype)) {
            fprintf(stderr, "bad input/output dtype\n"); return 2;
        }
        if (o.weight_dtype == "none") {
            p.weight_layout = ::ps::execution::RmsNormSelectorWeightLayout::None;
            p.weight_dtype = VD::BF16;
        } else if (!parse_dtype(o.weight_dtype, p.weight_dtype)) {
            fprintf(stderr, "bad weight dtype\n"); return 2;
        }
        if (!parse_layout(o.weight_layout, p.weight_layout)) { fprintf(stderr, "bad weight layout\n"); return 2; }
        p.weight_mode = (o.mode == "direct") ? 1u : 0u;
        p.features = o.features;
        p.group_size = (o.group_size != 0 && o.group_size <= o.features) ? o.group_size : o.features;
        profiles.push_back(p);
    }

    if (!o.out.empty()) {
        FILE* f = std::fopen(o.out.c_str(), "a");
        if (f) {
            if (std::ftell(f) == 0)
                std::fprintf(f, "variant,input_dtype,output_dtype,weight_dtype,weight_layout,weight_mode,"
                                 "features,group_size,rows,warmup,samples,launches_per_sample,min_us,p50_us,p95_us,mean_us\n");
            std::fclose(f);
        }
    }

    int rc = 0;
    for (const Profile& prof : profiles) {
        for (uint32_t rows : o.rows) {
            RmsEnv e;
            e.p = prof;
            e.rows = rows;
            gen_data(e, o.seed);
            if (!upload(e)) { fprintf(stderr, "upload failed rows=%u\n", rows); rc = 1; continue; }
            bool need_corr = (o.variant == "correctness" || o.variant == "auto");
            if (need_corr) { if (!build_correctness(e)) { fprintf(stderr, "build_correctness failed\n"); rc = 1; e.teardown(); continue; } }
            if (hipDeviceSynchronize() != hipSuccess) { rc = 1; e.teardown(); continue; }

            // auto: decide path once
            bool auto_opt = true;
            if (o.variant == "auto") {
                ::ps::execution::RmsNormSelectorInput si;
                si.input_dtype = prof.input_dtype;
                si.output_dtype = prof.output_dtype;
                si.weight_dtype = prof.weight_dtype;
                si.weight_layout = prof.weight_layout;
                si.weight_mode = prof.weight_mode;
                si.features = prof.features;
                si.group_size = prof.group_size;
                si.rows = rows;
                auto impl = ::ps::execution::select_rmsnorm_implementation(si);
                auto_opt = (impl == ::ps::execution::RmsNormImplementation::Optimized);
            }
            bool use_opt = (o.variant == "optimized") || (o.variant == "auto" && auto_opt);

            auto launch = [&, use_opt](hipStream_t s) -> hipError_t {
                return use_opt ? launch_optimized(e, s) : launch_correctness(e, s);
            };

            if (o.check) {
                hipError_t lr = launch(stream);
                if (lr != hipSuccess) { fprintf(stderr, "check launch failed rows=%u (%s)\n", rows, hipGetErrorString(lr)); rc = 1; e.teardown(); continue; }
                if (hipDeviceSynchronize() != hipSuccess) { rc = 1; e.teardown(); continue; }
                bool ok = check_output(e);
                std::printf("CHECK %-9s %s->%s %s/%s F=%u G=%u rows=%-5u %s\n",
                            o.variant.c_str(), vd_name(prof.input_dtype), vd_name(prof.output_dtype),
                            layout_name(prof.weight_layout), mode_name(prof.weight_mode), prof.features,
                            prof.group_size, rows, ok ? "PASS" : "FAIL");
                if (!ok) rc = 1;
            }

            for (int i = 0; i < o.warmup; ++i) if (launch(stream) != hipSuccess) rc = 1;
            if (hipStreamSynchronize(stream) != hipSuccess) { rc = 1; e.teardown(); continue; }
            hipEvent_t ev0 = nullptr, ev1 = nullptr;
            if (hipEventCreate(&ev0) != hipSuccess || hipEventCreate(&ev1) != hipSuccess) { rc = 1; e.teardown(); continue; }
            std::vector<double> per;
            per.reserve(o.samples);
            for (int s = 0; s < o.samples; ++s) {
                if (hipEventRecord(ev0, stream) != hipSuccess) break;
                for (int l = 0; l < o.launches; ++l) if (launch(stream) != hipSuccess) break;
                if (hipEventRecord(ev1, stream) != hipSuccess) break;
                if (hipEventSynchronize(ev1) != hipSuccess) break;
                float ms = 0.0f;
                if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess) break;
                per.push_back(ms * 1000.0 / (double)o.launches);
            }
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev0));
            ps::gpu::discard_cleanup_result(hipEventDestroy(ev1));
            if (per.empty()) { rc = 1; e.teardown(); continue; }
            const double mn = percentile(per, 0.0);
            const double p50 = percentile(per, 0.5);
            const double p95 = percentile(per, 0.95);
            double mean = 0.0; for (double v : per) mean += v; mean /= (double)per.size();
            const std::string vtag = (o.variant == "auto") ? (auto_opt ? "optimized" : "correctness") : o.variant;
            if (!o.out.empty()) {
                FILE* f = std::fopen(o.out.c_str(), "a");
                if (f) {
                    std::fprintf(f, "%s,%s,%s,%s,%s,%s,%u,%u,%u,%d,%d,%d,%.3f,%.3f,%.3f,%.3f\n",
                                 vtag.c_str(), vd_name(prof.input_dtype), vd_name(prof.output_dtype),
                                 vd_name(prof.weight_dtype), layout_name(prof.weight_layout),
                                 mode_name(prof.weight_mode), prof.features, prof.group_size, rows,
                                 o.warmup, o.samples, o.launches, mn, p50, p95, mean);
                    std::fclose(f);
                }
            }
            std::printf("%-9s %s->%s %s/%s F=%-5u G=%-5u rows=%-5u min=%.3f p50=%.3f p95=%.3f mean=%.3f us\n",
                        vtag.c_str(), vd_name(prof.input_dtype), vd_name(prof.output_dtype),
                        layout_name(prof.weight_layout), mode_name(prof.weight_mode), prof.features,
                        prof.group_size, rows, mn, p50, p95, mean);
            e.teardown();
        }
    }
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return rc;
}
