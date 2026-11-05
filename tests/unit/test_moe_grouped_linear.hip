#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/optimized/moe/router.h>
#include <shelf/models/qwen35moe/weights/moe_expert_load.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/quantization/codec/e4m3.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

#include "moe_expert_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace quant = shelf::quantization;
namespace rt = shelf::runtime;
namespace moe = shelf::kernel;
using namespace moe_expert_fixture;
using shelf::qwen35moe::MoeExpertGroupPlan;

namespace {

constexpr uint32_t kRowTile = moe::kMxfp4GemmRowTile;
constexpr std::uint16_t kSentinel = 0x7f7fu;

int g_fail = 0;
bool g_w13_fold_exact = false;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

float bf16_to_f32(shelf::bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

template <typename T>
T* alloc(std::size_t count) {
    T* p = nullptr;
    if (hipMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)) != hipSuccess) {
        std::printf("FAIL hipMalloc %zu\n", count * sizeof(T));
        std::exit(1);
    }
    return p;
}

template <typename T>
std::vector<T> d2h(const T* dev, std::size_t count) {
    std::vector<T> host(count);
    if (count == 0)
        return host;
    if (hipMemcpy(host.data(), dev, count * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL d2h %zu\n", count);
        std::exit(1);
    }
    return host;
}

struct Device {
    uint8_t* w13_codes = nullptr;
    uint8_t* w13_scales = nullptr;
    uint8_t* w13_ref = nullptr;
    uint8_t* w2_codes = nullptr;
    uint8_t* w2_scales = nullptr;
    uint8_t* w2_ref = nullptr;
    float* logits = nullptr;
    int32_t* sel_expert = nullptr;
    float* sel_weight = nullptr;
    uint32_t* tile_counts = nullptr;
    uint32_t* row_begin = nullptr;
    uint32_t* tile_base = nullptr;
    uint32_t* group_begin = nullptr;
    uint32_t* group_expert = nullptr;
    uint32_t* group_rows = nullptr;
    uint32_t* group_count = nullptr;
    int32_t* row_token = nullptr;
    int32_t* slot_row_of = nullptr;
    shelf::bf16_t* hidden13 = nullptr;
    shelf::bf16_t* hidden2 = nullptr;
    uint8_t* plane_codes = nullptr;
    float* plane_scales = nullptr;
    shelf::bf16_t* out13 = nullptr;
    shelf::bf16_t* out2 = nullptr;
};

struct Case {
    uint32_t rows = 1;
    uint32_t top_k = 2;
    uint32_t experts = kExperts;
    uint32_t group_capacity = 0;
    std::vector<float> logits;
};

std::vector<float> make_logits(uint32_t rows, uint32_t experts, bool skew) {
    std::vector<float> out(static_cast<std::size_t>(rows) * experts);
    uint32_t s = 31u + rows;
    auto rnd = [&s]() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xffffu) / 4096.0f - 8.0f;
    };
    for (uint32_t t = 0; t < rows; ++t) {
        for (uint32_t e = 0; e < experts; ++e)
            out[static_cast<std::size_t>(t) * experts + e] = rnd();
        if (skew) {
            out[static_cast<std::size_t>(t) * experts + 0u] = 7.0f;
            out[static_cast<std::size_t>(t) * experts + 1u] = 6.0f;
        }
    }
    return out;
}

struct ExpertWeights {
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    uint32_t rows = 0;
    uint32_t k_padded = 0;

    uint8_t code(uint32_t row, uint32_t k) const {
        const uint8_t packed = codes[static_cast<std::size_t>(row) * (k_padded / 2u) + (k >> 1)];
        return (k & 1u) == 0u ? static_cast<uint8_t>(packed & 0x0Fu)
                              : static_cast<uint8_t>(packed >> 4);
    }
    uint8_t scale(uint32_t row, uint32_t block) const {
        return scales[static_cast<std::size_t>(row) * (k_padded / 32u) + block];
    }
};

ExpertWeights fused_w13_expert(const Fixture& f, uint32_t layer, uint32_t expert,
                               const MoeExpertGroupPlan& plan) {
    ExpertWeights w;
    w.rows = plan.w13_rows;
    w.k_padded = plan.w13_k_padded;
    w.codes.assign(static_cast<std::size_t>(w.rows) * (w.k_padded / 2u), 0u);
    w.scales.assign(static_cast<std::size_t>(w.rows) * (w.k_padded / 32u), 0u);
    const uint32_t inter = plan.w13_gate_rows;
    for (uint32_t half = 0; half < 2u; ++half) {
        const std::string name = shelf::qwen35moe::moe_expert_tensor_name(
            layer, expert, half == 0u ? "gate_proj" : "up_proj", false);
        const std::vector<uint8_t>& src_codes = f.codes.at(name);
        const std::vector<uint8_t>& src_scales = f.scales.at(name + "_scale");
        std::memcpy(w.codes.data() + static_cast<std::size_t>(half) * inter * (w.k_padded / 2u),
                    src_codes.data(), src_codes.size());
        std::memcpy(w.scales.data() + static_cast<std::size_t>(half) * inter * (w.k_padded / 32u),
                    src_scales.data(), src_scales.size());
    }
    return w;
}

ExpertWeights w2_expert(const Fixture& f, uint32_t layer, uint32_t expert,
                        const MoeExpertGroupPlan& plan) {
    ExpertWeights w;
    w.rows = plan.w2_rows;
    w.k_padded = plan.w2_k_padded;
    w.codes = f.codes.at(shelf::qwen35moe::moe_expert_tensor_name(layer, expert, "down_proj",
                                                                  false));
    w.scales = f.scales.at(shelf::qwen35moe::moe_expert_tensor_name(layer, expert, "down_proj",
                                                                    true));
    return w;
}

struct PlanView {
    std::vector<uint32_t> group_expert;
    std::vector<uint32_t> group_rows;
    std::vector<int32_t> row_token;
    uint32_t group_count = 0;
};

// Double reference over exactly the operands the GPU consumes: canonical weight codes and
// E8M0 scales, and the E4M3 activation plane the quantizer wrote. Output rows are group
// rows, so rows outside a group's count stay 0.
std::vector<double> expected_grouped(const std::vector<ExpertWeights>& weights,
                                     const std::vector<uint8_t>& plane_codes,
                                     const std::vector<float>& plane_scales,
                                     const rt::Int8ActivationWorkspaceLayout& layout,
                                     const PlanView& plan, uint32_t rows_cap,
                                     uint32_t out_features) {
    const uint32_t nb = layout.k_padded / quant::kMxfp4ScaleBlockK;
    const uint32_t scale_stride_floats = layout.scale_row_stride_bytes / sizeof(float);
    std::vector<double> out(static_cast<std::size_t>(rows_cap) * out_features, 0.0);
    for (uint32_t g = 0; g < plan.group_count; ++g) {
        const ExpertWeights& w = weights[plan.group_expert[g]];
        for (uint32_t i = 0; i < plan.group_rows[g]; ++i) {
            const uint32_t row = g * kRowTile + i;
            for (uint32_t o = 0; o < out_features; ++o) {
                double acc = 0.0;
                for (uint32_t ib = 0; ib < nb; ++ib) {
                    const double ws = quant::mxfp4::e8m0_decode_f32(w.scale(o, ib));
                    double p = 0.0;
                    for (uint32_t j = 0; j < quant::kMxfp4ScaleBlockK; ++j) {
                        const uint32_t kc = ib * quant::kMxfp4ScaleBlockK + j;
                        const double wv = quant::mxfp4::e2m1_decode_f32(w.code(o, kc));
                        const size_t off = static_cast<size_t>(kc >> 6) *
                                quant::fp8::kFragBlockBytes +
                            quant::fp8::fp8_frag_offset(row & 15u, kc & 63u);
                        const size_t base = static_cast<size_t>(row >> 4) *
                            (static_cast<size_t>(layout.code_row_stride_bytes) * 16u);
                        p += wv * static_cast<double>(
                                       quant::codec::e4m3_decode_f32(plane_codes[base + off]));
                    }
                    acc += p * ws;
                }
                acc *= static_cast<double>(
                    plane_scales[static_cast<std::size_t>(row) * scale_stride_floats]);
                out[static_cast<std::size_t>(row) * out_features + o] = acc;
            }
        }
    }
    return out;
}

void compare_bf16(const std::vector<shelf::bf16_t>& got, const std::vector<double>& want,
                  const std::vector<char>& written, uint32_t out_features,
                  const std::string& tag) {
    double l2_want = 0.0;
    double l2_diff = 0.0;
    std::size_t exact = 0;
    std::size_t compared = 0;
    for (uint32_t r = 0; r < written.size(); ++r) {
        if (!written[r])
            continue;
        for (uint32_t o = 0; o < out_features; ++o) {
            const std::size_t at = static_cast<std::size_t>(r) * out_features + o;
            const double a = want[at];
            const double b = bf16_to_f32(got[at]);
            l2_want += a * a;
            l2_diff += (a - b) * (a - b);
            if (got[at].data == shelf::f32_to_bf16_rne(static_cast<float>(a)))
                ++exact;
            ++compared;
        }
    }
    const double rel = std::sqrt(l2_diff) / std::max(std::sqrt(l2_want), 1e-9);
    const double ratio = compared ? static_cast<double>(exact) / static_cast<double>(compared) : 0.0;
    check(ratio >= 0.999, tag + ": bf16 exact on >= 99.9% of the double model (" +
                               std::to_string(ratio) + ")");
    // A bf16 output is within 2^-9 relative of the double model per element, so the RMS
    // of correct rounding alone sits at ~1.2e-3 and can pass 2e-3 when only a couple of
    // rows are written (rows == 1, top_k == 2). The primary gate is the bf16-exact ratio.
    check(rel < 4.0e-3, tag + ": rel_l2 < 4e-3 against the double model (" +
                                std::to_string(rel) + ")");
    std::printf("%s: rows %zu, elements %zu, bf16 exact %.4f, rel_l2 %.3e\n", tag.c_str(),
                written.size(), compared, ratio, rel);
}

PlanView run_router_and_plan(const Case& c, Device& d, hipStream_t stream,
                             const std::string& tag) {
    PlanView view;
    if (hipMemcpy(d.logits, c.logits.data(), c.logits.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess)
        check(false, tag + ": logits upload");
    const std::size_t counts_bytes =
        static_cast<std::size_t>(c.experts) * moe::moe_slot_tile_count(c.rows) * sizeof(uint32_t);
    if (hipMemset(d.tile_counts, 0, counts_bytes) != hipSuccess)
        check(false, tag + ": counts memset");
    if (moe::launch_moe_router(d.logits, c.experts, d.sel_expert, d.sel_weight, d.tile_counts,
                               c.rows, c.experts, c.top_k, 0u, c.experts, nullptr,
                               stream) != hipSuccess)
        check(false, tag + ": launch_moe_router");
    if (moe::launch_moe_slot_scan(d.tile_counts, d.row_begin, d.tile_base, d.group_begin,
                                  d.group_expert, d.group_rows, d.group_count, d.row_token, c.rows,
                                  c.experts, c.top_k, stream) != hipSuccess)
        check(false, tag + ": launch_moe_slot_scan");
    if (moe::launch_moe_slot_scatter(d.sel_expert, d.tile_base, d.row_begin, d.group_begin,
                                     d.row_token, d.slot_row_of, c.rows, c.experts, c.top_k,
                                     stream) != hipSuccess)
        check(false, tag + ": launch_moe_slot_scatter");
    if (hipStreamSynchronize(stream) != hipSuccess)
        check(false, tag + ": plan sync");

    view.group_expert = d2h(d.group_expert, c.group_capacity);
    view.group_rows = d2h(d.group_rows, c.group_capacity);
    view.row_token = d2h(d.row_token, static_cast<std::size_t>(c.group_capacity) * kRowTile);
    view.group_count = d2h(d.group_count, 1).at(0);
    check(view.group_count <= c.group_capacity, tag + ": group count within the host bound");
    uint32_t slots = 0u;
    for (uint32_t g = 0; g < view.group_count; ++g)
        slots += view.group_rows[g];
    check(slots == c.rows * c.top_k, tag + ": groups cover every selection");
    return view;
}

hipError_t moe_launch_quantize(const shelf::bf16_t* hidden, const Device& d,
                               const rt::Int8ActivationWorkspaceLayout& layout, uint32_t cols,
                               uint32_t rows_cap, hipStream_t stream) {
    return moe::launch_activation_quantize_mxfp4_rowref(
        hidden, d.plane_codes, d.plane_scales, rows_cap, cols, layout.k_padded, cols,
        layout.code_row_stride_bytes, layout.scale_row_stride_bytes, d.row_token, stream);
}

void run_grouped_case(const Fixture& f, const MoeExpertGroupPlan& plan, const Case& c, Device& d,
                      hipStream_t stream, const std::string& tag) {
    const PlanView view = run_router_and_plan(c, d, stream, tag);
    const uint32_t rows_cap = c.group_capacity * kRowTile;
    std::vector<char> written(static_cast<std::size_t>(rows_cap), 0);
    for (uint32_t g = 0; g < view.group_count; ++g)
        for (uint32_t i = 0; i < view.group_rows[g]; ++i)
            written[g * kRowTile + i] = 1;

    const uint32_t cols13 = plan.w13_cols;
    const auto layout13 = rt::Int8ActivationWorkspaceLayout::make(plan.w13_k_padded, rows_cap);
    if (moe_launch_quantize(d.hidden13, d, layout13, cols13, rows_cap, stream) != hipSuccess)
        check(false, tag + ": quantize w13 activation");
    const std::vector<uint8_t> plane_codes13 = d2h(d.plane_codes, layout13.codes_bytes());
    const std::vector<float> plane_scales13 = d2h(d.plane_scales, layout13.scales_bytes() / 4u);

    bool padding_zero = true;
    for (uint32_t g = 0; g < view.group_count; ++g) {
        for (uint32_t i = view.group_rows[g]; i < kRowTile; ++i) {
            const uint32_t row = g * kRowTile + i;
            if (view.row_token[row] != -1)
                padding_zero = false;
            const size_t base =
                static_cast<size_t>(row >> 4) * layout13.code_row_stride_bytes * 16u;
            for (uint32_t kc = 0; kc < layout13.k_padded; ++kc) {
                const size_t off = static_cast<size_t>(kc >> 6) * quant::fp8::kFragBlockBytes +
                                   quant::fp8::fp8_frag_offset(row & 15u, kc & 63u);
                if (plane_codes13[base + off] != 0u)
                    padding_zero = false;
            }
        }
    }
    check(padding_zero, tag + ": padding group rows carry row_token -1 and zero operands");

    std::vector<ExpertWeights> w13;
    for (uint32_t e = 0; e < plan.experts; ++e)
        w13.push_back(fused_w13_expert(f, 0, e, plan));
    std::vector<shelf::bf16_t> sentinel(static_cast<std::size_t>(rows_cap) * plan.w13_rows,
                                        shelf::bf16_t{kSentinel});

    auto launch13 = [&](const uint8_t* ref) {
        return moe::launch_gemm_mxfp4_w4a8_grouped(
            d.w13_codes, d.w13_scales, ref, reinterpret_cast<const int8_t*>(d.plane_codes),
            d.plane_scales, d.out13,
            moe::Mxfp4GemmOutputDType::BF16, plan.w13_rows, plan.w13_k_padded, plan.w13_rows,
            plan.w13_codes_batch_stride_bytes, plan.w13_scales_batch_stride_bytes,
            plan.w13_scales_row_stride_bytes, layout13.code_row_stride_bytes,
            layout13.scale_row_stride_bytes, plan.w13_rows, d.group_expert, d.group_rows,
            d.group_count, c.group_capacity, stream);
    };

    if (hipMemcpy(d.out13, sentinel.data(), sentinel.size() * sizeof(shelf::bf16_t),
                  hipMemcpyHostToDevice) != hipSuccess)
        check(false, tag + ": sentinel fill w13");
    check(launch13(nullptr) == hipSuccess, tag + ": launch grouped w13 unfolded");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        check(false, tag + ": grouped w13 sync");
        return;
    }
    const std::vector<shelf::bf16_t> got13 = d2h(d.out13, sentinel.size());
    const std::vector<double> want13 =
        expected_grouped(w13, plane_codes13, plane_scales13, layout13, view,
                         view.group_count * kRowTile, plan.w13_rows);
    compare_bf16(got13, want13, written, plan.w13_rows, tag + "/w13 unfold");

    bool guard_ok = true;
    for (uint32_t row = 0; row < rows_cap; ++row) {
        if (written[row])
            continue;
        for (uint32_t o = 0; o < plan.w13_rows; ++o)
            if (got13[static_cast<std::size_t>(row) * plan.w13_rows + o].data != kSentinel)
                guard_ok = false;
    }
    check(guard_ok, tag + ": rows outside group_rows[g] keep the sentinel (never stored)");

    check(g_w13_fold_exact, tag + ": the fixture w13 plane is fold-exact");
    if (hipMemcpy(d.out13, sentinel.data(), sentinel.size() * sizeof(shelf::bf16_t),
                  hipMemcpyHostToDevice) != hipSuccess)
        check(false, tag + ": sentinel refill w13");
    check(launch13(d.w13_ref) == hipSuccess, tag + ": launch grouped w13 folded");
    const std::vector<shelf::bf16_t> got_fold = d2h(d.out13, sentinel.size());
    compare_bf16(got_fold, want13, written, plan.w13_rows, tag + "/w13 fold");
    std::size_t same = 0;
    for (std::size_t i = 0; i < got13.size(); ++i)
        if (got13[i].data == got_fold[i].data)
            ++same;
    std::printf("%s/w13 fold-vs-unfold bf16 identical: %.4f\n", tag.c_str(),
                static_cast<double>(same) / static_cast<double>(got13.size()));

    const auto layout2 = rt::Int8ActivationWorkspaceLayout::make(plan.w2_k_padded, rows_cap);
    if (moe_launch_quantize(d.hidden2, d, layout2, plan.w2_cols, rows_cap, stream) != hipSuccess)
        check(false, tag + ": quantize w2 activation");
    const std::vector<uint8_t> plane_codes2 = d2h(d.plane_codes, layout2.codes_bytes());
    const std::vector<float> plane_scales2 = d2h(d.plane_scales, layout2.scales_bytes() / 4u);

    std::vector<ExpertWeights> w2;
    for (uint32_t e = 0; e < plan.experts; ++e)
        w2.push_back(w2_expert(f, 0, e, plan));

    std::vector<shelf::bf16_t> sentinel2(static_cast<std::size_t>(rows_cap) * plan.w2_rows,
                                         shelf::bf16_t{kSentinel});
    if (hipMemcpy(d.out2, sentinel2.data(), sentinel2.size() * sizeof(shelf::bf16_t),
                  hipMemcpyHostToDevice) != hipSuccess)
        check(false, tag + ": sentinel fill w2");
    const hipError_t w2_launch = moe::launch_gemm_mxfp4_w4a8_grouped(
        d.w2_codes, d.w2_scales, nullptr, reinterpret_cast<const int8_t*>(d.plane_codes),
            d.plane_scales, d.out2,
        moe::Mxfp4GemmOutputDType::BF16, plan.w2_rows, plan.w2_k_padded, plan.w2_rows,
        plan.w2_codes_batch_stride_bytes, plan.w2_scales_batch_stride_bytes,
        plan.w2_scales_row_stride_bytes, layout2.code_row_stride_bytes,
        layout2.scale_row_stride_bytes, plan.w2_rows, d.group_expert, d.group_rows, d.group_count,
        c.group_capacity, stream);
    check(w2_launch == hipSuccess, tag + ": launch grouped w2");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        check(false, tag + ": grouped w2 sync");
        return;
    }
    const std::vector<shelf::bf16_t> got2 = d2h(d.out2, sentinel2.size());
    const std::vector<double> want2 =
        expected_grouped(w2, plane_codes2, plane_scales2, layout2, view,
                         view.group_count * kRowTile, plan.w2_rows);
    compare_bf16(got2, want2, written, plan.w2_rows, tag + "/w2");
}

}  // namespace

int main() {
    std::printf("=== MoE grouped MXFP4 linear (GPU) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess)
        return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess)
        return 1;

    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("shelf_moe_grouped_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    Fixture f;
    if (!write_fixture(f, root / "mxfp4")) {
        std::printf("FAIL fixture write\n");
        return 1;
    }
    std::vector<std::string> names;
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        for (const auto& name : expert_names(layer))
            names.push_back(name);
        for (const auto& name : mlp_aux_names(layer))
            names.push_back(name);
    }
    auto source = quant::bundle::synthesize_hf_quantized_source(root / "mxfp4", names, false);
    if (!source.ok()) {
        std::printf("FAIL hf source: %s\n", source.status().message().c_str());
        return 1;
    }
    auto reader = quant::bundle::QuantizedModelReader::open_hf_source(source.release(), false);
    if (!reader.ok()) {
        std::printf("FAIL reader: %s\n", reader.status().message().c_str());
        return 1;
    }
    auto plan = shelf::qwen35moe::build_moe_expert_group_plan(f.config, 0);
    if (!plan.ok()) {
        std::printf("FAIL plan: %s\n", plan.status().message().c_str());
        return 1;
    }
    const MoeExpertGroupPlan& p = plan.value();

    auto arena = shelf::gpu::GpuArena::create(0, 64ull * 1024ull * 1024ull);
    if (!arena.ok()) {
        std::printf("FAIL arena\n");
        return 1;
    }
    shelf::gpu::GpuArena gpu_arena = arena.release();
    auto loaded = shelf::qwen35moe::load_moe_expert_group(reader.value(), p, gpu_arena, stream);
    if (!loaded.ok()) {
        std::printf("FAIL load: %s\n", loaded.status().message().c_str());
        return 1;
    }
    g_w13_fold_exact = loaded.value().w13.row_ref_exact;
    check(loaded.value().w13.row_ref_exact && !loaded.value().w2.row_ref_exact,
          "fixture w13 is fold-exact and w2 is not");

    constexpr uint32_t kMaxRows = 64u;
    constexpr uint32_t kMaxTopK = 4u;
    const uint32_t max_group_capacity = moe::moe_group_count_upper_bound(kMaxRows, kExperts,
                                                                        kMaxTopK);
    const uint32_t max_plane_rows = max_group_capacity * kRowTile;

    Device d;
    d.w13_codes = alloc<uint8_t>(p.w13_codes_bytes);
    d.w13_scales = alloc<uint8_t>(p.w13_scales_bytes);
    d.w13_ref = alloc<uint8_t>(static_cast<std::size_t>(p.experts) * p.w13_rows);
    d.w2_codes = alloc<uint8_t>(p.w2_codes_bytes);
    d.w2_scales = alloc<uint8_t>(p.w2_scales_bytes);
    d.w2_ref = alloc<uint8_t>(static_cast<std::size_t>(p.experts) * p.w2_rows);
    d.logits = alloc<float>(static_cast<std::size_t>(kMaxRows) * kExperts);
    d.sel_expert = alloc<int32_t>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);
    d.sel_weight = alloc<float>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);
    d.tile_counts = alloc<uint32_t>(
        static_cast<std::size_t>(kExperts) * moe::moe_slot_tile_count(kMaxRows));
    d.row_begin = alloc<uint32_t>(kExperts + 1u);
    d.tile_base = alloc<uint32_t>(
        static_cast<std::size_t>(kExperts) * moe::moe_slot_tile_count(kMaxRows));
    d.group_begin = alloc<uint32_t>(kExperts + 1u);
    d.group_expert = alloc<uint32_t>(max_group_capacity);
    d.group_rows = alloc<uint32_t>(max_group_capacity);
    d.group_count = alloc<uint32_t>(1);
    d.row_token = alloc<int32_t>(max_plane_rows);
    d.slot_row_of = alloc<int32_t>(static_cast<std::size_t>(kMaxRows) * kMaxTopK);
    d.hidden13 = alloc<shelf::bf16_t>(static_cast<std::size_t>(kMaxRows) * p.w13_cols);
    d.hidden2 = alloc<shelf::bf16_t>(static_cast<std::size_t>(kMaxRows) * p.w2_cols);
    d.plane_codes = alloc<uint8_t>(static_cast<std::size_t>(max_plane_rows) * p.w13_k_padded);
    d.plane_scales = alloc<float>(static_cast<std::size_t>(max_plane_rows) * 4u);
    d.out13 = alloc<shelf::bf16_t>(static_cast<std::size_t>(max_plane_rows) * p.w13_rows);
    d.out2 = alloc<shelf::bf16_t>(static_cast<std::size_t>(max_plane_rows) * p.w2_rows);

    auto copy_planes = [&](const shelf::weights::MatrixWeight& w, uint8_t* codes, uint8_t* scales,
                           uint8_t* ref, std::size_t codes_bytes, std::size_t scales_bytes,
                           std::size_t ref_bytes) {
        if (hipMemcpy(codes, w.codes.data(), codes_bytes, hipMemcpyDeviceToDevice) != hipSuccess ||
            hipMemcpy(scales, w.scales.data(), scales_bytes, hipMemcpyDeviceToDevice) !=
                hipSuccess ||
            hipMemcpy(ref, w.row_ref.data(), ref_bytes, hipMemcpyDeviceToDevice) != hipSuccess) {
            std::printf("FAIL plane copy\n");
            std::exit(1);
        }
    };
    copy_planes(loaded.value().w13, d.w13_codes, d.w13_scales, d.w13_ref, p.w13_codes_bytes,
                p.w13_scales_bytes, static_cast<std::size_t>(p.experts) * p.w13_rows);
    copy_planes(loaded.value().w2, d.w2_codes, d.w2_scales, d.w2_ref, p.w2_codes_bytes,
                p.w2_scales_bytes, static_cast<std::size_t>(p.experts) * p.w2_rows);

    uint32_t seed = 1u;
    auto next_bf16 = [&](shelf::bf16_t* dev, std::size_t count) {
        std::vector<shelf::bf16_t> host(count);
        for (std::size_t i = 0; i < count; ++i) {
            seed = seed * 1103515245u + 12345u;
            const float value = static_cast<float>((seed >> 16) % 2001u) / 1000.0f - 1.0f;
            host[i] = shelf::bf16_t{shelf::f32_to_bf16_rne(value)};
        }
        if (hipMemcpy(dev, host.data(), count * sizeof(shelf::bf16_t), hipMemcpyHostToDevice) !=
            hipSuccess) {
            std::printf("FAIL hidden upload\n");
            std::exit(1);
        }
    };

    struct Spec {
        uint32_t rows;
        uint32_t top_k;
        bool skew;
    };
    const Spec specs[] = {
        {1, 2, false}, {1, 4, false}, {3, 2, true}, {16, 2, false},
        {17, 4, false}, {33, 4, false}, {64, 2, true},
    };
    for (const auto& sp : specs) {
        Case c;
        c.rows = sp.rows;
        c.top_k = sp.top_k;
        c.group_capacity = moe::moe_group_count_upper_bound(sp.rows, kExperts, sp.top_k);
        c.logits = make_logits(sp.rows, kExperts, sp.skew);
        next_bf16(d.hidden13, static_cast<std::size_t>(sp.rows) * p.w13_cols);
        next_bf16(d.hidden2, static_cast<std::size_t>(sp.rows) * p.w2_cols);
        const std::string tag = "rows" + std::to_string(sp.rows) + "/top" +
                                std::to_string(sp.top_k) + (sp.skew ? "/skew" : "");
        run_grouped_case(f, p, c, d, stream, tag);
    }

    const auto probe_layout = rt::Int8ActivationWorkspaceLayout::make(p.w13_k_padded, kRowTile);
    check(moe::launch_gemm_mxfp4_w4a8_grouped(
              d.w13_codes, d.w13_scales, nullptr, reinterpret_cast<const int8_t*>(d.plane_codes),
              d.plane_scales, d.out13,
              moe::Mxfp4GemmOutputDType::BF16, 17u, p.w13_k_padded, p.w13_rows,
              p.w13_codes_batch_stride_bytes, p.w13_scales_batch_stride_bytes,
              p.w13_scales_row_stride_bytes, probe_layout.code_row_stride_bytes,
              probe_layout.scale_row_stride_bytes, p.w13_rows, d.group_expert, d.group_rows,
              d.group_count, 1u, stream) != hipSuccess,
          "out_features that is not a 16 tile is rejected");
    check(moe::launch_gemm_mxfp4_w4a8_grouped(
              d.w13_codes, d.w13_scales, nullptr, reinterpret_cast<const int8_t*>(d.plane_codes),
              d.plane_scales, d.out13,
              moe::Mxfp4GemmOutputDType::BF16, p.w13_rows, p.w13_k_padded, p.w13_rows,
              p.w13_codes_batch_stride_bytes + 256u, p.w13_scales_batch_stride_bytes,
              p.w13_scales_row_stride_bytes, probe_layout.code_row_stride_bytes,
              probe_layout.scale_row_stride_bytes, p.w13_rows, d.group_expert, d.group_rows,
              d.group_count, 1u, stream) != hipSuccess,
          "a batch stride that is not the rank-3 native layout is rejected");
    check(moe::launch_gemm_mxfp4_w4a8_grouped(
              d.w13_codes, d.w13_scales, nullptr, reinterpret_cast<const int8_t*>(d.plane_codes),
              d.plane_scales, d.out13,
              moe::Mxfp4GemmOutputDType::BF16, p.w13_rows, p.w13_k_padded, p.w13_rows,
              p.w13_codes_batch_stride_bytes, p.w13_scales_batch_stride_bytes,
              p.w13_scales_row_stride_bytes, probe_layout.code_row_stride_bytes,
              probe_layout.scale_row_stride_bytes, p.w13_rows, d.group_expert, d.group_rows,
              d.group_count, 0u, stream) != hipSuccess,
          "zero group capacity is rejected");

    hipFree(d.w13_codes);
    hipFree(d.w13_scales);
    hipFree(d.w13_ref);
    hipFree(d.w2_codes);
    hipFree(d.w2_scales);
    hipFree(d.w2_ref);
    hipFree(d.logits);
    hipFree(d.sel_expert);
    hipFree(d.sel_weight);
    hipFree(d.tile_counts);
    hipFree(d.row_begin);
    hipFree(d.tile_base);
    hipFree(d.group_begin);
    hipFree(d.group_expert);
    hipFree(d.group_rows);
    hipFree(d.group_count);
    hipFree(d.row_token);
    hipFree(d.slot_row_of);
    hipFree(d.hidden13);
    hipFree(d.hidden2);
    hipFree(d.plane_codes);
    hipFree(d.plane_scales);
    hipFree(d.out13);
    hipFree(d.out2);
    fs::remove_all(root, ec);
    gpu_arena.shutdown();
    if (hipStreamDestroy(stream) != hipSuccess)
        check(false, "stream destroy");

    std::printf("test_moe_grouped_linear: %s (fail=%d)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
