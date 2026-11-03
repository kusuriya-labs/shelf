#include <shelf/core/memory/types.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/quantization/codec/e4m3.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/quantization/mxfp4/mxfp4.h>
#include <shelf/runtime/graph/value_type.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace quant = shelf::quantization;
namespace rt = shelf::runtime;

static float bf16_to_f32(shelf::bf16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static int g_fail = 0;

static void fail(const std::string& msg) {
    ++g_fail;
    std::printf("FAIL %s\n", msg.c_str());
}

static void check(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

static bool hip_ok(hipError_t e, const std::string& what) {
    if (e == hipSuccess) return true;
    fail(what + ": " + hipGetErrorString(e));
    return false;
}

static constexpr uint32_t kRows = 32u;
static constexpr uint32_t kCols = 128u;
static constexpr uint32_t kTokens = 5u;

struct Fixture {
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    std::vector<shelf::bf16_t> input;
    std::vector<float> weight_f32;
};

static Fixture make_fixture() {
    Fixture f;
    std::mt19937 rng(20261103u);
    std::normal_distribution<float> weight_dist(0.0f, 0.04f);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);
    f.weight_f32.assign(static_cast<std::size_t>(kRows) * kCols, 0.0f);
    for (float& value : f.weight_f32) value = weight_dist(rng);
    f.codes.assign(static_cast<std::size_t>(kRows) * (kCols / 2u), 0u);
    f.scales.assign(static_cast<std::size_t>(kRows) * (kCols / quant::kMxfp4ScaleBlockK), 0u);
    quant::mxfp4::quantize_mxfp4_matrix(
        f.weight_f32.data(), kCols, kRows, kCols, kCols, f.codes.data(), f.scales.data());
    f.input.assign(static_cast<std::size_t>(kTokens) * kCols, shelf::bf16_t{0});
    for (std::size_t i = 0; i < f.input.size(); ++i)
        f.input[i] = shelf::bf16_t{shelf::f32_to_bf16_rne(input_dist(rng))};
    return f;
}

// Rebuilds the expected output from the activation codes the GPU itself wrote
// into the workspace, so the test isolates the weight decode and the group-32
// rescaling instead of re-testing the E4M3 activation quantizer.
static std::vector<double> expected_output(
    const Fixture& f, const std::vector<uint8_t>& ws_codes,
    const std::vector<float>& ws_scales,
    const rt::Int8ActivationWorkspaceLayout& layout) {
    std::vector<double> out(static_cast<std::size_t>(kTokens) * kRows, 0.0);
    const uint32_t groups = kCols / quant::kMxfp4ScaleBlockK;
    for (uint32_t m = 0; m < kTokens; ++m) {
        const uint32_t m_tile = m >> 4;
        const uint32_t m_slot = m & 15u;
        for (uint32_t o = 0; o < kRows; ++o) {
            double acc = 0.0;
            for (uint32_t ib = 0; ib < groups; ++ib) {
                const double ws =
                    quant::mxfp4::e8m0_decode_f32(f.scales[o * groups + ib]);
                const double as = ws_scales[static_cast<std::size_t>(m) *
                        (layout.scale_row_stride_bytes / sizeof(float)) + (ib >> 2)];
                double p = 0.0;
                for (uint32_t j = 0; j < quant::kMxfp4ScaleBlockK; ++j) {
                    const uint32_t kc = ib * quant::kMxfp4ScaleBlockK + j;
                    const uint8_t packed = f.codes[o * (kCols / 2u) + (kc >> 1)];
                    const uint8_t code = (kc & 1u) == 0u ? (packed & 0x0Fu) : (packed >> 4);
                    const size_t aoff = static_cast<size_t>(kc >> 6) *
                                            quant::fp8::kFragBlockBytes +
                                        quant::fp8::fp8_frag_offset(m_slot, kc & 63u);
                    const size_t abase = static_cast<size_t>(m_tile) *
                        (static_cast<size_t>(layout.code_row_stride_bytes) * 16u);
                    const double av = quant::codec::e4m3_decode_f32(ws_codes[abase + aoff]);
                    p += static_cast<double>(quant::mxfp4::e2m1_decode_f32(code)) * av;
                }
                acc += p * ws * as;
            }
            out[static_cast<std::size_t>(m) * kRows + o] = acc;
        }
    }
    return out;
}

struct DeviceSetup {
    uint8_t* ws = nullptr;
    uint8_t* codes = nullptr;
    uint8_t* scales = nullptr;
    shelf::bf16_t* in = nullptr;
    shelf::bf16_t* out = nullptr;
    rt::WeightSlot* weights = nullptr;
    rt::DeviceValueBinding* values = nullptr;
    rt::DeviceWorkspaceRange* ranges = nullptr;
    rt::DeviceBatchContext* ctx = nullptr;
    float* scratch = nullptr;
    uint32_t* error_word = nullptr;
};

static void release(DeviceSetup& d) {
    if (d.ws) (void)hipFree(d.ws);
    if (d.codes) (void)hipFree(d.codes);
    if (d.scales) (void)hipFree(d.scales);
    if (d.in) (void)hipFree(d.in);
    if (d.out) (void)hipFree(d.out);
    if (d.weights) (void)hipFree(d.weights);
    if (d.values) (void)hipFree(d.values);
    if (d.ranges) (void)hipFree(d.ranges);
    if (d.ctx) (void)hipFree(d.ctx);
    if (d.scratch) (void)hipFree(d.scratch);
    if (d.error_word) (void)hipFree(d.error_word);
}

static bool allocate(DeviceSetup& d, std::size_t ws_bytes, const Fixture& f,
                     const rt::Int8ActivationWorkspaceLayout& layout) {
    bool ok = true;
    ok = ok && hip_ok(hipMalloc(&d.ws, ws_bytes), "malloc ws");
    ok = ok && hip_ok(hipMalloc(&d.codes, f.codes.size()), "malloc codes");
    ok = ok && hip_ok(hipMalloc(&d.scales, f.scales.size()), "malloc scales");
    ok = ok && hip_ok(hipMalloc(&d.in, f.input.size() * sizeof(shelf::bf16_t)), "malloc in");
    ok = ok && hip_ok(hipMalloc(&d.out, static_cast<std::size_t>(kTokens) * kRows *
                        sizeof(shelf::bf16_t)), "malloc out");
    ok = ok && hip_ok(hipMalloc(&d.weights, sizeof(rt::WeightSlot)), "malloc weights");
    ok = ok && hip_ok(hipMalloc(&d.values, 2 * sizeof(rt::DeviceValueBinding)), "malloc values");
    ok = ok && hip_ok(hipMalloc(&d.ranges, sizeof(rt::DeviceWorkspaceRange)), "malloc ranges");
    ok = ok && hip_ok(hipMalloc(&d.ctx, sizeof(rt::DeviceBatchContext)), "malloc ctx");
    ok = ok && hip_ok(hipMalloc(&d.scratch, 2048u * sizeof(float)), "malloc scratch");
    ok = ok && hip_ok(hipMalloc(&d.error_word, sizeof(uint32_t)), "malloc error");
    if (!ok) return false;
    ok = ok && hip_ok(hipMemcpy(d.codes, f.codes.data(), f.codes.size(),
                                hipMemcpyHostToDevice), "h2d codes");
    ok = ok && hip_ok(hipMemcpy(d.scales, f.scales.data(), f.scales.size(),
                                hipMemcpyHostToDevice), "h2d scales");
    ok = ok && hip_ok(hipMemcpy(d.in, f.input.data(), f.input.size() * sizeof(shelf::bf16_t),
                                hipMemcpyHostToDevice), "h2d in");
    ok = ok && hip_ok(hipMemset(d.ws, 0, ws_bytes), "memset ws");
    ok = ok && hip_ok(hipMemset(d.out, 0, static_cast<std::size_t>(kTokens) * kRows *
                        sizeof(shelf::bf16_t)), "memset out");
    ok = ok && hip_ok(hipMemset(d.error_word, 0, sizeof(uint32_t)), "memset error");
    if (!ok) return false;

    rt::DeviceValueBinding values[2] = {};
    values[0].logical_value = 0;
    values[0].storage = rt::ValueStorage::EXTERNAL_INPUT;
    values[0].dtype = rt::ValueDType::BF16;
    values[0].feature_count = kCols;
    values[0].row_stride = kCols;
    values[0].offset = reinterpret_cast<uint64_t>(d.in);
    values[0].bytes = f.input.size() * sizeof(shelf::bf16_t);
    values[1].logical_value = 1;
    values[1].storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    values[1].dtype = rt::ValueDType::BF16;
    values[1].feature_count = kRows;
    values[1].row_stride = kRows;
    values[1].offset = reinterpret_cast<uint64_t>(d.out);
    values[1].bytes = static_cast<std::size_t>(kTokens) * kRows * sizeof(shelf::bf16_t);
    ok = ok && hip_ok(hipMemcpy(d.values, values, sizeof(values), hipMemcpyHostToDevice),
                      "h2d values");

    rt::DeviceWorkspaceRange range{};
    range.offset = 0;
    range.bytes = ws_bytes;
    ok = ok && hip_ok(hipMemcpy(d.ranges, &range, sizeof(range), hipMemcpyHostToDevice),
                      "h2d ranges");

    rt::WeightSlot slot{};
    slot.rows = kRows;
    slot.cols = kCols;
    slot.k_padded = kCols;
    slot.encoding = 3u;
    slot.compute_spec = static_cast<uint8_t>(quant::ComputeSpecId::MXFP4_W4A8_F32_BF16);
    slot.codes = d.codes;
    slot.scales = d.scales;
    slot.codes_row_stride_bytes = kCols / quant::kMxfp4WeightsPerCode;
    slot.scale_row_stride_bytes = kCols / quant::kMxfp4ScaleBlockK;
    slot.storage_scale_stride_bytes = slot.scale_row_stride_bytes;
    slot.weight_scale_group = quant::kMxfp4ScaleBlockK;
    slot.preshuffled = true;
    ok = ok && hip_ok(hipMemcpy(d.weights, &slot, sizeof(slot), hipMemcpyHostToDevice),
                      "h2d weight slot");

    rt::DeviceBatchContext ctx{};
    ctx.actual_rows = kTokens;
    ok = ok && hip_ok(hipMemcpy(d.ctx, &ctx, sizeof(ctx), hipMemcpyHostToDevice), "h2d ctx");
    (void)layout;
    return ok;
}

static uint32_t run_dispatch(const DeviceSetup& d, const rt::DispatchBinding& binding,
                             hipStream_t stream) {
    if (!hip_ok(hipMemset(d.error_word, 0, sizeof(uint32_t)), "memset error"))
        return 0xFFFFFFFFu;
    rt::DeviceProgramView program{};
    program.values = d.values;
    program.value_count = 2;
    program.weight_table = d.weights;
    program.weight_slot_count = 1;
    program.workspace_ranges = d.ranges;
    program.workspace_range_count = 1;
    program.workspace_base = reinterpret_cast<uint64_t>(d.ws);
    auto status = ::shelf::kernel::launch_model_dispatch_correctness(
        program, binding, d.ctx, ::shelf::kernel::ModelDispatchStateView{},
        d.scratch, d.error_word, stream);
    if (!status.ok()) {
        fail("launch: " + status.message());
        return 0xFFFFFFFFu;
    }
    if (!hip_ok(hipStreamSynchronize(stream), "sync")) return 0xFFFFFFFFu;
    uint32_t error = 0;
    if (!hip_ok(hipMemcpy(&error, d.error_word, sizeof(error), hipMemcpyDeviceToHost),
                "read error"))
        return 0xFFFFFFFFu;
    return error;
}

static void upload_slot(const DeviceSetup& d, const rt::WeightSlot& slot) {
    hip_ok(hipMemcpy(d.weights, &slot, sizeof(slot), hipMemcpyHostToDevice), "h2d slot");
}

int main() {
    if (!hip_ok(hipSetDevice(0), "hipSetDevice")) return 1;
    hipStream_t stream = nullptr;
    if (!hip_ok(hipStreamCreate(&stream), "stream create")) return 1;

    const Fixture f = make_fixture();
    const rt::Int8ActivationWorkspaceLayout layout =
        rt::Int8ActivationWorkspaceLayout::make(kCols, kTokens);
    check(layout.k_padded == kCols, "activation layout keeps a 64-aligned K");

    DeviceSetup d;
    if (!allocate(d, layout.total_bytes, f, layout)) {
        release(d);
        std::printf("FAIL mxfp4 linear: device setup\n");
        return 1;
    }

    rt::DispatchBinding quantize{};
    quantize.kernel_id = rt::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED;
    quantize.workspace_slot = 0;
    quantize.group_size = 32u;
    quantize.activation_kp = layout.k_padded;
    quantize.activation_max_rows = kTokens;
    quantize.activation_code_stride = layout.code_row_stride_bytes;
    quantize.activation_scale_stride = layout.scale_row_stride_bytes;
    quantize.input_slots[0] = 0;
    quantize.input_count = 1;
    check(run_dispatch(d, quantize, stream) == 0u, "activation quantize dispatch completes");

    rt::DispatchBinding linear{};
    linear.kernel_id = rt::KernelId::LINEAR_MXFP4_W4A8;
    linear.workspace_slot = 0;
    linear.weight_index = 0;
    linear.compute_spec = static_cast<uint32_t>(quant::ComputeSpecId::MXFP4_W4A8_F32_BF16);
    linear.activation_kp = layout.k_padded;
    linear.activation_max_rows = kTokens;
    linear.activation_code_stride = layout.code_row_stride_bytes;
    linear.activation_scale_stride = layout.scale_row_stride_bytes;
    linear.output_slots[0] = 1;
    linear.output_count = 1;
    check(run_dispatch(d, linear, stream) == 0u, "mxfp4 linear dispatch completes");

    std::vector<uint8_t> workspace(layout.total_bytes, 0u);
    if (!hip_ok(hipMemcpy(workspace.data(), d.ws, workspace.size(), hipMemcpyDeviceToHost),
                 "d2h workspace")) {
        release(d);
        return 1;
    }
    std::vector<uint8_t> ws_codes(workspace.begin(),
                                 workspace.begin() + static_cast<std::ptrdiff_t>(layout.codes_bytes()));
    std::vector<float> ws_scales(kTokens * (layout.scale_row_stride_bytes / 4u), 0.0f);
    std::memcpy(ws_scales.data(), workspace.data() + layout.scales_offset,
                ws_scales.size() * sizeof(float));

    const std::vector<double> expected = expected_output(f, ws_codes, ws_scales, layout);
    std::vector<shelf::bf16_t> got(static_cast<std::size_t>(kTokens) * kRows, shelf::bf16_t{0});
    if (!hip_ok(hipMemcpy(got.data(), d.out, got.size() * sizeof(shelf::bf16_t),
                          hipMemcpyDeviceToHost), "d2h out")) {
        release(d);
        return 1;
    }

    double l2_expected = 0.0;
    double l2_diff = 0.0;
    double worst = 0.0;
    bool all_zero = true;
    std::size_t exact = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double value = bf16_to_f32(got[i]);
        const double want = expected[i];
        if (want != 0.0) all_zero = false;
        if (got[i].data == shelf::f32_to_bf16_rne(static_cast<float>(want))) ++exact;
        l2_expected += want * want;
        l2_diff += (value - want) * (value - want);
        worst = std::max(worst, std::fabs(value - want));
    }
    std::printf("mxfp4 linear: bf16-exact elements %zu / %zu\n", exact, got.size());
    check(!all_zero, "mxfp4 linear produced non-zero output");
    check(exact == got.size(), "mxfp4 linear matches a double reference after bf16 rounding");
    const double rel = std::sqrt(l2_diff) / std::max(std::sqrt(l2_expected), 1e-9);
    std::printf("mxfp4 linear: rel_l2=%.3e max_abs=%.3e\n", rel, worst);
    check(rel < 2.0e-3, "mxfp4 linear relative L2 within the bf16 output rounding");
    check(worst < 2.0e-2, "mxfp4 linear max absolute difference");

    rt::WeightSlot slot{};
    slot.rows = kRows;
    slot.cols = kCols;
    slot.k_padded = kCols;
    slot.encoding = 3u;
    slot.compute_spec = static_cast<uint8_t>(quant::ComputeSpecId::MXFP4_W4A8_F32_BF16);
    slot.codes = d.codes;
    slot.scales = d.scales;
    slot.codes_row_stride_bytes = kCols / quant::kMxfp4WeightsPerCode;
    slot.scale_row_stride_bytes = kCols / quant::kMxfp4ScaleBlockK;
    slot.storage_scale_stride_bytes = slot.scale_row_stride_bytes;
    slot.weight_scale_group = quant::kMxfp4ScaleBlockK;
    slot.preshuffled = true;

    rt::WeightSlot bad_group = slot;
    bad_group.weight_scale_group = 64u;
    upload_slot(d, bad_group);
    check(run_dispatch(d, linear, stream) != 0u, "mxfp4 linear rejects a group 64 weight");

    rt::WeightSlot bad_encoding = slot;
    bad_encoding.encoding = 1u;
    upload_slot(d, bad_encoding);
    check(run_dispatch(d, linear, stream) != 0u, "mxfp4 linear rejects an fp8 weight slot");

    rt::WeightSlot bad_stride = slot;
    bad_stride.codes_row_stride_bytes = kCols;
    upload_slot(d, bad_stride);
    check(run_dispatch(d, linear, stream) != 0u, "mxfp4 linear rejects a one-byte-per-weight stride");

    rt::WeightSlot bad_kp = slot;
    bad_kp.k_padded = 96u;
    upload_slot(d, bad_kp);
    check(run_dispatch(d, linear, stream) != 0u, "mxfp4 linear rejects k_padded that is not 64-aligned");

    upload_slot(d, slot);
    check(run_dispatch(d, linear, stream) == 0u, "mxfp4 linear accepts the reference slot again");

    release(d);
    (void)hipStreamDestroy(stream);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
