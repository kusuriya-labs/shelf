#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/linear/fp8_ptpc.h>
#include <shelf/quantization/codec/e4m3.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/quantization/fp8/fp8_native.h>
#include <shelf/runtime/program/int8_activation_workspace.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace q = shelf::quantization;
namespace rt = shelf::runtime;

namespace {

int failed = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::printf("FAIL: %s\n", message);
        ++failed;
    }
}

bool hip_ok(hipError_t error, const char* what) {
    if (error == hipSuccess) return true;
    std::printf("FAIL %s: %s\n", what, hipGetErrorString(error));
    ++failed;
    return false;
}

float bf16_to_f32(shelf::bf16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value.data) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

}  // namespace

int main() {
    constexpr std::uint32_t kRows = 5u;
    constexpr std::uint32_t kOut = 64u;
    constexpr std::uint32_t kK = 256u;
    constexpr std::uint32_t kMaxRows = 16u;

    std::mt19937 rng(20261106u);
    std::uniform_int_distribution<int> code_dist(0, 126);
    std::uniform_real_distribution<float> scale_dist(0.0005f, 0.002f);
    std::normal_distribution<float> input_dist(0.0f, 1.0f);

    std::vector<std::uint8_t> codes(static_cast<std::size_t>(kOut) * kK);
    for (auto& c : codes) c = static_cast<std::uint8_t>(code_dist(rng));
    std::vector<float> weight_scales(kOut);
    for (float& s : weight_scales) s = scale_dist(rng);

    std::vector<shelf::bf16_t> input(static_cast<std::size_t>(kRows) * kK);
    for (auto& v : input) v = shelf::bf16_t{shelf::f32_to_bf16_rne(input_dist(rng))};

    q::fp8::Fp8BlockCanonicalView cv;
    cv.batch = 1;
    cv.n = kOut;
    cv.padded_k = kK;
    cv.codes = codes.data();
    cv.codes_bytes = codes.size();
    std::vector<float> dummy(2u, 0.0f);
    cv.scales = dummy.data();
    cv.scales_bytes = dummy.size() * sizeof(float);
    cv.scale_n = 1u;
    cv.scale_k = kK / 128u;
    cv.scale_block_k = 128u;
    check(cv.validate(), "canonical fp8 view validates");
    q::fp8::Fp8NativeHost native;
    check(q::fp8::preshuffle_fp8_native(cv, native), "fp8 preshuffle");

    const rt::Int8ActivationWorkspaceLayout layout =
        rt::Int8ActivationWorkspaceLayout::make(kK, kMaxRows);
    const std::size_t codes_bytes = layout.codes_bytes();
    const std::size_t scales_bytes = layout.scales_bytes();

    std::uint8_t* d_weight_codes = nullptr;
    float* d_weight_scales = nullptr;
    std::uint8_t* d_activation_codes = nullptr;
    float* d_activation_scales = nullptr;
    shelf::bf16_t* d_input = nullptr;
    shelf::bf16_t* d_output = nullptr;
    bool ok = true;
    ok = ok && hip_ok(hipMalloc(&d_weight_codes, native.codes_bytes), "malloc weight codes");
    ok = ok && hip_ok(hipMalloc(&d_weight_scales, kOut * sizeof(float)), "malloc weight scales");
    ok = ok && hip_ok(hipMalloc(&d_activation_codes, codes_bytes), "malloc activation codes");
    ok = ok && hip_ok(hipMalloc(&d_activation_scales, scales_bytes), "malloc activation scales");
    ok = ok && hip_ok(hipMalloc(&d_input, input.size() * sizeof(shelf::bf16_t)), "malloc input");
    ok = ok && hip_ok(hipMalloc(&d_output, kRows * kOut * sizeof(shelf::bf16_t)), "malloc output");
    if (!ok) return 1;

    ok = ok && hip_ok(hipMemcpy(d_weight_codes, native.codes.get(), native.codes_bytes,
                                hipMemcpyHostToDevice), "copy weight codes");
    ok = ok && hip_ok(hipMemcpy(d_weight_scales, weight_scales.data(),
                                kOut * sizeof(float), hipMemcpyHostToDevice),
                      "copy weight scales");
    ok = ok && hip_ok(hipMemcpy(d_input, input.data(), input.size() * sizeof(shelf::bf16_t),
                                hipMemcpyHostToDevice), "copy input");
    ok = ok && hip_ok(hipMemset(d_activation_codes, 0, codes_bytes), "clear activation codes");
    ok = ok && hip_ok(hipMemset(d_activation_scales, 0, scales_bytes), "clear activation scales");

    hipStream_t stream = nullptr;
    ok = ok && hip_ok(hipStreamCreate(&stream), "stream");
    ok = ok && hip_ok(shelf::kernel::launch_activation_quantize_fp8_pertoken(
                          d_input, d_activation_codes, d_activation_scales, kRows, kMaxRows,
                          kMaxRows, kK, kK, layout.code_row_stride_bytes,
                          layout.scale_row_stride_bytes, stream),
                      "quantize");
    ok = ok && hip_ok(shelf::kernel::launch_gemm_fp8_ptpc(
                          d_weight_codes, d_weight_scales, d_activation_codes,
                          d_activation_scales, d_output,
                          shelf::kernel::Fp8PertokenGemmOutputDType::BF16, kRows, kOut, kK,
                          layout.code_row_stride_bytes, layout.scale_row_stride_bytes, kOut,
                          stream),
                      "gemm");
    ok = ok && hip_ok(hipStreamSynchronize(stream), "sync");
    if (!ok) return 1;

    std::vector<std::uint8_t> act_codes(codes_bytes);
    std::vector<float> act_scales(kMaxRows *
                                  (layout.scale_row_stride_bytes / sizeof(float)));
    std::vector<shelf::bf16_t> output(kRows * kOut);
    ok = ok && hip_ok(hipMemcpy(act_codes.data(), d_activation_codes, codes_bytes,
                                hipMemcpyDeviceToHost), "read activation codes");
    ok = ok && hip_ok(hipMemcpy(act_scales.data(), d_activation_scales, scales_bytes,
                                hipMemcpyDeviceToHost), "read activation scales");
    ok = ok && hip_ok(hipMemcpy(output.data(), d_output, output.size() * sizeof(shelf::bf16_t),
                                hipMemcpyDeviceToHost), "read output");
    if (!ok) return 1;

    const std::uint32_t scale_row_floats = layout.scale_row_stride_bytes / sizeof(float);
    double worst = 0.0;
    for (std::uint32_t row = 0; row < kRows; ++row) {
        const float act_scale = act_scales[row * scale_row_floats];
        for (std::uint32_t o = 0; o < kOut; ++o) {
            double acc = 0.0;
            for (std::uint32_t k = 0; k < kK; ++k) {
                const std::size_t act_offset =
                    static_cast<std::size_t>(row >> 4u) * layout.code_row_stride_bytes * 16u +
                    static_cast<std::size_t>(k >> 6u) * q::fp8::kFragBlockBytes +
                    q::fp8::fp8_frag_offset(row & 15u, k & 63u);
                const double a = q::codec::e4m3_decode_f32(act_codes[act_offset]);
                const double w = q::codec::e4m3_decode_f32(codes[static_cast<std::size_t>(o) * kK + k]);
                acc += w * a;
            }
            acc *= static_cast<double>(weight_scales[o]) * act_scale;
            const double got = bf16_to_f32(output[static_cast<std::size_t>(row) * kOut + o]);
            worst = std::max(worst, std::fabs(got - acc));
        }
    }
    std::printf("fp8 ptpc max_abs=%.6g\n", worst);
    check(worst <= 4.0e-2, "fp8 per-token per-channel gemm matches the CPU reference");

    (void)hipFree(d_weight_codes);
    (void)hipFree(d_weight_scales);
    (void)hipFree(d_activation_codes);
    (void)hipFree(d_activation_scales);
    (void)hipFree(d_input);
    (void)hipFree(d_output);
    (void)hipStreamDestroy(stream);
    std::printf(failed == 0 ? "PASS\n" : "FAIL (%d)\n", failed);
    return failed == 0 ? 0 : 1;
}
