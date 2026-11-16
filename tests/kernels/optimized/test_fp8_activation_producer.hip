#include <shelf/inference/execution/fp8_activation_producer.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/output_gather.h>
#include <shelf/kernels/optimized/rmsnorm.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/int8_activation_workspace.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++passed;
        std::printf("PASS: %s\n", name);
    } else {
        ++failed;
        std::printf("FAIL: %s\n", name);
    }
}

uint32_t hash32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return static_cast<uint32_t>(x);
}

float value(uint64_t seed, uint32_t i) {
    const int32_t n = static_cast<int32_t>(hash32(seed * 131u + i) % 4001u) - 2000;
    return static_cast<float>(n) / 500.0f;
}

shelf::bf16_t to_bf16(float x) {
    return shelf::bf16_t{shelf::f32_to_bf16_rne(x)};
}

bool copy_to(void* dst, const void* src, size_t bytes) {
    return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) == hipSuccess;
}

template <typename T>
bool copy_from(std::vector<T>& dst, const void* src) {
    return hipMemcpy(dst.data(), src, dst.size() * sizeof(T), hipMemcpyDeviceToHost) == hipSuccess;
}

bool run_residual(uint32_t kRows) {
    constexpr uint32_t kFeatures = 5120;
    constexpr float kEps = 1e-6f;
    const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, kRows);
    const uint32_t code_stride = layout.code_row_stride_bytes;
    const uint32_t scale_stride = layout.scale_row_stride_bytes;
    const uint32_t rows_padded = (kRows + 15u) & ~15u;

    std::vector<shelf::bf16_t> ha(kRows * kFeatures), hb(kRows * kFeatures);
    std::vector<shelf::bf16_t> hw(kFeatures);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t i = 0; i < kFeatures; ++i) {
            ha[r * kFeatures + i] = to_bf16(value(11u + r, i));
            hb[r * kFeatures + i] = to_bf16(value(101u + r, i) * 0.5f);
        }
    }
    for (uint32_t i = 0; i < kFeatures; ++i)
        hw[i] = to_bf16(value(7u, i) * 0.1f);

    void *da = nullptr, *db = nullptr, *dw = nullptr;
    void *d_res_ref = nullptr, *d_res_fused = nullptr;
    void *d_norm = nullptr;
    void *d_codes_ref = nullptr, *d_codes_fused = nullptr;
    void *d_scales_ref = nullptr, *d_scales_fused = nullptr;
    auto cleanup = [&]() {
        void* all[] = {da, db, dw, d_res_ref, d_res_fused, d_norm,
                       d_codes_ref, d_codes_fused, d_scales_ref, d_scales_fused};
        for (void* p : all)
            if (p != nullptr) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    const size_t elem = sizeof(shelf::bf16_t);
    if (hipMalloc(&da, ha.size() * elem) != hipSuccess ||
        hipMalloc(&db, hb.size() * elem) != hipSuccess ||
        hipMalloc(&dw, hw.size() * elem) != hipSuccess ||
        hipMalloc(&d_res_ref, ha.size() * elem) != hipSuccess ||
        hipMalloc(&d_res_fused, ha.size() * elem) != hipSuccess ||
        hipMalloc(&d_norm, ha.size() * elem) != hipSuccess ||
        hipMalloc(&d_codes_ref, rows_padded * code_stride) != hipSuccess ||
        hipMalloc(&d_codes_fused, rows_padded * code_stride) != hipSuccess ||
        hipMalloc(&d_scales_ref, kRows * scale_stride) != hipSuccess ||
        hipMalloc(&d_scales_fused, kRows * scale_stride) != hipSuccess) {
        cleanup();
        return false;
    }
    if (!copy_to(da, ha.data(), ha.size() * elem) ||
        !copy_to(db, hb.data(), hb.size() * elem) ||
        !copy_to(dw, hw.data(), hw.size() * elem)) {
        cleanup();
        return false;
    }
    shelf::gpu::discard_cleanup_result(hipMemset(d_codes_ref, 0xAB,
        rows_padded * code_stride));
    shelf::gpu::discard_cleanup_result(hipMemset(d_codes_fused, 0xAB,
        rows_padded * code_stride));
    shelf::gpu::discard_cleanup_result(hipMemset(d_scales_ref, 0xAB,
        kRows * scale_stride));
    shelf::gpu::discard_cleanup_result(hipMemset(d_scales_fused, 0xAB,
        kRows * scale_stride));

    const auto a = static_cast<const shelf::bf16_t*>(da);
    const auto b = static_cast<const shelf::bf16_t*>(db);
    const auto w = static_cast<const shelf::bf16_t*>(dw);
    auto* res_ref = static_cast<shelf::bf16_t*>(d_res_ref);
    auto* res_fused = static_cast<shelf::bf16_t*>(d_res_fused);
    auto* norm = static_cast<shelf::bf16_t*>(d_norm);
    auto* codes_ref = static_cast<uint8_t*>(d_codes_ref);
    auto* codes_fused = static_cast<uint8_t*>(d_codes_fused);
    auto* scales_ref = static_cast<float*>(d_scales_ref);
    auto* scales_fused = static_cast<float*>(d_scales_fused);

    hipError_t err = shelf::kernel::launch_residual_add_bf16(
        a, kFeatures, b, kFeatures, res_ref, kFeatures, kRows, kFeatures, nullptr);
    if (err == hipSuccess)
        err = shelf::kernel::launch_rmsnorm(
            res_ref, shelf::kernel::RmsNormDataType::BF16, w,
            shelf::kernel::RmsNormDataType::BF16,
            shelf::kernel::RmsNormWeightLayout::PER_FEATURE,
            shelf::kernel::RmsNormWeightMode::ONE_PLUS, norm,
            shelf::kernel::RmsNormDataType::BF16, kRows, kFeatures, kFeatures,
            kFeatures, kFeatures, kEps, nullptr);
    if (err == hipSuccess)
        err = shelf::kernel::launch_activation_quantize_fp8_finegrained(
            norm, codes_ref, scales_ref, kRows, kFeatures, kFeatures, kFeatures,
            code_stride, scale_stride, nullptr, nullptr);
    if (err != hipSuccess) {
        std::printf("  residual reference launch failed: %s\n", hipGetErrorString(err));
        cleanup();
        return false;
    }

    shelf::execution::Fp8ResidualNormArgs args{};
    args.a = a;
    args.b = b;
    args.residual = res_fused;
    args.weight = w;
    args.codes = codes_fused;
    args.scales = scales_fused;
    args.rows = kRows;
    args.k = kFeatures;
    args.k_padded = kFeatures;
    args.a_stride = kFeatures;
    args.b_stride = kFeatures;
    args.residual_stride = kFeatures;
    args.code_stride = code_stride;
    args.scale_stride = scale_stride;
    args.eps = kEps;
    err = shelf::execution::launch_fp8_residual_norm_quantize(args, nullptr);
    if (err != hipSuccess) {
        std::printf("  residual fused launch failed: %s\n", hipGetErrorString(err));
        cleanup();
        return false;
    }

    std::vector<shelf::bf16_t> h_res_ref(kRows * kFeatures), h_res_fused(kRows * kFeatures);
    std::vector<uint8_t> h_codes_ref(rows_padded * code_stride), h_codes_fused(rows_padded * code_stride);
    std::vector<float> h_scales_ref(kRows * scale_stride / 4u), h_scales_fused(kRows * scale_stride / 4u);
    bool ok = copy_from(h_res_ref, d_res_ref) && copy_from(h_res_fused, d_res_fused) &&
              copy_from(h_codes_ref, d_codes_ref) && copy_from(h_codes_fused, d_codes_fused) &&
              copy_from(h_scales_ref, d_scales_ref) && copy_from(h_scales_fused, d_scales_fused);
    if (ok) {
        ok = std::memcmp(h_res_ref.data(), h_res_fused.data(),
                         h_res_ref.size() * elem) == 0 &&
             std::memcmp(h_codes_ref.data(), h_codes_fused.data(),
                         h_codes_ref.size()) == 0 &&
             std::memcmp(h_scales_ref.data(), h_scales_fused.data(),
                         h_scales_ref.size() * sizeof(float)) == 0;
    }
    cleanup();
    return ok;
}

bool run_final(uint32_t kOutputs) {
    const uint32_t kRows = kOutputs + 2u;
    constexpr uint32_t kFeatures = 5120;
    constexpr float kEps = 1e-6f;
    const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(kFeatures, kOutputs);
    const uint32_t code_stride = layout.code_row_stride_bytes;
    const uint32_t scale_stride = layout.scale_row_stride_bytes;
    const uint32_t rows_padded = (kOutputs + 15u) & ~15u;

    std::vector<shelf::bf16_t> hin(kRows * kFeatures);
    std::vector<shelf::bf16_t> hw(kFeatures);
    for (uint32_t r = 0; r < kRows; ++r)
        for (uint32_t i = 0; i < kFeatures; ++i)
            hin[r * kFeatures + i] = to_bf16(value(211u + r, i));
    for (uint32_t i = 0; i < kFeatures; ++i)
        hw[i] = to_bf16(value(17u, i) * 0.1f);
    std::vector<uint32_t> h_rows(kOutputs);
    for (uint32_t i = 0; i < kOutputs; ++i)
        h_rows[i] = (i * 7u + 2u) % kRows;

    void *din = nullptr, *dw = nullptr, *drows = nullptr, *dctx = nullptr;
    void *d_gathered_ref = nullptr, *d_gathered_fused = nullptr, *d_norm = nullptr;
    void *d_codes_ref = nullptr, *d_codes_fused = nullptr;
    void *d_scales_ref = nullptr, *d_scales_fused = nullptr;
    auto cleanup = [&]() {
        void* all[] = {din, dw, drows, dctx, d_gathered_ref, d_gathered_fused, d_norm,
                       d_codes_ref, d_codes_fused, d_scales_ref, d_scales_fused};
        for (void* p : all)
            if (p != nullptr) shelf::gpu::discard_cleanup_result(hipFree(p));
    };
    const size_t elem = sizeof(shelf::bf16_t);
    if (hipMalloc(&din, hin.size() * elem) != hipSuccess ||
        hipMalloc(&dw, hw.size() * elem) != hipSuccess ||
        hipMalloc(&drows, h_rows.size() * sizeof(uint32_t)) != hipSuccess ||
        hipMalloc(&dctx, sizeof(shelf::runtime::DeviceBatchContext)) != hipSuccess ||
        hipMalloc(&d_gathered_ref, kOutputs * kFeatures * elem) != hipSuccess ||
        hipMalloc(&d_gathered_fused, kOutputs * kFeatures * elem) != hipSuccess ||
        hipMalloc(&d_norm, kOutputs * kFeatures * elem) != hipSuccess ||
        hipMalloc(&d_codes_ref, rows_padded * code_stride) != hipSuccess ||
        hipMalloc(&d_codes_fused, rows_padded * code_stride) != hipSuccess ||
        hipMalloc(&d_scales_ref, kOutputs * scale_stride) != hipSuccess ||
        hipMalloc(&d_scales_fused, kOutputs * scale_stride) != hipSuccess) {
        cleanup();
        return false;
    }
    if (!copy_to(din, hin.data(), hin.size() * elem) ||
        !copy_to(dw, hw.data(), hw.size() * elem) ||
        !copy_to(drows, h_rows.data(), h_rows.size() * sizeof(uint32_t))) {
        cleanup();
        return false;
    }
    shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = kRows;
    ctx.num_outputs = kOutputs;
    ctx.output_rows = static_cast<uint32_t*>(drows);
    if (!copy_to(dctx, &ctx, sizeof(ctx))) {
        cleanup();
        return false;
    }
    shelf::gpu::discard_cleanup_result(hipMemset(d_codes_ref, 0xAB,
        rows_padded * code_stride));
    shelf::gpu::discard_cleanup_result(hipMemset(d_codes_fused, 0xAB,
        rows_padded * code_stride));
    shelf::gpu::discard_cleanup_result(hipMemset(d_scales_ref, 0xAB,
        kOutputs * scale_stride));
    shelf::gpu::discard_cleanup_result(hipMemset(d_scales_fused, 0xAB,
        kOutputs * scale_stride));

    const auto input = static_cast<const shelf::bf16_t*>(din);
    const auto w = static_cast<const shelf::bf16_t*>(dw);
    const auto* batch = static_cast<const shelf::runtime::DeviceBatchContext*>(dctx);
    auto* gathered_ref = static_cast<shelf::bf16_t*>(d_gathered_ref);
    auto* gathered_fused = static_cast<shelf::bf16_t*>(d_gathered_fused);
    auto* norm = static_cast<shelf::bf16_t*>(d_norm);
    auto* codes_ref = static_cast<uint8_t*>(d_codes_ref);
    auto* codes_fused = static_cast<uint8_t*>(d_codes_fused);
    auto* scales_ref = static_cast<float*>(d_scales_ref);
    auto* scales_fused = static_cast<float*>(d_scales_fused);

    hipError_t err = shelf::kernel::launch_output_gather(
        input, kFeatures, gathered_ref, kFeatures, batch, kOutputs, kFeatures,
        shelf::runtime::ValueDType::BF16, nullptr);
    if (err == hipSuccess)
        err = shelf::kernel::launch_rmsnorm(
            gathered_ref, shelf::kernel::RmsNormDataType::BF16, w,
            shelf::kernel::RmsNormDataType::BF16,
            shelf::kernel::RmsNormWeightLayout::PER_FEATURE,
            shelf::kernel::RmsNormWeightMode::ONE_PLUS, norm,
            shelf::kernel::RmsNormDataType::BF16, kOutputs, kFeatures, kFeatures,
            kFeatures, kFeatures, kEps, nullptr);
    if (err == hipSuccess)
        err = shelf::kernel::launch_activation_quantize_fp8_finegrained(
            norm, codes_ref, scales_ref, kOutputs, kFeatures, kFeatures, kFeatures,
            code_stride, scale_stride, nullptr, nullptr);
    if (err != hipSuccess) {
        std::printf("  final reference launch failed: %s\n", hipGetErrorString(err));
        cleanup();
        return false;
    }

    shelf::execution::Fp8FinalProducer args{};
    args.input = input;
    args.weight = w;
    args.gathered = gathered_fused;
    args.codes = codes_fused;
    args.scales = scales_fused;
    args.batch = batch;
    args.rows = kOutputs;
    args.k = kFeatures;
    args.k_padded = kFeatures;
    args.input_stride = kFeatures;
    args.gathered_stride = kFeatures;
    args.code_stride = code_stride;
    args.scale_stride = scale_stride;
    args.eps = kEps;
    err = shelf::execution::launch_fp8_final_gather_norm_quantize(args, nullptr);
    if (err != hipSuccess) {
        std::printf("  final fused launch failed: %s\n", hipGetErrorString(err));
        cleanup();
        return false;
    }

    std::vector<shelf::bf16_t> h_gathered_ref(kOutputs * kFeatures), h_gathered_fused(kOutputs * kFeatures);
    std::vector<uint8_t> h_codes_ref(rows_padded * code_stride), h_codes_fused(rows_padded * code_stride);
    std::vector<float> h_scales_ref(kOutputs * scale_stride / 4u), h_scales_fused(kOutputs * scale_stride / 4u);
    bool ok = copy_from(h_gathered_ref, d_gathered_ref) &&
              copy_from(h_gathered_fused, d_gathered_fused) &&
              copy_from(h_codes_ref, d_codes_ref) && copy_from(h_codes_fused, d_codes_fused) &&
              copy_from(h_scales_ref, d_scales_ref) && copy_from(h_scales_fused, d_scales_fused);
    if (ok) {
        ok = std::memcmp(h_gathered_ref.data(), h_gathered_fused.data(),
                         h_gathered_ref.size() * elem) == 0 &&
             std::memcmp(h_codes_ref.data(), h_codes_fused.data(),
                         h_codes_ref.size()) == 0 &&
             std::memcmp(h_scales_ref.data(), h_scales_fused.data(),
                         h_scales_ref.size() * sizeof(float)) == 0;
    }
    cleanup();
    return ok;
}

}

int main() {
    check(run_residual(3u), "fp8 residual norm quantize rows=3");
    check(run_residual(17u), "fp8 residual norm quantize rows=17");
    check(run_final(3u), "fp8 final gather norm quantize outputs=3");
    check(run_final(16u), "fp8 final gather norm quantize outputs=16");
    std::printf("passed=%d failed=%d\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
