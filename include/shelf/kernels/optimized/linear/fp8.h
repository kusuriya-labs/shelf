#pragma once

#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

enum class Fp8FinegrainedGemmOutputDType : uint8_t {
    BF16,
    F32,
};

// FP8 E4M3 W8A8 WMMA GEMM for weights with a 128 (N) x 128 (K) FP32 scale tile.
// Weight codes are consumed in the gfx1201 FP8 WMMA fragment order (16 output
// rows x 64-K block, one contiguous 1 KiB block per wave, 32 bytes per lane;
// see shelf/quantization/fp8/fp8_frag.h), while the FP32 scales stay row-major
// [ceil(N/128)][Kp/128]. The activation E4M3 codes and per-row FP32 scales use
// the same fragment-order layout.
hipError_t launch_gemm_fp8_finegrained_w8a8_wmma(
    const Fp8FinegrainedGemmConfig& config,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8FinegrainedGemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream,
    uint32_t weight_scale_group = 128);

hipError_t launch_gemm_fp8_finegrained_w8a8_wmma_auto(
    const Fp8FinegrainedGemmConfig& config,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8FinegrainedGemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream,
    uint32_t weight_scale_group = 128);

}  // namespace ps::kernel
