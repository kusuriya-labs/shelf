#pragma once

#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

enum class Psq5GemmOutputDType : uint8_t {
    BF16,
    F32,
};

// PSQ5-CB28 W8A8 WMMA GEMM. Weights stay5-bit packed (magnitude 4-bit +
// sign 1-bit) and are unpacked to E4M3 in registers; the per-row per-32-K
// ue4m4 scale decodes to F32 in the kernel. The activation E4M3 codes and
// per-128-K FP32 scales use the shared fragment-order activation layout.
hipError_t launch_gemm_psq5_w8a8_wmma(
    const Psq5GemmConfig& config,
    const uint8_t* weight_magnitude,
    const uint8_t* weight_sign,
    const uint8_t* weight_scale,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq5GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

hipError_t launch_gemm_psq5_w8a8_wmma_auto(
    const Psq5GemmConfig& config,
    const uint8_t* weight_magnitude,
    const uint8_t* weight_sign,
    const uint8_t* weight_scale,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq5GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

}
