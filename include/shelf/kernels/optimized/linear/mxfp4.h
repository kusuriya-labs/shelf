#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/linear/config.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

enum class Mxfp4GemmOutputDType : uint8_t {
    BF16,
    F32,
};

// MXFP4 W4A8 linear on the gfx1201 FP8 WMMA unit.
//
// Weight codes are consumed in the native MXFP4 layout (16-output x 32-K blocks,
// byte b of an output row holding code[k = b] in the low nibble and
// code[k = 16 + b] in the high nibble; see quantization/mxfp4/mxfp4_native.h)
// and E8M0 scales are row-major one byte per 32-K block. The kernel expands
// E2M1 nibbles to exact E4M3 operands in registers with v_perm_b32 and applies
// weight_scale * activation_scale after each 32-K block. Activation E4M3 codes
// and FP32 per-128 scales use the fragment-order Int8ActivationWorkspaceLayout
// shared with the FP8 W8A8 path.
hipError_t launch_gemm_mxfp4_w4a8_wmma(
    const Mxfp4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

// Single-token decode specialization (rows == 1, out_features % 16 == 0,
// k_padded % 128 == 0), BF16 output only. unroll == 0 selects the measured
// default.
bool mxfp4_decode1_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded);

uint32_t mxfp4_decode1_unroll(uint32_t out_features, uint32_t k_padded);

hipError_t launch_gemm_mxfp4_w4a8_decode1(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    shelf::bf16_t* output,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t unroll,
    hipStream_t stream);

// Small-M (2 <= rows <= 128) split-K variant with an in-block LDS reduction.
bool mxfp4_splitk_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded);

hipError_t launch_gemm_mxfp4_w4a8_splitk(
    uint32_t split_k,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

// Applies the config selected by the linear selector. A config whose geometry
// is not supported falls back to the generic row-block path.
hipError_t launch_gemm_mxfp4_w4a8_auto(
    const Mxfp4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

}  // namespace shelf::kernel
