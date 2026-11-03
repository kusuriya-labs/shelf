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

// How the activation plane is laid out. The producer (an activation quantize kernel)
// and the consumer (a MXFP4 GEMM) must agree.
//   GroupF32: the FP8 W8A8 shape - E4M3 codes plus one f32 scale per 128 K group.
//     The GEMM applies that scale inside the K loop.
//   RowRefFolded: the MXFP4 W4A8 row-reference shape produced by
//     launch_activation_quantize_mxfp4_rowref - each 128 K group E8M0 is folded into
//     the E4M3 codes relative to the row's maximum group exponent, and
//     activation_scales[row * stride_floats + 0] holds 2^ref. The GEMM applies no
//     activation scale inside the K loop and multiplies the row reference once in
//     the epilogue.
enum class Mxfp4ActivationForm : uint8_t {
    GroupF32,
    RowRefFolded,
};

// MXFP4 W4A8 linear on the gfx1201 FP8 WMMA unit.
//
// Weight codes are consumed in the native MXFP4 layout (16-output x 32-K blocks,
// byte b of an output row holding code[k = b] in the low nibble and
// code[k = 16 + b] in the high nibble; see quantization/mxfp4/mxfp4_native.h)
// and E8M0 scales are row-major one byte per 32-K block. Activation E4M3 codes
// and FP32 per-128 scales use the fragment-order Int8ActivationWorkspaceLayout
// shared with the FP8 W8A8 path.
//
// weight_ref is one E8M0 byte per output row (the row's maximum scale exponent)
// and selects the accumulation order:
//   - non-null: each 32-K block's E8M0 is folded into the E4M3 weight operand as
//     2^(e8m0 - weight_ref[row]), the four blocks of a 128-K activation group
//     share one accumulator, and 2^(weight_ref[row] - 127) is applied once. The
//     quantized values are unchanged (d <= 8 is exact), the summation order is not.
//   - null: each 32-K block is rescaled by its own E8M0 and the activation group
//     scale, which is the accumulation order of the correctness path.
hipError_t launch_gemm_mxfp4_w4a8_wmma(
    const Mxfp4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* weight_ref,
    const int8_t* activation_codes,
    const float* activation_scales,
    Mxfp4ActivationForm activation_form,
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

// Small row-count split-K variant with an in-block LDS reduction.
bool mxfp4_splitk_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded);

hipError_t launch_gemm_mxfp4_w4a8_splitk(
    uint32_t split_k,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* weight_ref,
    const int8_t* activation_codes,
    const float* activation_scales,
    Mxfp4ActivationForm activation_form,
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
    const uint8_t* weight_ref,
    const int8_t* activation_codes,
    const float* activation_scales,
    Mxfp4ActivationForm activation_form,
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
