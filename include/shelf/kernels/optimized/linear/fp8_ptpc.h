#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

enum class Fp8PertokenGemmOutputDType : uint8_t {
    BF16,
    F32,
};

// FP8 E4M3 W8A8 WMMA GEMM with one FP32 weight scale per output row and one
// FP32 activation scale per token. Weight and activation codes are consumed in
// the gfx1201 FP8 WMMA fragment order (16 rows x 64 K per block, one contiguous
// 1 KiB block per wave); see shelf/quantization/fp8/fp8_frag.h.
hipError_t launch_gemm_fp8_ptpc(
    const uint8_t* weight_codes,
    const float* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8PertokenGemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream);

}  // namespace shelf::kernel
