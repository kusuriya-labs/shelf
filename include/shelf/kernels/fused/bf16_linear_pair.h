#pragma once

#include <shelf/kernels/optimized/linear/bf16.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_gemm_bf16_exact_rows_pair(
    const bf16_t* weight0,
    void* output0,
    Bf16GemmOutputDType output_dtype0,
    uint32_t out_features0,
    uint32_t output_row_stride0,
    const bf16_t* weight1,
    void* output1,
    Bf16GemmOutputDType output_dtype1,
    uint32_t out_features1,
    uint32_t output_row_stride1,
    const bf16_t* input,
    uint32_t input_row_stride,
    uint32_t rows,
    uint32_t k,
    bool dot2,
    hipStream_t stream);

}  // namespace shelf::kernel
