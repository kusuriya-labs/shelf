#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::execution {

// BF16 GEMM through hipBLASLt: output[r, o] = sum_k input[r, k] * weight[o, k].
// weight is row-major [out_features, k], input row-major [rows, input_row_stride],
// output row-major [rows, output_row_stride].
hipError_t gemm_bf16_lt(const shelf::bf16_t* weight,
                        const shelf::bf16_t* input,
                        void* output,
                        uint32_t rows,
                        uint32_t out_features,
                        uint32_t k,
                        uint32_t input_row_stride,
                        uint32_t output_row_stride,
                        bool output_bf16,
                        hipStream_t stream);

}  // namespace shelf::execution
