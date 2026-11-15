#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_exl3_mul24_rotate_quantize(
    const ::shelf::bf16_t* input, uint32_t input_row_stride, const uint16_t* suh, int8_t* codes,
    float* scales, uint32_t rows, uint32_t k, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, hipStream_t stream);

}  // namespace shelf::kernel
