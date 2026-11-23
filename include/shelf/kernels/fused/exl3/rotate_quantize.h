#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// `reset_count`, when non-null, is set to zero by the first block. The fused dense row
// preparation (launch_exl3_mul24_rowscale_fused) uses it as the atomic compact-slot counter.
hipError_t launch_exl3_mul24_rotate_quantize(
    const ::shelf::bf16_t* input, uint32_t input_row_stride, const uint16_t* suh, int8_t* codes,
    float* scales, uint32_t rows, uint32_t k, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, hipStream_t stream, uint32_t* reset_count = nullptr);

}  // namespace shelf::kernel
