#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// Fused x <- x * scale before SiLU, i.e. siLU(x * scale).
hipError_t launch_silu_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, float scale, hipStream_t stream);

}  // namespace shelf::kernel
