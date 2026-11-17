#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

// Writes two floats per row: the row maximum and sum(exp(x - max)) over the row.
// The maximum is reduced first so the sum stays in a usable range whatever the
// logits are.
//
// `stats` must hold rows * 2 floats.
hipError_t launch_dflash2_row_logsumexp(
    const float* logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t row_stride,
    float* stats,
    hipStream_t stream);

}  // namespace shelf::kernel
