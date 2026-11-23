#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

// One launch for the dense EXL3 row preparation: compute the row peak / squared sum,
// append flagged rows to a compact plane with an atomic slot, and re-round the other rows
// in place. `count` must be zero before the launch; the dense path lets rotate_quantize's
// reset_count do it. Slot order is arbitrary (rotate_out scatters by slot).
hipError_t launch_exl3_mul24_rowscale_fused(
    int8_t* codes, const float* scales, float* rowscale, int32_t* slots, uint32_t* count,
    int8_t* codes_f, float* scales_f, uint32_t rows, uint32_t cap, uint32_t k,
    uint32_t code_row_stride_bytes, uint32_t scale_row_stride_bytes, float share,
    hipStream_t stream);

}  // namespace shelf::kernel
