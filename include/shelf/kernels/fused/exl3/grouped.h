#pragma once

#include <shelf/kernels/optimized/linear/exl3_mul24.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_exl3_mul24_grouped(const Exl3GemmArgs& args, uint32_t n_tiles,
                                     uint32_t grid_x, uint32_t grid_z, hipStream_t stream);

// Pre-rotates a grouped row plane once: INT8 codes [group_row][k] and one F32 scale per
// 128-wide chunk [group_row][k / 128]. Each row uses the suh of its group's expert.
hipError_t launch_exl3_grouped_activate(
    const void* input,
    const void* suh,
    void* codes,
    float* scales,
    const uint32_t* group_expert,
    const uint32_t* group_rows,
    const uint32_t* group_count,
    const uint32_t* group_row_base,
    const int32_t* row_token,
    uint32_t row_capacity,
    uint32_t k,
    uint32_t input_row_stride,
    uint64_t suh_batch_stride,
    hipStream_t stream);

}  // namespace shelf::kernel
