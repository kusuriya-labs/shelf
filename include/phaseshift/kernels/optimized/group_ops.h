#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kGroupOpsBlock = 256u;

hipError_t launch_group_mean_f32(
    const float* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, uint32_t group,
    hipStream_t stream);

hipError_t launch_group_broadcast_mul_f32(
    const float* a, uint32_t a_row_stride,
    const float* b, uint32_t b_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, uint32_t group,
    hipStream_t stream);

}
