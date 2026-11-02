#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kMoeOpsBlock = 256u;

hipError_t launch_moe_router_f32(
    const float* x, uint32_t x_row_stride,
    const void* router_w,
    uint32_t experts, uint32_t hidden,
    uint32_t top_k, bool renorm,
    int32_t* top_idx, float* top_w,
    uint32_t rows, hipStream_t stream);

hipError_t launch_moe_experts_f32(
    const float* x, uint32_t x_row_stride,
    const void* gate_up,
    const void* down,
    uint32_t hidden, uint32_t inter, uint32_t experts,
    const int32_t* top_idx, const float* top_w, uint32_t top_k,
    float* out, uint32_t out_row_stride,
    uint32_t rows, hipStream_t stream);

}
