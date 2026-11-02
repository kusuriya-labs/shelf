#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace ps::kernel::detail {

__device__ __forceinline__ float target_swiglu_f32(float gate, float up) {
    return up * (gate / (1.0f + expf(-gate)));
}

}  // namespace ps::kernel::detail
