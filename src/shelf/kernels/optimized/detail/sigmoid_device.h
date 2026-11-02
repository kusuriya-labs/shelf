#pragma once

#include <hip/hip_runtime.h>
#include <cmath>

namespace ps::kernel::detail {

__device__ __forceinline__ float sigmoid_f32(float x) { return 1.0f / (1.0f + expf(-x)); }

}  // namespace ps::kernel::detail
