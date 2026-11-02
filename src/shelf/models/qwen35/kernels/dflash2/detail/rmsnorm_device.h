#pragma once

#include <shelf/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel::detail {

constexpr uint32_t kDflash2NormThreads = 128u;

__device__ __forceinline__ float dflash2_norm_block_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step > 0u; step >>= 1u) {
        if (threadIdx.x < step) {
            scratch[threadIdx.x] += scratch[threadIdx.x + step];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ void dflash2_rmsnorm_group_body(
    const bf16* x,
    const bf16* weight,
    bf16* out,
    uint32_t group_size,
    uint32_t weight_index_base,
    float eps,
    float* scratch) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        const float v = __bfloat162float(x[i]);
        sum += v * v;
    }
    const float mean =
        dflash2_norm_block_reduce_sum(sum, scratch) / static_cast<float>(group_size);
    const float inv = rsqrtf(mean + eps);

    for (uint32_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        const float norm = __bfloat162float(x[i]) * inv;
        const float norm_rounded = __bfloat162float(__float2bfloat16(norm));
        const float w = __bfloat162float(weight[weight_index_base + i]);
        out[i] = __float2bfloat16(w * norm_rounded);
    }
}

}  // namespace ps::kernel::detail
