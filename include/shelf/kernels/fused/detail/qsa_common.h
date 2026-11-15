#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel::detail::qsa {

__device__ __forceinline__ float bf16_value(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<std::uint32_t>(p->data) << 16);
}

__device__ __forceinline__ shelf::bf16_t bf16_from(float value) {
    return shelf::bf16_t{
        static_cast<std::uint16_t>(__bfloat16_as_ushort(__float2bfloat16(value)))};
}

__device__ __forceinline__ float block_sum(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = kQsaThreads / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

}  // namespace shelf::kernel::detail::qsa
