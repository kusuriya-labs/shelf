#pragma once

// Two distinct FWHT implementations. They are kept separate on purpose: the lane
// mapping, the shared-memory synchronisation and the resulting instruction stream
// differ, and either can win depending on the surrounding kernel.
//
// Both compute the unnormalised Sylvester Walsh-Hadamard transform of one 128-wide
// block; callers apply 1/sqrt(128) themselves.

#include <cstdint>

#include <hip/hip_runtime.h>

namespace shelf::kernel::exl3_detail {

// float4 form: one wave owns the 128-element block, four elements per lane.
// Stage 0-1 are local, stages 2-6 are warp shuffles. No shared memory, no barrier.
__device__ __forceinline__ void fwht128(float& v0, float& v1, float& v2, float& v3,
                                        uint32_t lane) {
    const float a = v0 + v1;
    const float b = v0 - v1;
    const float c = v2 + v3;
    const float d = v2 - v3;
    v0 = a + c;
    v1 = b + d;
    v2 = a - c;
    v3 = b - d;
#pragma unroll
    for (uint32_t step = 1u; step < 32u; step <<= 1u) {
        const float w0 = __shfl_xor(v0, step, 32);
        const float w1 = __shfl_xor(v1, step, 32);
        const float w2 = __shfl_xor(v2, step, 32);
        const float w3 = __shfl_xor(v3, step, 32);
        if ((lane & step) != 0u) {
            v0 = w0 - v0;
            v1 = w1 - v1;
            v2 = w2 - v2;
            v3 = w3 - v3;
        } else {
            v0 += w0;
            v1 += w1;
            v2 += w2;
            v3 += w3;
        }
    }
}

// scalar form: one element per thread (128 threads per block), five warp-shuffle
// stages followed by two shared-memory stages (two barriers per stage).
__device__ __forceinline__ float fwht128_smem(float v, float* smem, uint32_t t) {
#pragma unroll
    for (uint32_t h = 1u; h < 32u; h <<= 1u) {
        const float b = __shfl_xor(v, h);
        v = (t & h) ? (b - v) : (v + b);
    }
#pragma unroll
    for (uint32_t h = 32u; h < 128u; h <<= 1u) {
        smem[t] = v;
        __syncthreads();
        const float b = smem[t ^ h];
        __syncthreads();
        v = (t & h) ? (b - v) : (v + b);
    }
    return v;
}

}  // namespace shelf::kernel::exl3_detail
