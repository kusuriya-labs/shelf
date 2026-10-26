// shelf HIP Tile primitives — wave32 lane view and warp reductions
// (tile-refactor Phase 6; procedure §7.2 / §8). Demonstrated consumers:
// attention paged decode (BF16), FP8 decode, BF16/FP8 split.
//
// GEMM ExactRows uses a shuffle-down tree reduce (different order) and is
// deliberately NOT routed through these helpers (contract §3).
//
// Freeze requirement: the xor order 16,8,4,2,1 matches the vendored
// PhaseShift kernels; changing it changes results (bitwise gate).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace shelf::hipk {
namespace tile {

// Static lane/wave view of a block that is a multiple of 32 threads.
struct Wave32 {
    uint32_t tid;
    uint32_t lane;   // tid & 31
    uint32_t wave;   // tid >> 5  (RDNA wavefront index)

    __device__ __forceinline__ static Wave32 current() {
        const uint32_t tid = threadIdx.x;
        return {tid, tid & 31u, tid >> 5u};
    }
};

__device__ __forceinline__ float warp_sum_all(float x) {
    x += __shfl_xor(x, 16, 32);
    x += __shfl_xor(x, 8, 32);
    x += __shfl_xor(x, 4, 32);
    x += __shfl_xor(x, 2, 32);
    x += __shfl_xor(x, 1, 32);
    return x;
}

__device__ __forceinline__ float warp_max_all(float x) {
    x = fmaxf(x, __shfl_xor(x, 16, 32));
    x = fmaxf(x, __shfl_xor(x, 8, 32));
    x = fmaxf(x, __shfl_xor(x, 4, 32));
    x = fmaxf(x, __shfl_xor(x, 2, 32));
    x = fmaxf(x, __shfl_xor(x, 1, 32));
    return x;
}

}  // namespace tile
}  // namespace shelf::hipk
