// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Extracted from PhaseShift's optimized/gdn/detail/gdn_prepare_device.h — the
// warp-wide (32-lane) L2 head normalizer. Math is unchanged; the names drop the
// upstream "gdn_" prefix so this file can coexist with the sibling GDN port of
// the same helpers (detail/gdn_prepare_device.h) in one translation unit.
#pragma once

#include <shelf/backend/hip/kernels/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::hipk::detail {

constexpr uint32_t kL2ElemsPerLane = 4u;
constexpr uint32_t kL2Threads = 32u;

__device__ __forceinline__ float l2_warp_reduce_sum(float v) {
    v += __shfl_xor(v, 16, 32);
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 4, 32);
    v += __shfl_xor(v, 2, 32);
    v += __shfl_xor(v, 1, 32);
    return v;
}

__device__ __forceinline__ float l2_head_partial_sum(
    const bf16* in, uint32_t lane, uint32_t group_size, bool aligned) {
    float sum = 0.0f;
    for (uint32_t base = lane * kL2ElemsPerLane; base < group_size;
         base += kL2Threads * kL2ElemsPerLane) {
        if (aligned && base + kL2ElemsPerLane <= group_size) {
            uint64_t bits = *reinterpret_cast<const uint64_t*>(in + base);
            bf16 b[4];
            *reinterpret_cast<uint64_t*>(b) = bits;
            const float x0 = __bfloat162float(b[0]);
            const float x1 = __bfloat162float(b[1]);
            const float x2 = __bfloat162float(b[2]);
            const float x3 = __bfloat162float(b[3]);
            sum += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
        } else {
            for (uint32_t j = 0; j < kL2ElemsPerLane; ++j) {
                const uint32_t h = base + j;
                if (h >= group_size) break;
                const float x = __bfloat162float(in[h]);
                sum += x * x;
            }
        }
    }
    return sum;
}

// eps is added to the sum of squares (not divided by the group length):
//   inv = rsqrt(sum(x^2) + eps)
// WithScale additionally writes y * scale into out_scaled.
template <bool WithScale>
__device__ __forceinline__ void l2_normalize_head_body(
    const bf16* in, float* out, float* out_scaled, float scale,
    uint32_t lane, uint32_t group_size, float eps, bool aligned) {
    const float sum = l2_warp_reduce_sum(
        l2_head_partial_sum(in, lane, group_size, aligned));
    const float inv = rsqrtf(sum + eps);
    for (uint32_t base = lane * kL2ElemsPerLane; base < group_size;
         base += kL2Threads * kL2ElemsPerLane) {
        const uint32_t n = (base + kL2ElemsPerLane <= group_size)
                               ? kL2ElemsPerLane
                               : (group_size - base);
        for (uint32_t j = 0u; j < n; ++j) {
            const uint32_t h = base + j;
            const float y = __bfloat162float(in[h]) * inv;
            out[h] = y;
            if constexpr (WithScale) {
                out_scaled[h] = y * scale;
            }
        }
    }
}

}  // namespace shelf::hipk::detail
