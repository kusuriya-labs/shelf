// vendored from PhaseShift (own work); namespace/type adapted only.
// (PhaseShift's include of optimized/detail/vector_io.h dropped: nothing in
// this header uses it; it is only needed by the not-yet-vendored fused GDN
// post kernels.)
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace shelf::hipk::detail {

constexpr uint32_t kGdnL2ElemsPerLane = 4u;
constexpr uint32_t kGdnL2Threads = 32u;

__device__ __forceinline__ float gdn_silu_f32(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float gdn_l2_warp_reduce_sum(float v) {
    v += __shfl_xor(v, 16, 32);
    v += __shfl_xor(v, 8, 32);
    v += __shfl_xor(v, 4, 32);
    v += __shfl_xor(v, 2, 32);
    v += __shfl_xor(v, 1, 32);
    return v;
}

__device__ __forceinline__ float gdn_l2_head_partial_sum(
    const __hip_bfloat16* in, uint32_t lane, uint32_t group_size, bool aligned) {
    float sum = 0.0f;
    for (uint32_t base = lane * kGdnL2ElemsPerLane; base < group_size;
         base += kGdnL2Threads * kGdnL2ElemsPerLane) {
        if (aligned && base + kGdnL2ElemsPerLane <= group_size) {
            uint64_t bits = *reinterpret_cast<const uint64_t*>(in + base);
            __hip_bfloat16 b[4];
            *reinterpret_cast<uint64_t*>(b) = bits;
            const float x0 = __bfloat162float(b[0]);
            const float x1 = __bfloat162float(b[1]);
            const float x2 = __bfloat162float(b[2]);
            const float x3 = __bfloat162float(b[3]);
            sum += x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
        } else {
            for (uint32_t j = 0; j < kGdnL2ElemsPerLane; ++j) {
                const uint32_t h = base + j;
                if (h >= group_size) break;
                const float x = __bfloat162float(in[h]);
                sum += x * x;
            }
        }
    }
    return sum;
}

template <bool WithScale>
__device__ __forceinline__ void gdn_l2_normalize_head_body(
    const __hip_bfloat16* in, float* out, float* out_scaled, float scale,
    uint32_t lane, uint32_t group_size, float eps, bool aligned) {
    const float sum = gdn_l2_warp_reduce_sum(
        gdn_l2_head_partial_sum(in, lane, group_size, aligned));
    const float inv = rsqrtf(sum + eps);
    for (uint32_t base = lane * kGdnL2ElemsPerLane; base < group_size;
         base += kGdnL2Threads * kGdnL2ElemsPerLane) {
        const uint32_t n = (base + kGdnL2ElemsPerLane <= group_size)
                               ? kGdnL2ElemsPerLane
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
