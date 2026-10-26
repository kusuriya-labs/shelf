// vendored from PhaseShift (own work); namespace/type adapted only.
// (PhaseShift's include of optimized/detail/vector_io.h dropped: nothing in
// this header uses it; it is only needed by the not-yet-vendored fused GDN
// post kernels.)
// (SYCL port: duplicate of l2_normalize_device.h carried from the sibling
// GDN TU so both upstream bodies coexist, exactly as in the HIP backend.
// xor butterfly -> sub_group collective; see sycl_util.h note.)
#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

#include <shelf/backend/sycl/kernels/detail/vector_io.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

namespace shelf::syclk::detail {

constexpr uint32_t kGdnL2ElemsPerLane = 4u;
constexpr uint32_t kGdnL2Threads = 32u;

inline float gdn_silu_f32(float x) { return x / (1.0f + ::sycl::exp(-x)); }

inline float gdn_l2_warp_reduce_sum(const ::sycl::sub_group& sg, float v) {
    return wave_sum_all(sg, v);
}

inline float gdn_l2_head_partial_sum(
    const bf16* in, uint32_t lane, uint32_t group_size, bool aligned) {
    float sum = 0.0f;
    for (uint32_t base = lane * kGdnL2ElemsPerLane; base < group_size;
         base += kGdnL2Threads * kGdnL2ElemsPerLane) {
        if (aligned && base + kGdnL2ElemsPerLane <= group_size) {
            const F4 x = load_bf16x4(in + base);
            sum += x.v[0] * x.v[0] + x.v[1] * x.v[1] + x.v[2] * x.v[2] + x.v[3] * x.v[3];
        } else {
            for (uint32_t j = 0; j < kGdnL2ElemsPerLane; ++j) {
                const uint32_t h = base + j;
                if (h >= group_size) break;
                const float x = ::shelf::bf16_to_f32(in[h]);
                sum += x * x;
            }
        }
    }
    return sum;
}

template <bool WithScale>
inline void gdn_l2_normalize_head_body(
    const bf16* in, float* out, float* out_scaled, float scale,
    const ::sycl::sub_group& sg, uint32_t lane, uint32_t group_size, float eps, bool aligned) {
    const float sum = gdn_l2_warp_reduce_sum(sg, gdn_l2_head_partial_sum(in, lane, group_size, aligned));
    const float inv = ::sycl::rsqrt(sum + eps);
    for (uint32_t base = lane * kGdnL2ElemsPerLane; base < group_size;
         base += kGdnL2Threads * kGdnL2ElemsPerLane) {
        const uint32_t n = (base + kGdnL2ElemsPerLane <= group_size)
                               ? kGdnL2ElemsPerLane
                               : (group_size - base);
        for (uint32_t j = 0u; j < n; ++j) {
            const uint32_t h = base + j;
            const float y = ::shelf::bf16_to_f32(in[h]) * inv;
            out[h] = y;
            if constexpr (WithScale) {
                out_scaled[h] = y * scale;
            }
        }
    }
}

} // namespace shelf::syclk::detail
