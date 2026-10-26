// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Extracted from PhaseShift's optimized/gdn/detail/gdn_prepare_device.h — the
// warp-wide (32-lane) L2 head normalizer. Math is unchanged; the names drop the
// upstream "gdn_" prefix so this file can coexist with the sibling GDN port of
// the same helpers (detail/gdn_prepare_device.h) in one translation unit.
// (SYCL port: xor butterfly -> sub_group collective; see sycl_util.h note.)
#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

#include <shelf/backend/sycl/kernels/detail/vector_io.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

namespace shelf::syclk::detail {

constexpr uint32_t kL2ElemsPerLane = 4u;
constexpr uint32_t kL2Threads = 32u;

inline float l2_warp_reduce_sum(const ::sycl::sub_group& sg, float v) {
    return wave_sum_all(sg, v);
}

inline float l2_head_partial_sum(
    const bf16* in, uint32_t lane, uint32_t group_size, bool aligned) {
    float sum = 0.0f;
    for (uint32_t base = lane * kL2ElemsPerLane; base < group_size;
         base += kL2Threads * kL2ElemsPerLane) {
        if (aligned && base + kL2ElemsPerLane <= group_size) {
            const F4 x = load_bf16x4(in + base);
            sum += x.v[0] * x.v[0] + x.v[1] * x.v[1] + x.v[2] * x.v[2] + x.v[3] * x.v[3];
        } else {
            for (uint32_t j = 0; j < kL2ElemsPerLane; ++j) {
                const uint32_t h = base + j;
                if (h >= group_size) break;
                const float x = ::shelf::bf16_to_f32(in[h]);
                sum += x * x;
            }
        }
    }
    return sum;
}

// inv = rsqrt(sum(x^2) + eps) over each group of `group_size` elements.
// WithScale additionally writes y * scale into out_scaled.
template <bool WithScale>
inline void l2_normalize_head_body(
    const bf16* in, float* out, float* out_scaled, float scale,
    const ::sycl::sub_group& sg, uint32_t lane, uint32_t group_size, float eps, bool aligned) {
    const float sum = l2_warp_reduce_sum(sg, l2_head_partial_sum(in, lane, group_size, aligned));
    const float inv = ::sycl::rsqrt(sum + eps);
    for (uint32_t base = lane * kL2ElemsPerLane; base < group_size;
         base += kL2Threads * kL2ElemsPerLane) {
        const uint32_t n = (base + kL2ElemsPerLane <= group_size)
                               ? kL2ElemsPerLane
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
