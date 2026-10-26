// vendored from PhaseShift (own work); namespace/type adapted only.
//
// Minimal pieces extracted from PhaseShift's optimized/gdn/detail/
// {gdn_prepare_device.h,gdn_post_device.h}: the silu scalar and the
// f32*f32->bf16 vector body used by the elementwise ops. Math is byte-for-byte
// unchanged; the helper names drop the "gdn_" prefix so these do not collide
// with the sibling GDN port (detail/gdn_prepare_device.h), which carries the
// same upstream bodies.
#pragma once

#include <shelf/backend/hip/kernels/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::hipk::detail {

__device__ __forceinline__ float silu_f32(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ void mul_f32_bf16_body(
    const float* a_row, const float* b_row, bf16* out_row, uint32_t col0,
    uint32_t features) {
    if (col0 >= features) return;
    if (col0 + kVec - 1u < features && aligned16(a_row + col0) && aligned16(b_row + col0) &&
        aligned8(out_row + col0)) {
        const F4 xa = load_f32x4(a_row + col0);
        const F4 xb = load_f32x4(b_row + col0);
        F4 y;
        y.v[0] = xa.v[0] * xb.v[0];
        y.v[1] = xa.v[1] * xb.v[1];
        y.v[2] = xa.v[2] * xb.v[2];
        y.v[3] = xa.v[3] * xb.v[3];
        store_bf16x4(out_row + col0, y);
        return;
    }
    for (uint32_t i = 0u; i < kVec; ++i) {
        const uint32_t c = col0 + i;
        if (c >= features) break;
        out_row[c] = __float2bfloat16(a_row[c] * b_row[c]);
    }
}

}  // namespace shelf::hipk::detail
