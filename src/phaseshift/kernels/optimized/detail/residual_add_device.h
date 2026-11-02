#pragma once
#include "vector_io.h"

namespace ps::kernel::detail {

__device__ __forceinline__ void residual_add_bf16_body(
    const bf16* a_row, const bf16* b_row, bf16* out_row,
    uint32_t col0, uint32_t features) {
    if (col0 >= features) return;
    if (col0 + kVec - 1u < features && aligned8(a_row + col0) && aligned8(b_row + col0) &&
        aligned8(out_row + col0)) {
        const F4 xa = load_bf16x4(a_row + col0);
        const F4 xb = load_bf16x4(b_row + col0);
        F4 y;
        y.v[0] = xa.v[0] + xb.v[0];
        y.v[1] = xa.v[1] + xb.v[1];
        y.v[2] = xa.v[2] + xb.v[2];
        y.v[3] = xa.v[3] + xb.v[3];
        store_bf16x4(out_row + col0, y);
        return;
    }
    for (uint32_t i = 0u; i < kVec; ++i) {
        const uint32_t c = col0 + i;
        if (c >= features) break;
        const float av = __bfloat162float(a_row[c]);
        const float bv = __bfloat162float(b_row[c]);
        out_row[c] = __float2bfloat16(av + bv);
    }
}

}  // namespace ps::kernel::detail
