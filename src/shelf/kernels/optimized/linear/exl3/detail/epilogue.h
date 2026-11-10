#pragma once

// Shared output contract. The kernels keep their own epilogue loops (fused dense
// epilogue, split-K reduce, mul24 rotate_out); only the small type/stride helpers and
// the documented ordering live here.
//
// Order for every EXL3 epilogue:
//   multiply by row scale (mul1: rowsum bias and kMul1Inv; mul24: rowscale and S_FIX),
//   apply H128, multiply by 1/sqrt(128) and svh, then round to the output type.
// Split-K reductions sum the split slices in split index order before this sequence.

#include <shelf/kernels/optimized/linear/exl3.h>

#include <cstdint>

namespace shelf::kernel::exl3_detail {

// Row strides are in elements, so the byte stride is elements * this value.
inline constexpr uint32_t output_element_bytes(Exl3OutputDType dtype) noexcept {
    return dtype == Exl3OutputDType::BF16 ? 2u : 4u;
}

}  // namespace shelf::kernel::exl3_detail
