// shelf HIP Tile primitives — BF16 WMMA fragment types and the gfx1201
// matrix-dot builtin wrapper (tile-refactor Phase 2/6, procedure §4.1-A).
// Private to the HIP backend. Layout contract (wave32, 16x16x16):
//   A/B fragment: lane holds {k_half*4..+3, 8+k_half*4..+3}; row = lane & 15.
//   C fragment:   lane row = lane & 15, cols 8*(lane>>4) + 0..7.
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

namespace shelf::hipk {
namespace tile {

using Bf16x8 = short __attribute__((ext_vector_type(8)));
using F32x8 = float __attribute__((ext_vector_type(8)));

// Thin wrapper over the WMMA builtin: must not add copies or change the
// instruction sequence (gate: ISA histogram, v_wmma counts).
__device__ __forceinline__ F32x8 dot_bf16_16x16x16(Bf16x8 a, Bf16x8 b, F32x8 acc) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
}

}  // namespace tile
}  // namespace shelf::hipk
