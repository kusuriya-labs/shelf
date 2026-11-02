#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace ps::kernel::detail {

__device__ __forceinline__ float dflash2_round_to_bf16_f32(float x) {
    uint32_t u = __float_as_uint(x);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        u = ((u & 0x007FFFFFu) != 0u) ? 0x7FC00000u : (u & 0xFF800000u);
        return __uint_as_float(u);
    }
    u += 0x7FFFu + ((u >> 16u) & 1u);
    return __uint_as_float(u & 0xFFFF0000u);
}

__device__ __forceinline__ float dflash2_swiglu_rounded_f32(float gate, float up) {
    const float sigmoid = 1.0f / (1.0f + expf(-gate));
    const float silu = dflash2_round_to_bf16_f32(gate * sigmoid);
    return dflash2_round_to_bf16_f32(silu * up);
}

}  // namespace ps::kernel::detail
