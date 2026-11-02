#pragma once

#include <phaseshift/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>
#include <cmath>
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

__device__ __forceinline__ void dflash2_rope_pair_body(
    const bf16* src, bf16* dst, uint32_t pair, uint32_t head_dim, uint32_t position,
    float theta) {
    const uint32_t half = head_dim / 2u;
    const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim);
    const float inv_freq = powf(theta, exponent);
    const float angle = static_cast<float>(position) * inv_freq;
    const float c = dflash2_round_to_bf16_f32(cosf(angle));
    const float s = dflash2_round_to_bf16_f32(sinf(angle));

    const float x1 = __bfloat162float(src[pair]);
    const float x2 = __bfloat162float(src[pair + half]);
    const float p1 = dflash2_round_to_bf16_f32(x1 * c);
    const float p2 = dflash2_round_to_bf16_f32(x2 * s);
    const float p3 = dflash2_round_to_bf16_f32(x2 * c);
    const float p4 = dflash2_round_to_bf16_f32(x1 * s);
    dst[pair] = __float2bfloat16(p1 - p2);
    dst[pair + half] = __float2bfloat16(p3 + p4);
}

}  // namespace ps::kernel::detail
