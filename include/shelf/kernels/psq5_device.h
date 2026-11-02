#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float psq5_scale_f32(uint8_t stored) {
    const uint32_t e = static_cast<uint32_t>(stored >> 4);
    const uint32_t m = static_cast<uint32_t>(stored & 0x0Fu);
    const uint32_t bits_norm = ((e + 120u) << 23) | (m << 19);
    const uint32_t bits_zero = __float_as_uint(static_cast<float>(m) * 9.765625e-4f);
    const uint32_t mask_zero = 0u - static_cast<uint32_t>(e == 0u);
    uint32_t bits = (bits_zero & mask_zero) | (bits_norm & ~mask_zero);
    const uint32_t mask_nan = 0u - static_cast<uint32_t>(e >= 15u);
    bits = (0x7FC00000u & mask_nan) | (bits & ~mask_nan);
    return __uint_as_float(bits);
}

__device__ __forceinline__ uint32_t psq5_cb28_4x(uint32_t m) {
    const uint32_t h = m & 0x08080808u;
    return m + ((m & 0x07070707u) & (h - (h >> 3)));
}

__device__ __forceinline__ uint32_t psq5_sign_4x(uint32_t s) {
    return ((s * 0x00204081u) & 0x01010101u) << 7;
}

__device__ __forceinline__ int32x2_t psq5_unpack_e4m3(uint32_t magnitude4, uint8_t sign_byte) {
    const uint32_t lo = magnitude4 & 0x0F0F0F0Fu;
    const uint32_t hi = (magnitude4 >> 4) & 0x0F0F0F0Fu;
    const uint32_t m0 = __builtin_amdgcn_perm(hi, lo, 0x05010400u);
    const uint32_t m1 = __builtin_amdgcn_perm(hi, lo, 0x07030602u);
    const uint32_t w0 = psq5_cb28_4x(m0) | psq5_sign_4x(sign_byte & 0x0Fu);
    const uint32_t w1 = psq5_cb28_4x(m1) | psq5_sign_4x(sign_byte >> 4);
    int32x2_t r;
    r[0] = static_cast<int32_t>(w0);
    r[1] = static_cast<int32_t>(w1);
    return r;
}

}
