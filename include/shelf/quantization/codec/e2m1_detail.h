#pragma once
#include <cstdint>

namespace shelf::quantization::codec::device {

__device__ __forceinline__ float e2m1_decode_f32(uint8_t code) {
    constexpr float kMagnitude[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float magnitude = kMagnitude[code & 0x7u];
    return (code & 0x8u) != 0u ? -magnitude : magnitude;
}

// Smallest e with 448 * 2^e >= peak, i.e. the E8M0 group exponent whose E4M3
// ceiling covers the group peak. -127 for a zero peak, a subnormal peak, NaN or inf.
__device__ __forceinline__ int32_t e8m0_exponent_for_peak(float peak) {
    const uint32_t bits = __float_as_uint(peak);
    if (!(peak > 0.0f) || bits < 0x00800000u)
        return -127;
    const int32_t e_field = static_cast<int32_t>((bits >> 23) & 0xffu) - 127;
    const float unit = __uint_as_float(static_cast<uint32_t>(e_field + 127) << 23);
    if (peak <= 0.875f * unit)
        return e_field - 9;
    if (peak <= 1.75f * unit)
        return e_field - 8;
    return e_field - 7;
}

// 2^e as f32 for e in [-127, 127]; 2^-127 is emulated exactly as e8m0_decode_f32 does.
__device__ __forceinline__ float pow2_f32(int32_t e) {
    const uint32_t bits = static_cast<uint32_t>(e + 127) << 23;
    return __uint_as_float(bits != 0u ? bits : 0x00400000u);
}

__device__ __forceinline__ float e8m0_decode_f32(uint8_t byte) {
    if (byte == 255u) return __uint_as_float(0x7FC00000u);
    if (byte == 0u) return __uint_as_float(0x00400000u);
    return __uint_as_float(static_cast<uint32_t>(byte) << 23);
}

}  // namespace shelf::quantization::codec::device
