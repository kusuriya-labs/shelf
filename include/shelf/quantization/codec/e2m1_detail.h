#pragma once
#include <cstdint>

namespace shelf::quantization::codec::device {

__device__ __forceinline__ float e2m1_decode_f32(uint8_t code) {
    constexpr float kMagnitude[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const float magnitude = kMagnitude[code & 0x7u];
    return (code & 0x8u) != 0u ? -magnitude : magnitude;
}

__device__ __forceinline__ float e8m0_decode_f32(uint8_t byte) {
    if (byte == 255u) return __uint_as_float(0x7FC00000u);
    if (byte == 0u) return __uint_as_float(0x00400000u);
    return __uint_as_float(static_cast<uint32_t>(byte) << 23);
}

}  // namespace shelf::quantization::codec::device
