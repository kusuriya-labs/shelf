#include <shelf/quantization/codec/e4m3.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace shelf::quantization::codec {

float e4m3_decode_f32(uint8_t x) {
    const uint32_t sign = (static_cast<uint32_t>(x) >> 7) & 1u;
    const uint32_t e = (static_cast<uint32_t>(x) >> 3) & 0xFu;
    const uint32_t m = static_cast<uint32_t>(x) & 7u;
    if (x == 0x7Fu || x == 0xFFu) {
        return std::nanf("");
    }
    const float mag = (e == 0u)
        ? static_cast<float>(std::ldexp(static_cast<double>(m), -9))
        : static_cast<float>(std::ldexp(static_cast<double>(8u + m), static_cast<int>(e) - 10));
    return sign ? -mag : mag;
}

uint8_t e4m3_encode_u8(float x) {
    if (std::isnan(x)) {
        return 0x7Fu;
    }
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const uint32_t mag = bits & 0x7FFFFFFFu;
    // 448.0f is 0x43E00000. Everything above it saturates to the largest finite code,
    // the same code that infinite input produces.
    if (mag >= 0x43E00000u) {
        return static_cast<uint8_t>(sign | 0x7Eu);
    }
    const int exp = static_cast<int>(mag >> 23) - 127;
    const uint32_t mant = mag & 0x7FFFFFu;
    if (exp >= -6) {
        uint32_t e4 = static_cast<uint32_t>(exp + 7);
        constexpr uint32_t kShift = 20u;
        uint32_t m3 = mant >> kShift;
        const uint32_t rem = mant & ((1u << kShift) - 1u);
        const uint32_t half = 1u << (kShift - 1u);
        if (rem > half || (rem == half && (m3 & 1u) != 0u)) ++m3;
        if (m3 >= 8u) {
            m3 = 0u;
            ++e4;
        }
        if (e4 >= 15u) {
            // 0x7F is NaN in E4M3FN, so the largest finite value is 0x7E.
            if (e4 > 15u || m3 > 6u) return static_cast<uint8_t>(sign | 0x7Eu);
            return static_cast<uint8_t>(sign | 0x78u | m3);
        }
        return static_cast<uint8_t>(sign | (e4 << 3) | m3);
    }
    // Subnormal range: the quantum is 2^-9 and codes 0..7 cover it.
    int code = static_cast<int>(std::nearbyintf(std::fabs(x) * 512.0f));
    if (code >= 8) return static_cast<uint8_t>(sign | 0x08u);
    if (code < 0) code = 0;
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(code));
}

}  // namespace shelf::quantization::codec
