#include <shelf/quantization/exl3/exl3.h>

#include <cmath>
#include <cstring>

namespace shelf::quantization::exl3 {
float f32_from_f16_bits(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1Fu;
    uint32_t mantissa = bits & 0x3FFu;
    uint32_t out = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3FFu;
            const uint32_t exp32 = static_cast<uint32_t>(127 - 15 - shift);
            out = sign | (exp32 << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1Fu) {
        out = sign | 0x7F800000u | (mantissa << 13);
    } else {
        const uint32_t exp32 = exponent + (127u - 15u);
        out = sign | (exp32 << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

uint16_t f16_bits_from_f64(double value) {
    const uint16_t sign = std::signbit(value) ? 0x8000u : 0u;
    double magnitude = std::fabs(value);
    if (std::isnan(value)) return static_cast<uint16_t>(sign | 0x7E00u);
    if (magnitude == 0.0) return sign;

    int exponent = 0;
    std::frexp(magnitude, &exponent);
    exponent -= 1;
    if (exponent > 15) return static_cast<uint16_t>(sign | 0x7C00u);

    if (exponent < -14) {
        const double scaled = std::nearbyint(magnitude * 16777216.0);
        if (scaled >= 1024.0) return static_cast<uint16_t>(sign | 0x0400u);
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(scaled));
    }

    double scaled = std::nearbyint(magnitude * std::ldexp(1.0, 10 - exponent));
    if (scaled >= 2048.0) {
        scaled = 1024.0;
        ++exponent;
    }
    if (exponent > 15) return static_cast<uint16_t>(sign | 0x7C00u);
    const uint16_t exponent_bits = static_cast<uint16_t>((exponent + 15) << 10);
    return static_cast<uint16_t>(sign | exponent_bits |
                                 static_cast<uint16_t>(scaled - 1024.0));
}

uint16_t f16_bits_from_f32(float value) {
    return f16_bits_from_f64(static_cast<double>(value));
}

uint64_t trellis_bytes(uint32_t k, uint32_t n, uint32_t ka, bool half) {
    if ((k % kTileSide) != 0u || (n % kTileSide) != 0u) return 0;
    const uint64_t tiles = static_cast<uint64_t>(k / kTileSide) * (n / kTileSide);
    return tiles * words_per_tile(ka, half) * sizeof(uint16_t);
}

}
