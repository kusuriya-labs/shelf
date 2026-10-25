// shelf foundation — scalar numeric representations (bf16 / fp16).
//
// bf16_to_f32 added by shelf (the rest existed in our earlier runtime).
#pragma once

#include <cstdint>
#include <cstring>

namespace shelf {

struct alignas(2) bf16_t {
    uint16_t data;
};

struct alignas(2) fp16_t {
    _Float16 value;
};

inline float fp16_to_f32(fp16_t x) {
    return static_cast<float>(x.value);
}

inline fp16_t f32_to_fp16(float x) {
    return fp16_t{static_cast<_Float16>(x)};
}

// round-to-nearest-even f32 -> bf16 bits
inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u) {
        if ((u & 0x007FFFFFu) != 0u) return 0x7FC0u;  // quiet NaN
        return static_cast<uint16_t>(u >> 16);
    }
    uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1u);
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

inline bf16_t f32_to_bf16(float f) {
    return bf16_t{f32_to_bf16_rne(f)};
}

inline float bf16_to_f32(bf16_t b) {
    uint32_t u = static_cast<uint32_t>(b.data) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

} // namespace shelf
