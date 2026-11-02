#pragma once
#include <cstdint>
#include <cstring>

namespace ps {

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

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7F800000u) == 0x7F800000u) {
        if ((u & 0x007FFFFFu) != 0u) return 0x7FC0u;
        return static_cast<uint16_t>(u >> 16);
    }
    uint32_t rounding_bias = 0x7FFF + ((u >> 16) & 1u);
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

} // namespace ps
