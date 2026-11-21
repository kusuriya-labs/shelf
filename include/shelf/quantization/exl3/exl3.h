#pragma once

#include <cstdint>

namespace shelf::quantization::exl3 {

constexpr uint32_t kTileSide = 16;
constexpr uint32_t kTileWeights = kTileSide * kTileSide;
constexpr uint32_t kHadamardBlock = 128;

constexpr uint32_t words_per_tile(uint32_t ka, bool half) {
    return 16u * ka + (half ? 8u : 0u);
}

constexpr uint32_t bits_per_tile(uint32_t ka, bool half) {
    return 256u * ka + (half ? 128u : 0u);
}

constexpr bool layout_valid(uint32_t ka, bool half) {
    if (ka < 1u || ka > 8u) return false;
    return !half || ka <= 3u;
}

uint16_t f16_bits_from_f32(float value);
float f32_from_f16_bits(uint16_t bits);
uint16_t f16_bits_from_f64(double value);

uint64_t trellis_bytes(uint32_t k, uint32_t n, uint32_t ka, bool half);

}
