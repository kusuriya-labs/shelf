#pragma once

#include <cstdint>

namespace shelf::quantization::exl3 {

enum class Codebook : uint8_t {
    ThreeInst = 0,
    Mcg = 1,
    Mul1 = 2,
    Unknown = 255,
};

constexpr uint32_t kTileSide = 16;
constexpr uint32_t kTileWeights = kTileSide * kTileSide;
constexpr uint32_t kHadamardBlock = 128;

constexpr uint32_t words_per_tile(uint32_t ka, bool half) {
    return 16u * ka + (half ? 8u : 0u);
}

constexpr uint32_t bits_per_tile(uint32_t ka, bool half) {
    return 256u * ka + (half ? 128u : 0u);
}

constexpr bool layout_valid(uint32_t ka, bool half, Codebook codebook) {
    if (ka < 1u || ka > 8u) return false;
    if (half && (ka < 1u || ka > 3u || codebook != Codebook::Mul1)) return false;
    return codebook != Codebook::Unknown;
}

uint16_t f16_bits_from_f32(float value);
float f32_from_f16_bits(uint16_t bits);
uint16_t f16_bits_from_f64(double value);

uint16_t codebook_f16_bits(uint16_t state, Codebook codebook);

uint16_t state_from_ring(const uint32_t* words, uint32_t word_count, uint32_t position,
                         uint32_t ka, bool half);

void tile_coords(uint32_t position, uint32_t* row, uint32_t* col);

bool decode_tile(const uint16_t* trellis_words, uint32_t ka, bool half, Codebook codebook,
                 float* out_row_major);

bool decode_rotated(const uint16_t* trellis, uint32_t k, uint32_t n, uint32_t ka, bool half,
                    Codebook codebook, float* out_row_major);

void hadamard128_inplace(float* values);

bool dequant(const uint16_t* trellis, const uint16_t* suh_f16, const uint16_t* svh_f16,
             uint32_t k, uint32_t n, uint32_t ka, bool half, Codebook codebook,
             float* out_row_major);

uint64_t trellis_bytes(uint32_t k, uint32_t n, uint32_t ka, bool half);

}
