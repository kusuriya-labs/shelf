#include <shelf/quantization/exl3/exl3.h>

#include <cmath>
#include <cstring>

namespace shelf::quantization::exl3 {
namespace {

constexpr uint32_t kMul1Multiplier = 0x83DCD12Du;

}

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

uint16_t codebook_f16_bits(uint16_t state) {
    const uint32_t x = static_cast<uint32_t>(state) * kMul1Multiplier;
    const int sum = static_cast<int>(x & 0xFFu) + static_cast<int>((x >> 8) & 0xFFu) +
                    static_cast<int>((x >> 16) & 0xFFu) + static_cast<int>(x >> 24);
    const double h = 1024.0 + static_cast<double>(sum);
    const double k_inv = f32_from_f16_bits(0x1EEEu);
    const double k_bias = f32_from_f16_bits(0xC931u);
    return f16_bits_from_f64(h * k_inv + k_bias);
}

uint16_t state_from_ring(const uint32_t* words, uint32_t word_count, uint32_t position,
                         uint32_t ka, bool half) {
    const uint32_t bit_length = bits_per_tile(ka, half);
    const uint32_t step = position + 1u;
    const uint32_t end = step * ka + (half ? step / 2u : 0u) + bit_length;
    const uint32_t lo = (end - 16u) >> 5;
    const uint32_t hi = (end - 1u) >> 5;
    const uint32_t shift = ((hi + 1u) << 5) - end;
    const uint64_t pair =
        (static_cast<uint64_t>(words[lo % word_count]) << 32) | words[hi % word_count];
    return static_cast<uint16_t>((pair >> shift) & 0xFFFFu);
}

void tile_coords(uint32_t position, uint32_t* row, uint32_t* col) {
    const uint32_t lane = position / 8u;
    const uint32_t j = position % 8u;
    const uint32_t q = lane & 3u;
    *row = 2u * q + (j & 1u) + ((j & 2u) ? 8u : 0u);
    *col = (lane >> 2) + ((j & 4u) ? 8u : 0u);
}

bool decode_tile(const uint16_t* trellis_words, uint32_t ka, bool half,
                 float* out_row_major) {
    if (trellis_words == nullptr || out_row_major == nullptr) return false;
    if (!layout_valid(ka, half)) return false;
    const uint32_t word_count = words_per_tile(ka, half) / 2u;
    uint32_t ring[64];
    for (uint32_t i = 0; i < word_count; ++i) {
        ring[i] = static_cast<uint32_t>(trellis_words[2u * i]) |
                  (static_cast<uint32_t>(trellis_words[2u * i + 1u]) << 16);
    }
    for (uint32_t t = 0; t < kTileWeights; ++t) {
        uint32_t row = 0;
        uint32_t col = 0;
        tile_coords(t, &row, &col);
        const uint16_t state = state_from_ring(ring, word_count, t, ka, half);
        out_row_major[row * kTileSide + col] =
            f32_from_f16_bits(codebook_f16_bits(state));
    }
    return true;
}

bool decode_rotated(const uint16_t* trellis, uint32_t k, uint32_t n, uint32_t ka, bool half,
                    float* out_row_major) {
    if (trellis == nullptr || out_row_major == nullptr) return false;
    if (!layout_valid(ka, half)) return false;
    if ((k % kTileSide) != 0u || (n % kTileSide) != 0u) return false;
    const uint32_t kt = k / kTileSide;
    const uint32_t nt = n / kTileSide;
    const uint32_t words = words_per_tile(ka, half);
    float tile[kTileWeights];
    for (uint32_t ti = 0; ti < kt; ++ti) {
        for (uint32_t tj = 0; tj < nt; ++tj) {
            const uint16_t* source = trellis + (static_cast<uint64_t>(ti) * nt + tj) * words;
            if (!decode_tile(source, ka, half, tile)) return false;
            for (uint32_t r = 0; r < kTileSide; ++r) {
                float* dst = out_row_major + (static_cast<uint64_t>(ti) * kTileSide + r) * n +
                             static_cast<uint64_t>(tj) * kTileSide;
                std::memcpy(dst, tile + r * kTileSide, kTileSide * sizeof(float));
            }
        }
    }
    return true;
}

void hadamard128_inplace(float* values) {
    for (uint32_t step = 1u; step < kHadamardBlock; step <<= 1u) {
        for (uint32_t base = 0u; base < kHadamardBlock; base += step << 1u) {
            for (uint32_t j = 0u; j < step; ++j) {
                const float a = values[base + j];
                const float b = values[base + j + step];
                values[base + j] = a + b;
                values[base + j + step] = a - b;
            }
        }
    }
    constexpr float kScale = 0.08838834764831844055f;
    for (uint32_t i = 0u; i < kHadamardBlock; ++i) values[i] *= kScale;
}

bool dequant(const uint16_t* trellis, const uint16_t* suh_f16, const uint16_t* svh_f16,
             uint32_t k, uint32_t n, uint32_t ka, bool half, float* out_row_major) {
    if (suh_f16 == nullptr || svh_f16 == nullptr) return false;
    if ((k % kHadamardBlock) != 0u || (n % kHadamardBlock) != 0u) return false;
    if (!decode_rotated(trellis, k, n, ka, half, out_row_major)) return false;

    float block[kHadamardBlock];
    for (uint32_t kb = 0u; kb < k / kHadamardBlock; ++kb) {
        const uint32_t row0 = kb * kHadamardBlock;
        for (uint32_t col = 0u; col < n; ++col) {
            for (uint32_t r = 0u; r < kHadamardBlock; ++r) {
                block[r] = out_row_major[static_cast<uint64_t>(row0 + r) * n + col];
            }
            hadamard128_inplace(block);
            for (uint32_t r = 0u; r < kHadamardBlock; ++r) {
                out_row_major[static_cast<uint64_t>(row0 + r) * n + col] = block[r];
            }
        }
    }

    for (uint32_t row = 0u; row < k; ++row) {
        float* base = out_row_major + static_cast<uint64_t>(row) * n;
        for (uint32_t nb = 0u; nb < n / kHadamardBlock; ++nb) {
            float* segment = base + nb * kHadamardBlock;
            std::memcpy(block, segment, kHadamardBlock * sizeof(float));
            hadamard128_inplace(block);
            std::memcpy(segment, block, kHadamardBlock * sizeof(float));
        }
    }

    for (uint32_t row = 0u; row < k; ++row) {
        const float suh = f32_from_f16_bits(suh_f16[row]);
        float* base = out_row_major + static_cast<uint64_t>(row) * n;
        for (uint32_t col = 0u; col < n; ++col) {
            base[col] *= suh * f32_from_f16_bits(svh_f16[col]);
        }
    }
    return true;
}

uint64_t trellis_bytes(uint32_t k, uint32_t n, uint32_t ka, bool half) {
    if ((k % kTileSide) != 0u || (n % kTileSide) != 0u) return 0;
    const uint64_t tiles = static_cast<uint64_t>(k / kTileSide) * (n / kTileSide);
    return tiles * words_per_tile(ka, half) * sizeof(uint16_t);
}

}
