#pragma once

#include <shelf/quantization/quant_format.h>

#include <cstdint>

namespace shelf::quantization::mxfp4 {

// OCP MX FP4: E2M1 codes packed two per byte along K, one E8M0 scale per
// 32-element block of an output row.
inline constexpr uint32_t kGroupElements = kMxfp4ScaleBlockK;
inline constexpr uint32_t kWeightsPerCode = kMxfp4WeightsPerCode;
inline constexpr float kE2m1Max = 6.0f;
inline constexpr uint8_t kE8m0NanByte = 255u;

float e2m1_decode_f32(uint8_t code);
uint8_t e2m1_encode_u8(float value);

// E8M0 is the biased exponent of a power-of-two scale: value = 2^(byte - 127).
// Byte 255 is the MX spec's NaN and byte 0 is 2^-127.
float e8m0_decode_f32(uint8_t byte);
uint8_t e8m0_scale_byte(float amax);

// Quantize a matrix `n` x `k` (row-major, `src_stride` floats per row) into
// codes[n][k_padded / 2] (two E2M1 codes per byte, low nibble = even column)
// and scales[n][k_padded / 32] (one E8M0 byte per group).
void quantize_mxfp4_matrix(
    const float* src,
    uint64_t src_stride,
    uint64_t n,
    uint64_t k,
    uint64_t k_padded,
    uint8_t* codes,
    uint8_t* scales);

void dequantize_mxfp4_matrix(
    const uint8_t* codes,
    const uint8_t* scales,
    uint64_t n,
    uint64_t k_padded,
    float* dst);

// Canonical on-disk payload for one or more stacked matrices:
// codes[..., N, Kp/2] U8 and scales[..., N, Kp/32] U8.
struct Mxfp4CanonicalView {
    uint64_t batch = 1;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    const uint8_t* codes = nullptr;
    uint64_t codes_bytes = 0;
    const uint8_t* scales = nullptr;
    uint64_t scales_bytes = 0;

    bool validate() const;
};

}  // namespace shelf::quantization::mxfp4
