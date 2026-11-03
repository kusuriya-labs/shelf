#include <shelf/quantization/mxfp4/mxfp4.h>

#include <cmath>
#include <cstring>

namespace shelf::quantization::mxfp4 {

namespace {

constexpr float kE2m1Magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constexpr float kTwoPowNeg127 = 5.877471754111438e-39f;

uint32_t f32_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_f32(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

float e2m1_decode_f32(uint8_t code) {
    const float magnitude = kE2m1Magnitude[code & 0x7u];
    return (code & 0x8u) != 0u ? -magnitude : magnitude;
}

uint8_t e2m1_encode_u8(float value) {
    const uint32_t bits = f32_bits(value);
    const uint32_t sign = bits & 0x80000000u;
    uint32_t exponent = (bits >> 23) & 0xFFu;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (!std::isfinite(value)) {
        return static_cast<uint8_t>((sign >> 28) | 0x7u);
    }
    if (exponent < 127u) {
        const uint32_t shift = 126u - exponent;
        mantissa = shift >= 32u ? 0u : ((0x400000u | (mantissa >> 1)) >> shift);
    }
    exponent = (exponent < 126u ? 126u : exponent) - 126u;
    uint32_t code = (((exponent << 2) | (mantissa >> 21)) + 1u) >> 1;
    if (code > 7u) code = 7u;
    return static_cast<uint8_t>((sign >> 28) | code);
}

float e8m0_decode_f32(uint8_t byte) {
    if (byte == kE8m0NanByte) return bits_f32(0x7FC00000u);
    if (byte == 0u) return kTwoPowNeg127;
    return bits_f32(static_cast<uint32_t>(byte) << 23);
}

uint8_t e8m0_scale_byte(float amax) {
    if (std::isnan(amax)) return kE8m0NanByte;
    if (amax == 0.0f) return 1u;
    const uint32_t bits = f32_bits(std::fabs(amax));
    const uint32_t biased = ((bits + (1u << 21)) >> 23) & 0xFFu;
    if (biased >= 255u) return 254u;
    int32_t exponent = static_cast<int32_t>(biased) - 127 - 2;
    if (exponent < -127) exponent = -127;
    if (exponent > 127) exponent = 127;
    return static_cast<uint8_t>(exponent + 127);
}

void quantize_mxfp4_matrix(
    const float* src,
    uint64_t src_stride,
    uint64_t n,
    uint64_t k,
    uint64_t k_padded,
    uint8_t* codes,
    uint8_t* scales)
{
    if (codes != nullptr) std::memset(codes, 0, n * (k_padded / kWeightsPerCode));
    if (scales != nullptr) std::memset(scales, 0, n * (k_padded / kGroupElements));
    if (scales == nullptr) return;
    const uint64_t code_stride = k_padded / kWeightsPerCode;
    const uint64_t scale_stride = k_padded / kGroupElements;
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t group = 0; group < scale_stride; ++group) {
            float amax = 0.0f;
            const uint64_t begin = group * kGroupElements;
            const uint64_t end = begin + kGroupElements < k ? begin + kGroupElements : k;
            for (uint64_t col = begin; col < end; ++col) {
                const float value = std::fabs(src[row * src_stride + col]);
                if (value > amax) amax = value;
            }
            const uint8_t byte = e8m0_scale_byte(amax);
            scales[row * scale_stride + group] = byte;
            const float scale = e8m0_decode_f32(byte);
            const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            if (codes == nullptr) continue;
            for (uint64_t col = begin; col < begin + kGroupElements; ++col) {
                const float value = col < k ? src[row * src_stride + col] : 0.0f;
                const uint8_t code = e2m1_encode_u8(value * inv);
                uint8_t& byte_ref = codes[row * code_stride + col / kWeightsPerCode];
                if ((col & 1u) == 0u) {
                    byte_ref = static_cast<uint8_t>((byte_ref & 0xF0u) | code);
                } else {
                    byte_ref = static_cast<uint8_t>((byte_ref & 0x0Fu) | (code << 4));
                }
            }
        }
    }
}

void dequantize_mxfp4_matrix(
    const uint8_t* codes,
    const uint8_t* scales,
    uint64_t n,
    uint64_t k_padded,
    float* dst)
{
    const uint64_t code_stride = k_padded / kWeightsPerCode;
    const uint64_t scale_stride = k_padded / kGroupElements;
    for (uint64_t row = 0; row < n; ++row) {
        for (uint64_t group = 0; group < scale_stride; ++group) {
            const float scale = e8m0_decode_f32(scales[row * scale_stride + group]);
            for (uint64_t col = group * kGroupElements; col < (group + 1) * kGroupElements; ++col) {
                const uint8_t byte = codes[row * code_stride + col / kWeightsPerCode];
                const uint8_t code = (col & 1u) == 0u ? (byte & 0x0Fu) : (byte >> 4);
                dst[row * k_padded + col] = e2m1_decode_f32(code) * scale;
            }
        }
    }
}

bool Mxfp4CanonicalView::validate() const {
    if (batch == 0 || n == 0 || padded_k == 0) return false;
    if ((padded_k % kGroupElements) != 0) return false;
    if (codes == nullptr || scales == nullptr) return false;
    const uint64_t rows = batch * n;
    if (codes_bytes != rows * (padded_k / kWeightsPerCode)) return false;
    if (scales_bytes != rows * (padded_k / kGroupElements)) return false;
    return true;
}

}  // namespace shelf::quantization::mxfp4
