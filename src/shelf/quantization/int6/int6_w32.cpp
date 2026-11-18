#include <shelf/quantization/int6/int6_w32.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace shelf {
namespace quantization {
namespace int6 {

namespace {

inline float bf16_bits_to_f32(std::uint16_t bits) noexcept {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16u;
    float value = 0.0f;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

inline shelf::fp16_t load_scale(const std::uint8_t* bytes) noexcept {
    shelf::fp16_t scale{};
    std::memcpy(&scale, bytes, sizeof(scale));
    return scale;
}

inline void store_scale(std::uint8_t* bytes, shelf::fp16_t scale) noexcept {
    std::memcpy(bytes, &scale, sizeof(scale));
}

}  // namespace

void quantize_bf16_row(const bf16_t* src, std::uint32_t groups,
                       std::uint8_t* dst) noexcept {
    for (std::uint32_t group = 0; group < groups; ++group) {
        const bf16_t* source = src + static_cast<std::size_t>(group) * kGroupSize;
        float values[kGroupSize];
        float positive = 0.0f;
        float negative = 0.0f;
        for (std::uint32_t i = 0; i < kGroupSize; ++i) {
            values[i] = bf16_bits_to_f32(source[i].data);
            if (values[i] > positive) positive = values[i];
            if (values[i] < negative) negative = values[i];
        }

        const float raw = std::max(positive / kMaxPositive,
                                   -negative / kMaxNegativeMagnitude);
        const shelf::fp16_t stored =
            raw > 0.0f ? shelf::f32_to_fp16(std::max(raw, kMinScale))
                       : shelf::f32_to_fp16(1.0f);
        const float inv = 1.0f / shelf::fp16_to_f32(stored);

        std::uint8_t* codes = dst + static_cast<std::size_t>(group) * kGroupBytes;
        std::uint64_t bits = 0;
        std::uint32_t available = 0;
        std::uint32_t out_bytes = 0;
        for (std::uint32_t i = 0; i < kGroupSize; ++i) {
            const float scaled = std::nearbyintf(values[i] * inv);
            const float clamped =
                std::min(std::max(scaled, -kMaxNegativeMagnitude), kMaxPositive);
            const std::uint32_t code = std::isfinite(clamped)
                ? static_cast<std::uint32_t>(static_cast<std::int32_t>(clamped)) & 0x3Fu
                : 0u;
            bits |= static_cast<std::uint64_t>(code) << available;
            available += 6u;
            while (available >= 8u) {
                codes[out_bytes++] = static_cast<std::uint8_t>(bits & 0xFFu);
                bits >>= 8u;
                available -= 8u;
            }
        }
        store_scale(codes + kGroupCodeBytes, stored);
    }
}

void dequantize_row(const std::uint8_t* src, std::uint32_t groups,
                    float* dst) noexcept {
    for (std::uint32_t group = 0; group < groups; ++group) {
        const std::uint8_t* codes = src + static_cast<std::size_t>(group) * kGroupBytes;
        const float scale = shelf::fp16_to_f32(load_scale(codes + kGroupCodeBytes));

        std::uint64_t bits = 0;
        std::uint32_t available = 0;
        std::uint32_t in_bytes = 0;
        float* out = dst + static_cast<std::size_t>(group) * kGroupSize;
        for (std::uint32_t i = 0; i < kGroupSize; ++i) {
            while (available < 6u) {
                bits |= static_cast<std::uint64_t>(codes[in_bytes++]) << available;
                available += 8u;
            }
            const std::uint32_t code = static_cast<std::uint32_t>(bits & 0x3Fu);
            bits >>= 6u;
            available -= 6u;
            const std::int32_t value =
                (code & 0x20u) ? static_cast<std::int32_t>(code) - 64
                               : static_cast<std::int32_t>(code);
            out[i] = static_cast<float>(value) * scale;
        }
    }
}

}  // namespace int6
}  // namespace quantization
}  // namespace shelf
