#pragma once

#include <shelf/core/memory/types.h>

#include <cstdint>

namespace shelf {
namespace quantization {
namespace int6 {

// Row-major signed INT6 codes with one FP16 scale per 32-element group. One group
// occupies 24 packed code bytes followed by the 2-byte FP16 scale. Codes are
// two's-complement values in [-32, 31]. The scale covers both sides without clipping:
// scale = max(max_positive / 31, -min_negative / 32), or 1 for an all-zero group.
inline constexpr std::uint32_t kGroupSize = 32u;
inline constexpr std::uint32_t kGroupCodeBytes = 24u;
inline constexpr std::uint32_t kScaleBytes = 2u;
inline constexpr std::uint32_t kGroupBytes = kGroupCodeBytes + kScaleBytes;
inline constexpr float kMaxPositive = 31.0f;
inline constexpr float kMaxNegativeMagnitude = 32.0f;
inline constexpr float kMinScale = 0x1p-24f;

// Quantizes one row of BF16 values into `groups` packed groups. `src` holds
// groups * kGroupSize values; `dst` receives groups * kGroupBytes bytes.
void quantize_bf16_row(const bf16_t* src, std::uint32_t groups,
                       std::uint8_t* dst) noexcept;

// Expands one packed row into groups * kGroupSize F32 values.
void dequantize_row(const std::uint8_t* src, std::uint32_t groups,
                    float* dst) noexcept;

}  // namespace int6
}  // namespace quantization
}  // namespace shelf
