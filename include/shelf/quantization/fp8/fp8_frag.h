#pragma once

#include <cstddef>
#include <cstdint>

namespace shelf::quantization::fp8 {

// gfx1201 FP8 WMMA fragment order for E4M3 codes (weight and activation).
// One block is 16 rows x 64 K and is stored as 32 lane slots of 32 bytes, so a
// wave reads the block as one 1 KiB contiguous request and every lane holds the
// eight code bytes of each of the four 16-K fragments it consumes.
inline constexpr uint32_t kFragBlockK = 64u;
inline constexpr uint32_t kFragBlockBytes = 1024u;
inline constexpr uint32_t kFragLaneBytes = 32u;

constexpr uint32_t fp8_frag_blocks(uint32_t k_padded) noexcept {
    return k_padded / kFragBlockK;
}

constexpr std::size_t fp8_frag_tile_bytes(uint32_t k_padded) noexcept {
    return static_cast<std::size_t>(k_padded) * 16u;
}

constexpr std::size_t fp8_frag_codes_bytes(uint32_t tiles, uint32_t k_padded) noexcept {
    return static_cast<std::size_t>(tiles) * fp8_frag_tile_bytes(k_padded);
}

// Byte offset inside one block of the code at (row, k), both block-local.
constexpr uint32_t fp8_frag_offset(uint32_t row, uint32_t k) noexcept {
    const uint32_t lane = row + ((k & 15u) >> 3u) * 16u;
    return lane * kFragLaneBytes + (k >> 4u) * 8u + (k & 7u);
}

}  // namespace shelf::quantization::fp8
