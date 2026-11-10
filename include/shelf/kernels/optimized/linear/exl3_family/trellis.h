#pragma once

// Shared trellis physical layout: the circular MSB-first bit stream and the 16x16 tile
// coordinate mapping. The mul1 and mul24 decode loops stay separate; only the layout
// helpers live here.

#include <cstdint>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace shelf::kernel::exl3_detail {

// 16-bit window of the tile's circular bit stream at `position`.
__device__ __forceinline__ uint16_t state_from_ring(const uint32_t* ring, uint32_t word_count,
                                                    uint32_t position, uint32_t ka,
                                                    uint32_t half) {
    const uint32_t bit_length = 256u * ka + (half ? 128u : 0u);
    const uint32_t step = position + 1u;
    const uint32_t end = step * ka + (half ? step / 2u : 0u) + bit_length;
    const uint32_t lo = (end - 16u) >> 5;
    const uint32_t hi = (end - 1u) >> 5;
    const uint32_t shift = ((hi + 1u) << 5) - end;
    const uint64_t pair =
        (static_cast<uint64_t>(ring[lo % word_count]) << 32) | ring[hi % word_count];
    return static_cast<uint16_t>((pair >> shift) & 0xFFFFu);
}

// Stream position -> (row, col) inside one 16x16 tile.
__device__ __forceinline__ void tile_coords(uint32_t position, uint32_t* row, uint32_t* col) {
    const uint32_t lane = position / 8u;
    const uint32_t j = position % 8u;
    const uint32_t q = lane & 3u;
    *row = 2u * q + (j & 1u) + ((j & 2u) ? 8u : 0u);
    *col = (lane >> 2) + ((j & 4u) ? 8u : 0u);
}

// One mul1 codebook value as fp16 bits.
__device__ __forceinline__ uint16_t codebook_f16_bits(uint32_t state) {
    const uint32_t x = state * 0x83DCD12Du;
    const int sum = static_cast<int>(x & 0xFFu) + static_cast<int>((x >> 8) & 0xFFu) +
                    static_cast<int>((x >> 16) & 0xFFu) + static_cast<int>(x >> 24);
    const __half h = __ushort_as_half(static_cast<unsigned short>(0x6400u + sum));
    const __half k_inv = __ushort_as_half(0x1EEEu);
    const __half k_bias = __ushort_as_half(0xC931u);
    return __half_as_ushort(__hfma(h, k_inv, k_bias));
}

// Two adjacent mul1 codebook values packed into one 32-bit tile word.
__device__ __forceinline__ uint32_t codebook_pair_h_bits(uint32_t state0, uint32_t state1) {
    const uint32_t p0 = state0 * 0x83DCD12Du;
    const uint32_t p1 = state1 * 0x83DCD12Du;
    const uint32_t sum0 = __builtin_amdgcn_sad_u8(p0, 0u, 0x64006400u);
    return __builtin_amdgcn_sad_hi_u8(p1, 0u, sum0);
}

}  // namespace shelf::kernel::exl3_detail
