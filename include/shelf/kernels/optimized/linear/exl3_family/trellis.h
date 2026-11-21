#pragma once

// Shared trellis physical layout: the circular MSB-first bit stream and the 16x16 tile

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

}  // namespace shelf::kernel::exl3_detail
