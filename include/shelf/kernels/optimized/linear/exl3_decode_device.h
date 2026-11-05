#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel::exl3_detail {

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

__device__ __forceinline__ void tile_coords(uint32_t position, uint32_t* row, uint32_t* col) {
    const uint32_t lane = position / 8u;
    const uint32_t j = position % 8u;
    const uint32_t q = lane & 3u;
    *row = 2u * q + (j & 1u) + ((j & 2u) ? 8u : 0u);
    *col = (lane >> 2) + ((j & 4u) ? 8u : 0u);
}

__device__ __forceinline__ uint16_t codebook_f16_bits(uint32_t state) {
    const uint32_t x = state * 0x83DCD12Du;
    const int sum = static_cast<int>(x & 0xFFu) + static_cast<int>((x >> 8) & 0xFFu) +
                    static_cast<int>((x >> 16) & 0xFFu) + static_cast<int>(x >> 24);
    const __half h = __ushort_as_half(static_cast<unsigned short>(0x6400u + sum));
    const __half k_inv = __ushort_as_half(0x1EEEu);
    const __half k_bias = __ushort_as_half(0xC931u);
    return __half_as_ushort(__hfma(h, k_inv, k_bias));
}

__device__ __forceinline__ void fwht128(float& v0, float& v1, float& v2, float& v3,
                                        uint32_t lane) {
    const float a = v0 + v1;
    const float b = v0 - v1;
    const float c = v2 + v3;
    const float d = v2 - v3;
    v0 = a + c;
    v1 = b + d;
    v2 = a - c;
    v3 = b - d;
#pragma unroll
    for (uint32_t step = 1u; step < 32u; step <<= 1u) {
        const float w0 = __shfl_xor(v0, step, 32);
        const float w1 = __shfl_xor(v1, step, 32);
        const float w2 = __shfl_xor(v2, step, 32);
        const float w3 = __shfl_xor(v3, step, 32);
        if ((lane & step) != 0u) {
            v0 = w0 - v0;
            v1 = w1 - v1;
            v2 = w2 - v2;
            v3 = w3 - v3;
        } else {
            v0 += w0;
            v1 += w1;
            v2 += w2;
            v3 += w3;
        }
    }
}

}
