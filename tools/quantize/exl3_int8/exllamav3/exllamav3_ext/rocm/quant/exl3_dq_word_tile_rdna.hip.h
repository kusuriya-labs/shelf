#pragma once
#include "exl3_dq_rdna.hip.h"

// A compressed 16x16 tile has 8*bits dwords. Distribute those words across
// the wave and keep the existing state extraction and FragB layout unchanged.
// All 32 lanes must execute every lookup, even when a bank is unused locally.
template <int bits>
struct Exl3WaveWords
{
    uint32_t lo, hi;

    __device__ __forceinline__ Exl3WaveWords(const uint32_t* ptr, int lane)
    {
        lo = lane < 8 * bits ? ptr[lane] : 0;
        if constexpr (bits > 4) hi = lane + 32 < 8 * bits ? ptr[lane + 32] : 0;
        else hi = 0;
    }

    __device__ __forceinline__ uint32_t operator[](int word) const
    {
        uint32_t a = __shfl(lo, word & 31, 32);
        if constexpr (bits <= 4) return a;
        const uint32_t b = __shfl(hi, word & 31, 32);
        return word < 32 ? a : b;
    }
};

template <int bits, int cb>
__device__ __forceinline__ void dq_dispatch_word_tile(
    const uint32_t* ptr, int lane, FragB& frag0, FragB& frag1)
{
    const Exl3WaveWords<bits> words(ptr, lane);
    dq_dispatch<bits, cb>(words, lane << 3, frag0, frag1);
}
