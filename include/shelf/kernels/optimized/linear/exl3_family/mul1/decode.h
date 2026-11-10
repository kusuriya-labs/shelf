#pragma once

// mul1 (W16A16) decode constants. The shared trellis layout and FWHT variants live in
// trellis.h / fwht.h; the mul1 decode loops stay in exl3.hip.

#include <shelf/kernels/optimized/linear/exl3_family/fwht.h>
#include <shelf/kernels/optimized/linear/exl3_family/types.h>
#include <shelf/kernels/optimized/linear/exl3_family/trellis.h>

namespace shelf::kernel::exl3_detail::mul1 {

constexpr float kMul1Inv = 0.00676727294921875f;
constexpr float kMul1Bias = -10.3828125f;
constexpr float kMul1BiasOverInv = kMul1Bias / kMul1Inv;

__device__ __forceinline__ float bf16_value(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<uint32_t>(p->data) << 16u);
}

__device__ __forceinline__ uint32_t pick4(uint32_t r, uint32_t a, uint32_t b, uint32_t c,
                                          uint32_t d) {
    uint32_t v = a;
    v = r == 1u ? b : v;
    v = r == 2u ? c : v;
    v = r == 3u ? d : v;
    return v;
}

__device__ __forceinline__ void decode_tile_ka4(const Exl3GemmArgs& args, uint32_t kt,
                                               uint32_t nt, uint32_t lane, uint32_t* ring,
                                               uint16_t* tile) {
    (void) ring;
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * 64u;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);
    const uint32_t b = source[lane];
    const uint32_t a = __shfl_sync(0xFFFFFFFFull, b, (lane + 31u) & 31u, 32);

    uint32_t st[8];
    st[0] = __builtin_amdgcn_alignbit(a, b, 28) & 0xFFFFu;
    st[1] = __builtin_amdgcn_alignbit(a, b, 24) & 0xFFFFu;
    st[2] = __builtin_amdgcn_alignbit(a, b, 20) & 0xFFFFu;
    st[3] = __builtin_amdgcn_alignbit(b, b, 16) & 0xFFFFu;
    st[4] = __builtin_amdgcn_alignbit(b, b, 12) & 0xFFFFu;
    st[5] = __builtin_amdgcn_alignbit(b, b, 8) & 0xFFFFu;
    st[6] = __builtin_amdgcn_alignbit(b, b, 4) & 0xFFFFu;
    st[7] = b & 0xFFFFu;

    uint32_t* tile32 = reinterpret_cast<uint32_t*>(tile);
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p) {
        const uint32_t position = lane * 8u + p * 2u;
        const uint32_t packed =
            exl3_detail::codebook_pair_h_bits(st[2u * p], st[2u * p + 1u]);
        uint32_t row = 0;
        uint32_t col = 0;
        exl3_detail::tile_coords(position, &row, &col);
        tile32[col * 8u + (row >> 1)] = packed;
    }
    __syncwarp();
}

__device__ __forceinline__ void decode_tile_ka5(const Exl3GemmArgs& args, uint32_t kt, uint32_t nt,
                                                uint32_t lane, uint32_t* ring, uint16_t* tile) {
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * 80u;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);
    for (uint32_t i = lane; i < 40u; i += 32u) ring[i] = source[i];
    __syncwarp();

    const uint32_t off0 = 40u * lane + 1269u;
    uint32_t w0 = off0 >> 5u;
    if (w0 >= 40u) w0 -= 40u;
    uint32_t i1 = w0 + 1u;
    uint32_t i2 = w0 + 2u;
    uint32_t i3 = w0 + 3u;
    if (i1 >= 40u) i1 -= 40u;
    if (i2 >= 40u) i2 -= 40u;
    if (i3 >= 40u) i3 -= 40u;
    const uint32_t v0 = ring[w0];
    const uint32_t v1 = ring[i1];
    const uint32_t v2 = ring[i2];
    const uint32_t v3 = ring[i3];
    const uint32_t r = off0 & 31u;

    uint32_t st[8];
#pragma unroll
    for (uint32_t j = 0u; j < 8u; ++j) {
        const uint32_t off = r + 5u * j;
        const uint32_t lo = off >> 5u;
        const uint32_t hi = (off + 15u) >> 5u;
        const uint32_t shift = ((hi + 1u) << 5u) - (off + 16u);
        st[j] = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo, v0, v1, v2, v3)) << 32) |
             pick4(hi, v0, v1, v2, v3)) >>
            shift);
    }

    uint32_t* tile32 = reinterpret_cast<uint32_t*>(tile);
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p) {
        const uint32_t position = lane * 8u + p * 2u;
        const uint32_t packed =
            exl3_detail::codebook_pair_h_bits(st[2u * p], st[2u * p + 1u]);
        uint32_t row = 0;
        uint32_t col = 0;
        exl3_detail::tile_coords(position, &row, &col);
        tile32[col * 8u + (row >> 1)] = packed;
    }
    __syncwarp();
}

template <uint32_t KA, bool HALF>
__device__ __forceinline__ void decode_tile_fixed(const Exl3GemmArgs& args, uint32_t kt,
                                                  uint32_t nt, uint32_t lane, uint32_t* ring,
                                                  uint16_t* tile) {
    constexpr uint32_t kWords = 16u * KA + (HALF ? 8u : 0u);
    constexpr uint32_t kWordCount = kWords / 2u;
    constexpr uint32_t kBitLength = 256u * KA + (HALF ? 128u : 0u);
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * kWords;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);
    for (uint32_t i = lane; i < kWordCount; i += 32u) ring[i] = source[i];
    __syncwarp();

    const uint32_t p0 = lane * 8u;
    const uint32_t step0 = p0 + 1u;
    const uint32_t end0 = step0 * KA + (HALF ? step0 / 2u : 0u) + kBitLength;
    const uint32_t w0 = (end0 - 16u) >> 5u;
    const uint32_t v0 = ring[(w0 + 0u) % kWordCount];
    const uint32_t v1 = ring[(w0 + 1u) % kWordCount];
    const uint32_t v2 = ring[(w0 + 2u) % kWordCount];
    const uint32_t v3 = ring[(w0 + 3u) % kWordCount];

    uint32_t* tile32 = reinterpret_cast<uint32_t*>(tile);
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p) {
        const uint32_t position = p0 + p * 2u;
        const uint32_t step = position + 1u;
        const uint32_t end = step * KA + (HALF ? step / 2u : 0u) + kBitLength;
        const uint32_t shift = (((end - 1u) >> 5) + 1u) * 32u - end;
        const uint32_t lo = ((end - 16u) >> 5) - w0;
        const uint32_t hi = ((end - 1u) >> 5) - w0;
        const uint32_t state0 = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo, v0, v1, v2, v3)) << 32) |
             pick4(hi, v0, v1, v2, v3)) >>
            shift);
        const uint32_t step1 = position + 2u;
        const uint32_t end1 = step1 * KA + (HALF ? step1 / 2u : 0u) + kBitLength;
        const uint32_t shift1 = (((end1 - 1u) >> 5) + 1u) * 32u - end1;
        const uint32_t lo1 = ((end1 - 16u) >> 5) - w0;
        const uint32_t hi1 = ((end1 - 1u) >> 5) - w0;
        const uint32_t state1 = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo1, v0, v1, v2, v3)) << 32) |
             pick4(hi1, v0, v1, v2, v3)) >>
            shift1);
        const uint32_t packed = exl3_detail::codebook_pair_h_bits(state0, state1);
        uint32_t row = 0;
        uint32_t col = 0;
        exl3_detail::tile_coords(position, &row, &col);
        tile32[col * 8u + (row >> 1)] = packed;
    }
    __syncwarp();
}

template <uint32_t kDecode>
__device__ __forceinline__ void decode_tile(const Exl3GemmArgs& args, uint32_t kt, uint32_t nt,
                                            uint32_t lane, uint32_t* ring, uint16_t* tile) {
    if constexpr (kDecode == 1u) {
        decode_tile_ka5(args, kt, nt, lane, ring, tile);
        return;
    }
    if constexpr (kDecode == 2u) {
        decode_tile_fixed<3u, false>(args, kt, nt, lane, ring, tile);
        return;
    }
    if constexpr (kDecode == 3u) {
        decode_tile_fixed<3u, true>(args, kt, nt, lane, ring, tile);
        return;
    }
    if (args.ka == 4u && args.half == 0u) {
        decode_tile_ka4(args, kt, nt, lane, ring, tile);
        return;
    }
    const uint32_t words = 16u * args.ka + (args.half != 0u ? 8u : 0u);
    const uint32_t word_count = words / 2u;
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * words;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);

    for (uint32_t i = lane; i < word_count; i += 32u) ring[i] = source[i];
    __syncwarp();

    const uint32_t bit_length = 256u * args.ka + (args.half != 0u ? 128u : 0u);
    const uint32_t p0 = lane * 8u;
    const uint32_t step0 = p0 + 1u;
    const uint32_t end0 =
        step0 * args.ka + (args.half != 0u ? step0 / 2u : 0u) + bit_length;
    const uint32_t w0 = (end0 - 16u) >> 5;
    uint32_t i0 = w0;
    uint32_t i1 = w0 + 1u;
    uint32_t i2 = w0 + 2u;
    uint32_t i3 = w0 + 3u;
    if (i0 >= word_count) i0 -= word_count;
    if (i1 >= word_count) i1 -= word_count;
    if (i2 >= word_count) i2 -= word_count;
    if (i3 >= word_count) i3 -= word_count;
    if (i0 >= word_count) i0 -= word_count;
    if (i1 >= word_count) i1 -= word_count;
    if (i2 >= word_count) i2 -= word_count;
    if (i3 >= word_count) i3 -= word_count;
    const uint32_t v0 = ring[i0];
    const uint32_t v1 = ring[i1];
    const uint32_t v2 = ring[i2];
    const uint32_t v3 = ring[i3];

    uint32_t* tile32 = reinterpret_cast<uint32_t*>(tile);
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p) {
        const uint32_t position = p0 + p * 2u;
        const uint32_t step = position + 1u;
        const uint32_t end =
            step * args.ka + (args.half != 0u ? step / 2u : 0u) + bit_length;
        const uint32_t shift = (((end - 1u) >> 5) + 1u) * 32u - end;
        const uint32_t lo = ((end - 16u) >> 5) - w0;
        const uint32_t hi = ((end - 1u) >> 5) - w0;
        const uint32_t state0 = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo, v0, v1, v2, v3)) << 32) |
             pick4(hi, v0, v1, v2, v3)) >>
            shift);
        const uint32_t step1 = position + 2u;
        const uint32_t end1 =
            step1 * args.ka + (args.half != 0u ? step1 / 2u : 0u) + bit_length;
        const uint32_t shift1 = (((end1 - 1u) >> 5) + 1u) * 32u - end1;
        const uint32_t lo1 = ((end1 - 16u) >> 5) - w0;
        const uint32_t hi1 = ((end1 - 1u) >> 5) - w0;
        const uint32_t state1 = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo1, v0, v1, v2, v3)) << 32) |
             pick4(hi1, v0, v1, v2, v3)) >>
            shift1);
        const uint32_t packed = exl3_detail::codebook_pair_h_bits(state0, state1);
        uint32_t row = 0;
        uint32_t col = 0;
        exl3_detail::tile_coords(position, &row, &col);
        tile32[col * 8u + (row >> 1)] = packed;
    }
    __syncwarp();
}



}  // namespace shelf::kernel::exl3_detail::mul1
