#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel::detail::octave {

__device__ __forceinline__ uint32_t octave_byte_mask(uint32_t bits,
                                                      uint32_t shift) {
    return (bits << (8u - shift)) - (bits >> shift);
}

__device__ __forceinline__ uint32_t octave_lut5x4_biased(uint32_t codes) {
    const uint32_t selector = codes & 0x07070707u;
    const uint32_t v0 = __builtin_amdgcn_perm(0xd1cac3bau, 0xb1a69781u, selector);
    const uint32_t v1 = __builtin_amdgcn_perm(0xfdf8f3eeu, 0xe8e3ddd7u, selector);
    const uint32_t v2 = __builtin_amdgcn_perm(0x29231d18u, 0x120d0803u, selector);
    const uint32_t v3 = __builtin_amdgcn_perm(0x7f695a4fu, 0x463d362fu, selector);
    const uint32_t m3 = octave_byte_mask(codes & 0x08080808u, 3u);
    const uint32_t m4 = octave_byte_mask(codes & 0x10101010u, 4u);
    const uint32_t lo = (v1 & m3) | (v0 & ~m3);
    const uint32_t hi = (v3 & m3) | (v2 & ~m3);
    return ((hi & m4) | (lo & ~m4)) ^ 0x80808080u;
}

template <uint32_t kBits>
__device__ __forceinline__ int32_t codebook_int8(uint32_t code) {
    uint32_t value = 0u;
    const uint32_t bank = code >> 3u;
    const uint32_t selector = code & 7u;
    if constexpr (kBits == 3u) {
        value = __builtin_amdgcn_perm(0x7f4f2d0eu, 0xf2d3b181u, selector);
    } else if constexpr (kBits == 4u) {
        const uint32_t v0 = __builtin_amdgcn_perm(0xfaeee1d4u, 0xc6b5a081u, selector);
        const uint32_t v1 = __builtin_amdgcn_perm(0x7f604b3au, 0x2c1f1206u, selector);
        value = bank == 0u ? v0 : v1;
    } else if constexpr (kBits == 5u) {
        const uint32_t v0 = __builtin_amdgcn_perm(0xd1cac3bau, 0xb1a69781u, selector);
        const uint32_t v1 = __builtin_amdgcn_perm(0xfdf8f3eeu, 0xe8e3ddd7u, selector);
        const uint32_t v2 = __builtin_amdgcn_perm(0x29231d18u, 0x120d0803u, selector);
        const uint32_t v3 = __builtin_amdgcn_perm(0x7f695a4fu, 0x463d362fu, selector);
        value = bank == 0u ? v0 : bank == 1u ? v1 : bank == 2u ? v2 : v3;
    } else {
        const uint32_t v0 = __builtin_amdgcn_perm(0xbcb7b2acu, 0xa59d9281u, selector);
        const uint32_t v1 = __builtin_amdgcn_perm(0xd8d5d2cfu, 0xcbc8c4c0u, selector);
        const uint32_t v2 = __builtin_amdgcn_perm(0xeceae8e5u, 0xe2e0dddau, selector);
        const uint32_t v3 = __builtin_amdgcn_perm(0xfffdfaf8u, 0xf6f3f1efu, selector);
        const uint32_t v4 = __builtin_amdgcn_perm(0x110f0d0au, 0x08060301u, selector);
        const uint32_t v5 = __builtin_amdgcn_perm(0x2623201eu, 0x1b181614u, selector);
        const uint32_t v6 = __builtin_amdgcn_perm(0x403c3835u, 0x312e2b28u, selector);
        const uint32_t v7 = __builtin_amdgcn_perm(0x7f6e635bu, 0x544e4944u, selector);
        value = bank == 0u ? v0 : bank == 1u ? v1 : bank == 2u ? v2
              : bank == 3u ? v3 : bank == 4u ? v4 : bank == 5u ? v5
              : bank == 6u ? v6 : v7;
    }
    return static_cast<int32_t>(static_cast<int8_t>(value & 0xffu));
}

template <uint32_t kBits>
__device__ __forceinline__ float codebook_scale() {
    if constexpr (kBits == 3u) return 0.0169444544f;
    if constexpr (kBits == 4u) return 0.0215164533f;
    if constexpr (kBits == 5u) return 0.0256750590f;
    return 0.0294811393f;
}

__device__ __forceinline__ float half_at(const uint8_t* source) {
    const uint16_t bits = static_cast<uint16_t>(source[0]) |
                          static_cast<uint16_t>(source[1]) << 8u;
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ void wht8(float (&x)[8]) {
    #pragma unroll
    for (uint32_t half = 1u; half < 8u; half <<= 1u) {
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            if ((i & half) == 0u) {
                const float a = x[i], b = x[i + half];
                x[i] = a + b;
                x[i + half] = a - b;
            }
        }
    }
}

__device__ __forceinline__ void inverse_wave(float (&x)[8], uint32_t lane,
                                             bool compact) {
    wht8(x);
    const uint32_t limit = compact ? 16u : 4u;
    for (uint32_t mask = 1u; mask <= 16u; mask <<= 1u) {
        if (mask > limit) break;
        #pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) {
            const float other = __shfl_xor(x[i], mask, 32);
            x[i] = (lane & mask) != 0u ? other - x[i] : x[i] + other;
        }
    }
    const float norm = compact ? 0.0625f : 0.125f;
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) x[i] *= norm;
}

__device__ __forceinline__ float sign_of(uint32_t index, uint32_t salt) {
    uint32_t x = index * 2654435761u + salt * 40503u + 12345u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return (x & 1u) != 0u ? -1.0f : 1.0f;
}

__device__ __forceinline__ uint32_t load_word(const uint8_t* source) {
    uint32_t word;
    __builtin_memcpy(&word, source, sizeof(word));
    return word;
}

__device__ __forceinline__ void unpack8(const uint8_t* source, uint32_t base,
                                        uint32_t dimensions, uint32_t bits,
                                        uint32_t (&codes)[8]) {
    if (bits >= 4u) {
        const uint32_t low = load_word(source + (base / 8u) * 4u);
        source += dimensions / 2u;
        uint32_t high = 0u;
        if (bits == 5u)
            high = load_word(source + (base / 32u) * 4u);
        else if (bits == 6u)
            high = load_word(source + (base / 16u) * 4u);
        #pragma unroll
        for (uint32_t e = 0u; e < 8u; ++e) {
            codes[e] = (low >> (4u * e)) & 15u;
            if (bits == 5u)
                codes[e] |= ((high >> ((base & 31u) + e)) & 1u) << 4u;
            else if (bits == 6u)
                codes[e] |= ((high >> (2u * ((base & 15u) + e))) & 3u) << 4u;
        }
        return;
    }
    const uint32_t low = load_word(source + (base / 16u) * 4u);
    source += dimensions / 4u;
    const uint32_t high = load_word(source + (base / 32u) * 4u);
    #pragma unroll
    for (uint32_t e = 0u; e < 8u; ++e)
        codes[e] = ((low >> (2u * ((base & 15u) + e))) & 3u) |
                   (((high >> ((base & 31u) + e)) & 1u) << 2u);
}

__device__ __forceinline__ uint32_t unpack_plane(const uint8_t* source,
                                                 uint32_t index,
                                                 uint32_t plane_bits) {
    const uint32_t fields = 32u / plane_bits;
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & 31u;
    const uint32_t leader = lane - lane % fields;
    uint32_t word = lane == leader
        ? load_word(source + (index / fields) * 4u) : 0u;
    word = __shfl(word, leader);
    return (word >> ((index % fields) * plane_bits)) & ((1u << plane_bits) - 1u);
}

__device__ __forceinline__ uint32_t unpack_code(const uint8_t* source,
                                                uint32_t index,
                                                uint32_t dimensions,
                                                uint32_t bits) {
    if (bits >= 4u) {
        uint32_t code = unpack_plane(source, index, 4u);
        source += dimensions / 2u;
        if (bits == 5u) code |= unpack_plane(source, index, 1u) << 4u;
        if (bits == 6u) code |= unpack_plane(source, index, 2u) << 4u;
        return code;
    }
    uint32_t code = unpack_plane(source, index, 2u);
    source += dimensions / 4u;
    return code | (unpack_plane(source, index, 1u) << 2u);
}

__device__ __forceinline__ void wht(float& value, float* shared,
                                    uint32_t segment) {
    const uint32_t tid = static_cast<uint32_t>(threadIdx.x);
    for (uint32_t half = 1u; half < segment; half <<= 1u) {
        shared[tid] = value;
        __syncthreads();
        const float other = shared[tid ^ half];
        __syncthreads();
        value = (tid & half) != 0u ? other - value : value + other;
    }
    value *= rsqrtf(static_cast<float>(segment));
}

}  // namespace shelf::kernel::detail::octave
