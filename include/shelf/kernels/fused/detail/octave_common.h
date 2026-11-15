#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel::detail::octave {

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
