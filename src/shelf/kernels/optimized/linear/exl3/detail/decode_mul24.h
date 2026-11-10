#pragma once

// Shared constants and device helpers for the mul24 (EXL3-INT8 weights) kernels. Internal
// to the kernel layer; include it from the decode / prefill / grouped translation units.

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace shelf::kernel::exl3_detail {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

inline constexpr uint32_t kBlock = 128u;
inline constexpr float kInvSqrt128 = 0.08838834764831845f;

// Opt-in A/B switch: 0 keeps the scalar/smem H128 and the row-per-CTA layout, 1 uses the
// float4 win32 H128 (fwht128) with one wave per 128-block and vectorized I/O.
inline bool rotate_f4_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHELF_EXL3_INT8_ROTATE_F4");
        return env != nullptr && std::strtoul(env, nullptr, 10) != 0ul;
    }();
    return enabled;
}

// The float4 forms need enough CTAs to fill the device. Below these row counts the
// scalar/row-per-CTA layout (grid = rows x k/128) has more parallelism and measured
// faster, so the f4 path is skipped.
inline constexpr uint32_t kRotateQuantizeF4MinRows = 32u;
inline constexpr uint32_t kRotateOutF4MinRows = 512u;

// N-tile width of the float4 output rotation in 128-column blocks (0 = the whole row in
// one CTA).
inline uint32_t rotate_tile_blocks() {
    static const uint32_t blocks = [] {
        const char* env = std::getenv("SHELF_EXL3_INT8_ROTATE_TILE");
        return env != nullptr ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    return blocks;
}

// Sylvester Walsh-Hadamard transform of one 128-wide block, one element per thread (128
// threads). Result is unnormalised.
__device__ __forceinline__ float fwht128_smem(float v, float* smem, uint32_t t) {
    #pragma unroll
    for (uint32_t h = 1u; h < 32u; h <<= 1u) {
        const float b = __shfl_xor(v, h);
        v = (t & h) ? (b - v) : (v + b);
    }
    #pragma unroll
    for (uint32_t h = 32u; h < kBlock; h <<= 1u) {
        smem[t] = v;
        __syncthreads();
        const float b = smem[t ^ h];
        __syncthreads();
        v = (t & h) ? (b - v) : (v + b);
    }
    return v;
}

__device__ __forceinline__ uint16_t f32_to_bf16_rne_device(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7F800000u) == 0x7F800000u)
        return (u & 0x007FFFFFu) != 0u ? 0x7FC0u : static_cast<uint16_t>(u >> 16);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float f16_bits_to_f32(uint16_t bits) {
    return static_cast<float>(__builtin_bit_cast(_Float16, bits));
}

__device__ __forceinline__ float bf16_to_f32(::shelf::bf16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x.data) << 16);
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

}  // namespace shelf::kernel::exl3_detail
