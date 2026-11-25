#pragma once

// Shared constants and device helpers for the mul24 (EXL3-INT8 weights) kernels. The
// FWHT variants live in fwht.h; decode / prefill / grouped translation units include
// this header.

#include <shelf/kernels/optimized/linear/exl3_family/fwht.h>
#include <shelf/quantization/exl3/exl3_int8.h>

#include <shelf/core/memory/types.h>

#include <cstdint>
#include <cstdlib>

#include <hip/hip_runtime.h>

namespace shelf::kernel::exl3_detail::mul24 {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

inline constexpr uint32_t kBlock = 128u;
inline constexpr float kInvSqrt128 = 0.08838834764831845f;

// The float4 win32 H128 (fwht128, one wave per 128-block, vectorized I/O) is the default.
// SHELF_EXL3_INT8_ROTATE_F4=0 selects the scalar/smem form for A/B measurements.
inline bool rotate_f4_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHELF_EXL3_INT8_ROTATE_F4");
        return env == nullptr || std::strtoul(env, nullptr, 10) != 0ul;
    }();
    return enabled;
}

inline constexpr uint32_t kRotateQuantizeF4MinRows = 32u;
inline constexpr uint32_t kRotateOutF4MinRows = 512u;

__device__ __forceinline__ uint32_t trellis_state_3bit(uint32_t w0, uint32_t w1, uint32_t offset,
                                                       uint32_t index) {
    const uint32_t first = __builtin_amdgcn_alignbit(w0, w1, 32u - offset);
    const uint32_t last = __builtin_amdgcn_alignbit(w0, w1, 27u - offset);
    return (index < 4u ? first >> (16u - 3u * index) : last >> (3u * (7u - index))) & 0xFFFFu;
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

// One wave turns a 128-wide block of the int32 row-scale accumulator into four
// BF16 pairs: rowscale * S_FIX, H128, 1/sqrt(128), svh. Both the standalone
// rotate-out and a fused GEMM epilogue call this so the arithmetic exists once.
__device__ __forceinline__ uint2 rotate_i32_block_bf16(int4 a, float rs, const uint16_t* svh,
                                                       uint32_t col, uint32_t lane) {
    float v0 = (static_cast<float>(a.x) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
    float v1 = (static_cast<float>(a.y) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
    float v2 = (static_cast<float>(a.z) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
    float v3 = (static_cast<float>(a.w) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
    fwht128(v0, v1, v2, v3, lane);
    v0 = (v0 * kInvSqrt128) * f16_bits_to_f32(svh[col + 0u]);
    v1 = (v1 * kInvSqrt128) * f16_bits_to_f32(svh[col + 1u]);
    v2 = (v2 * kInvSqrt128) * f16_bits_to_f32(svh[col + 2u]);
    v3 = (v3 * kInvSqrt128) * f16_bits_to_f32(svh[col + 3u]);
    uint2 packed;
    packed.x = static_cast<uint32_t>(f32_to_bf16_rne_device(v0)) |
               (static_cast<uint32_t>(f32_to_bf16_rne_device(v1)) << 16u);
    packed.y = static_cast<uint32_t>(f32_to_bf16_rne_device(v2)) |
               (static_cast<uint32_t>(f32_to_bf16_rne_device(v3)) << 16u);
    return packed;
}

// Same for the flagged rows, whose input is already the per-group F32 result.
__device__ __forceinline__ uint2 rotate_f32_block_bf16(float4 y, const uint16_t* svh,
                                                       uint32_t col, uint32_t lane) {
    float v0 = y.x, v1 = y.y, v2 = y.z, v3 = y.w;
    fwht128(v0, v1, v2, v3, lane);
    v0 = (v0 * kInvSqrt128) * f16_bits_to_f32(svh[col + 0u]);
    v1 = (v1 * kInvSqrt128) * f16_bits_to_f32(svh[col + 1u]);
    v2 = (v2 * kInvSqrt128) * f16_bits_to_f32(svh[col + 2u]);
    v3 = (v3 * kInvSqrt128) * f16_bits_to_f32(svh[col + 3u]);
    uint2 packed;
    packed.x = static_cast<uint32_t>(f32_to_bf16_rne_device(v0)) |
               (static_cast<uint32_t>(f32_to_bf16_rne_device(v1)) << 16u);
    packed.y = static_cast<uint32_t>(f32_to_bf16_rne_device(v2)) |
               (static_cast<uint32_t>(f32_to_bf16_rne_device(v3)) << 16u);
    return packed;
}

__device__ __forceinline__ float bf16_to_f32(::shelf::bf16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x.data) << 16);
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

}  // namespace shelf::kernel::exl3_detail::mul24
