#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

namespace ps::kernel::detail {

using bf16 = __hip_bfloat16;

constexpr uint32_t kVec = 4u;

__device__ __forceinline__ bool aligned8(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 7u) == 0u;
}

__device__ __forceinline__ bool aligned16(const void* p) {
    return (reinterpret_cast<uintptr_t>(p) & 15u) == 0u;
}

struct F4 {
    float v[kVec];
};

__device__ __forceinline__ float bf16_to_f32(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f32_to_bf16(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ F4 load_bf16x4(const bf16* p) {
    const uint64_t bits = *reinterpret_cast<const uint64_t*>(p);
    F4 r;
    r.v[0] = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>(bits & 0xFFFFu)));
    r.v[1] = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>((bits >> 16u) & 0xFFFFu)));
    r.v[2] = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>((bits >> 32u) & 0xFFFFu)));
    r.v[3] = __bfloat162float(__ushort_as_bfloat16(static_cast<uint16_t>((bits >> 48u) & 0xFFFFu)));
    return r;
}

__device__ __forceinline__ void store_bf16x4(bf16* p, const F4& r) {
    uint64_t bits = 0u;
    bits |= static_cast<uint64_t>(
        static_cast<uint16_t>(__bfloat16_as_ushort(__float2bfloat16(r.v[0]))));
    bits |= static_cast<uint64_t>(
        static_cast<uint16_t>(__bfloat16_as_ushort(__float2bfloat16(r.v[1]))))
        << 16u;
    bits |= static_cast<uint64_t>(
        static_cast<uint16_t>(__bfloat16_as_ushort(__float2bfloat16(r.v[2]))))
        << 32u;
    bits |= static_cast<uint64_t>(
        static_cast<uint16_t>(__bfloat16_as_ushort(__float2bfloat16(r.v[3]))))
        << 48u;
    *reinterpret_cast<uint64_t*>(p) = bits;
}

__device__ __forceinline__ F4 load_f32x4(const float* p) {
    const float4 t = *reinterpret_cast<const float4*>(p);
    F4 r;
    r.v[0] = t.x;
    r.v[1] = t.y;
    r.v[2] = t.z;
    r.v[3] = t.w;
    return r;
}

__device__ __forceinline__ void store_f32x4(float* p, const F4& r) {
    float4 t;
    t.x = r.v[0];
    t.y = r.v[1];
    t.z = r.v[2];
    t.w = r.v[3];
    *reinterpret_cast<float4*>(p) = t;
}

}  // namespace ps::kernel::detail
