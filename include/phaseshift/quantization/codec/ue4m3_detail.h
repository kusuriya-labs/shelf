#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::quantization::codec::device {

__device__ __forceinline__ float ue4m3_decode_f32(uint8_t e) {
    const uint32_t expf = e >> 3;
    const uint32_t man = e & 7u;
    if (expf != 0u) {
        return __uint_as_float(((expf + 119u) << 23) | (man << 20));
    }
    if (man == 0u) {
        return 0.0f;
    }
    const uint32_t msb = 31u - static_cast<uint32_t>(__clz(man));
    return __uint_as_float(((117u + msb) << 23) | ((man - (1u << msb)) << (23u - msb)));
}

}  // namespace ps::quantization::codec::device
