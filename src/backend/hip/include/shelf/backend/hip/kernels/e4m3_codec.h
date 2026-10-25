// vendored from PhaseShift (own work); namespace adapted only
// (was ps::quantization::codec::device).
#pragma once
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace shelf::hipk {

__device__ __forceinline__ float e4m3_decode_f32(uint8_t x) {
    const __half_raw hr = __hip_cvt_fp8_to_halfraw(static_cast<__hip_fp8_storage_t>(x), __HIP_E4M3);
    return __half2float(*reinterpret_cast<const __half*>(&hr));
}

__device__ __forceinline__ uint8_t e4m3_encode_u8(float x) {
    return static_cast<uint8_t>(__hip_cvt_float_to_fp8(x, __HIP_SATFINITE, __HIP_E4M3));
}

}  // namespace shelf::hipk
