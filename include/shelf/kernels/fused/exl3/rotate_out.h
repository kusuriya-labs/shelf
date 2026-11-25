#pragma once

#include <shelf/kernels/optimized/linear/exl3_mul24.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_exl3_mul24_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream);

// BF16 only, rows with slots[m] < 0 are skipped: the companion of a fused GEMM
// epilogue that already wrote them.
hipError_t launch_exl3_mul24_rotate_out_i32_flagged(
    const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream);

}  // namespace shelf::kernel
