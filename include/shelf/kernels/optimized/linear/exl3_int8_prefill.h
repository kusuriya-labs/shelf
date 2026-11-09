#pragma once

#include <shelf/kernels/optimized/linear/exl3_int8.h>

namespace shelf::kernel {

hipError_t launch_exl3_int8_prefill_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3Int8OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream);

}
