#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" hipError_t shelf_hy4_linear(
    const uint16_t* input, const uint16_t* suh, const uint16_t* svh,
    const uint32_t* trellis, int8_t* codes, float* scales, float* partial,
    uint16_t* output, int rows, int input_width, int output_width,
    int padded_input, int padded_output, int bits, hipStream_t stream);

struct Hy4ExpertProjection {
    const uint16_t* suh;
    const uint16_t* svh;
    const uint32_t* trellis;
};
hipError_t shelf_hy4_expert_linear(const uint16_t* x,const Hy4ExpertProjection* matrices,int stride,
    int8_t* codes,float* scales,float* partial,uint16_t* y,int rows);
