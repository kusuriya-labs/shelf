#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" hipError_t shelf_hy4_linear(const uint16_t *input, const uint16_t *suh, const uint16_t *svh,
                                       const uint32_t *trellis, int8_t *codes, float *scales, float *partial,
                                       uint16_t *output, int rows, int input_width, int output_width,
                                       int padded_input, int padded_output, int bits, hipStream_t stream,
                                       int wmma = 0);

struct Hy4ExpertBatch {
    int first, count;
};
struct Hy4Route {
    int row, expert;
    float weight;
};

struct Hy4ExpertProjection {
    const uint16_t *suh;
    const uint16_t *svh;
    const uint32_t *trellis;
};
extern "C" hipError_t shelf_hy4_expert_linear(const uint16_t *x, const Hy4ExpertProjection *matrices,
                                              int stride, int8_t *codes, float *scales, float *partial,
                                              uint16_t *y, int rows, const Hy4Route *routes = nullptr,
                                              const Hy4ExpertBatch *batches = nullptr, int num_batches = 0,
                                              int wmma = 0);

hipError_t shelf_hy4_head(const uint16_t *x, const uint16_t *suh, const uint16_t *svh, const uint32_t *trellis,
                          int8_t *codes, float *scales, float *partial, float *logits, int rows);
