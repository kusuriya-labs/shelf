#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

extern "C" hipError_t shelf_hy4_linear(const uint16_t *input, const uint16_t *suh, const uint16_t *svh,
                                       const uint32_t *trellis, int8_t *codes, float *scales, float *partial,
                                       uint16_t *output, int rows, int input_width, int output_width,
                                       int padded_input, int padded_output, int bits, hipStream_t stream,
                                       int splits = 1);
extern "C" hipError_t shelf_hy4_linear_fused_gemm(const uint16_t *input, const uint16_t *suh, const uint16_t *svh,
                                                  const uint32_t *trellis, float *partial, uint16_t *output,
                                                  int rows, int input_width, int output_width, int padded_input,
                                                  int padded_output, int bits, int splits, hipStream_t stream);

hipError_t shelf_hy4_linear_workspace(const uint16_t *input, const uint16_t *suh, const uint16_t *svh,
                                       const uint32_t *trellis, int8_t *codes, float *scales, float *partial,
                                       uint16_t *output, int rows, int input_width, int output_width,
                                       int padded_input, int padded_output, int bits, hipStream_t stream,
                                       int splits, int8_t *expanded, float *f32_output = nullptr);

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
    uint64_t bits;
    int k;
    int n;
};
extern "C" hipError_t shelf_hy4_expert_linear(const uint16_t *x, const Hy4ExpertProjection *matrices,
                                              int stride, int8_t *codes, float *scales, float *partial,
                                              uint16_t *y, int rows, const Hy4Route *routes,
                                              const Hy4ExpertBatch *batches, int num_batches, int mixed, int k,
                                              int n, hipStream_t stream = nullptr);

hipError_t shelf_hy4_shared_gate_up(const uint16_t *input, Hy4ExpertProjection gate, Hy4ExpertProjection up,
                                    float *partial, uint16_t *output, int rows, hipStream_t stream);

hipError_t shelf_hy4_head(const uint16_t *x, const uint16_t *suh, const uint16_t *svh, const uint32_t *trellis,
                          int8_t *codes, float *scales, float *partial, float *logits, int rows,
                          hipStream_t stream = nullptr, int vocab_width = 120832);

hipError_t shelf_hy4_head_argmax(const uint16_t *x, const uint16_t *suh, const uint16_t *svh,
                                 const uint32_t *trellis, int8_t *codes, float *scales, float *partial,
                                 float *tile_vals, int *tile_ids, int *tokens, int rows,
                                 hipStream_t stream = nullptr, int vocab_width = 120832, int vocab_offset = 0,
                                 float *best_vals = nullptr, int *best_ids = nullptr);

hipError_t shelf_hy4_expert_decode(const uint16_t *latent, const Hy4ExpertProjection *matrices,
                                   const Hy4Route *routes, const int *count, int capacity, int intermediate,
                                   int8_t *codes, float *scales, float *partial, uint16_t *output,
                                   hipStream_t stream, const Hy4ExpertBatch *batches = nullptr,
                                   const int *num_batches = nullptr);

hipError_t shelf_hy4_expert_decode(const uint16_t *latent, const Hy4ExpertProjection *matrices,
                                   const Hy4Route *routes, const int *count, int capacity, int intermediate,
                                   int8_t *codes, float *scales, float *partial, uint16_t *output,
                                   hipStream_t stream, const Hy4ExpertBatch *batches,
                                   const int *num_batches, bool prefill);
