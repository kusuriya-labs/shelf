#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

hipError_t shelf_hy4_mla_prefill(const uint16_t *q, const uint16_t *kv, const uint16_t *rotary,
                                 const uint16_t *gate, const void *sink, bool sink_fp32, uint16_t *out,
                                 float *scores, uint16_t *probs, int rows, int heads, int slots, int first_key,
                                 int first_query, bool sliding);

hipError_t shelf_hy4_mla_absorbed(const uint16_t *q, const void *folded, const uint16_t *rotary,
                                  const uint16_t *gate, const void *sink, bool sink_fp32, const int *indices,
                                  int stride, const void *input, void *query, void *values, float *scores,
                                  uint16_t *probs, uint16_t *out, int rows, int heads);
