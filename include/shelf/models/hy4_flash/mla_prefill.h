#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <shelf/models/hy4_flash/residency.h>

hipError_t shelf_hy4_mla_absorbed(const uint16_t *q, const void *folded, const uint16_t *rotary,
                                   const uint16_t *gate, const void *sink, bool sink_fp32, const int *indices,
                                   int stride, const void *input, void *query, void *values, float *scores,
                                   uint16_t *probs, uint16_t *out, int rows, int heads,
                                   hipStream_t stream = nullptr, const shelf::hy4::Hy4StepParams *params = nullptr,
                                   bool contiguous = true, bool sliding = false, int row_offset = 0,
                                   int chunk_rows = 0, bool projected = false);

hipError_t shelf_hy4_mla_absorbed_batch(const uint16_t *q, const void *folded, const uint16_t *rotary,
                                         const uint16_t *gate, const void *sinks, bool sink_fp32,
                                         const int *indices, int stride, const void *input, void *query, void *values,
                                         float *scores, uint16_t *probs, uint16_t *out, int rows, int heads,
                                         hipStream_t stream, const int *row_positions, bool sliding);

hipError_t shelf_hy4_mla_absorbed_sparse_batch(const uint16_t *q, const void *folded, const uint16_t *rotary,
                                                const uint16_t *gate, const void *sinks, bool sink_fp32,
                                                const int *indices, int stride, int id_stride, const void *input,
                                                void *query, void *values, float *scores, uint16_t *probs,
                                                uint16_t *out, int sparse_blocks, int block_rows, int heads,
                                                hipStream_t stream, const int *row_positions,
                                                const int *block_map, bool draft);

hipError_t shelf_hy4_mla_project_query(const uint16_t *q, const void *folded, void *query, int rows, int heads, hipStream_t stream);
hipError_t shelf_hy4_mla_project_value(const void *values, const void *folded, const uint16_t *gate, uint16_t *out, int rows, int heads, hipStream_t stream);
