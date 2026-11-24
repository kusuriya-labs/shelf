#pragma once
#include <shelf/models/hy4_flash/ops.h>

namespace shelf::hy4 {
void fold_kv_weights(Residency &gpu, int rank);
void attend_absorbed_batch(Ops &op, const std::string &projection, Mat latent, Mat rotary, Mat q, Mat gate,
                           const int *indices, int stride, const int *row_positions, Param sinks, Mat output,
                           bool sliding, const _Float16 *prequant = nullptr);
void attend_absorbed_sparse_batch(Ops &op, const std::string &projection, const _Float16 *input,
                                  const uint16_t *rotary, Mat q, Mat gate, const int *indices, int stride,
                                  int id_stride, const int *row_positions, const int *block_map, Param sinks,
                                  Mat output, int sparse_blocks, int block_rows, bool draft);
void attend_absorbed(Ops &op, const std::string &projection, Mat latent, Mat rotary, Mat q, Mat gate,
                     const int *positions, int stride, Param sinks, Mat output, bool contiguous = true,
                     bool sliding = false, int row_offset = 0, int chunk_rows = 0, const int *meta = nullptr, const _Float16 *prequant = nullptr, void *batch_query = nullptr, void *batch_values = nullptr);
}
