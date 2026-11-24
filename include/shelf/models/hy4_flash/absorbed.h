#pragma once
#include <shelf/models/hy4_flash/ops.h>

namespace shelf::hy4 {
void fold_kv_weights(Residency &gpu, int rank);
void attend_absorbed_batch(Ops &op, const std::string &projection, Mat latent, Mat rotary, Mat q, Mat gate,
                           const int *indices, int stride, const int *row_positions, Param sinks, Mat output,
                           bool sliding);
void attend_absorbed(Ops &op, const std::string &projection, Mat latent, Mat rotary, Mat q, Mat gate,
                     const int *positions, int stride, Param sinks, Mat output, bool contiguous = true,
                     bool sliding = false, int row_offset = 0, int chunk_rows = 0, const int *meta = nullptr);
}
