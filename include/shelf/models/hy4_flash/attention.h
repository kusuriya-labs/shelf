#pragma once

#include <shelf/models/hy4_flash/ops.h>

namespace shelf::hy4 {
using Selection = std::vector<std::vector<int>>;
inline size_t index_workspace_floats(int rows, int stride) { return size_t(rows) * (stride + 2) + 1; }
void index_select(Mat q, Mat weights, const uint16_t *groups, int first, int rows, int start, int stride,
                  float *scores, int *selected, const Hy4StepParams *params, bool wmma, hipStream_t stream, const unsigned *shared_peak = nullptr);
Mat attention_batch(Ops &op, const std::string &prefix, int layer, const std::vector<int> &slots,
                    const std::vector<int> &positions, const int *device_slots,
                    const int *device_positions, Mat hidden, int block_rows = 1, bool draft = false);
Mat attention(Ops &op, const std::string &prefix, int layer, int start, Mat hidden, Selection &selected,
              bool draft);
}
