#pragma once

#include <shelf/models/hy4_flash/ops.h>

namespace shelf::hy4 {
using Selection = std::vector<std::vector<int>>;
Mat attention_batch(Ops &op, const std::string &prefix, int layer, const std::vector<int> &slots,
                    const std::vector<int> &positions, const int *device_slots,
                    const int *device_positions, Mat hidden, int block_rows = 1, bool draft = false);
Mat attention(Ops &op, const std::string &prefix, int layer, int start, Mat hidden, Selection &selected,
              bool draft);
}
