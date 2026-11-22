#pragma once

#include <shelf/models/hy4_flash/attention.h>

namespace shelf::hy4 {
void decoder_layer(Ops &op, int layer, int start, Mat x, const std::vector<int> &history,
                   const std::vector<int> &ids, Selection &selected);
Mat engram(Ops &op, int layer, Mat x, const std::vector<int> &prior, const std::vector<int> &ids);
Mat feedforward(Ops &op, const std::string &prefix, Mat x, bool clip = false);
Mat moe(Ops &op, int layer, Mat x);
}
