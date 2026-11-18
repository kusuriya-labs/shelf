#pragma once

#include <shelf/models/hy4_flash/attention.h>

namespace shelf::hy4 {
Mat engram(Ops &op, int layer, Mat x, const std::vector<int> &prior, const std::vector<int> &ids);
Mat feedforward(Ops &op, const std::string &prefix, Mat x, bool clip = false);
Mat moe(Ops &op, int layer, Mat x);
}
