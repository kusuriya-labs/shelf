#pragma once

#include <shelf/models/hy4_flash/attention.h>

namespace shelf::hy4 {
void decoder_layer(Ops &op, int layer, int start, Mat x, const std::vector<int> &history,
                   const std::vector<int> &ids, Selection &selected);
std::vector<uint8_t> gather_batch_engram(const Checkpoint &model, int layer,
        const std::vector<const std::vector<int> *> &priors, const std::vector<int> &ids);
Mat engram_batch(Ops &op, int layer, Mat x, const std::vector<const std::vector<int> *> &priors,
                 const std::vector<int> &ids);
Mat engram(Ops &op, int layer, Mat x, const std::vector<int> &prior, const std::vector<int> &ids);
Mat feedforward(Ops &op, const std::string &prefix, Mat x, bool clip = false);
Mat moe(Ops &op, int layer, Mat x);
}
