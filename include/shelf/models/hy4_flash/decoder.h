#pragma once

#include <shelf/models/hy4_flash/attention.h>

namespace shelf::hy4 {
void decoder_layer(Ops &op, int layer, int start, Mat x, const std::vector<int> &history,
                   const std::vector<int> &ids, Selection &selected);
std::vector<uint8_t> gather_batch_engram(const Checkpoint &model, int layer,
        const std::vector<const std::vector<int> *> &priors, const std::vector<int> &ids, std::span<const int> row_offsets = {});
Mat engram_batch(Ops &op, int layer, Mat x, const std::vector<const std::vector<int> *> &priors,
                 const std::vector<int> &ids, std::span<const int> row_offsets = {});
Mat engram(Ops &op, int layer, Mat x, const std::vector<int> &prior, const std::vector<int> &ids);
Mat feedforward(Ops &op, const std::string &prefix, Mat x, bool clip = false);
Mat moe(Ops &op, int layer, Mat x);
void route_select(Ops &op, Mat x, const uint16_t *weight, const float *bias, const float *w_norms, Hy4Route *picks,
                  bool exact_only = false);
}
