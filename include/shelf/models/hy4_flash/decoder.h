#pragma once

#include <shelf/models/hy4_flash/attention.h>

namespace shelf::hy4 {
void decoder_layer(Ops &op, int layer, int start, Mat x, const std::vector<int> &history,
                   const std::vector<int> &ids, Selection &selected);
struct NgramLayout {
    const uint16_t *books[16];
    unsigned bits[16], offsets[16], stride;
};
struct EngramNorms {
    const uint16_t *head[16];
};
void launch_decode_heads(const uint8_t *packed, EngramNorms norms, Mat embed, NgramLayout layout, bool ngram);
void prefetch_engram(Ops &op, const std::vector<int> &prior, const std::vector<int> &ids);
Mat engram(Ops &op, int layer, Mat x, const std::vector<int> &prior, const std::vector<int> &ids);
Mat feedforward(Ops &op, const std::string &prefix, Mat x, bool clip = false);
Mat moe(Ops &op, int layer, Mat x);
}
