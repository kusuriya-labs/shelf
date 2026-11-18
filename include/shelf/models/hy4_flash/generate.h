#pragma once
#include <shelf/models/hy4_flash/residency.h>

namespace shelf::hy4 {
class Ops;
int verify_speculation(Ops &op, const float *target_probs, const float *draft_probs, int position,
                       uint64_t seed, const std::vector<int> &draft, std::vector<int> &actual);
void generate(Residency &gpu, const std::vector<int> &ids, int count, int chunk, float temperature,
              uint64_t seed, const Path &logits_file);
}
