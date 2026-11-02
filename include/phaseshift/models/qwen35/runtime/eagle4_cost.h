#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>

namespace ps::qwen35::runtime {

inline constexpr int32_t kEagle4MaxDraftDepth = 15;
inline constexpr int32_t kEagle4MaxVerifyWidth = 16;
inline constexpr int32_t kEagle4CostWarmupRounds = 8;
inline constexpr int32_t kEagle4WidthChoices = 5;
inline constexpr int32_t kEagle4WidthChoicesList[kEagle4WidthChoices] = {1, 2, 4, 8, 16};

struct Eagle4CostTable {
    float draft_step_us[kEagle4MaxDraftDepth + 1] = {};
    float verify_us[kEagle4MaxVerifyWidth + 1] = {};
    float acceptance[kEagle4MaxDraftDepth + 1] = {};
    int32_t observed_rounds = 0;
    int32_t batch_bucket = 0;
    int32_t context_bucket = 0;
    bool frozen = false;
};

struct Eagle4TreePlan {
    int32_t depth = 1;
    int32_t width = 1;
};

struct Eagle4RoundSample {
    float draft_us = 0.0f;
    float verify_us = 0.0f;
    int32_t accepted = 0;
    int32_t depth_cap = 1;
    int32_t width = 1;
};

void eagle4_cost_observe(Eagle4CostTable& table, const Eagle4RoundSample& sample);

float eagle4_cost_gain(const Eagle4CostTable& table, int32_t depth, int32_t width);

Eagle4TreePlan eagle4_cost_choose(const Eagle4CostTable& table, int32_t max_depth,
                                  int32_t max_width);

}  // namespace ps::qwen35::runtime
