#include <shelf/models/qwen35/runtime/eagle4_cost.h>

#include <algorithm>

namespace ps::qwen35::runtime {

namespace {

int32_t clamp_depth(int32_t d) {
    if (d < 1) return 1;
    if (d > kEagle4MaxDraftDepth) return kEagle4MaxDraftDepth;
    return d;
}

int32_t clamp_width(int32_t w) {
    if (w < 1) return 1;
    if (w > kEagle4MaxVerifyWidth) return kEagle4MaxVerifyWidth;
    return w;
}

}  // namespace

void eagle4_cost_observe(Eagle4CostTable& table, const Eagle4RoundSample& sample) {
    if (table.frozen) return;
    const int32_t depth = clamp_depth(sample.depth_cap);
    const int32_t width = clamp_width(sample.width);
    const float n = static_cast<float>(table.observed_rounds);
    const bool first = table.observed_rounds == 0;
    const auto ema = [first, n](float current, float value) {
        return first ? value : (current * n + value) / (n + 1.0f);
    };
    table.draft_step_us[depth] = ema(table.draft_step_us[depth], sample.draft_us);
    table.verify_us[width] = ema(table.verify_us[width], sample.verify_us);
    for (int32_t i = 1; i <= depth; ++i) {
        const float hit = sample.accepted >= i ? 1.0f : 0.0f;
        table.acceptance[i] = ema(table.acceptance[i], hit);
    }
    table.observed_rounds += 1;
    if (table.observed_rounds >= kEagle4CostWarmupRounds) table.frozen = true;
}

float eagle4_cost_gain(const Eagle4CostTable& table, int32_t depth, int32_t width) {
    const int32_t d = clamp_depth(depth);
    const int32_t w = clamp_width(width);
    const int32_t limit = std::min(d, w);
    float gain = 1.0f;
    for (int32_t i = 1; i <= limit; ++i) {
        gain += static_cast<float>(i) * table.acceptance[i];
    }
    return gain;
}

Eagle4TreePlan eagle4_cost_choose(const Eagle4CostTable& table, int32_t max_depth,
                                  int32_t max_width) {
    Eagle4TreePlan best;
    float best_rate = -1.0f;
    const int32_t d_cap = clamp_depth(max_depth);
    const int32_t w_cap = clamp_width(max_width);
    for (int32_t d = 1; d <= d_cap; ++d) {
        const float draft_cost = table.draft_step_us[d];
        for (int32_t ci = 0; ci < kEagle4WidthChoices; ++ci) {
            const int32_t w = kEagle4WidthChoicesList[ci];
            if (w > w_cap) break;
            const float verify_cost = table.verify_us[w];
            const float cost = draft_cost + verify_cost;
            if (cost <= 0.0f) continue;
            const float rate = eagle4_cost_gain(table, d, w) / cost;
            if (rate > best_rate) {
                best_rate = rate;
                best.depth = d;
                best.width = w;
            }
        }
    }
    if (best_rate < 0.0f) {
        best.depth = 1;
        best.width = 1;
    }
    return best;
}

}  // namespace ps::qwen35::runtime
