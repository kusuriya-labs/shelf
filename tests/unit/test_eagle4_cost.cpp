#include <shelf/models/qwen35/runtime/eagle4_cost.h>

#include <cstdio>

namespace {

int g_fail = 0;

void check(bool cond, const char* name) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", name);
    }
}

}  // namespace

int main() {
    using ps::qwen35::runtime::Eagle4CostTable;
    using ps::qwen35::runtime::Eagle4RoundSample;
    using ps::qwen35::runtime::Eagle4TreePlan;
    using ps::qwen35::runtime::eagle4_cost_choose;
    using ps::qwen35::runtime::eagle4_cost_gain;
    using ps::qwen35::runtime::eagle4_cost_observe;
    using ps::qwen35::runtime::kEagle4CostWarmupRounds;
    using ps::qwen35::runtime::kEagle4WidthChoices;
    using ps::qwen35::runtime::kEagle4WidthChoicesList;

    {
        Eagle4CostTable table;
        Eagle4RoundSample s;
        s.draft_us = 1000.0f;
        s.verify_us = 400.0f;
        s.accepted = 2;
        s.depth_cap = 3;
        s.width = 4;
        for (int i = 0; i < kEagle4CostWarmupRounds - 1; ++i) {
            eagle4_cost_observe(table, s);
        }
        check(!table.frozen, "not frozen before warmup rounds");
        check(table.observed_rounds == kEagle4CostWarmupRounds - 1, "round count");
        eagle4_cost_observe(table, s);
        check(table.frozen, "frozen at warmup rounds");
        check(table.draft_step_us[3] == 1000.0f, "draft cost recorded");
        check(table.verify_us[4] == 400.0f, "verify cost recorded");
        check(table.acceptance[1] == 1.0f && table.acceptance[2] == 1.0f,
              "acceptance prefix hits");
        check(table.acceptance[3] == 0.0f, "depth3 below accepted stays zero");
        const int32_t before = table.observed_rounds;
        Eagle4RoundSample other;
        other.draft_us = 10.0f;
        other.verify_us = 10.0f;
        other.depth_cap = 1;
        other.width = 1;
        eagle4_cost_observe(table, other);
        check(table.observed_rounds == before, "frozen table ignores observe");
        check(table.draft_step_us[3] == 1000.0f, "frozen value immutable");
    }

    {
        Eagle4CostTable table;
        table.draft_step_us[1] = 100.0f;
        table.draft_step_us[2] = 300.0f;
        table.draft_step_us[3] = 600.0f;
        table.verify_us[1] = 100.0f;
        table.verify_us[2] = 200.0f;
        table.verify_us[4] = 400.0f;
        table.verify_us[8] = 900.0f;
        table.verify_us[1] = 600.0f;
        table.acceptance[1] = 0.8f;
        table.acceptance[2] = 0.4f;
        table.acceptance[3] = 0.1f;

        const float g11 = eagle4_cost_gain(table, 1, 1);
        const float expect_g11 = 1.0f + 1 * 0.8f;
        check(g11 == expect_g11, "gain formula depth1 width1");
        const float g32 = eagle4_cost_gain(table, 3, 2);
        const float expect_g32 = 1.0f + 1 * 0.8f + 2 * 0.4f;
        check(g32 == expect_g32, "gain formula depth3 width2 clamped by width");
        const float g31 = eagle4_cost_gain(table, 3, 1);
        check(g31 == expect_g11, "gain clamped by min depth width");

        Eagle4TreePlan p = eagle4_cost_choose(table, 3, 16);
        check(p.depth >= 1 && p.depth <= 3, "choose respects depth cap");
        bool allowed = false;
        for (int32_t i = 0; i < kEagle4WidthChoices; ++i) {
            if (kEagle4WidthChoicesList[i] == p.width) allowed = true;
        }
        check(allowed, "width from discrete choices");

        Eagle4TreePlan p_cap = eagle4_cost_choose(table, 3, 3);
        check(p_cap.width <= 3, "choose respects width cap");
        check(p_cap.width == 2, "cap-3 choose picks discrete width 2");
    }

    {
        Eagle4CostTable table;
        table.draft_step_us[1] = 1000.0f;
        table.draft_step_us[2] = 9000.0f;
        table.verify_us[1] = 100.0f;
        table.verify_us[2] = 100.0f;
        table.verify_us[16] = 100.0f;
        table.acceptance[1] = 0.9f;
        table.acceptance[2] = 0.9f;
        Eagle4TreePlan p = eagle4_cost_choose(table, 2, 16);
        check(p.depth == 1, "expensive deep draft keeps depth shallow");
    }

    {
        Eagle4CostTable empty;
        Eagle4TreePlan p = eagle4_cost_choose(empty, 4, 8);
        check(p.depth == 1 && p.width == 1, "empty table falls back to 1 1");
    }

    if (g_fail == 0) std::printf("test_eagle4_cost: all ok\n");
    return g_fail == 0 ? 0 : 1;
}
