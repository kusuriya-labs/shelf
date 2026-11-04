#include <shelf/models/qwen35/model/stage_range.h>

#include <cstdio>
#include <string>

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        std::printf("FAIL: %s\n", msg.c_str());
        ++g_fail;
    }
}

}  // namespace

int main() {
    using shelf::qwen35::StageRange;
    using shelf::qwen35::even_stage_ranges;

    {
        const auto r = even_stage_ranges(64, 2);
        check(r.size() == 2, "pp2: two stages");
        if (r.size() == 2) {
            check(r[0] == StageRange{0, 32}, "pp2: first half");
            check(r[1] == StageRange{32, 64}, "pp2: second half");
        }
    }
    {
        const auto r = even_stage_ranges(5, 2);
        check(r.size() == 2, "uneven: two stages");
        if (r.size() == 2) {
            check(r[0] == StageRange{0, 2}, "uneven: first stage");
            check(r[1] == StageRange{2, 5}, "uneven: remainder on last stage");
        }
    }
    {
        const auto r = even_stage_ranges(5, 4);
        check(r.size() == 4, "pp4: four stages");
        if (r.size() == 4) {
            check(r[0] == StageRange{0, 1}, "pp4: stage 0");
            check(r[1] == StageRange{1, 2}, "pp4: stage 1");
            check(r[2] == StageRange{2, 3}, "pp4: stage 2");
            check(r[3] == StageRange{3, 5}, "pp4: remainder on last stage");
        }
    }
    check(even_stage_ranges(0, 1).empty(), "zero layers");
    check(even_stage_ranges(4, 0).empty(), "zero stages");
    check(even_stage_ranges(4, 5).empty(), "more stages than layers");

    if (g_fail == 0) {
        std::printf("QWEN35_PIPELINE_STAGE: PASS\n");
        return 0;
    }
    std::printf("QWEN35_PIPELINE_STAGE: FAIL (%d)\n", g_fail);
    return 1;
}
