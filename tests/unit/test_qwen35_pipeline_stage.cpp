#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/common/qwen_hybrid/model_weights.h>
#include <shelf/models/common/qwen_hybrid/weight_validation.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/tensor_partition.h>

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
    using shelf::qwen_hybrid::StageRange;
    using shelf::qwen_hybrid::even_stage_ranges;

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

    {
        using shelf::qwen_hybrid::HybridModelLevelWeights;
        using shelf::qwen_hybrid::QwenHybridConfig;
        using shelf::weights::TensorPartitionDesc;

        QwenHybridConfig cfg;
        cfg.vocab_size = 128;
        cfg.hidden_size = 64;
        cfg.num_hidden_layers = 4;

        auto status_of = [&](const HybridModelLevelWeights& w, std::size_t lo,
                             std::size_t hi) {
            return shelf::qwen_hybrid::validate_hybrid_model_level(w, cfg, lo, hi);
        };
        auto message_has = [&](const shelf::Status& st, const char* needle) {
            return !st.ok() && st.message().find(needle) != std::string::npos;
        };

        // A middle stage carries neither the embedding nor the head.
        {
            HybridModelLevelWeights middle;
            check(status_of(middle, 1u, 3u).ok(), "middle stage: no model-level tensors");
        }
        // The first stage carries the embedding, and a vocabulary-sharded model-level tensor
        // reports local rows: the invariant is stated for the model, so the partition has to
        // be read back to the global count.
        {
            HybridModelLevelWeights first;
            first.embed_tokens.rows = 64;
            first.embed_tokens.cols = 64;
            first.embed_tokens.partition = TensorPartitionDesc{0, 0, 2, {128, 64}, {}};
            first.lm_head_tied = false;
            check(status_of(first, 0u, 2u).ok(), "first stage: sharded embedding passes");
            HybridModelLevelWeights broken;
            broken.embed_tokens.rows = 64;
            broken.embed_tokens.cols = 64;
            broken.lm_head_tied = false;
            check(message_has(status_of(broken, 0u, 2u), "embed_tokens vocab size mismatch"),
                  "first stage: unpartitioned sharded embedding is rejected");
            HybridModelLevelWeights absent;
            absent.lm_head_tied = false;
            check(message_has(status_of(absent, 0u, 2u), "embed_tokens is not loaded"),
                  "first stage: a missing embedding is rejected");
        }
        // The last stage carries the head.
        {
            HybridModelLevelWeights last;
            last.lm_head_tied = false;
            check(message_has(status_of(last, 2u, 4u), "lm_head is not loaded"),
                  "last stage: a missing head is rejected");
            last.lm_head.rows = 64;
            last.lm_head.cols = 64;
            last.lm_head.partition = TensorPartitionDesc{0, 1, 2, {128, 64}, {}};
            check(!status_of(last, 2u, 4u).ok(),
                  "last stage: a sharded head still needs the final norm");
        }
        check(message_has(status_of(HybridModelLevelWeights{}, 0u, 5u),
                          "loaded layer range is outside the model"),
              "a stage range past the model is rejected");
        check(message_has(status_of(HybridModelLevelWeights{}, 2u, 2u),
                          "loaded layer range is outside the model"),
              "an empty stage range is rejected");
    }

    if (g_fail == 0) {
        std::printf("QWEN35_PIPELINE_STAGE: PASS\n");
        return 0;
    }
    std::printf("QWEN35_PIPELINE_STAGE: FAIL (%d)\n", g_fail);
    return 1;
}
