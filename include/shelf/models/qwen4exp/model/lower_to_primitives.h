#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <cstdint>

namespace shelf {
namespace qwen4exp {

struct Qwen4ExpLowerOptions {
    // QSA indexer capacity. The graph sizes its block scores, top-k and expanded token
    // mask from this value, so it must be the runtime's maximum context length.
    std::uint32_t max_context_tokens = 512;
    std::uint32_t tp_rank = 0;
    float rope_attention_factor = 1.0f;
};

// Lowers one loaded rank: embedding repeated into the hc stream stack, every decoder layer
// (HyperConnection, GDN or QSA, PLE, MoE), the final mixer, and the LM head. The PLE
// embedding rows are the second external input when the model has PLE.
Result<::shelf::qwen_hybrid::HybridLoweredPrimitives> lower_qwen4exp_to_primitives(
    const Qwen4ExpTextConfig& config,
    const Qwen4ExpRankWeights& weights,
    const Qwen4ExpLowerOptions& options = {});

}  // namespace qwen4exp
}  // namespace shelf
