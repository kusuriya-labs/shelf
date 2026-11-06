#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <cstdint>
#include <span>

namespace shelf {
namespace qwen4exp {

struct Qwen4ExpLowerOptions {
    // QSA indexer capacity. The graph sizes its block scores, top-k and expanded token
    // mask from this value, so it must be the runtime's maximum context length.
    std::uint32_t max_context_tokens = 512;
    std::uint32_t tp_rank = 0;
    float rope_attention_factor = 1.0f;
    // Target layers whose completed output, averaged over the hyper-connection streams,
    // is exposed as a DFlash2 target feature. Appended after the head's four outputs.
    std::span<const std::uint32_t> hidden_taps;
    // Expose the pre-mixer wide residual (hc * hidden) after the hidden taps. The MTP
    // drafter consumes it; the Qwen4Exp trunk has no separate post-norm hidden.
    bool expose_wide_hidden = false;
};

// Lowers one loaded rank: embedding repeated into the hc stream stack, every decoder layer
// (HyperConnection, GDN or QSA, PLE, MoE), the final mixer, and the LM head. The PLE
// embedding rows are the second external input when the model has PLE.
Result<::shelf::qwen_hybrid::HybridLoweredPrimitives> lower_qwen4exp_to_primitives(
    const Qwen4ExpTextConfig& config,
    const Qwen4ExpRankWeights& weights,
    const Qwen4ExpLowerOptions& options = {});

// Lowers the rank-local MTP draft head. The graph reads the target token id and the wide
// hyper-connection residual (hc * hidden) and writes the next-token logits. The block
// attends densely: the target verify remains exact, and the draft head does not own a QSA
// key-state pool.
Result<::shelf::qwen_hybrid::HybridLoweredPrimitives> lower_qwen4exp_mtp_to_primitives(
    const Qwen4ExpTextConfig& config,
    const Qwen4ExpRankWeights& weights,
    const Qwen4ExpLowerOptions& options = {});

}  // namespace qwen4exp
}  // namespace shelf
