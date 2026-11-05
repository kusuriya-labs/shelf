#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>

#include <cstdint>
#include <span>

namespace shelf::qwen35moe {

// What a caller asks the MoE lowering to add. The tensor-partition fields are tp_size and
// tp_rank: an MoE model lowers to one rank of an expert-axis split, where the routed expert
// planes are cut along the expert axis (MoeExpertGroupPlan) and the shared expert and the
// backbone along their matrix axes (the rank's partition plan).
struct Qwen35MoeLowerOptions {
    std::span<const uint32_t> hidden_taps;
    // Layers whose input value the runtime wants to replace at run time. Like the hidden
    // taps, these are exposed in the lowered form's layer_input_values.
    std::span<const uint32_t> override_layers;
    float rope_attention_factor = 1.0f;
    // One more external output: the final norm applied to every token row of the last
    // layer. An MTP drafter reads it as the previous hidden of its own block.
    bool emit_normed_token_hidden = false;
    // One rank of an expert-axis split. The routed experts divide along the expert axis, the
    // shared expert and the backbone along their matrix axes, and the graph's grouped row
    // shape carries this rank's expert count. The weights have to have been loaded with the
    // matching partition plan. tp_size == 1 lowers the whole model on one device.
    uint32_t tp_size = 1;
    uint32_t tp_rank = 0;
};

// Lowers a loaded MoE model to the shared lowered form: the hybrid backbone of every layer
// (qwen_hybrid::emit_hybrid_block) and this family's FFN, which routes over rank-3 expert
// planes and adds a shared expert. The graph declares its grouped row shape once, so every
// group-row value and every MoE node reads the expert count and selection width from there.
Result<shelf::qwen_hybrid::HybridLoweredPrimitives> lower_qwen35_moe_to_primitives(
    const Qwen35MoeTextConfig& config,
    const Qwen35MoeModelWeights& weights,
    const Qwen35MoeLowerOptions& options = {});

// The checkpoint's MTP block as its own graph: the block's norms and fc over the token
// embedding and the previous hidden, one full-attention hybrid block with this family's
// MoE FFN, then the model's lm_head and sampling. Its external inputs are the token ids
// and the previous hidden; its state is the block's KV cache only.
Result<shelf::qwen_hybrid::HybridLoweredPrimitives> lower_qwen35_moe_mtp_to_primitives(
    const Qwen35MoeTextConfig& config,
    const Qwen35MoeModelWeights& weights,
    float rope_attention_factor = 1.0f);

}  // namespace shelf::qwen35moe
