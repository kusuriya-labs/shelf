#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
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
    // One rank of a TP / EP split. TP splits the backbone, the shared expert and each local
    // expert's matrix; EP splits routed expert IDs. The graph's grouped row shape carries
    // this rank's expert count. The weights must have been loaded with the matching plan.
    // tp_size == 1 and ep_size == 1 lower the whole model on one device.
    uint32_t tp_size = 1;
    uint32_t tp_rank = 0;
    // EP=1 (or 0, meaning unset) keeps every expert on every TP rank.
    uint32_t ep_size = 0;
    uint32_t ep_rank = 0;
    bool ep_enabled = false;
    // One pipeline stage: only the stage's own layers are lowered, the embedding is emitted
    // by the first stage, and the final norm + lm_head + sampling by the last. The other
    // stages take the hidden of the previous stage as their external input and hand their
    // last layer's output out. The weights must have been loaded with the same range.
    shelf::qwen_hybrid::PipelineStage stage;
};

// Lowers a loaded MoE model to the shared lowered form: the hybrid backbone of every layer
// (qwen_hybrid::emit_hybrid_block) and this family's FFN, which routes over rank-3 expert
// planes and adds a shared expert. The graph declares its grouped row shape once, so every
// group-row value and every MoE node reads the expert count and selection width from there.
Result<shelf::qwen_hybrid::HybridLoweredPrimitives> lower_qwen35_moe_to_primitives(
    const Qwen35MoeTextConfig& config,
    const Qwen35MoeModelWeights& weights,
    const Qwen35MoeLowerOptions& options = {});

}  // namespace shelf::qwen35moe
