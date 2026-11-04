#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>

#include <cstdint>
#include <span>

namespace shelf::qwen35moe {

// What a caller asks the MoE lowering to add. There is no tensor-partition field: an MoE
// model lowers to one device, and the routed expert planes are not part of the partition
// contract. There is no MTP field either; speculative decode uses the DFlash2 drafter.
struct Qwen35MoeLowerOptions {
    std::span<const uint32_t> hidden_taps;
    float rope_attention_factor = 1.0f;
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
