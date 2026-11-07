#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <cstdint>

namespace shelf {
namespace qwen4exp {

struct Qwen4ExpMoeLoweringGeometry {
    std::uint32_t global_experts = 0;
    std::uint32_t local_experts = 0;
    std::uint32_t expert_begin = 0;
    std::uint32_t top_k = 0;
    std::uint32_t hidden = 0;
    std::uint32_t intermediate = 0;
    std::uint32_t shared_intermediate = 0;
    // The shared expert is replicated on every rank. Only one rank contributes it before
    // the TP sum; the others scale it to zero.
    float shared_scale = 1.0f;
};

struct Qwen4ExpMoeLoweringIndices {
    std::uint32_t w_router = 0;
    std::uint32_t w_shared_gate = 0;
    std::uint32_t w_shared_gate_proj = 0;
    std::uint32_t w_shared_up_proj = 0;
    std::uint32_t w_shared_down_proj = 0;
    std::uint32_t w_gate_experts = 0;
    std::uint32_t w_up_experts = 0;
    std::uint32_t w_down_experts = 0;
};

// Qwen4Exp MoE FFN: global top-k routing, separate gate/up/down EXL3 expert planes,
// a dense shared expert, and the combine that folds the expert rows and the shared
// expert back to one row per token. The caller sets graph.grouped_row_shape to the
// rank's local expert count and top_k.
Result<::shelf::runtime::ValueId> emit_qwen4exp_moe(
    ::shelf::qwen_hybrid::Emitter& em,
    const Qwen4ExpMoeLoweringGeometry& geometry,
    const Qwen4ExpMoeWeights& weights,
    const Qwen4ExpMoeLoweringIndices& indices,
    ::shelf::runtime::ValueId post_norm,
    const char* debug_prefix);

}  // namespace qwen4exp
}  // namespace shelf
