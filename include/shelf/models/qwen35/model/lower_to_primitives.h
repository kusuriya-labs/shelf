#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/runtime/program/program.h>

#include <shelf/runtime/graph/primitive_graph.h>

#include <cstdint>
#include <span>

namespace shelf::qwen35 {

// The lowered form is family neutral: whatever FFN a model emits, the graph, its weight
// table and its parameter table have the same shape, and the executor consumes one type.
using Qwen35LoweredPrimitives = shelf::qwen_hybrid::HybridLoweredPrimitives;

struct Qwen35LowerOptions {
    std::span<const uint32_t> hidden_taps;
    const Qwen35TensorParallelContext* tp = nullptr;
    bool tp_combine_f32 = false;
    std::span<const uint32_t> override_layers;
    EffectiveRopeProfile rope_profile;
    PipelineStage stage;
    bool emit_normed_token_hidden = false;
};

Result<Qwen35LoweredPrimitives> lower_qwen35_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const Qwen35LowerOptions& options = {});

Result<Qwen35LoweredPrimitives> lower_qwen35_mtp_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const EffectiveRopeProfile& rope_profile = {},
    const shelf::weights::MatrixWeight* lm_head_override = nullptr);

}
