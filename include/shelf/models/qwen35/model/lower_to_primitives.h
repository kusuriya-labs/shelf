#pragma once
#include <shelf/core/status.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/runtime/program/program.h>

#include <shelf/runtime/graph/primitive_graph.h>

#include <cstdint>
#include <span>

namespace shelf::qwen35 {

struct Qwen35LoweredPrimitives {
    shelf::runtime::PrimitiveGraph graph;
    std::vector<shelf::runtime::WeightSlot> weights;
    std::vector<shelf::runtime::StaticParameterSlot> parameters;
    std::vector<std::pair<std::uint32_t, shelf::runtime::ValueId>> layer_input_values;
};

struct Qwen35LowerOptions {
    std::span<const uint32_t> hidden_taps;
    const Qwen35TensorParallelContext* tp = nullptr;
    bool tp_combine_f32 = false;
    std::span<const uint32_t> override_layers;
    EffectiveRopeProfile rope_profile;
};

Result<Qwen35LoweredPrimitives> lower_qwen35_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const Qwen35LowerOptions& options = {});

Result<Qwen35LoweredPrimitives> lower_qwen35_mtp_to_primitives(
    const Qwen35TextConfig& config,
    const Qwen35ModelWeights& weights,
    const EffectiveRopeProfile& rope_profile = {});

}
