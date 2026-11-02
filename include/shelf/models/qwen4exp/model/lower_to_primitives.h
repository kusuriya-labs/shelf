#pragma once
#include <shelf/core/status.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/model/qwen4exp_weights.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ps::qwen4exp {

struct Qwen4ExpLoweredPrimitives {
    ps::runtime::PrimitiveGraph graph;
    std::vector<ps::runtime::WeightSlot> weights;
    std::vector<ps::runtime::StaticParameterSlot> parameters;
    std::vector<std::pair<std::uint32_t, ps::runtime::ValueId>> layer_input_values;
    bool has_ple_prefix = false;
    ps::runtime::PrimitiveGraph ple_prefix_graph;
    std::vector<ps::runtime::WeightSlot> ple_prefix_weights;
    std::vector<ps::runtime::StaticParameterSlot> ple_prefix_parameters;
};

struct Qwen4ExpLowerOptions {
    std::span<const std::uint32_t> hidden_taps;
    std::span<const std::uint32_t> override_layers;
    ps::qwen35::EffectiveRopeProfile rope_profile;
    bool connect_moe = true;
    bool connect_ple = false;
    std::uint32_t stage_begin = 0;
    std::uint32_t stage_end = 0;
};

Result<Qwen4ExpLoweredPrimitives> lower_qwen4exp_to_primitives(
    const Qwen4ExpTextConfig& config,
    const Qwen4ExpModelWeights& weights,
    const Qwen4ExpLowerOptions& options = {});

}
