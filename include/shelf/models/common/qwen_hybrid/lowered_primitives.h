#pragma once

#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace shelf {
namespace qwen_hybrid {

// What a model family's lowering produces: the primitive graph, its weight table in the
// order the graph indexes it, its static parameters, and the layer input values exposed for
// hidden taps. The shape is the same whatever the FFN is, so the executor consumes one type
// for every family.
struct HybridLoweredPrimitives {
    ::shelf::runtime::PrimitiveGraph graph;
    std::vector<::shelf::runtime::WeightSlot> weights;
    std::vector<::shelf::runtime::StaticParameterSlot> parameters;
    std::vector<std::pair<std::uint32_t, ::shelf::runtime::ValueId>> layer_input_values;
};

}
}
