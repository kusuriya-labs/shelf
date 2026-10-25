// shelf model/qwen35 — single decode-step Model Graph lowering.
//
// Builds ONE primitive Graph (T=1 step) from the TextConfig: prefill and
// decode both replay this graph token by token on the CPU reference path
// (slow-but-exact; fused prefill kernels come with the HIP backend).
//
// The plan object tells the driver exactly which checkpoint tensor names map
// to which weight ids, and which zero-initialized states to allocate.
#pragma once

#include <string>
#include <vector>

#include <shelf/ir/graph.hpp>
#include <shelf/model/qwen35/text_config.hpp>

namespace shelf::model::qwen35 {

struct WeightRef {
    TensorId id;
    std::string checkpoint_name;
    std::vector<std::size_t> shape;   // IR contract shape; driver checks element counts
};

struct StateRef {
    ir::StateId id;
    ir::StateKind kind;
    std::vector<std::size_t> shape;
};

struct StepGraphPlan {
    ir::Graph graph;

    ir::ValueId tokens;   // i32 [1]
    ir::ValueId cos;      // f32 [rotary_dim]
    ir::ValueId sin;      // f32 [rotary_dim]
    ir::ValueId logits;   // f32 [1, vocab]

    std::vector<WeightRef> weights;   // load exactly these
    std::vector<StateRef> states;     // zero-initialize exactly these

    std::size_t kv_capacity = 0;
    bool tied_embeddings = false;     // lm_head reuses embed_tokens weight id
};

// kv_capacity: max tokens resident in each full-attention KV state.
Result<StepGraphPlan> build_step_graph(const TextConfig& cfg, std::size_t kv_capacity);

} // namespace shelf::model::qwen35
