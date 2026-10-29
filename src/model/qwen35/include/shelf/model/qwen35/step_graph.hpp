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

// pipeline-parallel のレイヤ範囲。`layer_hi == 0` は「全層の末尾」を意味する。
struct StageRange {
    std::size_t layer_lo = 0;
    std::size_t layer_hi = 0;
};

struct StepGraphPlan {
    ir::Graph graph;

    // 先頭ステージのみ有効（i32、行数ぶんのトークン id）。
    ir::ValueId tokens;
    // 非先頭ステージの入力（bf16 [rows, hidden]。rows=1 のグラフは [hidden]）。
    // 前段の `hidden_out` のビットをそのまま流し込む（PP は数値を変えない）。
    ir::ValueId hidden_in;
    // 非最終ステージの出力（bf16、次段の `hidden_in` に渡す）。
    ir::ValueId hidden_out;
    ir::ValueId cos;      // f32 [rotary_dim]（CPU 参照経路が使う。HIP は無視）
    ir::ValueId sin;      // f32 [rotary_dim]
    ir::ValueId logits;   // f32 [rows, vocab]（最終ステージのみ有効）

    // このプランが担当するレイヤ範囲（[layer_lo, layer_hi)）。
    std::size_t layer_lo = 0;
    std::size_t layer_hi = 0;
    bool is_first = true;   // embedding を持つ
    bool is_final = true;   // final norm + lm_head を持つ

    std::vector<WeightRef> weights;   // load exactly these
    std::vector<StateRef> states;     // zero-initialize exactly these

    std::size_t kv_capacity = 0;
    bool tied_embeddings = false;     // lm_head reuses embed_tokens weight id
};

// kv_capacity: max tokens resident in each full-attention KV state.
// rows = activation batch leading dim (1 = decode step; >1 = prefill chunk).
// range: pipeline-parallel のレイヤ範囲（既定は全層 = 単一ステージ）。
//   範囲を絞ると、そのレイヤ範囲の重み・状態（KV / conv / recurrent）だけが
//   プランに入る（= ステージ分割で自然に部分ロードになる）。
Result<StepGraphPlan> build_step_graph(const TextConfig& cfg, std::size_t kv_capacity,
                                       std::size_t rows = 1, StageRange range = {});

} // namespace shelf::model::qwen35
