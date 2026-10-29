// shelf runtime/pipeline — pipeline-parallel の編成（エンジン側）。
//
// 役割: レイヤ分割された K ステージを**順に回す**ことだけを担う。
//       ステージの生成（グラフ・実行器・重み投入）は composition root（apps）が行い、
//       ここには**非所有のポインタ**で渡す（P2: 境界は型で作る）。
//
// 契約（draft_docs/shelf_validate.md §6）:
//   * 先頭ステージが embedding、最終ステージが head（final norm + lm_head）
//   * ステージ間は bf16 の生ビットを host 経由で渡す（直列 hand-off。数値を変えない）
//   * 正しさの門は `exact(PP=1)` と `exact(PP=K)` の T1 一致
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <shelf/backend/common/stage_executor.hpp>
#include <shelf/ir/graph.hpp>
#include <shelf/status.hpp>

namespace shelf::runtime::pipeline {

// ステージのグラフ（エンジンは ir だけを見る。モデル固有の型は知らない）。
struct StageGraphs {
    const ir::Graph* decode = nullptr;       // rows = 1（必須）
    const ir::Graph* prefill = nullptr;      // rows = chunk（任意）
    ir::ValueId hidden_out_decode;           // 非最終ステージの出力（decode）
    ir::ValueId hidden_out_prefill;          // 同（prefill。無ければ invalid）
};

struct Stage {
    StageGraphs graphs;
    backend::StageExecutor* exec = nullptr;                  // 非所有
    std::size_t layer_lo = 0;
    std::size_t layer_hi = 0;
};

struct Pipeline {
    std::vector<Stage> stages;
    std::vector<bf16_t> handoff;   // ステージ間の hidden（1 本を再利用）
    std::size_t hidden_size = 0;
    std::size_t arena_bytes = 0;   // 先頭ステージの確保量（表示用）
    double last_step_ms = 0.0;

    bool empty() const noexcept { return stages.empty(); }
};

struct StageRange {
    std::size_t lo = 0;
    std::size_t hi = 0;
};

// 均等分割。余りは**最後のステージ**に寄せて全層を覆う。k == 0 / k > layers は空。
std::vector<StageRange> even_stage_ranges(std::size_t num_layers, std::size_t k);

// hand-off バッファを確保する（hidden_size と chunk_rows を設定済みの状態で呼ぶ）。
Status prepare(Pipeline& p, std::size_t chunk_rows);

Status reset(Pipeline& p);

// 1 ステップ。最終ステージのみ logits を返す（nullptr なら読み戻さない）。
Status step(Pipeline& p, const std::int32_t* tokens, std::size_t n, std::uint32_t base_pos,
            std::vector<float>* logits);

}  // namespace shelf::runtime::pipeline
