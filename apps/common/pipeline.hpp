// shelf apps/common — pipeline-parallel（レイヤ分割・1 ステージ = 1 デバイス）。
//
// 設計（draft_docs/shelf_validate.md §6 / §6.1）:
//   * レイヤ境界で K ステージに均等分割（余りは最後のステージへ寄せて全層を覆う）
//   * 各ステージは自分のレイヤ範囲の重みと状態（KV / conv / recurrent）だけを持つ
//   * ステージ間の受け渡しは **bf16 の生ビット**（数値を変えない）。直列 hand-off:
//     前段の `run_stage` が同期で終わってから次段へ渡す（Strata の既定と同じ）
//   * 正しさの担保は `exact(PP=1)` と `exact(PP=K)` の T1 一致
//   * head（final norm + lm_head）は最終ステージ、embedding は先頭ステージ
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <shelf/model/qwen35/step_graph.hpp>

#include "app_support.hpp"

#ifdef SHELF_HAVE_HIP
#include <shelf/backend/hip/step_executor.hpp>
#endif

namespace shelf::app {

#ifdef SHELF_HAVE_HIP

// 均等分割。`k` が層数より大きい場合はエラー（呼び手が検査する）。
// 余りは**最後のステージ**に寄せる（全層を必ず覆う）。
std::vector<model::qwen35::StageRange> even_stage_ranges(std::size_t num_layers, std::size_t k);

struct PipelineStage {
    model::qwen35::StepGraphPlan decode;                    // rows = 1
    std::unique_ptr<model::qwen35::StepGraphPlan> prefill;   // rows = chunk（不要なら null）
    std::unique_ptr<backend::hip::HipStepExecutor> exec;
    std::size_t layer_lo = 0;
    std::size_t layer_hi = 0;
};

struct Pipeline {
    std::vector<PipelineStage> stages;
    std::vector<bf16_t> handoff;   // ステージ間の hidden（再利用する 1 本のバッファ）
    std::size_t hidden_size = 0;
    std::size_t arena_bytes = 0;   // 先頭ステージの確保量（表示用）
    double last_step_ms = 0.0;

    bool is_first(std::size_t s) const { return stages[s].layer_lo == 0u; }
    bool is_final(std::size_t s) const {
        return s + 1 == stages.size();
    }
};

struct PipelineOptions {
    std::vector<int> devices;        // ステージ毎の GPU 番号（stages と同じ長さ）
    std::size_t kv_capacity = 0;
    std::size_t chunk_rows = 1;      // >1 なら prefill 用グラフも作る
    std::size_t arena_bytes = 0;     // 0 = グラフから自動
    double rope_theta = 1.0e7;
    std::size_t rope_rotary_dim = 64;
    std::uint32_t fp8_act_group = 0;
    bool verbose = false;
};

// 各ステージのプラン構築 → 重みのアップロード（そのレイヤ範囲だけ）→ 状態宣言。
Result<Pipeline> make_pipeline(const model::qwen35::TextConfig& cfg, const ShardSet& shards,
                               const Fp8ModelFormat& fmt, const PipelineOptions& opts);

Status pipeline_reset(Pipeline& p);

// 1 ステップ（n 行 = 1 or prefill チャンク）。最終ステージのみ logits を返す
// （`logits` が nullptr でも実行でき、その場合は読み戻さない）。
Status pipeline_step(Pipeline& p, const std::int32_t* tokens, std::size_t n,
                     std::uint32_t base_pos, std::vector<float>* logits);

#endif  // SHELF_HAVE_HIP

}  // namespace shelf::app
