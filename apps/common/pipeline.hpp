// shelf apps/common — pipeline-parallel の**組み立て**（composition root）。
//
// 実行そのものは `runtime/pipeline`（エンジン側）が担う。ここは
//   * ステージ毎のグラフ生成（レイヤ範囲）
//   * 実行器の生成と重み投入（そのステージの範囲だけ）
//   * エンジンへ非所有ポインタで編成を渡す
// だけを行い、ステップ実行・hand-off・時間計測はエンジンに任せる。
//
// 設計（draft_docs/shelf_validate.md §6 / §6.1）:
//   * レイヤ境界で K ステージに均等分割（余りは最後のステージへ寄せて全層を覆う）
//   * ステージ間は bf16 の生ビット（数値を変えない）・直列 hand-off
//   * head（final norm + lm_head）は最終ステージ、embedding は先頭ステージ
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <shelf/model/qwen35/step_graph.hpp>
#include <shelf/placement/budget.hpp>
#include <shelf/runtime/pipeline/pipeline.hpp>

#include "app_support.hpp"

#ifdef SHELF_HAVE_HIP
#include <shelf/backend/hip/step_executor.hpp>
#endif

namespace shelf::app {

#ifdef SHELF_HAVE_HIP

// 均等分割（エンジン側の実装をそのまま使う）。
inline std::vector<model::qwen35::StageRange> even_stage_ranges(std::size_t num_layers,
                                                                std::size_t k) {
    const auto r = runtime::pipeline::even_stage_ranges(num_layers, k);
    std::vector<model::qwen35::StageRange> out;
    out.reserve(r.size());
    for (const auto& s : r) out.push_back(model::qwen35::StageRange{s.lo, s.hi});
    return out;
}

// 1 ステージ分の所有物（apps が持つ。エンジンはポインタだけ参照する）。
struct PipelineStage {
    model::qwen35::StepGraphPlan decode;                    // rows = 1
    std::unique_ptr<model::qwen35::StepGraphPlan> prefill;   // rows = chunk（不要なら null）
    // 連続バッチング用: rows = 2..max_sequences の decode グラフ（index = rows - 2）。
    std::vector<std::unique_ptr<model::qwen35::StepGraphPlan>> decode_batches;
    std::unique_ptr<backend::hip::HipStepExecutor> exec;
    std::size_t layer_lo = 0;
    std::size_t layer_hi = 0;
};

struct Pipeline {
    std::vector<PipelineStage> stages;            // 所有（生成順で安定）
    runtime::pipeline::Pipeline engine;           // 編成（非所有ポインタ）
    std::size_t hidden_size = 0;
    std::size_t arena_bytes = 0;                  // 先頭ステージの確保量（表示用）
    std::size_t weight_bytes = 0;                 // checkpoint 上の重みの生バイト数（表示用）
    std::size_t host_weight_bytes = 0;            // pinned host に置いた重み（T14）
    std::size_t promoted_weight_bytes = 0;        // 実行中に VRAM へ移した重み
    std::size_t host_kv_bytes = 0;                // pinned host に置いた KV（K+V）
    std::size_t max_sequences = 1;                // T15: バッチの最大スロット数
    std::size_t device_bytes = 0;               // 全ステージの device 確保合計（予算の分子）
    placement::Report budget;                   // D2 の判定結果（確保前の算術）
    double last_step_ms = 0.0;

    bool empty() const noexcept { return engine.empty(); }
};

struct PipelineOptions {
    std::vector<int> devices;        // ステージ毎の GPU 番号（stages と同じ長さ）
    std::size_t kv_capacity = 0;
    std::size_t chunk_rows = 1;      // >1 なら prefill 用グラフも作る
    // prefill の logits を最終行だけにする（lm_head の重複計算を消す）。
    // 行ごと logits を読む経路（shelf-validate の T1 比較）では false を保つ。
    bool logits_last_row_only = false;
    std::size_t arena_bytes = 0;     // 0 = グラフから自動
    // D2: 確保の前に予算を判定する（既定 on）。`device_free_bytes` が 0 なら
    // 1 台目の空きを実測する（実測できない場合は判定しない）。
    bool check_budget = true;
    std::uint64_t device_free_bytes = 0;
    // device メモリの予算（モデル込み）。ステージ毎の実行器が確保する合計への上限
    // （0 = 無制限）。超過は hipMalloc 前に insufficient_memory として止まる。
    // 寿命の重ならない出力に同じ offset を割り当てる（D4）。中間値をステップ後に
    // 読むデバッグ経路では false にする（別のテンソルが入っているため）。
    bool arena_reuse = true;
    double rope_theta = 1.0e7;
    std::size_t rope_rotary_dim = 64;
    std::uint32_t fp8_act_group = 0;
    // N01: この行数以下の decode は行ごとに decode1 で回す（0 = 常にバッチ経路）。
    std::uint32_t decode1_max_rows = 4u;
    // T14: 重みの置き場を VRAM と pinned host に分ける。
    //   offload_weights_bytes > 0 : その分だけ明示的に host へ落とす（検証用）
    //   allow_host_weights        : 予算超過時に host へ落として起動を続ける
    // 予算は `device_free_bytes`（0 なら実測）を使う。
    std::uint64_t offload_weights_bytes = 0;
    bool allow_host_weights = false;
    // KV プールを pinned host に置く（T14 の online KV 第 1 段）。
    //   kv_pinned_host : 強制 / allow_host_kv : 予算不足時に自動
    bool kv_pinned_host = false;
    // T14: pinned 正本のうち直近 N ページを VRAM にミラーする（online KV）。
    // >0 のときは KV を pinned host に置く（kv_pinned_host を自動で立てる）。
    std::size_t online_kv_pages = 0;
    bool allow_host_kv = false;
    // T16: KV のグローバルブロック数（層ごと。0 = 自動）。prefix cache 用。
    std::size_t kv_blocks = 0;
    // T15: 連続バッチングの最大スロット数（1 = 従来の直列）。
    // 2 以上なら rows = 2..N の decode グラフも作る。
    std::size_t max_sequences = 1;
    bool verbose = false;
};

// 各ステージのプラン構築 → 重みのアップロード（そのレイヤ範囲だけ）→ 状態宣言 →
// エンジンへ編成を渡す。
// 重みの実体化計画は正準カタログ（weights/catalog）を入口にする（D3）。
Result<Pipeline> make_pipeline(const model::qwen35::TextConfig& cfg, const ShardSet& shards,
                               const weights::TensorCatalog& catalog, const Fp8ModelFormat& fmt,
                               const PipelineOptions& opts);

inline Status pipeline_reset(Pipeline& p) { return runtime::pipeline::reset(p.engine); }

// T15: 1 スロット分の状態を 0 に戻す（新しい要求をスロットへ入れるとき）。
inline Status pipeline_reset_slot(Pipeline& p, std::size_t slot) {
    for (auto& st : p.stages) SHELF_RETURN_IF_ERROR(st.exec->reset_slot(slot));
    return Status::make_ok();
}

// T15: 行ごとの slot / 位置を渡すステップ（prefill は 1 リクエストのチャンク）。
inline Status pipeline_step_batch_prefill(Pipeline& p, const std::int32_t* tokens,
                                          std::size_t n,
                                          backend::StageExecutor::RowBatch rows,
                                          std::vector<float>* logits) {
    const Status st = runtime::pipeline::step_batch_prefill(p.engine, tokens, n, rows, logits);
    p.last_step_ms = p.engine.last_step_ms;
    return st;
}
inline Status pipeline_step_batch_decode(Pipeline& p, const std::int32_t* tokens,
                                         std::size_t n,
                                         backend::StageExecutor::RowBatch rows,
                                         std::vector<float>* logits) {
    const Status st = runtime::pipeline::step_batch_decode(p.engine, tokens, n, rows, logits);
    p.last_step_ms = p.engine.last_step_ms;
    return st;
}

// T16: スロットの block table を全ステージへ書き込む。
inline Status pipeline_assign_blocks(Pipeline& p, std::size_t slot,
                                     const std::uint32_t* blocks, std::size_t count,
                                     std::size_t offset) {
    for (auto& st : p.stages)
        SHELF_RETURN_IF_ERROR(st.exec->assign_kv_blocks(slot, blocks, count, offset));
    return Status::make_ok();
}
// T16: GDN 状態の退避/復元（ステージごとの連結 float 列）。
inline Result<std::vector<std::vector<float>>> pipeline_export_gdn(Pipeline& p,
                                                                   std::size_t slot) {
    std::vector<std::vector<float>> out;
    out.reserve(p.stages.size());
    for (auto& st : p.stages) {
        auto r = st.exec->export_gdn_state(slot);
        if (!r.ok()) return r.status();
        out.push_back(std::move(r.value()));
    }
    return out;
}
inline Status pipeline_import_gdn(Pipeline& p, std::size_t slot,
                                  const std::vector<std::vector<float>>& data) {
    if (data.size() != p.stages.size())
        return Status::invalid_argument("import_gdn: ステージ数が違う", __FILE__, __LINE__);
    for (std::size_t s = 0; s < p.stages.size(); ++s)
        SHELF_RETURN_IF_ERROR(p.stages[s].exec->import_gdn_state(slot, data[s]));
    return Status::make_ok();
}

// T14: pinned 正本の重みを VRAM へ移す（転送 + 再配置）。host 側は正本として残る。
// 次ステップから device の複製を読む。呼ぶのはリクエスト境界など、ステップの外。
inline Status pipeline_promote(Pipeline& p) {
    for (auto& st : p.stages) {
        const std::size_t before = st.exec->promoted_weight_bytes();
        SHELF_RETURN_IF_ERROR(st.exec->promote_host_weights());
        p.promoted_weight_bytes += st.exec->promoted_weight_bytes() - before;
    }
    return Status::make_ok();
}

// 1 ステップ（エンジンへ委譲し、計測値を持ち帰る）。
Status pipeline_step(Pipeline& p, const std::int32_t* tokens, std::size_t n,
                     std::uint32_t base_pos, std::vector<float>* logits);

#endif  // SHELF_HAVE_HIP

}  // namespace shelf::app
