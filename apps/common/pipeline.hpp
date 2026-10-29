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
    double last_step_ms = 0.0;

    bool empty() const noexcept { return engine.empty(); }
};

struct PipelineOptions {
    std::vector<int> devices;        // ステージ毎の GPU 番号（stages と同じ長さ）
    std::size_t kv_capacity = 0;
    std::size_t chunk_rows = 1;      // >1 なら prefill 用グラフも作る
    std::size_t arena_bytes = 0;     // 0 = グラフから自動
    // 寿命の重ならない出力に同じ offset を割り当てる（D4）。中間値をステップ後に
    // 読むデバッグ経路では false にする（別のテンソルが入っているため）。
    bool arena_reuse = true;
    double rope_theta = 1.0e7;
    std::size_t rope_rotary_dim = 64;
    std::uint32_t fp8_act_group = 0;
    bool verbose = false;
};

// 各ステージのプラン構築 → 重みのアップロード（そのレイヤ範囲だけ）→ 状態宣言 →
// エンジンへ編成を渡す。
// 重みの実体化計画は正準カタログ（weights/catalog）を入口にする（D3）。
Result<Pipeline> make_pipeline(const model::qwen35::TextConfig& cfg, const ShardSet& shards,
                               const weights::TensorCatalog& catalog, const Fp8ModelFormat& fmt,
                               const PipelineOptions& opts);

inline Status pipeline_reset(Pipeline& p) { return runtime::pipeline::reset(p.engine); }

// 1 ステップ（エンジンへ委譲し、計測値を持ち帰る）。
Status pipeline_step(Pipeline& p, const std::int32_t* tokens, std::size_t n,
                     std::uint32_t base_pos, std::vector<float>* logits);

#endif  // SHELF_HAVE_HIP

}  // namespace shelf::app
