// shelf backend/common — ステージ実行の抽象（pipeline-parallel の境界）。
//
// runtime はこのインターフェース越しにステージを編成する（HIP 型は見ない）。
// 実装は backend 側（HIP なら HipStepExecutor）が持つ。
//
// 契約:
//   * 先頭ステージは `tokens` を受け取り embedding から始める
//   * 以降のステージは `hidden`（前段の `fetch_value_bf16` の結果）を受け取る
//   * ステージ間の受け渡しは **bf16 の生ビット**（数値を変えない）
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <shelf/foundation/numeric.hpp>
#include <shelf/ir/graph.hpp>
#include <shelf/status.hpp>

namespace shelf::backend {

class StageExecutor {
public:
    StageExecutor() = default;
    StageExecutor(const StageExecutor&) = delete;
    StageExecutor& operator=(const StageExecutor&) = delete;
    virtual ~StageExecutor() = default;

    // 1 ステップ。n 行はグラフのバッチと一致させること。
    virtual Status run_stage(const ir::Graph& g, const std::int32_t* tokens,
                             const bf16_t* hidden, std::size_t n,
                             std::uint32_t base_pos) = 0;

    // 行ごとの状態スロットと位置（T15 の連続バッチング）。slots[i] は状態
    // （KV・conv・recurrent）のスロット、positions[i] はその行の絶対位置。
    struct RowBatch {
        const std::uint32_t* slots = nullptr;
        const std::uint32_t* positions = nullptr;
    };

    // 複数スロットの 1 ステップ。既定は未対応（参照実装は run_stage だけ）。
    virtual Status run_stage_batch(const ir::Graph& g, const std::int32_t* tokens,
                                   const bf16_t* hidden, std::size_t n,
                                   const RowBatch& rows) {
        (void)g;
        (void)tokens;
        (void)hidden;
        (void)n;
        (void)rows;
        return Status::unsupported("run_stage_batch is not supported by this executor",
                                   __FILE__, __LINE__);
    }

    // 非最終ステージの出力（bf16 のまま。f32 値は受け付けない）。
    virtual Result<std::vector<bf16_t>> fetch_value_bf16(const ir::Graph& g,
                                                         ir::ValueId id) = 0;

    // 最終ステージの logits。
    virtual Result<std::vector<float>> fetch_logits(const ir::Graph& g) = 0;

    virtual Status reset_states() = 0;

    // 確保済みのアリーナ量（表示・予算用）。
    virtual std::size_t arena_bytes() const = 0;
    // 直近ステップの所要時間 [ms]。
    virtual double last_step_ms() const = 0;
};

}  // namespace shelf::backend
