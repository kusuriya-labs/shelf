// shelf apps/common — shelf-run と shelf-bench が共有するモデル読取と重みアップロード。
//
// 位置づけ: apps/ 配下の内部共有ヘルパ。D3（runtime/load/ への分離）の対象。
// 依存は model / artifact / foundation / status のみ。HIP を include しない。
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <shelf/artifact/safetensors/reader.hpp>
#include <shelf/foundation/dtype.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/model/qwen35/step_graph.hpp>
#include <shelf/status.hpp>

namespace shelf::app {

using shelf::bf16_t;

// ファイル全体を文字列で読む（config.json / index.json 用）。
std::string read_file_str(const std::string& path);

// 要素数の積。
std::size_t numel(const std::vector<std::size_t>& shape);

// model.safetensors.index.json から shard を開き、名前 -> reader の対応を作る。
// ポインタの安定性のため reader は unique_ptr で所有する（返却時のムーブでも不変）。
struct ShardSet {
    std::vector<std::unique_ptr<artifact::safetensors::Reader>> readers;
    std::map<std::string, artifact::safetensors::Reader*> where;
};
Status open_shards(const std::string& model_dir, ShardSet* out);

// checkpoint-native bf16 のまま読む（変換なし）。
Result<std::vector<bf16_t>> read_bf16(const std::map<std::string, artifact::safetensors::Reader*>& where,
                                      const std::string& name);
// f32 で読む（bf16 の checkpoint は変換する）。
Result<std::vector<float>> read_f32(const std::map<std::string, artifact::safetensors::Reader*>& where,
                                    const std::string& name);

// plan.weights を全部実行器へアップロードし、状態を宣告する。
// upload_weight_bf16 / upload_weight / declare_states を持つ実行器向け
// （現在は HipStepExecutor。HOST 実装が同じ 3 API を持てばそのまま使える）。
template <class Exec>
Status upload_plan(Exec& ex, const model::qwen35::StepGraphPlan& plan, const ShardSet& shards) {
    for (const auto& w : plan.weights) {
        auto tb = read_bf16(shards.where, w.checkpoint_name);
        if (tb.ok()) {
            if (tb.value().size() != numel(w.shape))
                return Status::invalid_state("shape mismatch: " + w.checkpoint_name,
                                             __FILE__, __LINE__);
            SHELF_RETURN_IF_ERROR(
                ex.upload_weight_bf16(w.id, tb.value().data(), tb.value().size()));
            continue;
        }
        auto t = read_f32(shards.where, w.checkpoint_name);
        if (!t.ok()) return t.status();
        if (t.value().size() != numel(w.shape))
            return Status::invalid_state("shape mismatch: " + w.checkpoint_name,
                                         __FILE__, __LINE__);
        SHELF_RETURN_IF_ERROR(ex.upload_weight(w.id, t.value()));
    }
    return ex.declare_states(plan.graph);
}

}  // namespace shelf::app
