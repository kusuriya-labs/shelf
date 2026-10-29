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
#include <shelf/quant/fp8_w8a8.hpp>
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
// f32 で読む（bf16 の checkpoint は変換する）。FP8 は対象外（明示的に unsupported）。
Result<std::vector<float>> read_f32(const std::map<std::string, artifact::safetensors::Reader*>& where,
                                    const std::string& name);
// F8_E4M3 コードをそのまま（行優先）読む。GEMM が読む断片順へは preshuffle する。
Result<std::vector<std::uint8_t>> read_u8(
    const std::map<std::string, artifact::safetensors::Reader*>& where, const std::string& name);

// 重みの種類を**コンテナの dtype**で決める（名前の推測はしない）。
enum class WeightKind : std::uint8_t {
    Bf16,        // bf16 のまま upload
    F32,         // f32 で読む（実行器側で bf16 化）
    Fp8W8A8,     // F8_E4M3 + 対応するスケールテンソル（粒度は config が正）
    Missing,     // checkpoint に無い
};
WeightKind classify(const std::map<std::string, artifact::safetensors::Reader*>& where,
                    const std::string& name);

// checkpoint の `quantization_config` から FP8 の形を読み出す（**宣言が正**。
// 実装側で粒度を決め打ちしない）。
//   * `quant_method: compressed-tensors`（ModelOpt 系）:
//       weights.strategy = channel / block（block_structure）、
//       input_activations.strategy = token / group（group_size）
//       スケール名は `<base>.weight_scale`
//   * `quant_method: fp8`（HF blockwise 系）:
//       weight_block_size = [128,128]、activation_scheme = dynamic -> 1×128
//       スケール名は `<base>.weight_scale_inv`（**実乗数**（dequant = code × inv））
struct Fp8ModelFormat {
    bool quantized = false;
    shelf::quant::fp8::WeightScaleGranularity weight =
        shelf::quant::fp8::WeightScaleGranularity::PerChannel;
    std::uint32_t act_group = 0u;              // 0 = 行全体、128 = K を 128 単位
    std::string scale_suffix = "_scale";       // weight_scale / weight_scale_inv
    std::string declared;                      // 表示用の要約（監査証跡）
};
Result<Fp8ModelFormat> read_fp8_format(const std::string& config_json_text);

// FP8 の重み（E4M3 断片順コード + f32 スケール）を組み立てる。
// 断片順への並べ替えは**無損 relayout**（値は変えない）。
// スケールテンソル名と粒度は `Fp8ModelFormat`（= config.json の宣言）から来る。
// 宣言と実データ（要素数）が合わない場合は推測せずエラーにする。
struct Fp8WeightPayload {
    std::vector<std::uint8_t> codes_tiled;   // quant/fp8 の tiled_bytes 分
    std::vector<float> scales;               // 宣言された粒度に応じた個数
    std::uint32_t features = 0u;             // shape[0]
    std::uint32_t k_padded = 0u;             // pad_k(shape[1])
    bool weight_block128 = false;            // true: 128×128 ブロック スケール
};
Result<Fp8WeightPayload> build_fp8_weight(
    const std::map<std::string, artifact::safetensors::Reader*>& where, const std::string& name,
    const std::vector<std::size_t>& shape, const Fp8ModelFormat& fmt);


// plan.weights を全部実行器へアップロードし、状態を宣告する。
// upload_weight_bf16 / upload_weight / upload_weight_fp8 / declare_states を持つ
// 実行器向け（現在は HipStepExecutor。HOST 実装が同じ API を持てばそのまま使える）。
template <class Exec>
Status upload_plan(Exec& ex, const model::qwen35::StepGraphPlan& plan, const ShardSet& shards,
                   const Fp8ModelFormat& fmt) {
    for (const auto& w : plan.weights) {
        switch (classify(shards.where, w.checkpoint_name)) {
        case WeightKind::Fp8W8A8: {
            auto fp = build_fp8_weight(shards.where, w.checkpoint_name, w.shape, fmt);
            if (!fp.ok()) return fp.status();
            SHELF_RETURN_IF_ERROR(ex.upload_weight_fp8(
                w.id, fp.value().codes_tiled.data(), fp.value().codes_tiled.size(),
                fp.value().scales.data(), fp.value().features, fp.value().k_padded,
                fp.value().weight_block128));
            continue;
        }
        case WeightKind::Bf16: {
            auto tb = read_bf16(shards.where, w.checkpoint_name);
            if (!tb.ok()) return tb.status();
            if (tb.value().size() != numel(w.shape))
                return Status::invalid_state("shape mismatch: " + w.checkpoint_name,
                                             __FILE__, __LINE__);
            SHELF_RETURN_IF_ERROR(
                ex.upload_weight_bf16(w.id, tb.value().data(), tb.value().size()));
            continue;
        }
        case WeightKind::F32: {
            auto t = read_f32(shards.where, w.checkpoint_name);
            if (!t.ok()) return t.status();
            if (t.value().size() != numel(w.shape))
                return Status::invalid_state("shape mismatch: " + w.checkpoint_name,
                                             __FILE__, __LINE__);
            SHELF_RETURN_IF_ERROR(ex.upload_weight(w.id, t.value()));
            continue;
        }
        default:
            return Status::not_found("weight not in shards: " + w.checkpoint_name,
                                     __FILE__, __LINE__);
        }
    }
    return ex.declare_states(plan.graph);
}

}  // namespace shelf::app
