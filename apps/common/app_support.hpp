// shelf apps/common — 実行ファイル間で共有する「モデルの読み方と材料の渡し方」。
//
// 位置づけ: composition root の共通部分。D1（起動）と D3（ロード）の移設が済んだ
// ので、ここに残るのは
//   * 供給元アダプタ（`ShardSource`: artifact の reader → `runtime::load`）
//   * 正準カタログの構築と実体化計画（`build_catalog` / `plan_materialization`）
//   * 実行器への受け渡し（`upload_plan` / `load_host_plan`）
//   * 起動 probe の実体（`probe_devices`。HIP ビルドのみ）
// だけ。HIP の型は公開面に出さない（`SHELF_HAVE_HIP` の内側だけ）。
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
#ifdef SHELF_HAVE_HIP
#include <shelf/runtime/boot/probe.hpp>
#endif
#include <shelf/runtime/load/materialize.hpp>
#include <shelf/weights/catalog/tensor_catalog.hpp>
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

// 重みの種類・FP8 の宣言・実体化の型は `src/runtime/load`（D3）が正本。
// アプリ側は別名で受ける（呼び出し側の記述を変えない）。
using WeightKind = runtime::load::WeightKind;
using Fp8ModelFormat = runtime::load::Fp8ModelFormat;
using Fp8WeightPayload = runtime::load::MaterializedWeight;
WeightKind classify(const std::map<std::string, artifact::safetensors::Reader*>& where,
                    const std::string& name);

// checkpoint の `quantization_config` の解析は runtime/load が持つ（D3）。
// ここは従来の名前で呼べるようにする薄い入口。
Result<Fp8ModelFormat> read_fp8_format(const std::string& config_json_text);

// FP8 の重みの実体化は runtime/load（読取 → 検証 → 無損 relayout）。
Result<Fp8WeightPayload> build_fp8_weight(
    const std::map<std::string, artifact::safetensors::Reader*>& where, const std::string& name,
    const std::vector<std::size_t>& shape, const Fp8ModelFormat& fmt);


// 起動時の probe（D1）。HIP ビルドで `want_device_execution` なら実デバイスを列挙し、
// 失敗・非対応なら空のまま返す（可否の判断は runtime/boot の evaluate が行う）。
#ifdef SHELF_HAVE_HIP
shelf::runtime::boot::Probe probe_devices(bool want_device_execution);
#endif

// `ShardSet` を runtime/load の供給元として見せるアダプタ（apps が持つ）。
class ShardSource : public runtime::load::TensorSource {
public:
    explicit ShardSource(const std::map<std::string, artifact::safetensors::Reader*>& where)
        : where_(where) {}
    bool has(const std::string& name) const override { return where_.count(name) != 0u; }
    Result<runtime::load::TensorBlob> read(const std::string& name) const override;

private:
    const std::map<std::string, artifact::safetensors::Reader*>& where_;
};

// ---- D3: 実体化計画の入口（weights/catalog）----
// 重みの実体化は「どのテンソルを、どの形・dtype で読むか」を先に決めてから行う。
// その決定を正準カタログ（weights/catalog）で受ける: 名前 → 安定 id・形状・dtype。
// ここで弾けば、デバイスに触る前に「checkpoint に無い / 形が違う」が分かる。

// shard 群の生カタログから正準カタログを作る（名前の重複・形状・dtype を検証）。
Result<weights::TensorCatalog> build_catalog(const ShardSet& shards);

// グラフの重み宣言 → 実体化要求（カタログで検証済み）。
struct MaterializationPlan {
    std::vector<runtime::load::WeightRequest> requests;
    std::size_t bf16 = 0, f32 = 0, fp8 = 0;
    std::size_t bytes = 0;   // checkpoint 上の生バイト数（確保量の見積りに使う）
    std::string summary;     // 例: "bf16 427 / fp8 0 / 0 B"
};
Result<MaterializationPlan> plan_materialization(const weights::TensorCatalog& catalog,
                                                 const model::qwen35::StepGraphPlan& plan);

// ホスト（CPU 参照経路）向けの実体化。重みを f32 で読み、状態を 0 で初期化する。
// GPU 経路の `upload_plan` と同じ入口（正準カタログ → runtime/load）を通すので、
// 「checkpoint に無い / 形が違う」はデバイスに触る前に分かる。
template <class HostStorageT>
Status load_host_plan(HostStorageT& storage, const model::qwen35::StepGraphPlan& plan,
                      const weights::TensorCatalog& catalog, const ShardSet& shards,
                      const Fp8ModelFormat& fmt) {
    (void)fmt;  // CPU 参照は fp8 を読まない（runtime/load が明示エラーにする）
    auto mp = plan_materialization(catalog, plan);
    if (!mp.ok()) return mp.status();
    ShardSource src(shards.where);
    for (std::size_t i = 0; i < plan.weights.size(); ++i) {
        const auto& w = plan.weights[i];
        auto t = runtime::load::read_f32(src, mp.value().requests[i].checkpoint_name);
        if (!t.ok())
            return Status::invalid_state(w.checkpoint_name + ": " + t.status().message(),
                                         __FILE__, __LINE__);
        SHELF_RETURN_IF_ERROR(storage.set_weight(w.id, t.release()));
    }
    for (const auto& s : plan.states) {
        std::size_t n = 1;
        for (const auto d : s.shape) n *= d;
        SHELF_RETURN_IF_ERROR(storage.init_state(s.id, std::vector<float>(n, 0.0f)));
    }
    return Status::make_ok();
}

// plan.weights を全部実行器へアップロードし、状態を宣告する。
// upload_weight_bf16 / upload_weight / upload_weight_fp8 / declare_states を持つ
// 実行器向け（現在は HipStepExecutor。HOST 実装が同じ API を持てばそのまま使える）。
// 実体化（読取 → 検証 → 無損 relayout）は runtime/load が行い、ここは
// 「材料を要求 → 実行器へ渡す」だけを担う（composition root の仕事）。
template <class Exec>
Status upload_plan(Exec& ex, const model::qwen35::StepGraphPlan& plan,
                   const weights::TensorCatalog& catalog, const ShardSet& shards,
                   const Fp8ModelFormat& fmt, MaterializationPlan* report = nullptr) {
    auto mp = plan_materialization(catalog, plan);
    if (!mp.ok()) return mp.status();
    if (report) *report = mp.value();
    ShardSource src(shards.where);
    for (std::size_t i = 0; i < plan.weights.size(); ++i) {
        const auto& w = plan.weights[i];
        const auto& req = mp.value().requests[i];
        auto m = runtime::load::materialize(src, req, fmt);
        if (!m.ok()) return m.status();
        switch (m.value().kind) {
        case WeightKind::Bf16:
            SHELF_RETURN_IF_ERROR(ex.upload_weight_bf16(w.id, m.value().bf16.data(),
                                                        m.value().bf16.size()));
            break;
        case WeightKind::F32:
            SHELF_RETURN_IF_ERROR(ex.upload_weight(w.id, m.value().f32));
            break;
        case WeightKind::Fp8W8A8:
            SHELF_RETURN_IF_ERROR(ex.upload_weight_fp8(
                w.id, m.value().codes.data(), m.value().codes.size(), m.value().scales.data(),
                m.value().features, m.value().k_padded, m.value().weight_block128));
            break;
        default:
            return Status::not_found("weight not in shards: " + w.checkpoint_name, __FILE__,
                                     __LINE__);
        }
    }
    return ex.declare_states(plan.graph);
}

}  // namespace shelf::app
