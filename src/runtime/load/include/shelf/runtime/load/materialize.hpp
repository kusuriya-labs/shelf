// shelf runtime/load — 検証と実体化（D3）。
//
// 契約（P5）: **ロード時に許す変換は無損 relayout のみ**。有損な dequant/量子化は
// ホットパス（GEMM 内）で行う。ここが返すのは次の 3 形態だけ:
//   * bf16 のまま（checkpoint の生ビットをコピー）
//   * f32（CPU 参照経路用。bf16 からは無損で拡張、fp8 は**明示エラー**）
//   * FP8 W8A8（E4M3 コードを断片順へ relayout ＋ スケールを f32 化）
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <shelf/foundation/numeric.hpp>
#include <shelf/quant/fp8_perchannel.hpp>
#include <shelf/quant/fp8_finegrained.hpp>
#include <shelf/runtime/load/tensor_source.hpp>
#include <shelf/status.hpp>

namespace shelf::runtime::load {

// 重みの種類は**コンテナの dtype**で決める（名前の推測はしない）。
enum class WeightKind : std::uint8_t {
    Bf16,
    F32,
    Fp8W8A8,
    // MXFP4（`quant_method: quark`）: 重みは E2M1 ニブル（断片順）、スケールは E8M0。
    // E8M0 は**f32 へ変換しない**（カーネルが `uint8_t` のまま食うため）。
    Mxfp4,
    Missing,
};


// checkpoint の `quantization_config`（FP8 の宣言）。**宣言が正**で、実装側で
// 粒度を決め打ちしない。解析は apps ではなくここ（ロードの契約）が持つ。
struct Fp8ModelFormat {
    bool quantized = false;
    // MXFP4（`quant_method: quark`・fp4 / group 32 / E8M0 スケール）。
    // `weight` / `act_group` は FP8 用なので、MXFP4 のときは `mxfp4` を見る。
    bool mxfp4 = false;
    std::uint32_t mxfp4_group = 32u;           // K 方向のブロック長
    shelf::quant::fp8::fine::WeightScaleGranularity weight =
        shelf::quant::fp8::fine::WeightScaleGranularity::PerChannel;
    std::uint32_t act_group = 0u;              // 0 = 行全体、128 = K を 128 単位
    std::string scale_suffix = "_scale";       // weight_scale / weight_scale_inv
    std::string declared;                      // 表示用の要約（監査証跡）
};
Result<Fp8ModelFormat> read_fp8_format(const std::string& config_json_text);

// 重みの種類は**コンテナの dtype**と宣言（`fmt`）で決める（名前の推測はしない）。
WeightKind classify(const TensorSource& src, const std::string& name,
                    const Fp8ModelFormat& fmt);


// 実体化の要求（グラフの重み 1 件分。id は呼び手が解釈する）。
struct WeightRequest {
    std::string checkpoint_name;
    std::vector<std::size_t> shape;   // 論理形状 [out, K]
};

// 実体化の結果（すべて所有権あり）。
struct MaterializedWeight {
    WeightKind kind = WeightKind::Missing;
    std::vector<bf16_t> bf16;          // kind == Bf16
    std::vector<float> f32;            // kind == F32
    std::vector<std::uint8_t> codes;   // kind == Fp8W8A8（断片順・tiled_bytes 分）
    std::vector<float> scales;         // kind == Fp8W8A8
    std::uint32_t features = 0u;
    std::uint32_t k_padded = 0u;
    bool weight_block128 = false;
    std::vector<std::uint8_t> mx_scales;   // kind == Mxfp4（E8M0・行優先 [out, K/32]）
};

// テンソルごとに 30 MB 級の vector を確保すると、29 GB のロードでページフォルトが
// 支配して 2.4 GB/s まで落ちる（使い回すと 18 GB/s）。呼び手が持ち回る作業領域。
struct MaterializeBuffers {
    TensorBlob src_blob;                // 供給元からの読み取り（容量を使い回す）
    std::vector<std::uint8_t> src;      // 読み取り先
    std::vector<std::uint8_t> codes;    // 断片順の出力先
    std::vector<float> scales;
    std::vector<bf16_t> bf16;
    std::vector<float> f32;
};

// 読取 → 検証 → 実体化。要素数の不一致や未対応 dtype は推測せず失敗させる。
// `bufs` を渡すと作業領域を使い回す（返る `MaterializedWeight` はその中身を
// move して受け取る。呼び手は使い終わったら戻すとよい）。
Result<MaterializedWeight> materialize(const TensorSource& src, const WeightRequest& req,
                                       const Fp8ModelFormat& fmt,
                                       MaterializeBuffers* bufs = nullptr);

// CPU 参照経路用: f32 として読む（bf16 は無損拡張、fp8 は明示エラー）。
Result<std::vector<float>> read_f32(const TensorSource& src, const std::string& name);

// bf16 のまま読む（checkpoint のビットをそのまま）。
Result<std::vector<bf16_t>> read_bf16(const TensorSource& src, const std::string& name);

// 8 ビットのペイロード（fp8 コード / u8）をそのまま読む。
Result<std::vector<std::uint8_t>> read_u8(const TensorSource& src, const std::string& name);

}  // namespace shelf::runtime::load
