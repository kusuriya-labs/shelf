// shelf quant — FP8 の**粗いスケール**形式（重み per-channel / 活性化 per-token）の契約。
//
// どちらの形式も W8A8（重み 8 bit・活性化 8 bit）で、行列命令も同じ。違いは
// **スケールの粒度だけ**なので、ファイル名は粒度で分ける:
//   * ここ      : 重み = 出力チャネルごと、活性化 = トークン（行）ごと（1 次元）
//   * fp8_finegrained.hpp : 重み = 128×128 ブロック、活性化 = K128 グループ（2 次元）
//
// checkpoint 側の正（`config.json` の `quantization_config`）:
//   * weights     : type float / num_bits 8 / strategy **channel** / symmetric / dynamic false
//   * activations : type float / num_bits 8 / strategy **token**   / symmetric / dynamic true
//   * 格納: `<name>.weight` が `F8_E4M3`、`<name>.weight_scale` が `BF16 [out, 1]`
//
// この形のスケール粒度は固定:
//   * 重みスケールは**出力チャネル単位**（K に依存しない。1 行に 1 個）
//   * 活性化スケールは**トークン（行）単位**で実行時に計算する（`amax / kE4m3Max`）
//     checkpoint に `input_scale` は存在しない（dynamic のため）
//
// 128(K)×128(N) のブロック状重みスケールと K128 グループの活性化スケールを使う
// **細かいスケール**形式は fp8_finegrained.hpp（`quant::fp8::fine`）。
//
// 数値コーデックと断片順レイアウトは fp8_codec.hpp（共通部）。
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/quant/fp8_codec.hpp>

namespace shelf::quant::fp8::perchannel {

// 活性化スケールは行（トークン）全体で 1 個。
inline constexpr std::uint32_t kActGroupPerToken = 0u;

// 重みスケールの要素数（= 出力特征数。K 方向は 1）。
constexpr std::size_t weight_scale_count(std::uint32_t features) noexcept {
    return features;
}

// 活性化スケールの要素数（= 行数）。
constexpr std::size_t activation_scale_count(std::uint32_t rows) noexcept {
    return rows;
}

}  // namespace shelf::quant::fp8::perchannel
