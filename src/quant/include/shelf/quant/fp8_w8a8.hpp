// shelf quant — FP8 W8A8（compressed-tensors / float-quantized）の契約とホスト側変換。
//
// checkpoint 側の正（`config.json` の `quantization_config`）:
//   * weights     : type float / num_bits 8 / strategy **channel** / symmetric / dynamic false
//   * activations : type float / num_bits 8 / strategy **token**   / symmetric / dynamic true
//   * 格納: `<name>.weight` が `F8_E4M3`、`<name>.weight_scale` が `BF16 [out, 1]`
//
// 実装側の契約（本ヘッダが定義するもの）:
//   1. 数値は `code(1 バイト) × scale(f32)`。実数値 = code * scale。
//   2. 重みスケールは**出力チャネル単位**（K に依存しない。1 行に 1 個）。
//   3. 活性化スケールは**トークン（行）単位**で実行時に計算する（`amax / kE4m3Max`）。
//      checkpoint に `input_scale` は存在しない（dynamic のため）。
//   4. 行列命令（gfx1201 の FP8 WMMA）が読むコード配置は**断片順**（下記 tiled_*）。
//      行優先 → 断片順の再配置は要素の値を変えない**無損 relayout** なので、
//      「ロード時の変換は無損な relayout のみ」という契約（directory_design P5）に従う。
//      有損の dequant は行わない（行わない限り、どこでやってもよい）。
//
// 依存: foundation のみ（境界ルール上、quant は model / backend / runtime / hip を
// include しない。デバイス側の e4m3 変換命令は backend/hip の codec にある）。
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/status.hpp>

namespace shelf::quant::fp8 {

// E4M3 の有限最大値（S1E4M3: 指数 4 ビット・bias 7・仮数 3 ビット、无限大なし、
// 0x7F が NaN）。動的活性化スケーリングの基準はこの最大値。
inline constexpr float kE4m3Max = 448.0f;

// code -> f32（この変換は厳密。E4M3 の全有限値は f32 で厳密に表せる）。
float e4m3_decode(std::uint8_t code) noexcept;

// f32 -> code（round-to-nearest-even、飽和 = satfinite で ±448 に clamp、
// NaN は 0x7F、無限大は ±448 へ丸める）。
std::uint8_t e4m3_encode(float x) noexcept;

// amax（対象範囲の最大絶対値）からスケールへ。amax が 0 の範囲はスケール 0
// （その範囲はすべて 0 コードになる）。host（relayout・テスト）と device
// （量子化カーネル）で**同じ式を共有**する（ずれると検証が通らなくなる）。
// hip のヘッダは include しない（境界ルール）。属性マクロだけ置く。
#if defined(__HIPCC__)
#define SHELF_QUANT_HOST_DEVICE __host__ __device__
#else
#define SHELF_QUANT_HOST_DEVICE
#endif
SHELF_QUANT_HOST_DEVICE inline float amax_scale(float amax) noexcept {
    return (amax > 0.0f) ? amax / kE4m3Max : 0.0f;
}
// 行全体（= トークン単位）の amax から作る場合の名残。式は同じ。
SHELF_QUANT_HOST_DEVICE inline float token_scale(float amax) noexcept {
    return amax_scale(amax);
}

// 観測最大値から決まるスケーリングの基準。
//   group = 0 のとき 1 行（= トークン）全体、group = kActGroupK のとき K を 128
//   要素のグループにしたもの。**どちらを使うかは checkpoint の宣言が正**で、
//   apps 側が `quantization_config` から読む（決め打ちしない）:
//     9B-FP8  : strategy=token        -> 0     / 重み strategy=channel
//     27B-FP8 : blockwise dynamic   -> 128   / 重み weight_block_size=[128,128]
//   粒度の実測差（同じ実活性化に対する相対 L2 誤差 1.14 倍改善）は
//   draft_docs/shelf_validate.md に記録してあるが、宣言に無い粒度は使わない。
inline constexpr std::uint32_t kActGroupWholeRow = 0u;
inline constexpr std::uint32_t kActGroupK = 128u;

// 活性化スケールの個数（行あたり）。
constexpr std::uint32_t act_scale_groups(std::uint32_t k_padded, std::uint32_t group) noexcept {
    return (group == 0u) ? 1u : (k_padded / group);
}

// 重みスケールの粒度。
//   PerChannel : 出力特征ごとに 1 個（この checkpoint の宣言 `strategy: channel`）
//   Block128   : 128(出力) × 128(K) タイルに 1 個（移植元 PhaseShift の native 形）
enum class WeightScaleGranularity : std::uint8_t {
    PerChannel,
    Block128,
};

// K 方向のスケールブロック数（128 単位。端数は切り上げ = 詰めを含むタイル数）。
constexpr std::uint32_t scale_k_blocks(std::uint32_t k_padded) noexcept {
    return (k_padded + kActGroupK - 1u) / kActGroupK;
}
// 出力方向のブロック数（Block128 のとき 128 単位、PerChannel のときは特征数そのもの）。
constexpr std::uint32_t scale_n_blocks(WeightScaleGranularity g, std::uint32_t features) noexcept {
    return (g == WeightScaleGranularity::Block128) ? ((features + kActGroupK - 1u) / kActGroupK)
                                                   : features;
}

// 重みスケール配列の要素数（f32）。
//   PerChannel : 出力特征ごと（K 方向は 1）
//   Block128   : [ceil(features/128)][ceil(k_padded/128)]
constexpr std::size_t weight_scale_count(WeightScaleGranularity g, std::uint32_t features,
                                         std::uint32_t k_padded) noexcept {
    if (g == WeightScaleGranularity::Block128)
        return static_cast<std::size_t>(scale_n_blocks(g, features)) * scale_k_blocks(k_padded);
    return features;
}
// 活性化スケール配列の要素数（f32、行優先 [rows][k_blocks]）。
constexpr std::size_t activation_scale_count(std::uint32_t rows, std::uint32_t k_padded,
                                             std::uint32_t act_group) noexcept {
    return static_cast<std::size_t>(rows) * act_scale_groups(k_padded, act_group);
}

// ---- 断片順のコード配置（WMMA のフラグメント順と一致させるレイアウト） ----------
//
// 16 行 × 32 K を 1 タイル（512 バイト）として、
//   [K の前半 16 の 16 行][K の後半 16 の 16 行]
// 行内は K が 16 バイト連続。タイルは (row/16) 優先、続いて (col/32)。
//   offset = (row/16)*(k_padded*16) + (col/32)*512 + ((col%32)/16)*256
//            + (row%16)*16 + (col%16)
// K 方向は k_padded（kTileK の倍数）まで 0 埋め、行方向は 16 行単位で切り上げた
// バイト数だけ確保する（詰め領域の値は 0）。
inline constexpr std::uint32_t kTileRows = 16u;
inline constexpr std::uint32_t kTileK = 32u;

constexpr std::uint32_t pad_k(std::uint32_t k) noexcept {
    return ((k + kTileK - 1u) / kTileK) * kTileK;
}

constexpr std::uint32_t pad_rows(std::uint32_t rows) noexcept {
    return ((rows + kTileRows - 1u) / kTileRows) * kTileRows;
}

// 確保が必要なバイト数（断片順、行方向の詰めを含む）。
constexpr std::size_t tiled_bytes(std::uint32_t rows, std::uint32_t k_padded) noexcept {
    return static_cast<std::size_t>(pad_rows(rows)) * static_cast<std::size_t>(k_padded);
}

// 要素 (row, col) のバイトオフセット。k_padded は pad_k() の結果であること。
constexpr std::size_t tiled_offset(std::uint32_t row, std::uint32_t col,
                                   std::uint32_t k_padded) noexcept {
    return static_cast<std::size_t>(row / kTileRows) *
               (static_cast<std::size_t>(k_padded) * kTileRows) +
           static_cast<std::size_t>(col / kTileK) * (kTileRows * kTileK) +
           static_cast<std::size_t>((col % kTileK) / 16u) * (kTileRows * 16u) +
           static_cast<std::size_t>(row % kTileRows) * 16u +
           static_cast<std::size_t>(col % 16u);
}

// 断片順レイアウトとして受理できる形か（k_padded の整合）。
Status validate_tiling(std::uint32_t rows, std::uint32_t cols, std::uint32_t k_padded);

// 行優先 [rows, cols] のコード配列を断片順へ再配置する（無損 relayout）。
// dst は tiled_bytes(rows, k_padded) 以上の容量が必要。詰め領域（K および行の
// 切り上げ分）は 0 で埋める。src と dst は重なってよい（同一バッファは不可）。
Status preshuffle_codes(const std::uint8_t* src, std::uint32_t rows, std::uint32_t cols,
                        std::uint32_t k_padded, std::uint8_t* dst);

// 検査・デバッグ用: 断片順を行優先 [rows, cols] に戻す（preshuffle の逆写像）。
Status untile_codes(const std::uint8_t* tiled, std::uint32_t rows, std::uint32_t cols,
                    std::uint32_t k_padded, std::uint8_t* dst);

// 重みスケールの型は f32 で統一する（checkpoint が BF16 [out,1] でも、実行器に
// 渡す時点で f32 に正規化する。値域は狭いので無損）。

}  // namespace shelf::quant::fp8
