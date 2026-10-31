// shelf quant — FP8（E4M3）の数値コーデックと断片順レイアウト（両形式の共通部）。
//
// ここだけは「通常 FP8」と「ブロック単位スケール形式」で共通:
//   * E4M3 の符号化・復号、amax からのスケール
//   * 行列命令（gfx1201 の FP8 WMMA）が読む**断片順**のレイアウトと relayout
//     （行優先 → 断片順は要素の値を変えない**無損**。有損の dequant はしない）
//
// 形式ごとの違い（スケールの粒度）は fp8_perchannel.hpp と fp8_finegrained.hpp が持つ。
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

// ---- 断片順のコード配置（WMMA のフラグメント順と一致させるレイアウト） ----------
//
// 1 タイル = **16 行 × 64 K（1024 バイト）= 32 レーン × 32 バイト**。
// レーン l（= 行の下位 4 ビット + 16×(K の 8 要素組)）の 32 バイトは、
// 4 つの断片（16 行 × 16 K）を back-to-back に並べたもので、バイト 8t が断片 t。
//   lane  = (row%16) + 16 * ((col%16)/8)
//   frag  = (col%64)/16          バイト位置 8*frag + (col%8)
//   offset = (row/16)*(k_padded*16) + (col/64)*1024 + lane*32 + frag*8 + (col%8)
//
// 16 K 断片 1 本（1 レーン 8 バイト = 1 波 256 バイトのリクエスト）では LDS 経由の
// 転置にすら負けるため、参照実装（radiance の r4d_fp8_frag.h）と同じく
// **1 レーン 32 バイト = 1 波 1 KiB のリクエスト**になるよう 64 K を単位にする。
// 数値の並びは置換なので、ロード時の relayout は依然として無損。
//
// K 方向は k_padded（kSuperK の倍数）まで 0 埋め、行方向は 16 行単位で切り上げた
// バイト数だけ確保する（詰め領域の値は 0）。
inline constexpr std::uint32_t kTileRows = 16u;
inline constexpr std::uint32_t kTileK = 16u;     // 1 断片の K 幅（WMMA 1 命令分）
inline constexpr std::uint32_t kSuperK = 64u;    // 1 タイルの K 幅（4 断片 = レーン 32 B）
inline constexpr std::uint32_t kLaneBytes = 32u;

constexpr std::uint32_t pad_k(std::uint32_t k) noexcept {
    return ((k + kSuperK - 1u) / kSuperK) * kSuperK;
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
    const std::uint32_t lane = (row % kTileRows) + kTileRows * ((col % 16u) / 8u);
    return static_cast<std::size_t>(row / kTileRows) *
               (static_cast<std::size_t>(k_padded) * kTileRows) +
           static_cast<std::size_t>(col / kSuperK) * (kTileRows * kSuperK) +
           static_cast<std::size_t>(lane) * kLaneBytes +
           static_cast<std::size_t>((col % kSuperK) / 16u) * 8u +
           static_cast<std::size_t>(col % 8u);
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
