// shelf quant — MXFP4（OCP MX の 4 bit 浮動小数）の契約とホスト側変換。
//
// 参照: `~/work/references/phaseshift-mxfp4` の
//   `include/phaseshift/quantization/mxfp4/mxfp4.h`（契約）
//   `src/phaseshift/quantization/mxfp4/mxfp4.cpp`（量子化・逆量子化）
//   `src/phaseshift/quantization/mxfp4/mxfp4_native.cpp`（断片順 relayout）
// を shelf の規約（`fp8_perchannel.hpp` と同じ構成）に写したもの。
//
// 数値の契約:
//   * 重みは **E2M1**（符号 1 bit + 指数 2 bit + 仮数 1 bit、絶対値は
//     {0, 0.5, 1, 1.5, 2, 3, 4, 6}）。2 値を 1 バイトに詰める（下位ニブルが先）。
//   * スケールは **E8M0**（値 = 2^(byte-127)、byte 255 は NaN）。K 方向 32 要素で 1 個。
//     したがって 32 重みあたり 16 B（コード）+ 1 B（スケール）= 17 B = 4.25 bpw。
//   * 活性化は E4M3 のまま（既存の `fp8_perchannel` 経路と同じ）。MXFP4 側で変わるのは
//     重みの表現だけ。
//
// 断片順（tiled）レイアウト — gfx1201 の FP8 WMMA の要求に合わせる:
//   1 タイル = 16 出力行 × 32 K。16 B/行 × 16 行 = 256 B。
//   行 ol のバイト b は `lo = code(k = ib*32 + b)`、`hi = code(k = ib*32 + 16 + b)`。
//   これは参照の `Mxfp4NativeView` と同じ配置で、E2M1 のニブルを
//   `v_perm_b32` で E4M3 のバイトへ展開するときに 8 B ロード 1 回で
//   16 K 断片 2 個（lo/hi）が得られる形になっている。
//
// 依存: foundation のみ（境界ルール。device 側の展開は backend/hip に置く）。
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/status.hpp>

namespace shelf::quant::mxfp4 {

// K 方向のスケール粒度（1 E8M0 バイトが覆う要素数）。
inline constexpr std::uint32_t kBlockK = 32u;
// 1 ブロックのコードのバイト数（32 要素 × 4 bit）。
inline constexpr std::uint32_t kCodesBytesPerBlock = 16u;
// タイルの形。
inline constexpr std::uint32_t kTileRows = 16u;
inline constexpr std::uint32_t kTileK = 32u;
inline constexpr std::uint32_t kLaneBytes = 16u;

// E2M1 の絶対値（3 bit の大きさフィールドで引く）。
inline constexpr float kE2M1Magnitudes[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

// ニブル（0..15）: bit3 が符号、bit0..2 が kE2M1Magnitudes の添字。
float e2m1_decode_nibble(std::uint8_t nibble) noexcept;
std::uint8_t e2m1_encode_nibble(float x) noexcept;

// E8M0 バイト: 値 = 2^(byte-127)。255 は NaN、0 は最小指数（-127）。
float e8m0_decode_u8(std::uint8_t byte) noexcept;
std::uint8_t e8m0_encode_f32(float x) noexcept;

// K を 32 の倍数へ、行を 16 の倍数へ切り上げる。
constexpr std::uint32_t pad_k(std::uint32_t k) noexcept {
    return ((k + kBlockK - 1u) / kBlockK) * kBlockK;
}

constexpr std::uint32_t pad_rows(std::uint32_t rows) noexcept {
    return ((rows + kTileRows - 1u) / kTileRows) * kTileRows;
}

// 行優先 [rows, cols] を詰めるときのコードのバイト数（K 方向は 2 値/バイト）。
constexpr std::size_t canonical_code_bytes(std::uint32_t rows, std::uint32_t k_padded) noexcept {
    return static_cast<std::size_t>(rows) * (static_cast<std::size_t>(k_padded) / 2u);
}

// 断片順に並べたときのコードのバイト数（行は 16 の倍数へ詰める）。
constexpr std::size_t tiled_bytes(std::uint32_t rows, std::uint32_t k_padded) noexcept {
    return static_cast<std::size_t>(pad_rows(rows)) *
           (static_cast<std::size_t>(k_padded) / 2u);
}

// スケールのバイト数（行数 × K ブロック数。行優先のまま）。
constexpr std::size_t scale_bytes(std::uint32_t rows, std::uint32_t k_padded) noexcept {
    return static_cast<std::size_t>(rows) * (static_cast<std::size_t>(k_padded) / kBlockK);
}

// 要素 (row, col) のコードが入るバイト位置とニブル位置（0 = 下位, 1 = 上位）。
// k_padded は pad_k() の結果であること。
constexpr std::size_t tiled_offset(std::uint32_t row, std::uint32_t col,
                                   std::uint32_t k_padded) noexcept {
    const std::size_t blocks_per_row_tile = static_cast<std::size_t>(k_padded) / kBlockK;
    return static_cast<std::size_t>(row / kTileRows) * (blocks_per_row_tile * 256u) +
           static_cast<std::size_t>(col / kTileK) * 256u +
           static_cast<std::size_t>(row % kTileRows) * kLaneBytes +
           static_cast<std::size_t>(col % 16u);
}

// tiled_offset が返すバイトのうち、要素が下位ニブルか上位ニブルか（0 か 4）。
constexpr std::uint32_t tiled_nibble_shift(std::uint32_t col) noexcept {
    return ((col % kTileK) / 16u) * 4u;
}

// スケールのバイト位置（行優先 [rows, k_padded/32]）。
constexpr std::size_t scale_offset(std::uint32_t row, std::uint32_t col,
                                   std::uint32_t k_padded) noexcept {
    return static_cast<std::size_t>(row) * (static_cast<std::size_t>(k_padded) / kBlockK) +
           static_cast<std::size_t>(col / kBlockK);
}

// 断片順レイアウトとして受理できる形か。
Status validate_tiling(std::uint32_t rows, std::uint32_t cols, std::uint32_t k_padded);

// 行優先のコード（[rows, k_padded/2]）を断片順へ再配置する（無損 relayout）。
// 詰め領域（K と行の切り上げ分）は 0 で埋める。
Status preshuffle_codes(const std::uint8_t* src, std::uint32_t rows, std::uint32_t cols,
                        std::uint32_t k_padded, std::uint8_t* dst);

// 検査・デバッグ用: 断片順を行優先へ戻す（preshuffle の逆写像）。
Status untile_codes(const std::uint8_t* tiled, std::uint32_t rows, std::uint32_t cols,
                    std::uint32_t k_padded, std::uint8_t* dst);

// 1 行を量子化する。`src` は logical_k 個の f32（k_padded まで 0 とみなす）。
// codes は k_padded/2 バイト、scales は k_padded/32 バイト（行優先・行ごとに呼ぶ）。
Status quantize_row(const float* src, std::uint32_t logical_k, std::uint32_t k_padded,
                    std::uint8_t* codes, std::uint8_t* scales);

// 断片順コードから f32 行列へ逆量子化する（テストと変換ツール用）。
// out は rows × cols の行優先 f32。
Status dequantize(const std::uint8_t* tiled_codes, const std::uint8_t* scales,
                  std::uint32_t rows, std::uint32_t cols, std::uint32_t k_padded,
                  float* out);

// 1 要素を読む（デバッグ・テスト用。断片順からのスカラー読み）。
float read_element(const std::uint8_t* tiled_codes, const std::uint8_t* scales,
                   std::uint32_t row, std::uint32_t col, std::uint32_t k_padded) noexcept;

}  // namespace shelf::quant::mxfp4
