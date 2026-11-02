// shelf backend/hip — MXFP4 W4A8 linear（E2M1 重み + E8M0 ブロックスケール、gfx1201）。
//
// 数値の契約（`src/quant/include/shelf/quant/mxfp4.hpp` と同一）:
//   out[m, n] = Σ_ib ( Σ_k a_code[m,k] · w_code[n,k] ) · w_scale[n][ib] · a_scale[m][kb]
//   * 重みは E2M1 を 2 値/バイトで詰めた**断片順**（16 出力 × 32 K = 256 B のタイル、
//     行 ol のバイト b が `lo = code(k = ib*32 + b)`、`hi = code(k = ib*32 + 16 + b)`）。
//   * 重みスケールは **E8M0 1 バイト**を K 32 要素ごとに（行優先 [features][k_padded/32]）。
//   * 活性化は E4M3 のまま（既存の `linear_fp8.h` と同じ断片順・同じスケール粒度）。
//     `act_group = 128` なら K を 128 要素ごとに 1 スケール、0 なら行全体で 1 個。
//
// 実装: 8 B の重みロードから `v_perm_b32` で E4M3 のバイトへ展開し、16 K 断片 2 個を
// 作って FP8 WMMA を 2 回。E2M1 の 8 値は E4M3 で厳密に表せるので展開は無損失。
//
// 元: `~/work/references/phaseshift-mxfp4` の
//   `kernels/optimized/linear/mxfp4.hip`（`gemv_mxfp4_w4a8_decode1_kernel`）。
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/foundation/numeric.hpp>

namespace shelf::hipk {

enum class Mxfp4GemmOutputDType : std::uint8_t {
    BF16,
    F32,
};

// スケール配置（活性化側は fp8 と同じ粒度を使う）。
struct Mxfp4GemmScaleArgs {
    std::uint32_t act_group = 0u;                 // 0 = 行全体、128 = K を 128 単位
    std::uint32_t activation_scale_stride = 1u;   // f32 個数 / 行
    std::uint32_t weight_scale_stride = 1u;       // E8M0 バイト数 / 行（= k_padded/32）
};

// rows=1 専用（GEMV 形）が使える形か。16 出力単位・K が 32 の倍数。
bool mxfp4_decode1_supported(std::uint32_t rows, std::uint32_t features,
                             std::uint32_t k_padded) noexcept;

// N09: 同一エキスパートへ複数行（1 行以上いくつでも）をまとめて流す形か。
bool mxfp4_decode_rows_supported(std::uint32_t rows, std::uint32_t features,
                                 std::uint32_t k_padded) noexcept;

// N09: 行単位の GEMV。1 波 = 1 行で、活性化の行と出力の行をデバイス上の
// 索引配列で指定する（`nullptr` なら 0,1,2,… の連番）。
// 行ごとの値は decode1 とビット同一。
hipError_t launch_gemm_mxfp4_w4a8_decode_rows(
    const std::uint8_t* weight_codes,
    const std::uint8_t* weight_scales,
    const std::uint8_t* activation_codes,
    const float* activation_scales,
    const std::uint32_t* act_rows,
    const std::uint32_t* out_rows,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    std::uint32_t rows,
    std::uint32_t features,
    std::uint32_t k_padded,
    std::uint32_t act_row_stride,
    std::uint32_t out_row_stride,
    const Mxfp4GemmScaleArgs& scales,
    hipStream_t stream);

// 2〜4 行の decode GEMV が使える形か。重みの読み出しを全行で共有する。
// 出力は「行ごとに decode1 を呼んだ結果」とビット同一（累加順が同じ）。
bool mxfp4_decode_n_supported(std::uint32_t rows, std::uint32_t features,
                              std::uint32_t k_padded) noexcept;

// 2〜4 行の GEMV。活性化コードは **行ごとに独立した断片順ブロック** を
// `quant::fp8::tiled_bytes(1, k_padded)` バイト間隔で並べる（1 行ずつ量子化した
// 結果をそのまま並べる）。出力の行ストライドは要素数で受ける。
hipError_t launch_gemm_mxfp4_w4a8_decode_n(
    const std::uint8_t* weight_codes,
    const std::uint8_t* weight_scales,
    const std::uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    std::uint32_t rows,
    std::uint32_t features,
    std::uint32_t k_padded,
    std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs& scales,
    hipStream_t stream);

// 断片順の重みコードに必要なバイト数（`quant::mxfp4::tiled_bytes` と同じ）。
std::size_t mxfp4_weight_bytes(std::uint32_t features, std::uint32_t k_padded) noexcept;

// rows=1 の GEMV。`weight_codes` は断片順、`weight_scales` は E8M0 の行優先。
hipError_t launch_gemm_mxfp4_w4a8_decode1(
    const std::uint8_t* weight_codes,
    const std::uint8_t* weight_scales,
    const std::uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    std::uint32_t rows,
    std::uint32_t features,
    std::uint32_t k_padded,
    std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs& scales,
    hipStream_t stream);

// prefill 用（128 行 × 128 出力、8 波・256 スレッド）。K は 128 の倍数。
// 活性化は 64 K 半の ping-pong LDS で、fp8 の wide 版と同じ。
hipError_t launch_gemm_mxfp4_w4a8_prefill(
    const std::uint8_t* weight_codes,
    const std::uint8_t* weight_scales,
    const std::uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    std::uint32_t rows,
    std::uint32_t features,
    std::uint32_t k_padded,
    std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs& scales,
    hipStream_t stream);

}  // namespace shelf::hipk
