// shelf backend/hip — 活性化を MXFP4（E2M1 + E8M0）へ量子化する。
//
// 契約は `src/quant/include/shelf/quant/mxfp4.hpp` と同じ:
//   * 入力は bf16 [rows, input_row_stride]（有効列は k 個。非有限値は 0 とみなす）
//   * 出力 codes は**断片順**のニブル（16 出力行 × 32 K = 256 B のタイル、
//     行 ol のバイト b が `lo = code(k = ib*32 + b)`、`hi = code(k = ib*32 + 16 + b)`）
//   * 出力 scales は **E8M0 1 バイト**を K 32 要素ごと（行優先 [rows][k_padded/32]）
//
// 重み側（`quant::mxfp4` の preshuffle）と**同じ配置**なので、GEMM は両側とも
// `mxfp4_low_high` で E4M3 に展開して FP8 WMMA に渡せる。
//
// 元: モデルカードの指示（Quark MXFP4 は A 側も E2M1 へ量子化する）。数値は
// `quant::mxfp4::quantize_row` + `preshuffle_codes` と一致することを
// tests/unit/test_hip_activation_mxfp4.cpp が readback で検証する。
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/foundation/numeric.hpp>

namespace shelf::hipk {

// 受理される形か（k_padded は 32 の倍数で k 以上）。
bool activation_quantize_mxfp4_supported(std::uint32_t k, std::uint32_t k_padded) noexcept;

// codes は quant::mxfp4::tiled_bytes(rows, k_padded) バイト、
// scales は quant::mxfp4::scale_bytes(rows, k_padded) バイトを呼び手が確保する。
hipError_t launch_activation_quantize_mxfp4(
    const shelf::bf16_t* input, std::uint8_t* codes, std::uint8_t* scales,
    std::uint32_t rows, std::uint32_t k, std::uint32_t k_padded,
    std::uint32_t input_row_stride, hipStream_t stream);

// SwiGLU 融合版（gate/up から直接 MXFP4 のコードとスケールを出す）。
// `out` が非 nullptr なら bf16 の swiglu 結果も書く（既存の融合と同じ値）。
hipError_t launch_swiglu_quantize_mxfp4(
    const shelf::bf16_t* gate, const shelf::bf16_t* up,
    shelf::bf16_t* out,
    std::uint8_t* codes, std::uint8_t* scales,
    std::uint32_t rows, std::uint32_t k, std::uint32_t k_padded,
    std::uint32_t gate_row_stride, std::uint32_t up_row_stride,
    std::uint32_t out_row_stride, hipStream_t stream);

}  // namespace shelf::hipk
