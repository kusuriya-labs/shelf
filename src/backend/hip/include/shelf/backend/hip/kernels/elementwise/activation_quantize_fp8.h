// shelf backend/hip — 活性化の FP8(E4M3) 動的量子化。
//
// 契約（src/quant/include/shelf/quant/fp8_w8a8.hpp）:
//   scale(row, kb) = amax(row, kb) / 448            … 実行時に計算（dynamic）
//   code(row,k)    = e4m3( x(row,k) / scale(row, kb) )
//   kb = K を 128 要素に切ったブロック番号（`act_group = 128`）。
//        `act_group = 0` のとき 1 行（= トークン）全体で 1 スケール。
//        どちらを使うかは checkpoint の宣言が正（apps 側が quantization_config から読む）。
//   codes は **断片順**（GEMM が読む形そのもの）、scales は f32[rows][k_blocks]。
//
// 元の実装（PhaseShift `activation_quantize`）と同じ手順
//   scale = amax/448、code = e4m3(x * (1/scale))。除算でなく逆数を掛ける点は
//   丸めの差になりうるため、実測（T2）で範囲を確認して使う（parity_policy §3）。
#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/foundation/numeric.hpp>

namespace shelf::hipk {

// codes 必要なバイト数（断片順。rows は 16 行単位で切り上げ、K は k_padded）。
std::size_t activation_quantize_fp8_bytes(std::uint32_t rows, std::uint32_t k_padded) noexcept;
// scales 必要な個数（f32）。
std::size_t activation_quantize_fp8_scale_count(std::uint32_t rows, std::uint32_t k_padded,
                                                std::uint32_t act_group) noexcept;
// 受理される形か（k_padded は 32 の倍数で k 以上、act_group は 0 または 128）。
bool activation_quantize_fp8_supported(std::uint32_t k, std::uint32_t k_padded,
                                       std::uint32_t act_group) noexcept;

// input  : bf16 [rows, input_row_stride]（有効列は k 個。非有限値は 0 として扱う）
// codes  : 断片順 u8（GEMM がそのまま読む）
// scales : f32[rows][k_blocks]（k_blocks = act_group ? k_padded/128 : 1）
hipError_t launch_activation_quantize_fp8(
    const shelf::bf16_t* input, std::uint8_t* codes, float* scales,
    std::uint32_t rows, std::uint32_t k, std::uint32_t k_padded,
    std::uint32_t input_row_stride, std::uint32_t act_group, hipStream_t stream);

}  // namespace shelf::hipk
