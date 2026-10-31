// shelf GEMM FP8 W8A8 — 内部入口（実装は役割ごとのファイル）。
//
//   * linear_fp8_decode.hip  — rows=1 の GEMV
//   * linear_fp8.hip         — prefill（RowBlock / Prefill2D / wide）と振り分け
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>

namespace shelf::hipk::detail {

// rows=1 専用（GEMV 形）。
hipError_t fp8_launch_decode1(const std::uint8_t* weight_codes, const float* weight_scales,
                              const std::uint8_t* activation_codes,
                              const float* activation_scales, void* output,
                              Fp8GemmOutputDType output_dtype, std::uint32_t features,
                              std::uint32_t k_padded, const Fp8GemmScaleArgs& scales,
                              hipStream_t stream);

// prefill（RowBlock / Prefill2D / wide）。config は Auto 以外を渡してよい。
hipError_t fp8_launch_prefill(const Fp8GemmConfig& config,
                              const std::uint8_t* weight_codes, const float* weight_scales,
                              const std::uint8_t* activation_codes,
                              const float* activation_scales, void* output,
                              Fp8GemmOutputDType output_dtype, std::uint32_t rows,
                              std::uint32_t features, std::uint32_t k_padded,
                              std::uint32_t output_row_stride, const Fp8GemmScaleArgs& scales,
                              hipStream_t stream);

}  // namespace shelf::hipk::detail
