// shelf GEMM FP8 W8A8 — decode（rows=1）の GEMV。
//
// prefill 側（RowBlock / Prefill2D / wide）と振り分けは linear_fp8.hip。
// 1 ブロック = 16 出力 × 32 スレッド（16 行タイル × 2 K 半）。行列命令 2 本で 32 K を
// 積み、32 K ごとにスケールを掛ける。
#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/gemm/detail/linear_fp8_device.h>
#include <shelf/backend/hip/kernels/gemm/detail/linear_fp8_launch.h>
#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>
#include <shelf/quant/fp8_codec.hpp>
#include <shelf/quant/fp8_finegrained.hpp>
#include <shelf/quant/fp8_perchannel.hpp>

namespace shelf::hipk {
namespace {

using namespace gemm_fp8_detail;

template <Fp8GemmOutputDType kOutputType, int kUnroll, bool kBlockW>
__global__ void __launch_bounds__(kThreadsPerBlock) gemv_fp8_w8a8_decode1_kernel(
    const std::uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const Fp8GemmScaleArgs sc) {
    fp8_decode1_body<kOutputType, kUnroll, kBlockW>(threadIdx.x, weight_codes, weight_scales,
                                                    activation_codes, activation_scales,
                                                    output, features, k_padded, sc);
}

constexpr int kDecode1Unroll = 16;
// N07: 複数行 decode の 1 ブロックあたりの行数（= 波数）。
constexpr std::uint32_t kDecodeNRows = 4u;


template <Fp8GemmOutputDType kOutputType>
hipError_t fp8_decode1_launch(
    const std::uint8_t* weight_codes, const float* weight_scales,
    const std::uint8_t* activation_codes, const float* activation_scales, void* output,
    std::uint32_t features, std::uint32_t k_padded, const Fp8GemmScaleArgs& scales,
    hipStream_t stream) {
    const dim3 grid(features >> 4u);
    if (scales.weight_block128)
        gemv_fp8_w8a8_decode1_kernel<kOutputType, kDecode1Unroll, true>
            <<<grid, kThreadsPerBlock, 0, stream>>>(weight_codes, weight_scales,
                                                    activation_codes, activation_scales,
                                                    output, features, k_padded, scales);
    else
        gemv_fp8_w8a8_decode1_kernel<kOutputType, kDecode1Unroll, false>
            <<<grid, kThreadsPerBlock, 0, stream>>>(weight_codes, weight_scales,
                                                    activation_codes, activation_scales,
                                                    output, features, k_padded, scales);
    return hipGetLastError();
}


// N07: 2〜4 行の decode。1 波 = 1 行で、ブロック内の波に行を割り当てる。
// 重みは 4 波が同じアドレスを読むので L2 に載り、DRAM の読み出しは 1 行ぶんに
// なる（1 行ずつ別カーネルで回すと 4 回 DRAM から読む）。行ごとの本文は decode1 と
// 同じ `fp8_decode1_body` なので、出力は行ごとの decode1 とビット同一。
template <Fp8GemmOutputDType kOutputType, bool kBlockW>
__global__ void __launch_bounds__(kDecodeNRows * 32u) gemv_fp8_w8a8_decode_n_kernel(
    const std::uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const std::uint32_t rows,
    const std::uint32_t output_row_stride,
    const Fp8GemmScaleArgs sc) {
    const std::uint32_t warp = threadIdx.x >> 5u;
    if (warp >= rows) return;
    const std::size_t row_bytes = static_cast<std::size_t>(k_padded) * 16u;
    const std::size_t out_off = static_cast<std::size_t>(warp) * output_row_stride;
    void* out;
    if constexpr (kOutputType == Fp8GemmOutputDType::BF16)
        out = static_cast<void*>(static_cast<shelf::bf16_t*>(output) + out_off);
    else
        out = static_cast<void*>(static_cast<float*>(output) + out_off);
    fp8_decode1_body<kOutputType, kDecode1Unroll, kBlockW>(
        threadIdx.x & 31u, weight_codes, weight_scales,
        activation_codes + static_cast<std::size_t>(warp) * row_bytes,
        activation_scales + static_cast<std::size_t>(warp) * sc.activation_scale_stride, out,
        features, k_padded, sc);
}

}  // namespace

bool fp8_decode_n_supported(std::uint32_t rows, std::uint32_t features,
                            std::uint32_t k_padded) noexcept {
    if (rows < 2u || rows > 4u) return false;
    return fp8_decode1_supported(1u, features, k_padded);
}

namespace {

template <Fp8GemmOutputDType kOutputType, bool kBlockW>
hipError_t fp8_decode_n_launch(const std::uint8_t* weight_codes, const float* weight_scales,
                               const std::uint8_t* activation_codes,
                               const float* activation_scales, void* output,
                               std::uint32_t rows, std::uint32_t features,
                               std::uint32_t k_padded, std::uint32_t output_row_stride,
                               const Fp8GemmScaleArgs& scales, hipStream_t stream) {
    const dim3 grid(features >> 4u);
    // 行数ぶんの波だけ起動する（rows = 2 で 4 波を出すと半分が空になる）。
    gemv_fp8_w8a8_decode_n_kernel<kOutputType, kBlockW>
        <<<grid, rows * 32u, 0, stream>>>(weight_codes, weight_scales,
                                                  activation_codes, activation_scales, output,
                                                  features, k_padded, rows, output_row_stride,
                                                  scales);
    return hipGetLastError();
}

template <Fp8GemmOutputDType kOutputType>
hipError_t fp8_decode_n_dtype(const std::uint8_t* weight_codes, const float* weight_scales,
                              const std::uint8_t* activation_codes,
                              const float* activation_scales, void* output,
                              std::uint32_t rows, std::uint32_t features,
                              std::uint32_t k_padded, std::uint32_t output_row_stride,
                              const Fp8GemmScaleArgs& scales, hipStream_t stream) {
    if (scales.weight_block128)
        return fp8_decode_n_launch<kOutputType, true>(weight_codes, weight_scales,
                                                      activation_codes, activation_scales,
                                                      output, rows, features, k_padded,
                                                      output_row_stride, scales, stream);
    return fp8_decode_n_launch<kOutputType, false>(weight_codes, weight_scales, activation_codes,
                                                   activation_scales, output, rows, features,
                                                   k_padded, output_row_stride, scales, stream);
}

}  // namespace

hipError_t launch_gemm_fp8_w8a8_decode_n(
    const std::uint8_t* weight_codes, const float* weight_scales,
    const std::uint8_t* activation_codes, const float* activation_scales, void* output,
    Fp8GemmOutputDType output_dtype, std::uint32_t rows, std::uint32_t features,
    std::uint32_t k_padded, std::uint32_t output_row_stride,
    const Fp8GemmScaleArgs& scales, hipStream_t stream) {
    if (!fp8_decode_n_supported(rows, features, k_padded)) return hipErrorInvalidValue;
    if (output_dtype == Fp8GemmOutputDType::BF16)
        return fp8_decode_n_dtype<Fp8GemmOutputDType::BF16>(
            weight_codes, weight_scales, activation_codes, activation_scales, output, rows,
            features, k_padded, output_row_stride, scales, stream);
    return fp8_decode_n_dtype<Fp8GemmOutputDType::F32>(
        weight_codes, weight_scales, activation_codes, activation_scales, output, rows,
        features, k_padded, output_row_stride, scales, stream);
}

hipError_t detail::fp8_launch_decode1(const std::uint8_t* weight_codes,
                                      const float* weight_scales,
                                      const std::uint8_t* activation_codes,
                                      const float* activation_scales, void* output,
                                      Fp8GemmOutputDType output_dtype, std::uint32_t features,
                                      std::uint32_t k_padded, const Fp8GemmScaleArgs& scales,
                                      hipStream_t stream) {
    if (output_dtype == Fp8GemmOutputDType::BF16)
        return fp8_decode1_launch<Fp8GemmOutputDType::BF16>(
            weight_codes, weight_scales, activation_codes, activation_scales, output,
            features, k_padded, scales, stream);
    return fp8_decode1_launch<Fp8GemmOutputDType::F32>(
        weight_codes, weight_scales, activation_codes, activation_scales, output, features,
        k_padded, scales, stream);
}

}  // namespace shelf::hipk
