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
    fp8_decode1_body<kOutputType, kUnroll, kBlockW>(weight_codes, weight_scales,
                                                    activation_codes, activation_scales,
                                                    output, features, k_padded, sc);
}

constexpr int kDecode1Unroll = 16;


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


}  // namespace

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
