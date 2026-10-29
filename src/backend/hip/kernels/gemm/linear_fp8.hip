// shelf GEMM FP8 W8A8 — カーネル入口とホスト側ランチャ。
//
// デバイス本体は detail/linear_fp8_device.h。活性化コードの断片順レイアウトと
// スケール配置の契約は src/quant/include/shelf/quant/fp8_w8a8.hpp。
// （定義は二重になるが、tests/unit/test_hip_gemm_fp8.cpp が readback で一致を
// 検証するので、片方を壊すとテストが落ちる。）
#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/gemm/detail/linear_fp8_device.h>
#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>
#include <shelf/quant/fp8_w8a8.hpp>

namespace shelf::hipk {
namespace {

using namespace gemm_fp8_detail;

template <int MB, bool kBlockW>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_fp8_w8a8_wmma_kernel(
    const std::uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t output_dtype,
    const std::uint32_t rows,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const std::uint32_t output_row_stride,
    const Fp8GemmScaleArgs sc) {
    fp8_wmma_body<MB, kBlockW>(weight_codes, weight_scales, activation_codes,
                               activation_scales, output, output_dtype, rows, features,
                               k_padded, output_row_stride, sc);
}

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

Fp8GemmConfigId pick_config(std::uint32_t rows) noexcept {
    if (rows <= kTileRow) return Fp8GemmConfigId::RowBlock1;
    if (rows <= 2u * kTileRow) return Fp8GemmConfigId::RowBlock2;
    return Fp8GemmConfigId::RowBlock4;
}

#define SHELF_FP8_MB_LAUNCH(MB)                                                        \
    do {                                                                               \
        const dim3 grid((features + kTileOut - 1u) / kTileOut,                         \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1);         \
        if (scales.weight_block128)                                                    \
            gemm_fp8_w8a8_wmma_kernel<(MB), true><<<grid, kThreadsPerBlock, 0, stream>>>(\
                weight_codes, weight_scales, activation_codes, activation_scales,       \
                output, static_cast<std::uint32_t>(output_dtype), rows, features,       \
                k_padded, output_row_stride, scales);                                  \
        else                                                                           \
            gemm_fp8_w8a8_wmma_kernel<(MB), false><<<grid, kThreadsPerBlock, 0, stream>>>(\
                weight_codes, weight_scales, activation_codes, activation_scales,       \
                output, static_cast<std::uint32_t>(output_dtype), rows, features,       \
                k_padded, output_row_stride, scales);                                  \
        return hipGetLastError();                                                      \
    } while (0)

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

bool fp8_gemm_shape_supported(std::uint32_t rows, std::uint32_t features,
                              std::uint32_t k_padded) noexcept {
    if (rows == 0u || features == 0u || k_padded == 0u) return false;
    if (rows > 64u) return false;                 // kMaxStepRows
    if ((k_padded % shelf::quant::fp8::kTileK) != 0u) return false;
    return true;
}

bool fp8_scale_args_supported(const Fp8GemmScaleArgs& sc, std::uint32_t rows,
                              std::uint32_t features, std::uint32_t k_padded) noexcept {
    using namespace shelf::quant::fp8;
    if (sc.act_group != kActGroupWholeRow && sc.act_group != kActGroupK) return false;
    if (sc.act_group == kActGroupK && (k_padded % kActGroupK) != 0u) return false;
    if (sc.activation_scale_stride != act_scale_groups(k_padded, sc.act_group)) return false;
    const std::uint32_t nb = scale_n_blocks(sc.weight_block128 ? WeightScaleGranularity::Block128
                                                               : WeightScaleGranularity::PerChannel,
                                            features);
    if (sc.weight_block128 && sc.weight_scale_k_blocks != scale_k_blocks(k_padded)) return false;
    if (!sc.weight_block128 && nb != features) return false;
    (void)rows;
    return true;
}

bool fp8_decode1_supported(std::uint32_t rows, std::uint32_t features,
                           std::uint32_t k_padded) noexcept {
    if (rows != 1u) return false;
    if ((features & 15u) != 0u) return false;     // 1 出力特征タイル = 16
    return (k_padded & 31u) == 0u;
}

std::size_t fp8_activation_bytes(std::uint32_t rows, std::uint32_t k_padded) noexcept {
    return shelf::quant::fp8::tiled_bytes(rows, k_padded);
}

hipError_t launch_gemm_fp8_w8a8(
    const Fp8GemmConfig& config,
    const std::uint8_t* weight_codes,
    const float* weight_scales,
    const std::uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8GemmOutputDType output_dtype,
    std::uint32_t rows,
    std::uint32_t features,
    std::uint32_t k_padded,
    std::uint32_t output_row_stride,
    const Fp8GemmScaleArgs& scales,
    hipStream_t stream) {
    if (rows == 0u || features == 0u || k_padded == 0u) return hipSuccess;
    if (!fp8_gemm_shape_supported(rows, features, k_padded)) return hipErrorNotSupported;
    if (!fp8_scale_args_supported(scales, rows, features, k_padded))
        return hipErrorNotSupported;

    // rows == 1 は GEMV 形が最速。ただし 16 出力特征単位でないと使えないため、
    // その場合は RowBlock1（1 ウェーブ = 16 トークン）に落ちる。
    const bool want_decode1 = (config.id == Fp8GemmConfigId::Auto ||
                               config.id == Fp8GemmConfigId::Decode1);
    if (rows == 1u && want_decode1 && fp8_decode1_supported(rows, features, k_padded)) {
        if (output_dtype == Fp8GemmOutputDType::BF16)
            return fp8_decode1_launch<Fp8GemmOutputDType::BF16>(
                weight_codes, weight_scales, activation_codes, activation_scales, output,
                features, k_padded, scales, stream);
        return fp8_decode1_launch<Fp8GemmOutputDType::F32>(
            weight_codes, weight_scales, activation_codes, activation_scales, output,
            features, k_padded, scales, stream);
    }

    Fp8GemmConfigId id = config.id;
    if (id == Fp8GemmConfigId::Auto || id == Fp8GemmConfigId::Decode1) id = pick_config(rows);
    switch (id) {
        case Fp8GemmConfigId::RowBlock4:
            if (rows <= 2u * kTileRow) { SHELF_FP8_MB_LAUNCH(2); }
            SHELF_FP8_MB_LAUNCH(4);
        case Fp8GemmConfigId::RowBlock2:
            SHELF_FP8_MB_LAUNCH(2);
        case Fp8GemmConfigId::RowBlock1:
        default:
            SHELF_FP8_MB_LAUNCH(1);
    }
}

#undef SHELF_FP8_MB_LAUNCH

}  // namespace shelf::hipk
