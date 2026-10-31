// shelf GEMM FP8 W8A8 — カーネル入口とホスト側ランチャ。
//
// デバイス本体は detail/linear_fp8_device.h。活性化コードの断片順レイアウトと
// スケール配置の契約は src/quant/include/shelf/quant/fp8_perchannel.hpp。
// （定義は二重になるが、tests/unit/test_hip_gemm_fp8.cpp が readback で一致を
// 検証するので、片方を壊すとテストが落ちる。）
#include <cstdint>

#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/gemm/detail/linear_fp8_device.h>
#include <shelf/backend/hip/kernels/gemm/detail/linear_fp8_launch.h>
#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>
#include <shelf/quant/fp8_perchannel.hpp>
#include <shelf/quant/fp8_finegrained.hpp>

namespace shelf::hipk {
namespace {

using namespace gemm_fp8_detail;

}  // namespace

bool fp8_gemm_shape_supported(std::uint32_t rows, std::uint32_t features,
                              std::uint32_t k_padded) noexcept {
    if (rows == 0u || features == 0u || k_padded == 0u) return false;
    if (rows > 2048u) return false;                // kMaxStepRows
    if ((k_padded % shelf::quant::fp8::kSuperK) != 0u) return false;   // 64 K 単位
    return true;
}

bool fp8_scale_args_supported(const Fp8GemmScaleArgs& sc, std::uint32_t rows,
                              std::uint32_t features, std::uint32_t k_padded) noexcept {
    using namespace shelf::quant::fp8;
using namespace shelf::quant::fp8::fine;
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

bool fp8_prefill2d_supported(std::uint32_t rows, std::uint32_t features,
                             std::uint32_t k_padded) noexcept {
    if (rows == 0u || features == 0u || k_padded == 0u) return false;
    if ((rows % kPrefill2dRows) != 0u) return false;
    if ((features % 64u) != 0u) return false;   // 32 幅は 64 幅の形の部分集合
    return (k_padded % 64u) == 0u;
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
        return detail::fp8_launch_decode1(weight_codes, weight_scales, activation_codes,
                                          activation_scales, output, output_dtype, features,
                                          k_padded, scales, stream);
    }

    return detail::fp8_launch_prefill(config, weight_codes, weight_scales, activation_codes,
                                      activation_scales, output, output_dtype, rows, features,
                                      k_padded, output_row_stride, scales, stream);
}

#undef SHELF_FP8_MB_LAUNCH

}  // namespace shelf::hipk
