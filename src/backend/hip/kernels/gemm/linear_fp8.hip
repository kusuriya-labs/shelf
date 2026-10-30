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

// NEGATIVE RESULT (2026-10-30): VGPR 176 / LDS 16,896 B の組み合わせは
// レジスタ側（65,536 ÷ (176×32) ≒ 11.6 ウェーブ ≒ 1.4 workgroup/CU）で束縛されており、
// LDS は 3 まで許す。`__launch_bounds__(256, 2)` で最低 2 workgroup を要求したが、
// 生成された VGPR は 176 のまま（実測 trace で確認）で prefill も中立
// （1,008.4 → 1,007.2 ms）。 occupancy を上げるには本体側でスレッドあたり
// 累加数を減らす（例: 4 行 × 2 出力 → 4 行 × 1 出力、64 → 32 acc）必要がある。
__global__ void __launch_bounds__(kPrefill2dWideThreads) gemm_fp8_w8a8_prefill_wide_kernel(
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
    fp8_prefill2d_wide_body(weight_codes, weight_scales, activation_codes,
                            activation_scales, output, output_dtype, rows, features,
                            k_padded, output_row_stride, sc);
}

template <std::uint32_t kOutW>
__global__ void __launch_bounds__(kPrefill2dThreadsFor(kOutW))
gemm_fp8_w8a8_prefill_2d_kernel(
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
    fp8_prefill2d_body<kOutW>(weight_codes, weight_scales, activation_codes,
                              activation_scales, output, output_dtype, rows, features,
                              k_padded, output_row_stride, sc);
}

// 注: radiance は `R4D_WAVES` でタイル形状ごとの waves/SIMD を固定してレジスタ予算を
// 制御する。この kernel について同一の実験（minBlocksPerMultiprocessor = 12）をしたが
// VGPR は 32 のまま（予算が紐解きにならない）で性能も不変だった。パイプライン化は
// 明示的な 2 段先読み（`fp8_decode1_body`）で実現したので、hint は付けない。

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
    if (rows > 2048u) return false;                // kMaxStepRows
    if ((k_padded % shelf::quant::fp8::kSuperK) != 0u) return false;   // 64 K 単位
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
        if (output_dtype == Fp8GemmOutputDType::BF16)
            return fp8_decode1_launch<Fp8GemmOutputDType::BF16>(
                weight_codes, weight_scales, activation_codes, activation_scales, output,
                features, k_padded, scales, stream);
        return fp8_decode1_launch<Fp8GemmOutputDType::F32>(
            weight_codes, weight_scales, activation_codes, activation_scales, output,
            features, k_padded, scales, stream);
    }

    Fp8GemmConfigId id = config.id;
    if (id == Fp8GemmConfigId::Auto) {
        // prefill（64 行の束）は 2 次元タイルで活性化の読みを 1/4 にする。
        id = fp8_prefill2d_supported(rows, features, k_padded) ? Fp8GemmConfigId::Prefill2D
                                                                : pick_config(rows);
    } else if (id == Fp8GemmConfigId::Decode1) {
        id = pick_config(rows);
    }
    if (id == Fp8GemmConfigId::Prefill2D) {
        if (!fp8_prefill2d_supported(rows, features, k_padded)) return hipErrorNotSupported;
        // 64 出力 × 64 行 = 16 波（512 スレッド）。32 幅版も試したが、ブロック数が
        // 増えても resident wave は変わらず、実測で差が出なかったので 64 幅に統一した。
        // 128 行・128 出力が揃う形は広いタイル（活性化の読みが半分、行列命令密度 2 倍）。
        // 実測で K が小さい形（5120 など）は wide タイルでも 147 TFLOP/s まで
        // 落ちる（K=17408 では 216）。ブロック数 4 倍の 64 出力タイル版へ回す実験を
        // したが、**大きく悪い**（chunk 2048 で 1,006 → 1,444 ms、+43%）。
        // よって K が小さくても wide を使うのが正しく、改善は wide 本体の内側
        // （LDS ステージングの二段化、K ブロック 256）でなければならない。
        // `SHELF_FP8_WIDE_MIN_K`（既定 0 = 常に wide）は再現用に残す。
        std::uint32_t wide_min_k = 0u;
        if (const char* e = std::getenv("SHELF_FP8_WIDE_MIN_K")) {
            const long v = std::strtol(e, nullptr, 10);
            if (v >= 0) wide_min_k = static_cast<std::uint32_t>(v);
        }
        if ((rows % kPrefill2dWideRows) == 0u && (features % kPrefill2dWideOut) == 0u &&
            (k_padded % 128u) == 0u && rows >= kPrefill2dWideRows &&
            k_padded >= wide_min_k) {
            // x = 行タイル、y = 出力タイル（重みの L2 再利用のため。本体の注記参照）。
            const dim3 g(rows / kPrefill2dWideRows, features / kPrefill2dWideOut, 1);
            gemm_fp8_w8a8_prefill_wide_kernel<<<g, kPrefill2dWideThreads, 0, stream>>>(
                weight_codes, weight_scales, activation_codes, activation_scales, output,
                static_cast<std::uint32_t>(output_dtype), rows, features, k_padded,
                output_row_stride, scales);
            return hipGetLastError();
        }
        const dim3 grid(features / 64u, rows / kPrefill2dRows, 1);
        gemm_fp8_w8a8_prefill_2d_kernel<64u>
            <<<grid, kPrefill2dThreadsFor(64u), 0, stream>>>(
                weight_codes, weight_scales, activation_codes, activation_scales, output,
                static_cast<std::uint32_t>(output_dtype), rows, features, k_padded,
                output_row_stride, scales);
        return hipGetLastError();
    }
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
