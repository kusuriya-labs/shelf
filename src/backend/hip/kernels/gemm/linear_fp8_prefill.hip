// shelf GEMM FP8 W8A8 — prefill 側（RowBlock / Prefill2D / wide タイル）。
//
// decode（rows=1 の GEMV）は linear_fp8_decode.hip。
//   * `gemm_fp8_w8a8_wmma_kernel<MB>`: MB 行 × 16 出力の RowBlock（rows <= 64）
//   * `gemm_fp8_w8a8_prefill_2d_kernel<kOutW>`: 64 行 × kOutW 出力の 2 次元タイル
//   * `gemm_fp8_w8a8_prefill_wide_kernel`: 128 行 × 128 出力（8 波）。prefill の主経路
#include <cstdint>
#include <cstdlib>

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
//
// NEGATIVE RESULT (2026-10-31): その「累加 64 → 32」を実際にやった。1 波の行数を
// 64 → 32 にして 16 波（512 スレッド、VGPR 118、1 workgroup/CU = 16 波）にすると、
// タイルは 128 × 128 のままで数値もビット一致するが、**実測は遅い**
// （rows=2048: qkv 172 → 138 TF/s、gate+up 162 → 153）。1 波あたりの出力幅が
// 変わらないので重み断片の読みが波数ぶん増え（2 倍）、さらに 16 波のバリアが
// 重くなるため。occupancy を上げるなら重みの再利用（出力幅を広げる）とセットで
// なければならない。よって 8 波のままが最良点。
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


}  // namespace

hipError_t detail::fp8_launch_prefill(const Fp8GemmConfig& config,
                                      const std::uint8_t* weight_codes,
                                      const float* weight_scales,
                                      const std::uint8_t* activation_codes,
                                      const float* activation_scales, void* output,
                                      Fp8GemmOutputDType output_dtype, std::uint32_t rows,
                                      std::uint32_t features, std::uint32_t k_padded,
                                      std::uint32_t output_row_stride,
                                      const Fp8GemmScaleArgs& scales, hipStream_t stream) {
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
        // getenv は 1 ステップで数百回呼ばれる（層 × GEMM）。初回だけ読んで
        // キャッシュする。プロセス起動後に環境変数を変える使い方はしない。
        static const std::uint32_t wide_min_k = [] {
            if (const char* e = std::getenv("SHELF_FP8_WIDE_MIN_K")) {
                const long v = std::strtol(e, nullptr, 10);
                if (v >= 0) return static_cast<std::uint32_t>(v);
            }
            return 0u;
        }();
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

}  // namespace shelf::hipk
#undef SHELF_FP8_MB_LAUNCH
