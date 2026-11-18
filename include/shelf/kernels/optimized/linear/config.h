#pragma once

#include <cstdint>

namespace shelf::kernel {

inline constexpr uint32_t kBf16GemmExactRowsMax = 64u;
inline constexpr uint32_t kBf16GemmExactRowsKernelMax = 16u;
// One block per output row, all 256 threads on K, up to 8 token accumulators.
// Used for the narrow-output small-M shapes where one warp per row leaves too
// few blocks on the device.
inline constexpr uint32_t kBf16GemmSmallRowsMax = 16u;
inline constexpr uint32_t kBf16GemmSplitKMaxOutFeatures = 256u;
inline constexpr uint32_t kBf16GemmWideMinRows = 65u;
inline constexpr uint32_t kBf16GemmExactKAlignment = 8u;
inline constexpr uint32_t kBf16GemmExactPointerAlignment = 16u;
inline constexpr uint32_t kBf16GemmExactRowStrideAlignment = 8u;
inline constexpr uint32_t kBf16GemmExactRowsThreads = 256u;
inline constexpr uint32_t kBf16GemmExactRowsOutFeaturesPerBlock = 8u;

constexpr uint32_t bf16_exact_rows_grid_x(uint32_t out_features) {
    return (out_features + kBf16GemmExactRowsOutFeaturesPerBlock - 1u) /
           kBf16GemmExactRowsOutFeaturesPerBlock;
}

enum class Bf16GemmConfigId : uint8_t {
    Wmma,
    WmmaWide,
    WmmaKPartition,
    ExactRows,
    ExactRowsDot2,
    SmallRows,
};

struct Bf16GemmConfig {
    Bf16GemmConfigId id = Bf16GemmConfigId::Wmma;
    uint8_t exact_rows = 0u;
};

inline constexpr uint32_t kBlockScaledGemmRowTile = 16u;
inline constexpr uint32_t kBlockScaledGemmRowBlockMinOutFeatures = 256u;
inline constexpr uint32_t kBlockScaledGemmPrefill2dRowsPerBlock = 16u * kBlockScaledGemmRowTile;
inline constexpr uint32_t kBlockScaledGemmPrefill2dMinRows = 512u;
inline constexpr uint32_t kBlockScaledGemmPrefill2dMinOutFeatures = 512u;

inline constexpr uint32_t kFp8FinegrainedGemmRowTile = 16u;
inline constexpr uint32_t kFp8FinegrainedGemmPrefill2dOutBlock = 64u;
inline constexpr uint32_t kFp8FinegrainedGemmPrefill2dKChunk = 64u;

enum class Fp8FinegrainedGemmConfigId : uint8_t {
    RowBlock1,
    RowBlock2,
    RowBlock4,
    Prefill2D,
    SplitK,
};

// Small-M split-K: the wave count of one 16-output column tile is below the
// device's fill point, so K is divided across the warps of one block and the
// partial sums are reduced through LDS in split order. The kernel uses a
// 16-row tile and grid.y covers the remaining row tiles.
inline constexpr uint32_t kFp8FinegrainedGemmSplitKMaxRows = 64u;
inline constexpr uint32_t kFp8FinegrainedGemmSplitKMaxWays = 16u;
inline constexpr uint32_t kFp8FinegrainedGemmWaveSlots = 768u;
inline constexpr uint32_t kFp8FinegrainedGemmSplitKMinGroups = 4u;

// MXFP4 W4A8 decode / verify row counts run on the WMMA row-block family:
// SplitK (small out_features, K split across waves) or RowBlock (16 outputs per
// wave, MB activation row tiles); the GEMV family covers the one-lane-per-output
// decode shapes whose WMMA block wastes 15 of its 16 output lanes. The
// row-count crossover is set from the measured kernel time and weight rate in
// docs/research (Gemv vs RowBlock1 at the same weight bytes).
inline constexpr uint32_t kMxfp4GemmRowTile = 16u;
// Measured on R9700 with shelf-bench gemm --dtype mxfp4 (docs/research).
inline constexpr uint32_t kMxfp4GemmSplitKMaxRows = 64u;
// The split is only a win while the output width is narrow and each split still
// owns enough 128-K activation groups; below 2048 K the 8-way reduction costs
// more than the extra parallelism it buys.
inline constexpr uint32_t kMxfp4GemmSplitKMinKPadded = 2048u;
inline constexpr uint32_t kMxfp4GemmSplitKWays = 8u;
inline constexpr uint32_t kMxfp4GemmRowBlock1MaxRows = 16u;
inline constexpr uint32_t kMxfp4GemmRowBlock1MaxOutFeatures = 8192u;
inline constexpr uint32_t kMxfp4GemmRowBlock2MaxRows = 192u;
inline constexpr uint32_t kMxfp4GemmGemvMaxRows = 8u;
// The GEMV holds one accumulator per activation row, so its instruction count
// grows with the row count while the WMMA block does not. Measured crossover on
// R9700 is 4 rows for the wide shapes. Narrow outputs stay launch bound up to 8
// rows: at rows == 8 the GEMV wins up to out_features 2560 (1.39x at 2048x5120,
// 1.41x at 2560x5120) and loses from 3072 (0.92x at 3072x5120).
inline constexpr uint32_t kMxfp4GemmGemvWideMaxRows = 4u;
inline constexpr uint32_t kMxfp4GemmGemvNarrowMaxOutFeatures = 2560u;
// The decision is the weight stream rate, not the geometry alone. Below the L3
// size the WMMA row block reaches 650-1400 GB/s (above the 636 GB/s DRAM
// ceiling), i.e. it is instruction bound (or block starved on narrow outputs),
// and the GEMV wins by 1.03x to 9.4x. At and above L3 the WMMA path drops to
// 570-740 GB/s == the DRAM ceiling and the GEMV is equal or slightly slower
// (0.93x to 0.99x), so the WMMA path keeps the shape. Measured in
// docs/research/20261118/mxfp4_gemv.md.
inline constexpr uint64_t kMxfp4GemmGemvMaxWeightBytes = 64ull * 1024ull * 1024ull;

enum class Mxfp4GemmConfigId : uint8_t {
    Prefill2D,
    SplitK,
    RowBlock1,
    RowBlock2,
    RowBlock4,
    Gemv,
};

struct Mxfp4GemmConfig {
    Mxfp4GemmConfigId id = Mxfp4GemmConfigId::RowBlock1;
    uint8_t split_k = 8u;
    // 0 selects the launcher's row-count rule, 1 or 2 forces the activation
    // row tile count for split-K.
    uint8_t split_mb = 0u;
};

inline constexpr uint32_t kMxfp4GemmPrefill2dKChunk = 128u;
inline constexpr uint32_t kMxfp4GemmPrefill2dOutBlock = 64u;
inline constexpr uint32_t kMxfp4GemmPrefill2dMinRows = 512u;
inline constexpr uint64_t kMxfp4GemmPrefill2dMinWeights = 40000000ull;
inline constexpr uint32_t kMxfp4GemmPrefill2dRowsPerBlock = 16u * kMxfp4GemmRowTile;

// Fold each 32-K block's E8M0 exponent into the E4M3 weight operand relative to
// the output row's reference exponent, and apply 2^ref once per output. For a
// fold distance <= 8 the folded operand is exact, so the result stays bit
// identical to the unfolded form (d > 8 is the E4M3 subnormal tail).
//
// Only the prefill 2D variant folds: it measured -22% to -36% on rows >= 512.
// The row-block and split-K variants measured a loss at rows == 1 (the table
// lookup per K block costs more than the rescale it removes), so they keep the
// unfolded form. Measured in docs/research/20261103/mxfp4_gate_a_prime.md.
inline constexpr bool mxfp4_gemm_folds_row_ref(Mxfp4GemmConfigId id) {
    return id == Mxfp4GemmConfigId::Prefill2D;
}

struct Fp8FinegrainedGemmConfig {
    Fp8FinegrainedGemmConfigId id = Fp8FinegrainedGemmConfigId::RowBlock1;
    uint8_t split_k = 8u;
    // 0 selects the launcher's row-count rule, 1 or 2 forces the activation
    // row tile count for split-K.
    uint8_t split_mb = 0u;
};

}  // namespace shelf::kernel
