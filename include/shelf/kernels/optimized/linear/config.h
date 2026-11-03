#pragma once

#include <cstdint>

namespace shelf::kernel {

inline constexpr uint32_t kBf16GemmExactRowsMax = 64u;
inline constexpr uint32_t kBf16GemmExactRowsKernelMax = 16u;
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
};

// MXFP4 W4A8 decode / verify row counts run on the WMMA row-block family:
// SplitK (small out_features, K split across waves) or RowBlock (16 outputs per
// wave, MB activation row tiles). rows == 1 has no dedicated variant: the FP8
// linear path removed its decode1 kernel and MXFP4 follows the same contract.
inline constexpr uint32_t kMxfp4GemmRowTile = 16u;
// Measured on R9700 with shelf-bench gemm --dtype mxfp4 (docs/research).
inline constexpr uint32_t kMxfp4GemmSplitKMaxRows = 32u;
inline constexpr uint32_t kMxfp4GemmSplitKMaxOutFeatures = 2560u;
inline constexpr uint32_t kMxfp4GemmSplitKWays = 8u;
inline constexpr uint32_t kMxfp4GemmRowBlock1MaxRows = 16u;
inline constexpr uint32_t kMxfp4GemmRowBlock2MaxRows = 192u;

enum class Mxfp4GemmConfigId : uint8_t {
    Prefill2D,
    SplitK,
    RowBlock1,
    RowBlock2,
    RowBlock4,
};

struct Mxfp4GemmConfig {
    Mxfp4GemmConfigId id = Mxfp4GemmConfigId::RowBlock1;
    uint8_t split_k = 8u;
};

inline constexpr uint32_t kMxfp4GemmPrefill2dKChunk = 128u;
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
};

}  // namespace shelf::kernel
