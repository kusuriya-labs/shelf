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

// MXFP4 W4A8 covers decode / verify row counts (Decode1, SplitK, RowBlock) with
// one wave per 16 outputs. Prefill row counts need the 2D staged variant, which
// is not implemented yet.
inline constexpr uint32_t kMxfp4GemmRowTile = 16u;
// Measured on R9700 with shelf-bench gemm --dtype mxfp4 (docs/research).
inline constexpr uint32_t kMxfp4GemmDecode1Unroll = 2u;
inline constexpr uint32_t kMxfp4GemmSplitKMaxRows = 32u;
inline constexpr uint32_t kMxfp4GemmSplitKMaxOutFeatures = 2560u;
inline constexpr uint32_t kMxfp4GemmSplitKWays = 8u;
inline constexpr uint32_t kMxfp4GemmRowBlock1MaxRows = 16u;
inline constexpr uint32_t kMxfp4GemmRowBlock2MaxRows = 192u;

enum class Mxfp4GemmConfigId : uint8_t {
    Decode1,
    SplitK,
    RowBlock1,
    RowBlock2,
    RowBlock4,
};

struct Mxfp4GemmConfig {
    Mxfp4GemmConfigId id = Mxfp4GemmConfigId::Decode1;
    uint8_t split_k = 4u;
    uint8_t unroll = 0u;
};

struct Fp8FinegrainedGemmConfig {
    Fp8FinegrainedGemmConfigId id = Fp8FinegrainedGemmConfigId::RowBlock1;
};

}  // namespace shelf::kernel
