#pragma once

#include <cstdint>

namespace ps::kernel {

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

inline constexpr uint32_t kFp8GemmRowTile = 16u;
inline constexpr uint32_t kFp8GemmPrefill2dOutBlock = 64u;
inline constexpr uint32_t kFp8GemmPrefill2dKChunk = 64u;

enum class Fp8GemmConfigId : uint8_t {
    RowBlock1,
    RowBlock2,
    RowBlock4,
    RowBlock8,
    Prefill2D,
};

struct Fp8GemmConfig {
    Fp8GemmConfigId id = Fp8GemmConfigId::RowBlock1;
};

inline constexpr uint32_t kPsq5GemmRowTile = 16u;
inline constexpr uint32_t kPsq5GemmPrefill2dOutBlock = 64u;
inline constexpr uint32_t kPsq5GemmPrefill2dKChunk = 64u;
inline constexpr uint32_t kPsq5GemmPrefill2dWideOutBlock = 128u;
inline constexpr uint32_t kPsq5GemmPrefill2dWideRowsPerBlock = 128u;

enum class Psq5GemmConfigId : uint8_t {
    RowBlock1,
    RowBlock2,
    RowBlock4,
    RowBlock8,
    Prefill2D,
    Prefill2DWide,
    Prefill2DWStage,
};

struct Psq5GemmConfig {
    Psq5GemmConfigId id = Psq5GemmConfigId::RowBlock1;
};

}  // namespace ps::kernel
