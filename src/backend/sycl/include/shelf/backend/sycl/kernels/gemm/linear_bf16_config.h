// (SYCL port of backend/hip kernels/gemm/linear_bf16_config.h — unchanged
// except namespace.)
#pragma once

#include <cstdint>

namespace shelf::syclk {

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

} // namespace shelf::syclk
