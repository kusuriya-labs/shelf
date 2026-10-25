#pragma once

#include <shelf/backend/hip/kernels/linear_bf16_config.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

using shelf::bf16_t;

inline constexpr const char* kBf16ExactRowsSymbol =
    "phaseshift_gemm_bf16_exact_rows_1";
inline constexpr const char* kBf16WmmaSymbol =
    "phaseshift_gemm_bf16_wmma";
inline constexpr const char* kBf16WmmaWideSymbol =
    "phaseshift_gemm_bf16_wmma_wide";
inline constexpr const char* kBf16SplitKWmmaSymbol =
    "phaseshift_gemm_bf16_splitk_wmma";

inline constexpr const char* kBf16ExactRowsSymbols[16] = {
    "phaseshift_gemm_bf16_exact_rows_1",
    "phaseshift_gemm_bf16_exact_rows_2",
    "phaseshift_gemm_bf16_exact_rows_3",
    "phaseshift_gemm_bf16_exact_rows_4",
    "phaseshift_gemm_bf16_exact_rows_5",
    "phaseshift_gemm_bf16_exact_rows_6",
    "phaseshift_gemm_bf16_exact_rows_7",
    "phaseshift_gemm_bf16_exact_rows_8",
    "phaseshift_gemm_bf16_exact_rows_9",
    "phaseshift_gemm_bf16_exact_rows_10",
    "phaseshift_gemm_bf16_exact_rows_11",
    "phaseshift_gemm_bf16_exact_rows_12",
    "phaseshift_gemm_bf16_exact_rows_13",
    "phaseshift_gemm_bf16_exact_rows_14",
    "phaseshift_gemm_bf16_exact_rows_15",
    "phaseshift_gemm_bf16_exact_rows_16",
};

struct Bf16GemmExactRowsArgs {
    const void* weight = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t output_dtype = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(Bf16GemmExactRowsArgs) == 48);
static_assert(alignof(Bf16GemmExactRowsArgs) == 8);
static_assert(offsetof(Bf16GemmExactRowsArgs, weight) == 0);
static_assert(offsetof(Bf16GemmExactRowsArgs, input) == 8);
static_assert(offsetof(Bf16GemmExactRowsArgs, output) == 16);
static_assert(offsetof(Bf16GemmExactRowsArgs, output_dtype) == 24);
static_assert(offsetof(Bf16GemmExactRowsArgs, n) == 28);
static_assert(offsetof(Bf16GemmExactRowsArgs, k) == 32);
static_assert(offsetof(Bf16GemmExactRowsArgs, input_row_stride) == 36);
static_assert(offsetof(Bf16GemmExactRowsArgs, output_row_stride) == 40);

struct Bf16GemmWmmaArgs {
    const void* weight = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t output_dtype = 0;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
};

static_assert(sizeof(Bf16GemmWmmaArgs) == 48);
static_assert(alignof(Bf16GemmWmmaArgs) == 8);
static_assert(offsetof(Bf16GemmWmmaArgs, weight) == 0);
static_assert(offsetof(Bf16GemmWmmaArgs, input) == 8);
static_assert(offsetof(Bf16GemmWmmaArgs, output) == 16);
static_assert(offsetof(Bf16GemmWmmaArgs, output_dtype) == 24);
static_assert(offsetof(Bf16GemmWmmaArgs, rows) == 28);
static_assert(offsetof(Bf16GemmWmmaArgs, out_features) == 32);
static_assert(offsetof(Bf16GemmWmmaArgs, k) == 36);
static_assert(offsetof(Bf16GemmWmmaArgs, input_row_stride) == 40);
static_assert(offsetof(Bf16GemmWmmaArgs, output_row_stride) == 44);

inline constexpr uint32_t kBf16WmmaOutTile = 16u;
inline constexpr uint32_t kBf16WmmaRowTile = 16u;
inline constexpr uint32_t kBf16WmmaWideRowTiles = 4u;
inline constexpr uint32_t kBf16WmmaThreads = 32u;
inline constexpr uint32_t kBf16SplitKWmmaThreads = 256u;

constexpr uint32_t bf16_wmma_grid_x(uint32_t out_features) noexcept {
    return (out_features + kBf16WmmaOutTile - 1u) / kBf16WmmaOutTile;
}

constexpr uint32_t bf16_wmma_grid_y(uint32_t rows, uint32_t row_tiles) noexcept {
    const uint32_t rows_per_block = row_tiles * kBf16WmmaRowTile;
    return (rows + rows_per_block - 1u) / rows_per_block;
}

enum class Bf16GemmOutputDType : uint8_t {
    BF16,
    F32,
};

hipError_t launch_gemm_bf16(
    const Bf16GemmConfig& config,
    const bf16_t* weight,
    const bf16_t* input,
    void* output,
    Bf16GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    hipStream_t stream);

}  // namespace shelf::hipk
