// (SYCL port of backend/hip kernels/gemm/linear_bf16.h.)
//
// Config identity is preserved for the executor's dispatch policy, but note
// the Wmma/WmmaWide/WmmaKPartition entries are now implemented with
// per-lane scalar FMA tiles (no gfx12 __builtin_amdgcn_wmma_*; SYCL 2020
// has no tensor-core access and AdaptiveCpp exposes no DPNAS extension).
// The ExactRowsDot2 config maps to the same scalar accumulation order as
// ExactRows in this port (the amdgcn fdot2 builtin has no portable
// equivalent; results are identical to the scalar path by construction).
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/backend/sycl/kernels/gemm/linear_bf16_config.h>
#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

using shelf::bf16_t;

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

::shelf::Status launch_gemm_bf16(
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
    const ::shelf::sycl::Stream& stream);

} // namespace shelf::syclk
