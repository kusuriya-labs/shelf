#pragma once

#include <shelf/kernels/optimized/linear/config.h>
#include <cstdint>
#include <optional>

namespace shelf::execution {

enum class LinearComputeFamily : uint8_t {
    Bf16,
    Fp8FinegrainedW8A8,
    Mxfp4W4A8,
};

struct Bf16GemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    bool weight_aligned16 = false;
    bool input_aligned16 = false;
    bool verify_exact = false;
    bool allow_exact_rows = true;
    bool allow_dot2_rows = false;
};

struct Fp8FinegrainedGemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

struct Mxfp4GemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

bool linear_shape_supported(
    LinearComputeFamily family,
    uint32_t out_features,
    uint32_t k);

std::optional<shelf::kernel::Bf16GemmConfig>
select_bf16_gemm_config(const Bf16GemmSelectorInput& input);

std::optional<shelf::kernel::Fp8FinegrainedGemmConfig>
select_fp8_finegrained_gemm_config(const Fp8FinegrainedGemmSelectorInput& input);

std::optional<shelf::kernel::Mxfp4GemmConfig>
select_mxfp4_gemm_config(const Mxfp4GemmSelectorInput& input);

}  // namespace shelf::execution
