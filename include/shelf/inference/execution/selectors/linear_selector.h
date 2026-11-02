#pragma once

#include <shelf/kernels/optimized/linear/config.h>
#include <cstdint>
#include <optional>

namespace ps::execution {

enum class LinearComputeFamily : uint8_t {
    Bf16,
    Fp8W8A8,
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

struct Fp8GemmSelectorInput {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

bool linear_shape_supported(
    LinearComputeFamily family,
    uint32_t out_features,
    uint32_t k);

std::optional<ps::kernel::Bf16GemmConfig>
select_bf16_gemm_config(const Bf16GemmSelectorInput& input);

std::optional<ps::kernel::Fp8GemmConfig>
select_fp8_gemm_config(const Fp8GemmSelectorInput& input);

}  // namespace ps::execution
