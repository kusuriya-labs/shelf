// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/elementwise/l2_normalize.h.)
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

using shelf::bf16_t;

inline constexpr uint32_t kL2NormalizeThreads = 32u;

constexpr uint32_t l2_normalize_groups(uint32_t features, uint32_t group_size) {
    return (group_size == 0u || features % group_size != 0u) ? 0u
                                                             : features / group_size;
}

struct L2NormalizeArgs {
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t group_size = 0;
    float eps = 0.0f;
};

static_assert(sizeof(L2NormalizeArgs) == 32);
static_assert(alignof(L2NormalizeArgs) == 8);
static_assert(offsetof(L2NormalizeArgs, input) == 0);
static_assert(offsetof(L2NormalizeArgs, output) == 8);
static_assert(offsetof(L2NormalizeArgs, input_row_stride) == 16);
static_assert(offsetof(L2NormalizeArgs, output_row_stride) == 20);
static_assert(offsetof(L2NormalizeArgs, group_size) == 24);
static_assert(offsetof(L2NormalizeArgs, eps) == 28);

// inv = rsqrt(sum(x^2) + eps) over each group of `group_size` elements.
::shelf::Status launch_l2_normalize_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, uint32_t group_size,
    float eps, const ::shelf::sycl::Stream& stream);

} // namespace shelf::syclk
