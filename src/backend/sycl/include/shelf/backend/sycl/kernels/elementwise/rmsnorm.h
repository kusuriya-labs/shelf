// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port; launchers return shelf::Status.)
#pragma once

#include <cstddef>
#include <cstdint>

#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

enum class RmsNormDataType : uint8_t {
    BF16,
    F32,
};

// DIRECT   : out = rmsnorm(x) * w                (plain gamma)
// ONE_PLUS : out = rmsnorm(x) * (1.0f + w)       (zero-centered gamma)
enum class RmsNormWeightMode : uint8_t {
    ONE_PLUS,
    DIRECT,
};

// NONE        : no weight applied
// PER_FEATURE : weight has `features` entries (indexed by global feature)
// PER_GROUP   : weight has `group_size` entries, reused for every group
enum class RmsNormWeightLayout : uint8_t {
    NONE,
    PER_FEATURE,
    PER_GROUP,
};

::shelf::Status launch_rmsnorm(
    const void* input,
    RmsNormDataType input_dtype,
    const void* weight,
    RmsNormDataType weight_dtype,
    RmsNormWeightLayout weight_layout,
    RmsNormWeightMode weight_mode,
    void* output,
    RmsNormDataType output_dtype,
    uint32_t rows,
    uint32_t features,
    uint32_t group_size,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    float eps,
    const ::shelf::sycl::Stream& stream);

// exposed for tests / launch config introspection (mirrors hip names)
uint32_t rmsnorm_bf16_pf_oneplus_block_size(uint32_t group_size) noexcept;
uint32_t rmsnorm_bf16_pf_oneplus_grid(uint32_t rows, uint32_t features,
                                      uint32_t group_size) noexcept;

} // namespace shelf::syclk
