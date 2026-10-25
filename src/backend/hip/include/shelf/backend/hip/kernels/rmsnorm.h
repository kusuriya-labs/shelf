// vendored from PhaseShift (own work); namespace/type adapted only.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

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

inline constexpr const char* kRmsNormBf16PfOnePlusSymbol =
    "phaseshift_rmsnorm_bf16_pf_oneplus";

inline constexpr const char* kRmsNormF32PgDirectSymbol =
    "phaseshift_rmsnorm_f32_pg_direct";

inline constexpr const char* kRmsNormBf16F32PgOnePlusSymbol =
    "phaseshift_rmsnorm_bf16_f32_pg_oneplus";

struct RmsNormF32PgDirectArgs {
    const void* input = nullptr;
    const void* weight = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    float eps = 0.0f;
};

static_assert(sizeof(RmsNormF32PgDirectArgs) == 48);
static_assert(alignof(RmsNormF32PgDirectArgs) == 8);
static_assert(offsetof(RmsNormF32PgDirectArgs, input) == 0);
static_assert(offsetof(RmsNormF32PgDirectArgs, weight) == 8);
static_assert(offsetof(RmsNormF32PgDirectArgs, output) == 16);
static_assert(offsetof(RmsNormF32PgDirectArgs, rows) == 24);
static_assert(offsetof(RmsNormF32PgDirectArgs, features) == 28);
static_assert(offsetof(RmsNormF32PgDirectArgs, group_size) == 32);
static_assert(offsetof(RmsNormF32PgDirectArgs, input_row_stride) == 36);
static_assert(offsetof(RmsNormF32PgDirectArgs, output_row_stride) == 40);
static_assert(offsetof(RmsNormF32PgDirectArgs, eps) == 44);

struct RmsNormBf16F32PgOnePlusArgs {
    const void* input = nullptr;
    const void* weight = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    float eps = 0.0f;
};

struct RmsNormBf16PfOnePlusArgs {
    const void* input = nullptr;
    const void* weight = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t features = 0;
    uint32_t group_size = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    float eps = 0.0f;
};

static_assert(sizeof(RmsNormBf16PfOnePlusArgs) == 48);
static_assert(alignof(RmsNormBf16PfOnePlusArgs) == 8);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, input) == 0);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, weight) == 8);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, output) == 16);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, rows) == 24);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, features) == 28);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, group_size) == 32);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, input_row_stride) == 36);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, output_row_stride) == 40);
static_assert(offsetof(RmsNormBf16PfOnePlusArgs, eps) == 44);

uint32_t rmsnorm_bf16_pf_oneplus_block_size(uint32_t group_size) noexcept;
uint32_t rmsnorm_bf16_pf_oneplus_grid(uint32_t rows, uint32_t features,
                                      uint32_t group_size) noexcept;

hipError_t launch_rmsnorm(
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
    hipStream_t stream);

}  // namespace shelf::hipk
