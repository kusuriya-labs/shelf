#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::kernel {

enum class ElementwiseSplitLayout : uint8_t { Halves, InterleavedHeads };

inline constexpr uint32_t kElementwiseBlock = 256u;
inline constexpr uint32_t kElementwiseVec = 4u;

constexpr uint32_t elementwise_grid_x(uint32_t features) {
    const uint64_t span =
        static_cast<uint64_t>(kElementwiseBlock) * kElementwiseVec;
    return static_cast<uint32_t>((static_cast<uint64_t>(features) + span - 1u) /
                                 span);
}

// Rows are addressed by blockIdx.y + blockIdx.z * gridDim.y: a HIP grid's y dimension is
// bounded by 65535 (a launch with grid.y = 69632 is rejected with `invalid configuration
// argument`), so rows above that bound are carried by grid.z and one launch still covers the
// whole row domain (MoE group rows reach 69632 at the largest production bucket). grid.y is
// the even split of rows over grid.z, so the launch wastes at most grid.z - 1 row slots; the
// kernels drop those with a row bound check.
inline constexpr uint32_t kElementwiseGridYMax = 65535u;

constexpr uint32_t elementwise_grid_z(uint32_t rows) {
    return (rows + kElementwiseGridYMax - 1u) / kElementwiseGridYMax;
}

constexpr uint32_t elementwise_grid_y(uint32_t rows) {
    const uint32_t gz = elementwise_grid_z(rows);
    return (rows + gz - 1u) / gz;
}

inline dim3 elementwise_grid(uint32_t features, uint32_t rows) {
    return dim3(elementwise_grid_x(features), elementwise_grid_y(rows),
                elementwise_grid_z(rows));
}

inline constexpr const char* kElementwiseResidualAddBf16Symbol =
    "shelf_residual_add_bf16";
inline constexpr const char* kElementwiseSwigluBf16Symbol =
    "shelf_swiglu_bf16";
inline constexpr const char* kElementwiseMulF32F32Bf16Symbol =
    "shelf_mul_f32_f32_bf16";
inline constexpr const char* kElementwiseScaleF32Symbol =
    "shelf_scale_f32";
inline constexpr const char* kElementwiseScaleBf16Symbol =
    "shelf_scale_bf16";
inline constexpr const char* kElementwiseSiluBf16ToF32Symbol =
    "shelf_silu_bf16_f32";
inline constexpr const char* kElementwiseSiluBf16Symbol =
    "shelf_silu_bf16";
inline constexpr const char* kElementwiseSigmoidBf16Symbol =
    "shelf_sigmoid_bf16";
inline constexpr const char* kElementwiseSigmoidF32Symbol =
    "shelf_sigmoid_f32";
inline constexpr const char* kElementwiseSiluF32ToBf16Symbol =
    "shelf_silu_f32_bf16";
inline constexpr const char* kElementwiseSigmoidBf16ToF32Symbol =
    "shelf_sigmoid_bf16_f32";
inline constexpr const char* kElementwiseSplitBf16Symbol =
    "shelf_split_bf16";

struct ElementwiseArgs {
    const void* in0 = nullptr;
    uint32_t in0_row_stride = 0;
    const void* in1 = nullptr;
    uint32_t in1_row_stride = 0;
    void* out0 = nullptr;
    uint32_t out0_row_stride = 0;
    void* out1 = nullptr;
    uint32_t out1_row_stride = 0;
    uint32_t features = 0;
    uint32_t features_b = 0;
    uint32_t aux = 0;
    uint32_t reserved = 0;
    float scalar_a = 0.0f;
    uint32_t rows = 0;
};

static_assert(sizeof(ElementwiseArgs) == 88);
static_assert(alignof(ElementwiseArgs) == 8);
static_assert(offsetof(ElementwiseArgs, in0) == 0);
static_assert(offsetof(ElementwiseArgs, in0_row_stride) == 8);
static_assert(offsetof(ElementwiseArgs, in1) == 16);
static_assert(offsetof(ElementwiseArgs, in1_row_stride) == 24);
static_assert(offsetof(ElementwiseArgs, out0) == 32);
static_assert(offsetof(ElementwiseArgs, out0_row_stride) == 40);
static_assert(offsetof(ElementwiseArgs, out1) == 48);
static_assert(offsetof(ElementwiseArgs, out1_row_stride) == 56);
static_assert(offsetof(ElementwiseArgs, features) == 60);
static_assert(offsetof(ElementwiseArgs, features_b) == 64);
static_assert(offsetof(ElementwiseArgs, aux) == 68);
static_assert(offsetof(ElementwiseArgs, scalar_a) == 76);
static_assert(offsetof(ElementwiseArgs, rows) == 80);

hipError_t launch_residual_add_bf16(
    const bf16_t* a, uint32_t a_row_stride,
    const bf16_t* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_swiglu_bf16(
    const bf16_t* gate, uint32_t gate_row_stride,
    const bf16_t* up, uint32_t up_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_sigmoid_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_sigmoid_f32(
    const float* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_sigmoid_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_silu_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_silu_f32_bf16(
    const float* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_mul_f32_f32_bf16(
    const float* a, uint32_t a_row_stride,
    const float* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_scale_f32(
    const float* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, float scale, hipStream_t stream);

hipError_t launch_scale_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, float scale, hipStream_t stream);

hipError_t launch_scale_bf16_f32(const bf16_t* input, uint32_t input_stride,
    float* output, uint32_t output_stride, uint32_t rows, uint32_t features,
    float scale, hipStream_t stream);

hipError_t launch_mul_bf16(
    const bf16_t* a, uint32_t a_stride, const bf16_t* b, uint32_t b_stride,
    bf16_t* out, uint32_t out_stride, uint32_t rows, uint32_t features, hipStream_t stream);

hipError_t launch_split_uneven_bf16(
    const bf16_t* input, uint32_t input_row_stride, bf16_t* o0, uint32_t o0_row_stride,
    bf16_t* o1, uint32_t o1_row_stride, uint32_t rows, uint32_t features_a,
    uint32_t features_b, hipStream_t stream);

hipError_t launch_split_bf16(
    const bf16_t* input, uint32_t input_row_stride,
    bf16_t* o0, uint32_t o0_row_stride,
    bf16_t* o1, uint32_t o1_row_stride,
    uint32_t rows, uint32_t features,
    ElementwiseSplitLayout layout, uint32_t head_dim, hipStream_t stream);

hipError_t launch_concat_bf16(
    const bf16_t* a, uint32_t a_row_stride,
    const bf16_t* b, uint32_t b_row_stride,
    bf16_t* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features_a, uint32_t features_b, hipStream_t stream);

}  // namespace shelf::kernel
