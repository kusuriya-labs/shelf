// vendored from PhaseShift (own work); namespace/type adapted only.
#pragma once

#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

using shelf::bf16_t;

enum class ElementwiseSplitLayout : uint8_t { Halves, InterleavedHeads };

inline constexpr uint32_t kElementwiseBlock = 256u;
inline constexpr uint32_t kElementwiseVec = 4u;

constexpr uint32_t elementwise_grid_x(uint32_t features) {
    const uint64_t span =
        static_cast<uint64_t>(kElementwiseBlock) * kElementwiseVec;
    return static_cast<uint32_t>((static_cast<uint64_t>(features) + span - 1u) /
                                 span);
}

inline constexpr const char* kElementwiseResidualAddBf16Symbol =
    "phaseshift_residual_add_bf16";
inline constexpr const char* kElementwiseSwigluBf16Symbol =
    "phaseshift_swiglu_bf16";
inline constexpr const char* kElementwiseMulF32F32Bf16Symbol =
    "phaseshift_mul_f32_f32_bf16";
inline constexpr const char* kElementwiseScaleF32Symbol =
    "phaseshift_scale_f32";
inline constexpr const char* kElementwiseSiluBf16ToF32Symbol =
    "phaseshift_silu_bf16_f32";
inline constexpr const char* kElementwiseSiluF32ToBf16Symbol =
    "phaseshift_silu_f32_bf16";
inline constexpr const char* kElementwiseSigmoidBf16ToF32Symbol =
    "phaseshift_sigmoid_bf16_f32";
inline constexpr const char* kElementwiseSplitBf16Symbol =
    "phaseshift_split_bf16";

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

}  // namespace shelf::hipk
