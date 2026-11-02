#pragma once
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::kernel {

inline constexpr const char* kActivationQuantizeA8Symbol =
    "phaseshift_activation_quantize_a8";

struct ActivationQuantizeA8Args {
    const void* input = nullptr;
    void* codes = nullptr;
    void* scales = nullptr;
    uint32_t rows = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t input_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
};

static_assert(sizeof(ActivationQuantizeA8Args) == 48);
static_assert(alignof(ActivationQuantizeA8Args) == 8);
static_assert(offsetof(ActivationQuantizeA8Args, input) == 0);
static_assert(offsetof(ActivationQuantizeA8Args, codes) == 8);
static_assert(offsetof(ActivationQuantizeA8Args, scales) == 16);
static_assert(offsetof(ActivationQuantizeA8Args, rows) == 24);
static_assert(offsetof(ActivationQuantizeA8Args, k) == 28);
static_assert(offsetof(ActivationQuantizeA8Args, k_padded) == 32);
static_assert(offsetof(ActivationQuantizeA8Args, input_row_stride) == 36);
static_assert(offsetof(ActivationQuantizeA8Args, code_row_stride) == 40);
static_assert(offsetof(ActivationQuantizeA8Args, scale_row_stride) == 44);

inline constexpr const char* kActivationQuantizeE4m3K5120Symbol =
    "phaseshift_activation_quantize_e4m3_k5120";

struct ActivationQuantizeE4m3K5120Args {
    const void* input = nullptr;
    void* codes = nullptr;
    void* scales = nullptr;
    uint32_t input_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
};

static_assert(sizeof(ActivationQuantizeE4m3K5120Args) == 40);
static_assert(alignof(ActivationQuantizeE4m3K5120Args) == 8);
static_assert(offsetof(ActivationQuantizeE4m3K5120Args, input) == 0);
static_assert(offsetof(ActivationQuantizeE4m3K5120Args, codes) == 8);
static_assert(offsetof(ActivationQuantizeE4m3K5120Args, scales) == 16);
static_assert(offsetof(ActivationQuantizeE4m3K5120Args, input_row_stride) == 24);
static_assert(offsetof(ActivationQuantizeE4m3K5120Args, code_row_stride) == 28);
static_assert(offsetof(ActivationQuantizeE4m3K5120Args, scale_row_stride) == 32);

inline constexpr const char* kActivationQuantizeE4m3K6144Symbol =
    "phaseshift_activation_quantize_e4m3_k6144";
inline constexpr const char* kActivationQuantizeE4m3K17408Symbol =
    "phaseshift_activation_quantize_e4m3_k17408";

using ActivationQuantizeE4m3KArgs = ActivationQuantizeE4m3K5120Args;

static_assert(sizeof(ActivationQuantizeE4m3KArgs) == 40);
static_assert(offsetof(ActivationQuantizeE4m3KArgs, input_row_stride) == 24);
static_assert(offsetof(ActivationQuantizeE4m3KArgs, code_row_stride) == 28);
static_assert(offsetof(ActivationQuantizeE4m3KArgs, scale_row_stride) == 32);

bool activation_quantize_e4m3_vec_supported(
    const ::ps::bf16_t* input,
    const uint8_t* codes,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride);

hipError_t launch_activation_quantize_a8(
    const ::ps::bf16_t* input,
    int8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream);

hipError_t launch_activation_quantize_e4m3(
    const ::ps::bf16_t* input,
    uint8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream,
    bool* specialized = nullptr);

hipError_t launch_activation_quantize_e4m3_generic(
    const ::ps::bf16_t* input,
    uint8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream);

hipError_t launch_activation_quantize_i8_row(
    const ::ps::bf16_t* input,
    int8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream);

}
