#pragma once
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::kernel {


inline constexpr const char* kActivationQuantizeFp8FinegrainedK5120Symbol =
    "shelf_activation_quantize_fp8_finegrained_k5120";

struct ActivationQuantizeFp8FinegrainedK5120Args {
    const void* input = nullptr;
    void* codes = nullptr;
    void* scales = nullptr;
    uint32_t input_row_stride = 0;
    uint32_t code_row_stride = 0;
    uint32_t scale_row_stride = 0;
};

static_assert(sizeof(ActivationQuantizeFp8FinegrainedK5120Args) == 40);
static_assert(alignof(ActivationQuantizeFp8FinegrainedK5120Args) == 8);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedK5120Args, input) == 0);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedK5120Args, codes) == 8);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedK5120Args, scales) == 16);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedK5120Args, input_row_stride) == 24);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedK5120Args, code_row_stride) == 28);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedK5120Args, scale_row_stride) == 32);

inline constexpr const char* kActivationQuantizeFp8FinegrainedK6144Symbol =
    "shelf_activation_quantize_fp8_finegrained_k6144";
inline constexpr const char* kActivationQuantizeFp8FinegrainedK17408Symbol =
    "shelf_activation_quantize_fp8_finegrained_k17408";

using ActivationQuantizeFp8FinegrainedKArgs = ActivationQuantizeFp8FinegrainedK5120Args;

static_assert(sizeof(ActivationQuantizeFp8FinegrainedKArgs) == 40);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedKArgs, input_row_stride) == 24);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedKArgs, code_row_stride) == 28);
static_assert(offsetof(ActivationQuantizeFp8FinegrainedKArgs, scale_row_stride) == 32);

bool activation_quantize_fp8_finegrained_vec_supported(
    const ::ps::bf16_t* input,
    const uint8_t* codes,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride);


hipError_t launch_activation_quantize_fp8_finegrained(
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

hipError_t launch_activation_quantize_fp8_finegrained_generic(
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

}
