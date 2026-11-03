#pragma once
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {


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
    const ::shelf::bf16_t* input,
    const uint8_t* codes,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride);


hipError_t launch_activation_quantize_fp8_finegrained(
    const ::shelf::bf16_t* input,
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

// MXFP4 W4A8 activation form: E8M0 group scales folded into the E4M3 codes against
// the row's maximum group exponent, plus one f32 row reference in scales column 0.
// Not yet reachable from the graph; see docs/research/20261103/mxfp4_act_rowref_fold_plan.md.
hipError_t launch_activation_quantize_mxfp4_rowref(
    const ::shelf::bf16_t* input,
    uint8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t k_padded,
    uint32_t input_row_stride,
    uint32_t code_row_stride,
    uint32_t scale_row_stride,
    hipStream_t stream);

hipError_t launch_activation_quantize_fp8_finegrained_generic(
    const ::shelf::bf16_t* input,
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
