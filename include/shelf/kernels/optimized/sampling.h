#pragma once

#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_apply_token_bitmask_f32(
    float* logits, const ::shelf::runtime::DeviceSamplingParams* sampling,
    uint32_t outputs, uint32_t vocab_size, uint32_t row_stride, hipStream_t stream);

struct ArgMaxPair {
    float value;
    uint32_t token;
};

static_assert(sizeof(ArgMaxPair) == 8);

struct ArgmaxF32Args {
    const float* logits = nullptr;
    const ::shelf::runtime::DeviceSamplingParams* sampling = nullptr;
    int32_t* sampled_tokens = nullptr;
    uint32_t outputs = 0;
    uint32_t vocab_size = 0;
    uint32_t logits_row_stride = 0;
    const uint32_t* output_rows = nullptr;
};

hipError_t launch_argmax_f32_single(
    const ArgmaxF32Args& args,
    hipStream_t stream);

hipError_t launch_argmax_f32_partitioned(
    const ArgmaxF32Args& args,
    ArgMaxPair* partials,
    uint32_t partial_capacity,
    uint32_t partitions,
    hipStream_t stream);

struct ArgMaxTop2 {
    float v1;
    uint32_t t1;
    float v2;
    uint32_t t2;
};

hipError_t launch_argmax_f32_top2(
    const float* logits,
    const ::shelf::runtime::DeviceSamplingParams* sampling,
    ArgMaxTop2* out,
    uint32_t outputs,
    uint32_t vocab_size,
    uint32_t logits_row_stride,
    hipStream_t stream);

}  // namespace shelf::kernel
