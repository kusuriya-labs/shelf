#pragma once

#include <shelf/kernels/optimized/sampling_common.h>
#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

struct StochasticSamplingF32Args {
    const float* logits = nullptr;
    const ::shelf::runtime::DeviceSamplingParams* sampling = nullptr;
    int32_t* sampled_tokens = nullptr;
    uint32_t outputs = 0;
    uint32_t vocab_size = 0;
    uint32_t logits_row_stride = 0;
    uint32_t* error_word = nullptr;
    uint32_t* attempts_out = nullptr;
};

struct StochasticTopKSamplingArgs {
    const float* logits = nullptr;
    const ::shelf::runtime::DeviceSamplingParams* sampling = nullptr;
    int32_t* sampled_tokens = nullptr;
    const int32_t* top_ids = nullptr;
    const float* top_logits = nullptr;
    uint32_t outputs = 0;
    uint32_t vocab_size = 0;
    uint32_t logits_row_stride = 0;
    uint32_t top_k = 0;
    uint32_t* error_word = nullptr;
    uint32_t* attempts_out = nullptr;
    uint32_t* active_count_out = nullptr;
};

hipError_t launch_stochastic_sampling_f32_singleblock(
    const StochasticSamplingF32Args& args,
    hipStream_t stream);

hipError_t launch_stochastic_topk_sampling_f32(
    const StochasticTopKSamplingArgs& args,
    hipStream_t stream);

}  // namespace shelf::kernel
