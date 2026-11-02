#include <shelf/kernels/optimized/stochastic_sampling.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {
namespace {

constexpr uint32_t kThreads = 256u;

__global__ void stochastic_sampling_impl(
    const float* __restrict__ logits,
    const ::ps::runtime::DeviceSamplingParams* __restrict__ sampling,
    int32_t* __restrict__ sampled_tokens,
    uint32_t vocab_size,
    uint32_t logits_row_stride,
    uint32_t* error_word,
    uint32_t* __restrict__ attempts_out) {
    const uint32_t row = blockIdx.x;
    const float* row_logits = logits + static_cast<std::size_t>(row) * logits_row_stride;
    const SamplingRowParams params = sampling_row_params(sampling[row]);
    sampling_sample_row(row_logits, vocab_size, params, sampled_tokens + row, error_word,
                        attempts_out == nullptr ? nullptr : attempts_out + row);
}

}  // namespace

hipError_t launch_stochastic_sampling_f32_singleblock(
    const StochasticSamplingF32Args& args,
    hipStream_t stream) {
    if (args.outputs == 0u) return hipSuccess;
    if (args.vocab_size == 0u) return hipErrorInvalidValue;
    if (args.logits == nullptr || args.sampling == nullptr ||
        args.sampled_tokens == nullptr)
        return hipErrorInvalidValue;
    stochastic_sampling_impl<<<args.outputs, kThreads, 0, stream>>>(
        args.logits, args.sampling, args.sampled_tokens, args.vocab_size,
        args.logits_row_stride, args.error_word, args.attempts_out);
    return hipGetLastError();
}

namespace {

__global__ void stochastic_topk_sampling_impl(
    const float* __restrict__ logits,
    const ::ps::runtime::DeviceSamplingParams* __restrict__ sampling,
    int32_t* __restrict__ sampled_tokens,
    const int32_t* __restrict__ top_ids,
    const float* __restrict__ top_logits,
    uint32_t vocab_size,
    uint32_t logits_row_stride,
    uint32_t top_k,
    uint32_t* error_word,
    uint32_t* __restrict__ attempts_out,
    uint32_t* __restrict__ active_count_out) {
    const uint32_t row = blockIdx.x;
    const float* row_logits = logits + static_cast<std::size_t>(row) * logits_row_stride;
    const SamplingRowParams params = sampling_row_params(sampling[row]);
    SamplingTopKView view;
    view.ids = top_ids + static_cast<std::size_t>(row) * top_k;
    view.logits = top_logits + static_cast<std::size_t>(row) * top_k;
    view.count = top_k;
    sampling_topk_active_row(
        row_logits, vocab_size, params, view, sampled_tokens + row, error_word,
        attempts_out == nullptr ? nullptr : attempts_out + row,
        active_count_out == nullptr ? nullptr : active_count_out + row);
}

}  // namespace

hipError_t launch_stochastic_topk_sampling_f32(
    const StochasticTopKSamplingArgs& args,
    hipStream_t stream) {
    if (args.outputs == 0u) return hipSuccess;
    if (args.vocab_size == 0u) return hipErrorInvalidValue;
    if (args.top_k == 0u || args.top_k > kSamplingMaxTopK) return hipErrorInvalidValue;
    if (args.logits == nullptr || args.sampling == nullptr ||
        args.sampled_tokens == nullptr || args.top_ids == nullptr ||
        args.top_logits == nullptr)
        return hipErrorInvalidValue;
    stochastic_topk_sampling_impl<<<args.outputs, kThreads, 0, stream>>>(
        args.logits, args.sampling, args.sampled_tokens, args.top_ids, args.top_logits,
        args.vocab_size, args.logits_row_stride, args.top_k, args.error_word,
        args.attempts_out, args.active_count_out);
    return hipGetLastError();
}

}  // namespace ps::kernel
