#include <shelf/inference/execution/selectors/sampling_selector.h>
#include <shelf/kernels/optimized/sampling_limits.h>

namespace shelf::execution {
namespace {

SamplingBatchMode derive_mode(const SamplingSelectorInput& in) {
    if (in.stochastic_outputs != 0) return SamplingBatchMode::HasStochastic;
    if (in.sampled_outputs == 0) return SamplingBatchMode::NoSampling;
    if (in.sampled_outputs == in.outputs) return SamplingBatchMode::AllGreedy;
    return SamplingBatchMode::Mixed;
}

const SamplingRule kRules[] = {
    {248320, SamplingBatchMode::HasStochastic, 1, 256, SamplingImplementation::StochasticSingleBlock, 0},
    {248320, SamplingBatchMode::AllGreedy, 1, 64, SamplingImplementation::Partitioned, 64},
    {248320, SamplingBatchMode::Mixed, 1, 64, SamplingImplementation::Partitioned, 64},
    {248320, SamplingBatchMode::NoSampling, 1, 256, SamplingImplementation::SingleBlock, 0},
};

}  // namespace

SamplingChoice select_sampling_implementation(const SamplingSelectorInput& in) {
    if (in.outputs == 0) return {SamplingImplementation::Correctness, 0};
    const SamplingBatchMode mode = derive_mode(in);
    if (mode == SamplingBatchMode::HasStochastic) {
        const bool topk_ready = in.stochastic_topk_eligible &&
                                in.stochastic_outputs == in.outputs &&
                                in.sampled_outputs == in.outputs &&
                                in.stochastic_top_k > 0u &&
                                in.stochastic_top_k <= ::shelf::kernel::kSamplingMaxTopK &&
                                in.stochastic_topk_workspace;
        if (topk_ready) return {SamplingImplementation::StochasticRadixTopK, 0};
    }
    for (const auto& r : kRules) {
        if (r.vocab_size != in.vocab_size) continue;
        if (r.batch_mode != mode) continue;
        if (in.outputs < r.min_outputs || in.outputs > r.max_outputs) continue;
        if (r.implementation == SamplingImplementation::Partitioned) {
            if (r.partitions == 0 ||
                static_cast<uint64_t>(in.outputs) * r.partitions >
                    in.scratch_pair_capacity)
                continue;
        }
        return {r.implementation, r.partitions};
    }
    return {SamplingImplementation::Correctness, 0};
}

}  // namespace shelf::execution
