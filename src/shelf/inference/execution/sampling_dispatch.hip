#include <shelf/inference/execution/sampling_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/optimized/sampling.h>
#include <shelf/kernels/optimized/stochastic_sampling.h>
#include <shelf/kernels/optimized/radix_topn.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/selectors/elementwise_selector.h>
#include <shelf/inference/execution/selectors/embedding_selector.h>
#include <shelf/inference/execution/selectors/gdn_conv_selector.h>
#include <shelf/inference/execution/selectors/gdn_recurrence_selector.h>
#include <shelf/inference/execution/selectors/kv_append_selector.h>
#include <shelf/inference/execution/selectors/l2_normalize_selector.h>
#include <shelf/inference/execution/selectors/linear_selector.h>
#include <shelf/inference/execution/selectors/paged_attention_selector.h>
#include <shelf/inference/execution/selectors/rmsnorm_selector.h>
#include <shelf/inference/execution/selectors/rope_selector.h>
#include <shelf/inference/execution/selectors/sampling_selector.h>
#include <shelf/inference/execution/selectors/selector_limits.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::execution {
namespace {

using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueRowDomain;

constexpr OptimizedLaunchResult kNone{OptimizedLaunchKind::NotApplicable};

Result<OptimizedLaunchResult> resolve_sampling(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u) return kNone;

    auto in_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return kNone;
    const HostResolvedValue in = in_res.value();
    if (in.dtype != ValueDType::F32) return kNone;
    if (in.row_domain != ValueRowDomain::OUTPUT_ROWS) return kNone;
    if (in.ptr == nullptr || in.feature_count == 0u) return kNone;
    if (in.row_stride != in.feature_count) return kNone;

    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return kNone;
    const HostResolvedValue out = out_res.value();
    if (out.dtype != ValueDType::I32) return kNone;
    if (out.row_domain != ValueRowDomain::OUTPUT_ROWS) return kNone;
    if (out.ptr == nullptr || out.feature_count != 1u) return kNone;

    const uint32_t vocab = in.feature_count;
    if (b.rotary_dim != vocab) return kNone;

    const uint32_t outputs = ctx.actual_outputs;
    if (outputs == 0u)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (ctx.output_sampling_params == nullptr) return kNone;

    const uint32_t sampled = ctx.actual_sampled_outputs;
    if (sampled > outputs) return kNone;

    SamplingSelectorInput si;
    si.vocab_size = vocab;
    si.outputs = outputs;
    si.sampled_outputs = sampled;
    si.stochastic_outputs = ctx.actual_stochastic_outputs;
    si.scratch_pair_capacity =
        ctx.scratch != nullptr ? (ctx.scratch_floats / 2u) : 0u;
    si.stochastic_topk_eligible = ctx.stochastic_topk_eligible;
    si.stochastic_top_k = ctx.stochastic_top_k;
    si.stochastic_topk_workspace =
        ctx.stochastic_topk_ids != nullptr && ctx.stochastic_topk_logits != nullptr &&
        ctx.stochastic_topk_scratch != nullptr && ctx.stochastic_topk_scratch_bytes > 0u;

    SamplingChoice choice = select_sampling_implementation(si);
    bool has_token_bitmask = false;
    for (uint32_t i = 0; i < ctx.staged_request_count; ++i) {
        if (ctx.staged_requests[i].sampling.token_bitmask != nullptr) {
            has_token_bitmask = true;
            const uint32_t stride = ctx.staged_requests[i].sampling.token_bitmask_row_stride;
            if (stride != 0u && stride < (vocab + 31u) / 32u) {
                return Status::invalid_argument("token bitmask row stride is too small",
                                                __FILE__, __LINE__);
            }
        }
    }
    if (choice.implementation == SamplingImplementation::Correctness) {
        if (has_token_bitmask)
            return Status::unsupported("token bitmask requires optimized sampling", __FILE__, __LINE__);
        return kNone;
    }
    if (has_token_bitmask) {
        const hipError_t err = ::shelf::kernel::launch_apply_token_bitmask_f32(
            static_cast<float*>(in.ptr), ctx.output_sampling_params, outputs, vocab,
            in.row_stride, stream);
        if (err != hipSuccess)
            return Status::hip_error("token bitmask launch", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        record_physical_launch(1);
    }

    if (choice.implementation == SamplingImplementation::StochasticRadixTopK) {
        if (ctx.stochastic_topk_ids == nullptr || ctx.stochastic_topk_logits == nullptr ||
            ctx.stochastic_topk_scratch == nullptr ||
            ctx.stochastic_topk_scratch_bytes == 0u ||
            ctx.stochastic_topk_partitions == 0u || ctx.stochastic_top_k == 0u)
            return kNone;
        const hipError_t rerr = ::shelf::kernel::launch_dflash2_radix_topn(
            static_cast<const float*>(in.ptr), outputs, vocab, in.row_stride,
            ctx.stochastic_top_k, ctx.stochastic_topk_ids, ctx.stochastic_topk_logits,
            ctx.stochastic_topk_partitions, ctx.stochastic_topk_scratch,
            ctx.stochastic_topk_scratch_bytes, stream);
        if (rerr != hipSuccess)
            return Status::hip_error("stochastic topk radix launch", hipGetErrorString(rerr),
                                     __FILE__, __LINE__);
        ::shelf::kernel::StochasticTopKSamplingArgs targs;
        targs.logits = static_cast<const float*>(in.ptr);
        targs.sampling = ctx.output_sampling_params;
        targs.sampled_tokens = static_cast<int32_t*>(out.ptr);
        targs.top_ids = ctx.stochastic_topk_ids;
        targs.top_logits = ctx.stochastic_topk_logits;
        targs.outputs = outputs;
        targs.vocab_size = vocab;
        targs.logits_row_stride = in.row_stride;
        targs.top_k = ctx.stochastic_top_k;
        targs.error_word = ctx.error_word;
        targs.active_count_out = ctx.stochastic_topk_active_counts;
        const hipError_t serr =
            ::shelf::kernel::launch_stochastic_topk_sampling_f32(targs, stream);
        if (serr != hipSuccess)
            return Status::hip_error("stochastic topk sampling launch",
                                     hipGetErrorString(serr), __FILE__, __LINE__);
        record_physical_launch(2);
        record_optimized_op();
        return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
    }

    if (choice.implementation == SamplingImplementation::StochasticSingleBlock) {
        ::shelf::kernel::StochasticSamplingF32Args sargs;
        sargs.logits = static_cast<const float*>(in.ptr);
        sargs.sampling = ctx.output_sampling_params;
        sargs.sampled_tokens = static_cast<int32_t*>(out.ptr);
        sargs.outputs = outputs;
        sargs.vocab_size = vocab;
        sargs.logits_row_stride = in.row_stride;
        sargs.error_word = ctx.error_word;
        const hipError_t serr =
            ::shelf::kernel::launch_stochastic_sampling_f32_singleblock(sargs, stream);
        if (serr != hipSuccess)
            return Status::hip_error("stochastic sampling launch", hipGetErrorString(serr),
                                     __FILE__, __LINE__);
        record_physical_launch(1);
        record_optimized_op();
        return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
    }

    ::shelf::kernel::ArgmaxF32Args args;
    args.logits = static_cast<const float*>(in.ptr);
    args.sampling = ctx.output_sampling_params;
    args.sampled_tokens = static_cast<int32_t*>(out.ptr);
    args.outputs = outputs;
    args.vocab_size = vocab;
    args.logits_row_stride = in.row_stride;

    hipError_t herr = hipSuccess;
    if (choice.implementation == SamplingImplementation::SingleBlock) {
        herr = ::shelf::kernel::launch_argmax_f32_single(args, stream);
    } else {
        if (ctx.scratch == nullptr ||
            (reinterpret_cast<uintptr_t>(ctx.scratch) &
             (alignof(::shelf::kernel::ArgMaxPair) - 1u)) != 0u)
            return kNone;
        auto* partials = reinterpret_cast<::shelf::kernel::ArgMaxPair*>(ctx.scratch);
        const uint32_t capacity = ctx.scratch_floats / 2u;
        herr = ::shelf::kernel::launch_argmax_f32_partitioned(
            args, partials, capacity, choice.partitions, stream);
    }
    if (herr != hipSuccess)
        return Status::hip_error("sampling optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(choice.implementation == SamplingImplementation::Partitioned
                               ? 2
                               : 1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace

Result<OptimizedLaunchResult> try_launch_sampling(
    const ::shelf::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (dispatch_index >= program.dispatches.size()) return kNone;
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != ::shelf::runtime::KernelId::SAMPLING) return kNone;
    return resolve_sampling(program, b, ctx, stream);
}

}  // namespace shelf::execution
