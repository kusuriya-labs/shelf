#include <shelf/inference/execution/gdn_conv_silu_fusion.h>

#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/kernels/optimized/gdn/fused_gdn_conv_silu.h>
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

namespace ps::execution {
namespace {

constexpr OptimizedLaunchResult kNone{OptimizedLaunchKind::NotApplicable, 0};
constexpr uint32_t kFusedGdnConvSiluChain = 2u;

struct FusedRange {
    uintptr_t start = 0;
    uintptr_t end = 0;
};

FusedRange fused_range(uint64_t ptr, uint32_t row_stride, uint32_t features,
                       size_t elem_bytes, uint32_t rows) {
    const size_t extent =
        (static_cast<size_t>(rows - 1u) * row_stride + features) * elem_bytes;
    const uintptr_t base = static_cast<uintptr_t>(ptr);
    return FusedRange{base, base + extent};
}

bool fused_disjoint(const FusedRange& a, const FusedRange& b) {
    return a.end <= b.start || b.end <= a.start;
}

bool aligned16(uint64_t ptr, uint32_t row_stride, size_t elem_bytes) {
    return (ptr & 15u) == 0u &&
           ((static_cast<uint64_t>(row_stride) * elem_bytes) & 15u) == 0u;
}

}

Result<OptimizedLaunchResult> try_launch_fused_gdn_conv_silu(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (ctx.value_trace != nullptr) return kNone;
    const size_t dispatch_count = program.dispatches.size();
    if (dispatch_index + kFusedGdnConvSiluChain > dispatch_count) return kNone;
    const auto& d0 = program.dispatches[dispatch_index];
    const auto& d1 = program.dispatches[dispatch_index + 1u];
    if (d0.kernel_id != ::ps::runtime::KernelId::STATEFUL_CAUSAL_CONV1D ||
        d1.kernel_id != ::ps::runtime::KernelId::SILU) {
        return kNone;
    }
    if (d0.input_count != 1u || d0.output_count != 1u || d1.input_count != 1u ||
        d1.output_count != 1u) {
        return kNone;
    }
    if (d1.input_slots[0] != d0.output_slots[0]) return kNone;

    PhysicalGdnConv1dLaunch conv{};
    const Result<PhysicalResolveStatus> r0 =
        resolve_gdn_conv1d_physical(program, d0, ctx, conv);
    if (!r0.ok() || r0.value() != PhysicalResolveStatus::Applicable) return kNone;

    PhysicalElementwiseLaunch silu{};
    const Result<PhysicalResolveStatus> r1 =
        resolve_elementwise_physical(program, d1, ctx, silu);
    if (!r1.ok() || r1.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (silu.profile != ElementwiseProfile::SiluF32ToBf16) return kNone;

    const uint32_t conv_dim = conv.args.conv_dim;
    const uint32_t rows = conv.args.actual_rows;
    if (rows == 0u || rows != silu.rows) return kNone;
    if (conv_dim == 0u || conv_dim != silu.features) return kNone;
    if (silu.in0 != reinterpret_cast<uint64_t>(conv.args.output)) return kNone;
    if (silu.in0_stride != conv.args.output_row_stride) return kNone;
    if (silu.out0_stride < conv_dim) return kNone;
    if (silu.out0 == 0u || silu.out0 == silu.in0) return kNone;

    if (!aligned16(silu.in0, silu.in0_stride, sizeof(float))) return kNone;
    if (!aligned16(silu.out0, silu.out0_stride, sizeof(::ps::bf16_t))) return kNone;
    if (!aligned16(reinterpret_cast<uint64_t>(conv.args.input),
                   conv.args.input_row_stride, sizeof(::ps::bf16_t)))
        return kNone;

    const FusedRange conv_in = fused_range(
        reinterpret_cast<uint64_t>(conv.args.input), conv.args.input_row_stride,
        conv_dim, sizeof(::ps::bf16_t), rows);
    const FusedRange conv_out = fused_range(
        reinterpret_cast<uint64_t>(conv.args.output), conv.args.output_row_stride,
        conv_dim, sizeof(float), rows);
    const FusedRange act_out =
        fused_range(silu.out0, silu.out0_stride, conv_dim,
                    sizeof(::ps::bf16_t), rows);
    if (!fused_disjoint(conv_in, conv_out)) return kNone;
    if (!fused_disjoint(conv_in, act_out)) return kNone;
    if (!fused_disjoint(conv_out, act_out)) return kNone;

    ::ps::kernel::FusedGdnConvSiluArgs args{};
    args.conv = conv.args;
    args.act_out = reinterpret_cast<::ps::bf16_t*>(silu.out0);
    args.act_row_stride = silu.out0_stride;
    const hipError_t herr = ::ps::kernel::launch_fused_gdn_conv_silu(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fused gdn conv silu launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched,
                                 kFusedGdnConvSiluChain};
}

}
