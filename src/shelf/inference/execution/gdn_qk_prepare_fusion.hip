#include <shelf/inference/execution/gdn_qk_prepare_fusion.h>

#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/kernels/optimized/gdn/fused_gdn_qk_prepare.h>
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
constexpr uint32_t kFusedGdnQkPrepareChain = 3u;

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

bool row_aligned16(uint64_t ptr, uint32_t row_stride, size_t elem_bytes) {
    return (ptr & 15u) == 0u &&
           ((static_cast<uint64_t>(row_stride) * elem_bytes) & 15u) == 0u;
}

}

Result<OptimizedLaunchResult> try_launch_fused_gdn_qk_prepare(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (ctx.value_trace != nullptr) return kNone;
    const size_t dispatch_count = program.dispatches.size();
    if (dispatch_index + kFusedGdnQkPrepareChain > dispatch_count) return kNone;
    const auto& d0 = program.dispatches[dispatch_index];
    const auto& d1 = program.dispatches[dispatch_index + 1u];
    const auto& d2 = program.dispatches[dispatch_index + 2u];
    if (d0.kernel_id != ::ps::runtime::KernelId::L2_NORMALIZE ||
        d1.kernel_id != ::ps::runtime::KernelId::L2_NORMALIZE ||
        d2.kernel_id != ::ps::runtime::KernelId::SCALE) {
        return kNone;
    }
    if (d0.input_count != 1u || d0.output_count != 1u || d1.input_count != 1u ||
        d1.output_count != 1u || d2.input_count != 1u || d2.output_count != 1u) {
        return kNone;
    }
    if (d2.input_slots[0] != d0.output_slots[0]) return kNone;
    if (d1.input_slots[0] == d0.input_slots[0]) return kNone;

    PhysicalL2NormalizeLaunch l2q{};
    const Result<PhysicalResolveStatus> r0 =
        resolve_l2_normalize_physical(program, d0, ctx, l2q);
    if (!r0.ok() || r0.value() != PhysicalResolveStatus::Applicable) return kNone;
    PhysicalL2NormalizeLaunch l2k{};
    const Result<PhysicalResolveStatus> r1 =
        resolve_l2_normalize_physical(program, d1, ctx, l2k);
    if (!r1.ok() || r1.value() != PhysicalResolveStatus::Applicable) return kNone;
    PhysicalElementwiseLaunch sc{};
    const Result<PhysicalResolveStatus> r2 =
        resolve_elementwise_physical(program, d2, ctx, sc);
    if (!r2.ok() || r2.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (sc.profile != ElementwiseProfile::ScaleF32) return kNone;

    if (l2q.group_size == 0u || l2q.group_size != l2k.group_size) return kNone;
    if (l2q.eps != l2k.eps) return kNone;
    if (l2q.features == 0u || l2q.features != l2k.features) return kNone;
    if (l2q.features != sc.features) return kNone;
    if (l2q.features % l2q.group_size != 0u) return kNone;
    if (l2q.group_size % 4u != 0u) return kNone;
    const uint32_t rows = l2q.rows;
    if (rows == 0u || rows != l2k.rows || rows != sc.rows) return kNone;

    if (sc.in0 != l2q.output) return kNone;
    if (sc.in0_stride != l2q.output_row_stride) return kNone;

    const uint64_t qk_elem_bytes = sizeof(::ps::bf16_t);
    if (l2k.input < l2q.input) return kNone;
    if (l2k.input - l2q.input != static_cast<uint64_t>(l2q.features) * qk_elem_bytes)
        return kNone;
    if (l2q.input_row_stride != l2k.input_row_stride) return kNone;
    if (l2q.features % 8u != 0u) return kNone;

    if (!row_aligned16(l2q.input, l2q.input_row_stride, qk_elem_bytes)) return kNone;
    if (!row_aligned16(l2q.output, l2q.output_row_stride, sizeof(float))) return kNone;
    if (!row_aligned16(l2k.output, l2k.output_row_stride, sizeof(float))) return kNone;
    if (!row_aligned16(sc.out0, sc.out0_stride, sizeof(float))) return kNone;

    const FusedRange q_in =
        fused_range(l2q.input, l2q.input_row_stride, l2q.features,
                    qk_elem_bytes, rows);
    const FusedRange k_in =
        fused_range(l2k.input, l2k.input_row_stride, l2k.features,
                    qk_elem_bytes, rows);
    const FusedRange nq_out =
        fused_range(l2q.output, l2q.output_row_stride, l2q.features,
                    sizeof(float), rows);
    const FusedRange nk_out =
        fused_range(l2k.output, l2k.output_row_stride, l2k.features,
                    sizeof(float), rows);
    const FusedRange sq_out =
        fused_range(sc.out0, sc.out0_stride, sc.features, sizeof(float), rows);
    if (!fused_disjoint(nq_out, nk_out)) return kNone;
    if (!fused_disjoint(nq_out, sq_out)) return kNone;
    if (!fused_disjoint(nk_out, sq_out)) return kNone;
    if (!fused_disjoint(q_in, nq_out) || !fused_disjoint(q_in, nk_out) ||
        !fused_disjoint(q_in, sq_out)) {
        return kNone;
    }
    if (!fused_disjoint(k_in, nq_out) || !fused_disjoint(k_in, nk_out) ||
        !fused_disjoint(k_in, sq_out)) {
        return kNone;
    }

    ::ps::kernel::FusedGdnQkPrepareArgs args{};
    args.q = reinterpret_cast<const ::ps::bf16_t*>(l2q.input);
    args.q_row_stride = l2q.input_row_stride;
    args.k = reinterpret_cast<const ::ps::bf16_t*>(l2k.input);
    args.k_row_stride = l2k.input_row_stride;
    args.nq_out = reinterpret_cast<float*>(l2q.output);
    args.nq_row_stride = l2q.output_row_stride;
    args.nk_out = reinterpret_cast<float*>(l2k.output);
    args.nk_row_stride = l2k.output_row_stride;
    args.sq_out = reinterpret_cast<float*>(sc.out0);
    args.sq_row_stride = sc.out0_stride;
    args.eps = l2q.eps;
    args.scale = sc.scalar_a;
    args.rows = rows;
    args.features = l2q.features;
    args.group_size = l2q.group_size;
    const hipError_t herr = ::ps::kernel::launch_fused_gdn_qk_prepare(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fused gdn qk prepare launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched,
                                 kFusedGdnQkPrepareChain};
}

}
