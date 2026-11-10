#include <shelf/inference/execution/rope_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/kernels/optimized/attention/rope.h>
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

uint32_t rope_rows_for(const HostResolvedValue& v, const HostExecutionContext& ctx) {
    if (v.row_domain == ValueRowDomain::OUTPUT_ROWS) return ctx.actual_outputs;
    return ctx.actual_rows != 0 ? ctx.actual_rows : 1u;
}

}  // namespace

Result<PhysicalResolveStatus> resolve_rope_physical(
    const ::shelf::runtime::Program& program, const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx, PhysicalRopeLaunch& out) {
    out = PhysicalRopeLaunch{};
    if (b.flags > 1u) return PhysicalResolveStatus::NotApplicable;
    if (b.input_count != 1u || b.output_count != 1u)
        return PhysicalResolveStatus::NotApplicable;

    auto in_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    auto out_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue dst = out_res.value();

    const bool rounded = b.flags == 1u && in.dtype == ValueDType::BF16 && dst.dtype == ValueDType::BF16;
    if (!rounded && (b.flags != 0u || in.dtype != ValueDType::F32 || dst.dtype != ValueDType::BF16))
        return PhysicalResolveStatus::NotApplicable;
    if (rounded && in.ptr == dst.ptr) return PhysicalResolveStatus::NotApplicable;
    const uint32_t F = in.feature_count;
    if (F == 0u || F != dst.feature_count) return PhysicalResolveStatus::NotApplicable;
    if (in.ptr == nullptr || dst.ptr == nullptr)
        return PhysicalResolveStatus::NotApplicable;
    if (in.row_stride < F || dst.row_stride < F)
        return PhysicalResolveStatus::NotApplicable;

    const uint32_t head_dim = b.group_size != 0u ? b.group_size : F;
    if (head_dim == 0u || F % head_dim != 0u) return PhysicalResolveStatus::NotApplicable;
    const uint32_t rotary =
        (b.rotary_dim != 0u && (b.rotary_dim & 1u) == 0u && b.rotary_dim <= head_dim)
            ? b.rotary_dim
            : (head_dim & ~1u);
    if (rotary == 0u) return PhysicalResolveStatus::NotApplicable;

    const uint32_t rows = rope_rows_for(dst, ctx);
    if (rows == 0u || (rounded && rows > 65535u)) return PhysicalResolveStatus::NotApplicable;
    if (rounded) {
        const uintptr_t ib = reinterpret_cast<uintptr_t>(in.ptr);
        const uintptr_t ob = reinterpret_cast<uintptr_t>(dst.ptr);
        const size_t isz = (static_cast<size_t>(rows - 1u) * in.row_stride + F) * sizeof(::shelf::bf16_t);
        const size_t osz = (static_cast<size_t>(rows - 1u) * dst.row_stride + F) * sizeof(::shelf::bf16_t);
        if (ib < ob + osz && ob < ib + isz) return PhysicalResolveStatus::NotApplicable;
    }

    const float theta = b.scalar_b > 0.0f ? b.scalar_b : 10000000.0f;

    RopeSelectorInput si;
    si.input_dtype = in.dtype;
    si.output_dtype = dst.dtype;
    si.features = F;
    si.head_dim = head_dim;
    si.rotary_dim = rotary;
    si.rows = rows;
    if (!rounded && select_rope_implementation(si) != RopeImplementation::Optimized)
        return PhysicalResolveStatus::NotApplicable;

    if (b.rope_table_index != 0u &&
        (ctx.rope_inv_freq == nullptr || b.rope_table_index >= ctx.rope_table_count ||
         ctx.rope_table_stride < rotary / 2u))
        return Status::invalid_argument("invalid RoPE table index", __FILE__, __LINE__);
    const double* table = ctx.rope_inv_freq == nullptr ? nullptr
        : ctx.rope_inv_freq + b.rope_table_index * ctx.rope_table_stride;
    const uint32_t* positions =
        ctx.rope_positions != nullptr ? ctx.rope_positions : ctx.row_positions;
    const bool pair =
        !rounded && table != nullptr &&
        ::shelf::kernel::rope_pair_shape_supported(
            static_cast<const float*>(in.ptr),
            static_cast<::shelf::bf16_t*>(dst.ptr), in.row_stride, dst.row_stride, F,
            head_dim, rotary);

    out.variant = rounded ? PhysicalRopeVariant::Bf16Rounded
                : pair ? PhysicalRopeVariant::F32Bf16Pair
                       : PhysicalRopeVariant::F32Bf16Generic;
    out.input = reinterpret_cast<uint64_t>(in.ptr);
    out.output = reinterpret_cast<uint64_t>(dst.ptr);
    out.positions = reinterpret_cast<uint64_t>(positions);
    out.inv_freq = reinterpret_cast<uint64_t>(table);
    out.input_row_stride = in.row_stride;
    out.output_row_stride = dst.row_stride;
    out.rows = rows;
    out.features = F;
    out.head_dim = head_dim;
    out.rotary_dim = rotary;
    out.theta = theta;
    out.attention_factor = b.scalar_a > 0.0f ? b.scalar_a : 1.0f;
    if (pair) {
        out.grid = rows;
        out.workgroup = ::shelf::kernel::kRopePairBlock;
    }
    return PhysicalResolveStatus::Applicable;
}

hipError_t launch_rope_physical(const PhysicalRopeLaunch& d, hipStream_t stream) {
    if (d.variant == PhysicalRopeVariant::Bf16Rounded) {
        return ::shelf::kernel::launch_rope_bf16_rounded(
            reinterpret_cast<const ::shelf::bf16_t*>(d.input), d.input_row_stride,
            reinterpret_cast<::shelf::bf16_t*>(d.output), d.output_row_stride,
            reinterpret_cast<const uint32_t*>(d.positions), d.rows, d.features, d.head_dim,
            d.rotary_dim, d.theta, reinterpret_cast<const double*>(d.inv_freq),
            d.attention_factor, stream);
    }
    if (d.variant == PhysicalRopeVariant::F32Bf16Pair) {
        return ::shelf::kernel::launch_rope_f32_bf16_pair(
            reinterpret_cast<const float*>(d.input), d.input_row_stride,
            reinterpret_cast<::shelf::bf16_t*>(d.output), d.output_row_stride,
            reinterpret_cast<const uint32_t*>(d.positions), d.rows, d.features,
            d.head_dim, d.rotary_dim, d.theta,
            reinterpret_cast<const double*>(d.inv_freq), d.attention_factor, stream);
    }
    if (d.variant == PhysicalRopeVariant::F32Bf16Generic) {
        return ::shelf::kernel::launch_rope_f32_bf16_generic(
            reinterpret_cast<const float*>(d.input), d.input_row_stride,
            reinterpret_cast<::shelf::bf16_t*>(d.output), d.output_row_stride,
            reinterpret_cast<const uint32_t*>(d.positions), d.rows, d.features,
            d.head_dim, d.rotary_dim, d.theta,
            reinterpret_cast<const double*>(d.inv_freq), d.attention_factor, stream);
    }
    return hipErrorNotSupported;
}

Result<OptimizedLaunchResult> try_launch_rope(
    const ::shelf::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (dispatch_index >= program.dispatches.size()) return kNone;
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != ::shelf::runtime::KernelId::ROPE) return kNone;
    PhysicalRopeLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_rope_physical(program, b, ctx, launch);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() == PhysicalResolveStatus::NotApplicable) return kNone;
    if (resolved.value() == PhysicalResolveStatus::Skipped)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    const hipError_t herr = launch_rope_physical(launch, stream);
    if (herr != hipSuccess)
        return Status::hip_error("rope optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    if (launch.variant == PhysicalRopeVariant::F32Bf16Generic)
        record_degraded_dispatch(b.kernel_id);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
