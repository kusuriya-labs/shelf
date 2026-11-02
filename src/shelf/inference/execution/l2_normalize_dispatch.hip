#include <shelf/inference/execution/l2_normalize_dispatch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/kernels/optimized/l2_normalize.h>
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

uint32_t l2_rows_for(const HostResolvedValue& v, const HostExecutionContext& ctx) {
    if (v.row_domain == ValueRowDomain::OUTPUT_ROWS) return ctx.actual_outputs;
    return ctx.actual_rows != 0 ? ctx.actual_rows : 1u;
}

}  // namespace

Result<PhysicalResolveStatus> resolve_l2_normalize_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalL2NormalizeLaunch& out) {
    out = PhysicalL2NormalizeLaunch{};
    if (b.input_count != 1u || b.output_count != 1u)
        return PhysicalResolveStatus::NotApplicable;

    auto in_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue in = in_res.value();
    auto out_res =
        resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return PhysicalResolveStatus::NotApplicable;
    const HostResolvedValue dst = out_res.value();

    if (in.dtype != ValueDType::BF16 || dst.dtype != ValueDType::F32)
        return PhysicalResolveStatus::NotApplicable;
    const uint32_t F = in.feature_count;
    if (F == 0u || in.feature_count != dst.feature_count)
        return PhysicalResolveStatus::NotApplicable;
    const uint32_t G = (b.group_size != 0u && b.group_size <= F) ? b.group_size : F;
    if (F % G != 0u) return PhysicalResolveStatus::NotApplicable;
    if (in.ptr == nullptr || dst.ptr == nullptr)
        return PhysicalResolveStatus::NotApplicable;
    if (in.row_stride < F || dst.row_stride < F)
        return PhysicalResolveStatus::NotApplicable;

    const uint32_t rows = l2_rows_for(dst, ctx);
    if (rows == 0u) return PhysicalResolveStatus::NotApplicable;

    L2NormalizeSelectorInput si;
    si.input_dtype = in.dtype;
    si.output_dtype = dst.dtype;
    si.features = F;
    si.group_size = G;
    si.rows = rows;
    if (select_l2_normalize_implementation(si) != L2NormalizeImplementation::Optimized)
        return PhysicalResolveStatus::NotApplicable;

    out.input = reinterpret_cast<uint64_t>(in.ptr);
    out.output = reinterpret_cast<uint64_t>(dst.ptr);
    out.input_row_stride = in.row_stride;
    out.output_row_stride = dst.row_stride;
    out.rows = rows;
    out.features = F;
    out.group_size = G;
    out.eps = b.scalar_a;
    out.grid_x = ::shelf::kernel::l2_normalize_groups(F, G);
    out.grid_y = rows;
    out.workgroup = ::shelf::kernel::kL2NormalizeThreads;
    return PhysicalResolveStatus::Applicable;
}

Result<OptimizedLaunchResult> try_launch_l2_normalize(
    const ::shelf::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (dispatch_index >= program.dispatches.size()) return kNone;
    const auto& b = program.dispatches[dispatch_index];
    if (b.kernel_id != ::shelf::runtime::KernelId::L2_NORMALIZE) return kNone;
    PhysicalL2NormalizeLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_l2_normalize_physical(program, b, ctx, launch);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable) return kNone;

    const hipError_t herr = ::shelf::kernel::launch_l2_normalize_bf16_f32(
        reinterpret_cast<const ::shelf::bf16_t*>(launch.input),
        launch.input_row_stride, reinterpret_cast<float*>(launch.output),
        launch.output_row_stride, launch.rows, launch.features, launch.group_size,
        launch.eps, stream);
    if (herr != hipSuccess)
        return Status::hip_error("l2_normalize optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
