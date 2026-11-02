#include <shelf/inference/execution/gdn_post_fusion.h>

#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/kernels/optimized/gdn/fused_gdn_post.h>
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
constexpr uint32_t kFusedGdnPostChain = 3u;

}

Result<OptimizedLaunchResult> try_launch_fused_gdn_post(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    const size_t dispatch_count = program.dispatches.size();
    if (dispatch_index + kFusedGdnPostChain > dispatch_count) return kNone;
    const auto& d0 = program.dispatches[dispatch_index];
    const auto& d1 = program.dispatches[dispatch_index + 1u];
    const auto& d2 = program.dispatches[dispatch_index + 2u];
    if (d0.kernel_id != ::ps::runtime::KernelId::RMS_NORM ||
        d1.kernel_id != ::ps::runtime::KernelId::SILU ||
        d2.kernel_id != ::ps::runtime::KernelId::MUL) {
        return kNone;
    }
    if (d0.input_count != 1u || d0.output_count != 1u || d1.input_count != 1u ||
        d1.output_count != 1u || d2.input_count != 2u || d2.output_count != 1u) {
        return kNone;
    }
    if (d1.input_slots[0] == d0.output_slots[0]) return kNone;
    if (d2.input_slots[0] != d0.output_slots[0] ||
        d2.input_slots[1] != d1.output_slots[0]) {
        return kNone;
    }

    PhysicalRmsNormLaunch rn{};
    const Result<PhysicalResolveStatus> r0 =
        resolve_rmsnorm_physical(program, d0, ctx, rn);
    if (!r0.ok() || r0.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (rn.variant != PhysicalRmsNormVariant::F32PgDirect) return kNone;
    if (rn.features != 6144u && rn.features != 3072u) return kNone;
    if (rn.group_size == 0u || rn.group_size > 128u || rn.group_size % 4u != 0u ||
        rn.features % rn.group_size != 0u) {
        return kNone;
    }
    if (rn.grid != rn.rows * (rn.features / rn.group_size)) return kNone;

    PhysicalElementwiseLaunch silu{};
    const Result<PhysicalResolveStatus> r1 =
        resolve_elementwise_physical(program, d1, ctx, silu);
    if (!r1.ok() || r1.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (silu.profile != ElementwiseProfile::SiluBf16ToF32) return kNone;

    PhysicalElementwiseLaunch mul{};
    const Result<PhysicalResolveStatus> r2 =
        resolve_elementwise_physical(program, d2, ctx, mul);
    if (!r2.ok() || r2.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (mul.profile != ElementwiseProfile::MulF32F32ToBf16) return kNone;

    if (rn.rows == 0u || rn.rows != silu.rows || rn.rows != mul.rows) return kNone;
    if (rn.features != silu.features || rn.features != mul.features) return kNone;
    if (silu.out0 != mul.in1 || mul.in0 != rn.output) return kNone;
    if (silu.in0_stride != silu.out0_stride) return kNone;

    ::ps::kernel::FusedGdnPostArgs args{};
    args.rec = reinterpret_cast<const float*>(rn.input);
    args.rec_row_stride = rn.input_row_stride;
    args.norm_weight = reinterpret_cast<const ::ps::bf16_t*>(rn.weight);
    args.z = reinterpret_cast<const ::ps::bf16_t*>(silu.in0);
    args.z_row_stride = silu.in0_stride;
    args.rms_out = reinterpret_cast<float*>(rn.output);
    args.rms_row_stride = rn.output_row_stride;
    args.gated_z_out = reinterpret_cast<float*>(silu.out0);
    args.gated_z_row_stride = silu.out0_stride;
    args.gated_out = reinterpret_cast<::ps::bf16_t*>(mul.out0);
    args.gated_row_stride = mul.out0_stride;
    args.eps = rn.eps;
    args.rows = rn.rows;
    args.features = rn.features;
    args.group_size = rn.group_size;
    const hipError_t herr = ::ps::kernel::launch_fused_gdn_post(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fused gdn post launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, kFusedGdnPostChain};
}

}
