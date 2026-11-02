#include <shelf/inference/execution/resid_rmsnorm_quantize_fusion.h>

#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/kernels/optimized/resid_rmsnorm_quantize.h>
#include <shelf/kernels/optimized/activation_quantize.h>
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
constexpr uint32_t kFusedResidRmsNormQuantizeChain = 3u;
constexpr uint32_t kFusedK = 5120u;

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

}

Result<OptimizedLaunchResult> try_launch_fused_resid_rmsnorm_quantize(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (ctx.value_trace != nullptr) return kNone;
    const size_t dispatch_count = program.dispatches.size();
    if (dispatch_index + kFusedResidRmsNormQuantizeChain > dispatch_count) return kNone;
    const auto& d0 = program.dispatches[dispatch_index];
    const auto& d1 = program.dispatches[dispatch_index + 1u];
    const auto& d2 = program.dispatches[dispatch_index + 2u];
    if (d0.kernel_id != ::ps::runtime::KernelId::RESIDUAL_ADD ||
        d1.kernel_id != ::ps::runtime::KernelId::RMS_NORM ||
        d2.kernel_id != ::ps::runtime::KernelId::ACTIVATION_QUANTIZE_W4A8) {
        return kNone;
    }
    if (d0.input_count != 2u || d0.output_count != 1u || d1.input_count != 1u ||
        d1.output_count != 1u || d2.input_count != 1u) {
        return kNone;
    }
    if (d1.input_slots[0] != d0.output_slots[0]) return kNone;
    if (d2.input_slots[0] != d1.output_slots[0]) return kNone;

    PhysicalElementwiseLaunch rs{};
    const Result<PhysicalResolveStatus> r0 =
        resolve_elementwise_physical(program, d0, ctx, rs);
    if (!r0.ok() || r0.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (rs.profile != ElementwiseProfile::ResidualBf16) return kNone;

    PhysicalRmsNormLaunch rn{};
    const Result<PhysicalResolveStatus> r1 =
        resolve_rmsnorm_physical(program, d1, ctx, rn);
    if (!r1.ok() || r1.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (rn.variant != PhysicalRmsNormVariant::Bf16PfOnePlus) return kNone;
    if (rn.features != kFusedK || rn.group_size != rn.features) return kNone;
    if (rn.grid != rn.rows) return kNone;
    if (rn.rows != rs.rows || rn.features != rs.features) return kNone;
    if (rs.out0 == 0u || rn.input != rs.out0) return kNone;
    if (rs.out0_stride != rn.input_row_stride) return kNone;
    if (rn.output == 0u) return kNone;

    PhysicalActivationQuantizeLaunch aq{};
    const Result<PhysicalResolveStatus> r2 =
        resolve_activation_quantize_physical(program, d2, ctx, aq);
    if (!r2.ok() || r2.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (aq.kind != PhysicalActivationQuantizeKind::E4M3) return kNone;
    if (aq.k != kFusedK || aq.k_padded != kFusedK) return kNone;
    if (aq.rows != rn.rows) return kNone;
    if (aq.input != rn.output || aq.input_row_stride != rn.output_row_stride) return kNone;
    if (!::ps::kernel::activation_quantize_e4m3_vec_supported(
            reinterpret_cast<const ::ps::bf16_t*>(rn.output),
            reinterpret_cast<const uint8_t*>(aq.codes), aq.k, aq.k_padded,
            aq.input_row_stride)) {
        return kNone;
    }

    const FusedRange ra = fused_range(rs.in0, rs.in0_stride, rs.features,
                                      sizeof(::ps::bf16_t), rs.rows);
    const FusedRange rb = fused_range(rs.in1, rs.in1_stride, rs.features,
                                      sizeof(::ps::bf16_t), rs.rows);
    const FusedRange residual = fused_range(rs.out0, rs.out0_stride, rs.features,
                                            sizeof(::ps::bf16_t), rs.rows);
    const FusedRange normed = fused_range(rn.output, rn.output_row_stride,
                                          rn.features, sizeof(::ps::bf16_t), rn.rows);
    const FusedRange weight = FusedRange{
        static_cast<uintptr_t>(rn.weight),
        static_cast<uintptr_t>(rn.weight) + static_cast<uintptr_t>(rn.features) *
                                                sizeof(::ps::bf16_t)};
    const FusedRange codes = FusedRange{
        static_cast<uintptr_t>(aq.codes),
        static_cast<uintptr_t>(aq.codes) +
            static_cast<uintptr_t>(((aq.rows + 15u) & ~15u)) * aq.code_row_stride};
    const FusedRange scales = FusedRange{
        static_cast<uintptr_t>(aq.scales),
        static_cast<uintptr_t>(aq.scales) +
            static_cast<uintptr_t>(aq.rows) * aq.scale_row_stride};
    if (!fused_disjoint(ra, codes) || !fused_disjoint(ra, scales) ||
        !fused_disjoint(rb, codes) || !fused_disjoint(rb, scales) ||
        !fused_disjoint(residual, codes) || !fused_disjoint(residual, scales) ||
        !fused_disjoint(normed, codes) || !fused_disjoint(normed, scales) ||
        !fused_disjoint(weight, codes) || !fused_disjoint(weight, scales)) {
        return kNone;
    }

    ::ps::kernel::FusedResidRmsNormQuantizeArgs args{};
    args.resid_a = reinterpret_cast<const void*>(rs.in0);
    args.resid_b = reinterpret_cast<const void*>(rs.in1);
    args.residual = reinterpret_cast<void*>(rs.out0);
    args.normed = reinterpret_cast<void*>(rn.output);
    args.weight = reinterpret_cast<const void*>(rn.weight);
    args.codes = reinterpret_cast<uint8_t*>(aq.codes);
    args.scales = reinterpret_cast<float*>(aq.scales);
    args.rows = rs.rows;
    args.features = rs.features;
    args.group_size = rn.group_size;
    args.resid_a_row_stride = rs.in0_stride;
    args.resid_b_row_stride = rs.in1_stride;
    args.residual_row_stride = rs.out0_stride;
    args.normed_row_stride = rn.output_row_stride;
    args.code_row_stride = aq.code_row_stride;
    args.scale_row_stride = aq.scale_row_stride;
    args.eps = rn.eps;
    const hipError_t herr =
        ::ps::kernel::launch_fused_resid_rmsnorm_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fused resid rmsnorm quantize launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched,
                                 kFusedResidRmsNormQuantizeChain};
}

}