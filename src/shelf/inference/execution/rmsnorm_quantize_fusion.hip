#include <shelf/inference/execution/rmsnorm_quantize_fusion.h>

#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/kernels/optimized/rmsnorm_quantize.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::execution {
namespace {

constexpr OptimizedLaunchResult kNone{OptimizedLaunchKind::NotApplicable, 0};
constexpr uint32_t kFusedRmsNormQuantizeChain = 2u;
constexpr uint32_t kFusedNormQuantizeK = 5120u;

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

Result<OptimizedLaunchResult> try_launch_fused_rmsnorm_quantize(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    const size_t dispatch_count = program.dispatches.size();
    if (dispatch_index + kFusedRmsNormQuantizeChain > dispatch_count) return kNone;
    const auto& d0 = program.dispatches[dispatch_index];
    const auto& d1 = program.dispatches[dispatch_index + 1u];
    if (d0.kernel_id != ::ps::runtime::KernelId::RMS_NORM ||
        d1.kernel_id != ::ps::runtime::KernelId::ACTIVATION_QUANTIZE_W4A8) {
        return kNone;
    }
    if (d0.input_count != 1u || d0.output_count != 1u || d1.input_count != 1u) {
        return kNone;
    }
    if (d1.input_slots[0] != d0.output_slots[0]) return kNone;

    PhysicalRmsNormLaunch rn{};
    const Result<PhysicalResolveStatus> r0 =
        resolve_rmsnorm_physical(program, d0, ctx, rn);
    if (!r0.ok() || r0.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (rn.variant != PhysicalRmsNormVariant::Bf16PfOnePlus) return kNone;
    if (rn.features != kFusedNormQuantizeK || rn.group_size != rn.features) return kNone;
    if (rn.grid != rn.rows) return kNone;

    PhysicalActivationQuantizeLaunch aq{};
    const Result<PhysicalResolveStatus> r1 =
        resolve_activation_quantize_physical(program, d1, ctx, aq);
    if (!r1.ok() || r1.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (aq.kind != PhysicalActivationQuantizeKind::E4M3) return kNone;
    if (aq.k != kFusedNormQuantizeK || aq.k_padded != kFusedNormQuantizeK) return kNone;
    if (aq.rows != rn.rows) return kNone;
    if (aq.input != rn.output || aq.input_row_stride != rn.output_row_stride) return kNone;
    if (!::ps::kernel::activation_quantize_e4m3_vec_supported(
            reinterpret_cast<const ::ps::bf16_t*>(rn.output),
            reinterpret_cast<const uint8_t*>(aq.codes), aq.k, aq.k_padded,
            aq.input_row_stride)) {
        return kNone;
    }

    const FusedRange input = fused_range(rn.input, rn.input_row_stride, rn.features,
                                         sizeof(::ps::bf16_t), rn.rows);
    const FusedRange output = fused_range(rn.output, rn.output_row_stride, rn.features,
                                          sizeof(::ps::bf16_t), rn.rows);
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
    if (!fused_disjoint(input, codes) || !fused_disjoint(input, scales) ||
        !fused_disjoint(output, codes) || !fused_disjoint(output, scales) ||
        !fused_disjoint(weight, codes) || !fused_disjoint(weight, scales)) {
        return kNone;
    }

    ::ps::kernel::FusedRmsNormQuantizeArgs args{};
    args.input = reinterpret_cast<const void*>(rn.input);
    args.weight = reinterpret_cast<const void*>(rn.weight);
    args.output = reinterpret_cast<void*>(rn.output);
    args.codes = reinterpret_cast<uint8_t*>(aq.codes);
    args.scales = reinterpret_cast<float*>(aq.scales);
    args.rows = rn.rows;
    args.features = rn.features;
    args.group_size = rn.group_size;
    args.input_row_stride = rn.input_row_stride;
    args.output_row_stride = rn.output_row_stride;
    args.code_row_stride = aq.code_row_stride;
    args.scale_row_stride = aq.scale_row_stride;
    args.eps = rn.eps;
    const hipError_t herr = ::ps::kernel::launch_fused_rmsnorm_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fused rmsnorm quantize launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched,
                                 kFusedRmsNormQuantizeChain};
}

}
