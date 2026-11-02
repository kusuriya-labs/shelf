#include <shelf/inference/execution/swiglu_quantize_fusion.h>

#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/swiglu_quantize.h>
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
constexpr uint32_t kFusedSwiGluQuantizeChain = 2u;

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

Result<OptimizedLaunchResult> try_launch_fused_swiglu_quantize(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    const size_t dispatch_count = program.dispatches.size();
    if (dispatch_index + kFusedSwiGluQuantizeChain > dispatch_count) return kNone;
    const auto& d0 = program.dispatches[dispatch_index];
    const auto& d1 = program.dispatches[dispatch_index + 1u];
    if (d0.kernel_id != ::ps::runtime::KernelId::SWIGLU ||
        d1.kernel_id != ::ps::runtime::KernelId::ACTIVATION_QUANTIZE_W4A8) {
        return kNone;
    }
    if (d0.input_count != 2u || d0.output_count != 1u || d1.input_count != 1u) {
        return kNone;
    }
    if (d1.input_slots[0] != d0.output_slots[0]) return kNone;

    PhysicalElementwiseLaunch ew{};
    const Result<PhysicalResolveStatus> r0 =
        resolve_elementwise_physical(program, d0, ctx, ew);
    if (!r0.ok() || r0.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (ew.profile != ElementwiseProfile::SwigluBf16) return kNone;

    PhysicalActivationQuantizeLaunch aq{};
    const Result<PhysicalResolveStatus> r1 =
        resolve_activation_quantize_physical(program, d1, ctx, aq);
    if (!r1.ok() || r1.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (aq.kind != PhysicalActivationQuantizeKind::E4M3) return kNone;
    if (aq.k != ew.features || aq.k_padded != ew.features) return kNone;
    if (aq.rows != ew.rows) return kNone;
    if (aq.input != ew.out0 || aq.input_row_stride != ew.out0_stride) return kNone;
    if (!::ps::kernel::activation_quantize_e4m3_vec_supported(
            reinterpret_cast<const ::ps::bf16_t*>(aq.input),
            reinterpret_cast<const uint8_t*>(aq.codes), aq.k, aq.k_padded,
            aq.input_row_stride)) {
        return kNone;
    }

    const FusedRange gate = fused_range(ew.in0, ew.in0_stride, ew.features,
                                        sizeof(::ps::bf16_t), ew.rows);
    const FusedRange up = fused_range(ew.in1, ew.in1_stride, ew.features,
                                      sizeof(::ps::bf16_t), ew.rows);
    const FusedRange output = fused_range(ew.out0, ew.out0_stride, ew.features,
                                          sizeof(::ps::bf16_t), ew.rows);
    const FusedRange codes = FusedRange{
        static_cast<uintptr_t>(aq.codes),
        static_cast<uintptr_t>(aq.codes) +
            static_cast<uintptr_t>(((aq.rows + 15u) & ~15u)) * aq.code_row_stride};
    const FusedRange scales = FusedRange{
        static_cast<uintptr_t>(aq.scales),
        static_cast<uintptr_t>(aq.scales) +
            static_cast<uintptr_t>(aq.rows) * aq.scale_row_stride};
    if (!fused_disjoint(gate, codes) || !fused_disjoint(gate, scales) ||
        !fused_disjoint(up, codes) || !fused_disjoint(up, scales) ||
        !fused_disjoint(output, codes) || !fused_disjoint(output, scales)) {
        return kNone;
    }

    ::ps::kernel::FusedSwiGluQuantizeArgs args{};
    args.gate = reinterpret_cast<const void*>(ew.in0);
    args.up = reinterpret_cast<const void*>(ew.in1);
    args.output = reinterpret_cast<void*>(ew.out0);
    args.codes = reinterpret_cast<uint8_t*>(aq.codes);
    args.scales = reinterpret_cast<float*>(aq.scales);
    args.rows = ew.rows;
    args.k = ew.features;
    args.gate_row_stride = ew.in0_stride;
    args.up_row_stride = ew.in1_stride;
    args.output_row_stride = ew.out0_stride;
    args.code_row_stride = aq.code_row_stride;
    args.scale_row_stride = aq.scale_row_stride;
    const hipError_t herr = ::ps::kernel::launch_fused_swiglu_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fused swiglu quantize launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched,
                                 kFusedSwiGluQuantizeChain};
}

}
