#include <phaseshift/inference/execution/attn_post_quantize_fusion.h>

#include <phaseshift/inference/execution/optimized_dispatch.h>
#include <phaseshift/inference/execution/physical_launch.h>
#include <phaseshift/inference/execution/program_executor.h>
#include <phaseshift/inference/execution/decode_perf_stats.h>
#include <phaseshift/kernels/optimized/attn_post_quantize.h>
#include <phaseshift/kernels/optimized/activation_quantize.h>
#include <phaseshift/inference/execution/selectors/elementwise_selector.h>
#include <phaseshift/inference/execution/selectors/embedding_selector.h>
#include <phaseshift/inference/execution/selectors/gdn_conv_selector.h>
#include <phaseshift/inference/execution/selectors/gdn_recurrence_selector.h>
#include <phaseshift/inference/execution/selectors/kv_append_selector.h>
#include <phaseshift/inference/execution/selectors/l2_normalize_selector.h>
#include <phaseshift/inference/execution/selectors/linear_selector.h>
#include <phaseshift/inference/execution/selectors/paged_attention_selector.h>
#include <phaseshift/inference/execution/selectors/rmsnorm_selector.h>
#include <phaseshift/inference/execution/selectors/rope_selector.h>
#include <phaseshift/inference/execution/selectors/sampling_selector.h>
#include <phaseshift/inference/execution/selectors/selector_limits.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::execution {
namespace {

constexpr OptimizedLaunchResult kNone{OptimizedLaunchKind::NotApplicable, 0};
constexpr uint32_t kFusedAttnPostQuantizeChain = 3u;

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

bool fused_slot_used_elsewhere(const ::ps::runtime::Program& program,
                               size_t keep_a, size_t keep_b, uint32_t slot) {
    for (size_t j = 0; j < program.dispatches.size(); ++j) {
        if (j == keep_a || j == keep_b) continue;
        const auto& d = program.dispatches[j];
        for (uint32_t i = 0u; i < d.input_count; ++i) {
            if (i < ::ps::runtime::kMaxIoSlots && d.input_slots[i] == slot) return true;
        }
        for (uint32_t o = 0u; o < d.output_count; ++o) {
            if (o < ::ps::runtime::kMaxIoSlots && d.output_slots[o] == slot) return true;
        }
    }
    return false;
}

}

Result<OptimizedLaunchResult> try_launch_fused_attn_post_quantize(
    const ::ps::runtime::Program& program, size_t dispatch_index,
    HostExecutionContext& ctx, hipStream_t stream) {
    if (ctx.value_trace != nullptr) return kNone;
    const size_t dispatch_count = program.dispatches.size();
    if (dispatch_index + kFusedAttnPostQuantizeChain > dispatch_count) return kNone;
    const auto& d0 = program.dispatches[dispatch_index];
    const auto& d1 = program.dispatches[dispatch_index + 1u];
    const auto& d2 = program.dispatches[dispatch_index + 2u];
    if (d0.kernel_id != ::ps::runtime::KernelId::SIGMOID ||
        d1.kernel_id != ::ps::runtime::KernelId::MUL ||
        d2.kernel_id != ::ps::runtime::KernelId::ACTIVATION_QUANTIZE_W4A8) {
        return kNone;
    }
    if (d0.input_count != 1u || d0.output_count != 1u || d1.input_count != 2u ||
        d1.output_count != 1u || d2.input_count != 1u) {
        return kNone;
    }
    const uint32_t gs_slot = d0.output_slots[0];
    const bool gs_is_in0 = d1.input_slots[0] == gs_slot;
    const bool gs_is_in1 = d1.input_slots[1] == gs_slot;
    if (!gs_is_in0 && !gs_is_in1) return kNone;
    if (gs_is_in0 && gs_is_in1) return kNone;
    if (d2.input_slots[0] != d1.output_slots[0]) return kNone;

    PhysicalElementwiseLaunch sig{};
    const Result<PhysicalResolveStatus> r0 =
        resolve_elementwise_physical(program, d0, ctx, sig);
    if (!r0.ok() || r0.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (sig.profile != ElementwiseProfile::SigmoidBf16ToF32) return kNone;
    if (sig.features != 6144u && sig.features != 3072u) return kNone;

    PhysicalElementwiseLaunch mul{};
    const Result<PhysicalResolveStatus> r1 =
        resolve_elementwise_physical(program, d1, ctx, mul);
    if (!r1.ok() || r1.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (mul.profile != ElementwiseProfile::MulF32F32ToBf16) return kNone;
    if (mul.rows != sig.rows || mul.features != sig.features) return kNone;
    const uint64_t gs_ptr = sig.out0;
    const uint32_t gs_stride = sig.out0_stride;
    uint64_t ctx_ptr = 0;
    uint32_t ctx_stride = 0;
    if (gs_is_in0) {
        if (mul.in0 != gs_ptr || mul.in0_stride != gs_stride) return kNone;
        ctx_ptr = mul.in1;
        ctx_stride = mul.in1_stride;
    } else {
        if (mul.in1 != gs_ptr || mul.in1_stride != gs_stride) return kNone;
        ctx_ptr = mul.in0;
        ctx_stride = mul.in0_stride;
    }
    if (ctx_ptr == 0u) return kNone;

    PhysicalActivationQuantizeLaunch aq{};
    const Result<PhysicalResolveStatus> r2 =
        resolve_activation_quantize_physical(program, d2, ctx, aq);
    if (!r2.ok() || r2.value() != PhysicalResolveStatus::Applicable) return kNone;
    if (aq.kind != PhysicalActivationQuantizeKind::E4M3) return kNone;
    if (aq.k != sig.features || aq.k_padded != aq.k) return kNone;
    if (aq.rows != sig.rows) return kNone;
    if (aq.input != mul.out0 || aq.input_row_stride != mul.out0_stride) return kNone;
    if (!::ps::kernel::activation_quantize_e4m3_vec_supported(
            reinterpret_cast<const ::ps::bf16_t*>(mul.out0),
            reinterpret_cast<const uint8_t*>(aq.codes), aq.k, aq.k_padded,
            aq.input_row_stride)) {
        return kNone;
    }

    if (fused_slot_used_elsewhere(program, dispatch_index, dispatch_index + 1u,
                                  gs_slot)) {
        return kNone;
    }
    const ::ps::runtime::ValueBinding* gs_binding =
        program.find_value(::ps::runtime::ValueId{gs_slot});
    if (gs_binding == nullptr ||
        gs_binding->storage == ::ps::runtime::ValueStorage::EXTERNAL_OUTPUT) {
        return kNone;
    }

    const FusedRange gate = fused_range(sig.in0, sig.in0_stride, sig.features,
                                        sizeof(::ps::bf16_t), sig.rows);
    const FusedRange v_ctx = fused_range(ctx_ptr, ctx_stride, sig.features,
                                         sizeof(float), sig.rows);
    const FusedRange v_gs = fused_range(gs_ptr, gs_stride, sig.features,
                                        sizeof(float), sig.rows);
    const FusedRange gated = fused_range(mul.out0, mul.out0_stride, mul.features,
                                         sizeof(::ps::bf16_t), mul.rows);
    const FusedRange codes = FusedRange{
        static_cast<uintptr_t>(aq.codes),
        static_cast<uintptr_t>(aq.codes) +
            static_cast<uintptr_t>(((aq.rows + 15u) & ~15u)) * aq.code_row_stride};
    const FusedRange scales = FusedRange{
        static_cast<uintptr_t>(aq.scales),
        static_cast<uintptr_t>(aq.scales) +
            static_cast<uintptr_t>(aq.rows) * aq.scale_row_stride};
    if (!fused_disjoint(gate, codes) || !fused_disjoint(gate, scales) ||
        !fused_disjoint(v_ctx, codes) || !fused_disjoint(v_ctx, scales) ||
        !fused_disjoint(v_gs, codes) || !fused_disjoint(v_gs, scales) ||
        !fused_disjoint(gated, codes) || !fused_disjoint(gated, scales)) {
        return kNone;
    }

    ::ps::kernel::FusedAttnPostQuantizeArgs args{};
    args.gate = reinterpret_cast<const ::ps::bf16_t*>(sig.in0);
    args.gate_row_stride = sig.in0_stride;
    args.ctx = reinterpret_cast<const float*>(ctx_ptr);
    args.ctx_row_stride = ctx_stride;
    args.gated_out = reinterpret_cast<::ps::bf16_t*>(mul.out0);
    args.gated_row_stride = mul.out0_stride;
    args.codes = reinterpret_cast<uint8_t*>(aq.codes);
    args.scales = reinterpret_cast<float*>(aq.scales);
    args.code_row_stride = aq.code_row_stride;
    args.scale_row_stride = aq.scale_row_stride;
    args.rows = sig.rows;
    args.features = sig.features;
    args.ctx_first = gs_is_in1 ? 1u : 0u;
    const hipError_t herr = ::ps::kernel::launch_fused_attn_post_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fused attn post quantize launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched,
                                 kFusedAttnPostQuantizeChain};
}

}
