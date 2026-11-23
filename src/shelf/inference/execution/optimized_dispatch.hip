#include <shelf/kernels/fused/exl3/decode.h>
#include <shelf/kernels/fused/exl3/rotate_out.h>
#include <shelf/kernels/fused/exl3/rotate_quantize.h>
#include <shelf/kernels/fused/exl3/row_prepare.h>
#include <shelf/kernels/fused/exl3/grouped.h>
#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/kernels/optimized/linear/lm_head_int8.h>
#include <shelf/inference/execution/boundary_deferred.h>
#include <shelf/inference/execution/final_producer.h>
#include <shelf/inference/execution/exl3_dispatch_decision.h>
#include <shelf/kernels/fused/fp8_activation_producer.h>
#include <shelf/kernels/fused/mxfp4_activation_producer.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/quantization/quant_contract.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/optimized/rmsnorm.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/linear/fp8.h>
#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/fused/bf16_linear_pair.h>
#include <shelf/kernels/fused/hyper_connection.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/inference/execution/hyper_connection_dispatch.h>
#include <shelf/inference/execution/ple_dispatch.h>
#include <shelf/inference/execution/qsa_dispatch.h>
#include <shelf/inference/execution/fp8_ptpc_dispatch.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/workspace.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/fused/exl3/producer_rotate.h>
#include <shelf/kernels/fused/exl3/producer_rotate_rows.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/exl3_dense_gemm.h>
#include <shelf/inference/execution/gemm_lt.h>
#include <shelf/inference/execution/kv_append_dispatch.h>
#include <shelf/inference/execution/paged_attention_dispatch.h>
#include <shelf/inference/execution/gdn_recurrence_dispatch.h>
#include <shelf/inference/execution/elementwise_dispatch.h>
#include <shelf/inference/execution/embedding_dispatch.h>
#include <shelf/inference/execution/sampling_dispatch.h>
#include <shelf/inference/execution/l2_normalize_dispatch.h>
#include <shelf/inference/execution/gdn_conv_dispatch.h>
#include <shelf/inference/execution/rope_dispatch.h>
#include <shelf/inference/execution/moe_dispatch.h>
#include <shelf/kernels/optimized/output_gather.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
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
#include <atomic>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>

namespace shelf::execution {

using ::shelf::runtime::KernelId;
using ::shelf::runtime::Int8ActivationWorkspaceLayout;
using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueRowDomain;

namespace {

uint32_t rows_for(const HostResolvedValue& v, const HostExecutionContext& ctx) {
    if (v.row_domain == ValueRowDomain::OUTPUT_ROWS)
        return ctx.actual_outputs;
    return ctx.actual_rows != 0 ? ctx.actual_rows : 1u;
}

}  // namespace

Result<OptimizedLaunchResult> try_launch_activation_quantize(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    PhysicalActivationQuantizeLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_activation_quantize_physical(program, b, ctx, launch);
    if (!resolved.ok())
        return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    bool specialized = true;
    const hipError_t herr =
        b.kernel_id == KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF
            ? ::shelf::kernel::launch_activation_quantize_mxfp4_rowref(
                  reinterpret_cast<const ::shelf::bf16_t*>(launch.input),
                  reinterpret_cast<uint8_t*>(launch.codes),
                  reinterpret_cast<float*>(launch.scales), launch.rows, launch.k,
                  launch.k_padded, launch.input_row_stride, launch.code_row_stride,
                  launch.scale_row_stride,
                  launch.row_index != 0u
                      ? reinterpret_cast<const int32_t*>(launch.row_index)
                      : nullptr, stream)
            : ::shelf::kernel::launch_activation_quantize_fp8_finegrained(
                  reinterpret_cast<const ::shelf::bf16_t*>(launch.input),
                  reinterpret_cast<uint8_t*>(launch.codes),
                  reinterpret_cast<float*>(launch.scales), launch.rows, launch.k,
                  launch.k_padded, launch.input_row_stride, launch.code_row_stride,
                  launch.scale_row_stride, stream, &specialized);
    if (herr != hipSuccess)
        return Status::hip_error("act-quant optimized launch", hipGetErrorString(herr),
                                   __FILE__, __LINE__);
    if (!specialized)
        record_degraded_dispatch(b.kernel_id);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

::shelf::kernel::RmsNormDataType rmsnorm_kd(ValueDType d);

Result<OptimizedLaunchResult> try_launch_rmsnorm(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    PhysicalRmsNormLaunch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_rmsnorm_physical(program, b, ctx, launch);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() == PhysicalResolveStatus::NotApplicable)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (resolved.value() == PhysicalResolveStatus::Skipped)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    const bool small_group = b.flags == static_cast<uint32_t>(::shelf::runtime::RmsNormWeightMode::OFFSET_BF16_SMALL_GROUP);
    const hipError_t herr = (b.flags == static_cast<uint32_t>(::shelf::runtime::RmsNormWeightMode::OFFSET_BF16) || small_group)
        ? ::shelf::kernel::launch_rmsnorm_offset_bf16(
            reinterpret_cast<const void*>(launch.input), reinterpret_cast<const void*>(launch.weight),
            reinterpret_cast<void*>(launch.output), launch.rows, launch.features, launch.group_size,
            launch.input_row_stride, launch.output_row_stride,
            launch.weight_layout == ::shelf::kernel::RmsNormWeightLayout::PER_FEATURE,
            launch.eps, launch.weight_offset, stream, small_group)
        : ::shelf::kernel::launch_rmsnorm(
        reinterpret_cast<const void*>(launch.input), launch.input_dtype,
        reinterpret_cast<const void*>(launch.weight), launch.weight_dtype,
        launch.weight_layout, launch.weight_mode,
        reinterpret_cast<void*>(launch.output), launch.output_dtype,
        launch.rows, launch.features, launch.group_size,
        launch.input_row_stride, launch.output_row_stride, launch.eps, stream);
    if (herr != hipSuccess)
        return Status::hip_error("rmsnorm optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_output_gather(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    auto in_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    auto out_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue out = out_res.value();
    if (in.dtype != out.dtype || in.feature_count != out.feature_count ||
        in.feature_count == 0u)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (out.row_domain != ValueRowDomain::OUTPUT_ROWS)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (in.ptr == nullptr || out.ptr == nullptr)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (in.row_stride < in.feature_count || out.row_stride < out.feature_count)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (ctx.actual_outputs == 0u)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (ctx.batch_context == nullptr)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (in.dtype != ValueDType::BF16 && in.dtype != ValueDType::F32 &&
        in.dtype != ValueDType::I32)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    const hipError_t herr = ::shelf::kernel::launch_output_gather(
        in.ptr, in.row_stride, out.ptr, out.row_stride, ctx.batch_context,
        ctx.actual_outputs, in.feature_count, in.dtype, stream);
    if (herr != hipSuccess)
        return Status::hip_error("output_gather optimized launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_linear_fp8(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.output_count != 1 || b.input_count != 0)
        return Status::invalid_state("fp8 linear: bad io count", __FILE__, __LINE__);
    if (b.compute_spec != 1u)
        return Status::invalid_state("fp8 linear: compute_spec != 1", __FILE__, __LINE__);
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return Status::invalid_state("fp8 linear: weight_index invalid", __FILE__, __LINE__);
    if (b.workspace_slot >= program.workspace.ranges.size())
        return Status::invalid_state("fp8 linear: workspace_slot invalid", __FILE__, __LINE__);
    const ::shelf::runtime::WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 1u)
        return Status::invalid_state("fp8 linear: encoding != Fp8", __FILE__, __LINE__);
    if (e.codes == nullptr || e.scales == nullptr)
        return Status::invalid_state("fp8 linear: weight view missing", __FILE__, __LINE__);
    if (e.rows == 0 || e.cols == 0)
        return Status::invalid_state("fp8 linear: weight dims == 0", __FILE__, __LINE__);
    const uint32_t wkp = e.k_padded;
    if (wkp < e.cols || (wkp & 127u) != 0u)
        return Status::invalid_state("fp8 linear: weight k_padded invalid", __FILE__, __LINE__);
    if (!quantization::weight_scale_group_matches(e.weight_scale_group, quantization::QuantKernelKind::LinearFp8Finegrained))
        return Status::invalid_state("fp8 linear: weight_scale_group unsupported", __FILE__, __LINE__);
    const uint32_t sstride = e.storage_scale_stride_bytes;
    if (sstride != (wkp / e.weight_scale_group) * 4u)
        return Status::invalid_state("fp8 linear: storage_scale_stride invalid", __FILE__, __LINE__);
    const uint32_t akp = b.activation_kp;
    if (akp != wkp)
        return Status::invalid_state("fp8 linear: activation_kp != weight k_padded", __FILE__, __LINE__);
    if (b.activation_code_stride < akp || (b.activation_code_stride & 15u) != 0u)
        return Status::invalid_state("fp8 linear: activation_code_stride invalid", __FILE__, __LINE__);
    if (b.activation_scale_stride < 4u || (b.activation_scale_stride & 15u) != 0u)
        return Status::invalid_state("fp8 linear: activation_scale_stride invalid", __FILE__, __LINE__);
    if (b.activation_max_rows == 0)
        return Status::invalid_state("fp8 linear: activation_max_rows == 0", __FILE__, __LINE__);
    if (ctx.workspace == nullptr)
        return Status::invalid_state("fp8 linear: workspace not bound", __FILE__, __LINE__);
    const Int8ActivationWorkspaceLayout layout =
        Int8ActivationWorkspaceLayout::make(akp, b.activation_max_rows);
    if (layout.k_padded != akp ||
        layout.code_row_stride_bytes != b.activation_code_stride ||
        layout.scale_row_stride_bytes != b.activation_scale_stride)
        return Status::invalid_state("fp8 linear: workspace layout mismatch", __FILE__, __LINE__);
    const uint32_t out_vid = b.output_slots[0];
    auto out_res = resolve_host_value(program, ::shelf::runtime::ValueId{out_vid}, ctx);
    if (!out_res.ok())
        return out_res.status();
    const HostResolvedValue out = out_res.value();
    if (out.feature_count != e.rows)
        return Status::invalid_state(
            "fp8 linear: output features != weight rows", __FILE__, __LINE__);
    if (out.row_stride != out.feature_count || e.output_col_offset != 0u)
        return Status::unsupported(
            "fp8 linear: padded row stride or column offset requires optimized path",
            __FILE__, __LINE__);
    const bool out_bf16 = out.dtype == ValueDType::BF16;
    const bool out_f32 = out.dtype == ValueDType::F32;
    if (!out_bf16 && !out_f32)
        return Status::invalid_state("fp8 linear: output dtype not BF16/F32", __FILE__, __LINE__);
    const uint32_t rows = rows_for(out, ctx);
    if (rows == 0)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (rows > b.activation_max_rows)
        return Status::invalid_state("fp8 linear: rows out of range", __FILE__, __LINE__);
    const Fp8FinegrainedGemmSelectorInput lin{rows, e.rows, e.cols, wkp};
    const auto config = select_fp8_finegrained_gemm_config(lin);
    if (!config) {
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    }
    if (!e.preshuffled)
        return Status::invalid_state("fp8 linear: weight not preshuffled to native layout",
                                   __FILE__, __LINE__);
    const uint64_t range_off = program.workspace.ranges[b.workspace_slot].offset;
    uint8_t* range_base = ctx.workspace + range_off;
    const uint8_t* acodes =
        reinterpret_cast<const uint8_t*>(range_base + layout.codes_offset);
    const float* ascales =
        reinterpret_cast<const float*>(range_base + layout.scales_offset);
    const ::shelf::kernel::Fp8FinegrainedGemmOutputDType odt =
        out_bf16 ? ::shelf::kernel::Fp8FinegrainedGemmOutputDType::BF16
                 : ::shelf::kernel::Fp8FinegrainedGemmOutputDType::F32;
    hipError_t herr = ::shelf::kernel::launch_gemm_fp8_finegrained_w8a8_wmma_auto(
            *config,
            static_cast<const uint8_t*>(e.codes),
            static_cast<const float*>(e.scales),
            reinterpret_cast<const int8_t*>(acodes), ascales,
            out.ptr, odt,
            rows, e.rows, wkp,
            sstride, b.activation_code_stride, b.activation_scale_stride,
            out.row_stride, stream, e.weight_scale_group);
    if (herr == hipErrorNotSupported)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (herr != hipSuccess)
        return Status::hip_error("fp8 gemm launch", hipGetErrorString(herr),
                                   __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_linear_mxfp4(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.output_count != 1 || b.input_count != 0)
        return Status::invalid_state("mxfp4 linear: bad io count", __FILE__, __LINE__);
    if (b.compute_spec != 2u)
        return Status::invalid_state("mxfp4 linear: compute_spec != MXFP4_W4A8", __FILE__, __LINE__);
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return Status::invalid_state("mxfp4 linear: weight_index invalid", __FILE__, __LINE__);
    if (b.workspace_slot >= program.workspace.ranges.size())
        return Status::invalid_state("mxfp4 linear: workspace_slot invalid", __FILE__, __LINE__);
    const ::shelf::runtime::WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 3u)
        return Status::invalid_state("mxfp4 linear: encoding != Mxfp4", __FILE__, __LINE__);
    if (e.codes == nullptr || e.scales == nullptr)
        return Status::invalid_state("mxfp4 linear: weight view missing", __FILE__, __LINE__);

    if (e.rows == 0 || e.cols == 0)
        return Status::invalid_state("mxfp4 linear: weight dims == 0", __FILE__, __LINE__);
    if (!e.preshuffled)
        return Status::invalid_state("mxfp4 linear: weight not preshuffled to native layout",
                                     __FILE__, __LINE__);
    const uint32_t wkp = e.k_padded;
    if (wkp < e.cols || (wkp & 127u) != 0u)
        return Status::invalid_state("mxfp4 linear: weight k_padded invalid", __FILE__, __LINE__);
    if (e.weight_scale_group != 32u)
        return Status::invalid_state("mxfp4 linear: weight_scale_group must be 32",
                                     __FILE__, __LINE__);
    const uint32_t nb = wkp >> 5;
    if (e.codes_row_stride_bytes != nb * 256u || e.scale_row_stride_bytes != nb ||
        e.storage_scale_stride_bytes != nb) {
        return Status::invalid_state("mxfp4 linear: weight strides are not the native layout",
                                     __FILE__, __LINE__);
    }
    const uint32_t akp = b.activation_kp;
    if (akp != wkp)
        return Status::invalid_state("mxfp4 linear: activation_kp != weight k_padded",
                                     __FILE__, __LINE__);
    if (b.activation_code_stride < akp || (b.activation_code_stride & 15u) != 0u)
        return Status::invalid_state("mxfp4 linear: activation_code_stride invalid",
                                     __FILE__, __LINE__);
    if (b.activation_scale_stride < 4u || (b.activation_scale_stride & 15u) != 0u)
        return Status::invalid_state("mxfp4 linear: activation_scale_stride invalid",
                                     __FILE__, __LINE__);
    if (b.activation_max_rows == 0)
        return Status::invalid_state("mxfp4 linear: activation_max_rows == 0", __FILE__, __LINE__);
    if (ctx.workspace == nullptr)
        return Status::invalid_state("mxfp4 linear: workspace not bound", __FILE__, __LINE__);
    const Int8ActivationWorkspaceLayout layout =
        Int8ActivationWorkspaceLayout::make(akp, b.activation_max_rows);
    if (layout.k_padded != akp ||
        layout.code_row_stride_bytes != b.activation_code_stride ||
        layout.scale_row_stride_bytes != b.activation_scale_stride)
        return Status::invalid_state("mxfp4 linear: workspace layout mismatch", __FILE__, __LINE__);
    const uint32_t out_vid = b.output_slots[0];
    auto out_res = resolve_host_value(program, ::shelf::runtime::ValueId{out_vid}, ctx);
    if (!out_res.ok())
        return out_res.status();
    const HostResolvedValue out = out_res.value();
    if (out.feature_count != e.rows)
        return Status::invalid_state(
            "mxfp4 linear: output features != weight rows", __FILE__, __LINE__);
    if (out.row_stride != out.feature_count || e.output_col_offset != 0u)
        return Status::unsupported(
            "mxfp4 linear: padded row stride or column offset requires optimized path",
            __FILE__, __LINE__);
    const bool out_bf16 = out.dtype == ValueDType::BF16;
    const bool out_f32 = out.dtype == ValueDType::F32;
    if (!out_bf16 && !out_f32)
        return Status::invalid_state("mxfp4 linear: output dtype not BF16/F32", __FILE__, __LINE__);
    const uint32_t rows = rows_for(out, ctx);
    if (rows == 0)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (rows > b.activation_max_rows)
        return Status::invalid_state("mxfp4 linear: rows out of range", __FILE__, __LINE__);
    const Mxfp4GemmSelectorInput sel{rows, e.rows, e.cols, wkp};
    const auto config = select_mxfp4_gemm_config(sel);
    if (!config) {
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    }
    const uint64_t range_off = program.workspace.ranges[b.workspace_slot].offset;
    uint8_t* range_base = ctx.workspace + range_off;
    const uint8_t* acodes =
        reinterpret_cast<const uint8_t*>(range_base + layout.codes_offset);
    const float* ascales =
        reinterpret_cast<const float*>(range_base + layout.scales_offset);
    const ::shelf::kernel::Mxfp4GemmOutputDType odt =
        out_bf16 ? ::shelf::kernel::Mxfp4GemmOutputDType::BF16
                 : ::shelf::kernel::Mxfp4GemmOutputDType::F32;
    hipError_t herr = ::shelf::kernel::launch_gemm_mxfp4_w4a8_auto(
        *config,
        static_cast<const uint8_t*>(e.codes),
        static_cast<const uint8_t*>(e.scales),
        shelf::kernel::mxfp4_gemm_folds_row_ref(config->id) && e.row_ref_exact &&
                e.row_ref != nullptr
            ? static_cast<const uint8_t*>(e.row_ref) : nullptr,
        reinterpret_cast<const int8_t*>(acodes), ascales,
        out.ptr, odt,
        rows, e.rows, wkp,
        e.scale_row_stride_bytes, b.activation_code_stride, b.activation_scale_stride,
        out.row_stride, stream);
    if (herr == hipErrorNotSupported)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (herr != hipSuccess)
        return Status::hip_error("mxfp4 gemm launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<PhysicalResolveStatus> resolve_bf16_physical(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalBf16Launch& out) {
    out = PhysicalBf16Launch{};
    if (b.output_count != 1 || b.input_count != 1)
        return Status::invalid_state("bf16 linear: bad io count", __FILE__, __LINE__);
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return Status::invalid_state("bf16 gemv: weight_index invalid", __FILE__, __LINE__);
    const ::shelf::runtime::WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 0u)
        return Status::invalid_state("bf16 gemv: encoding != 0", __FILE__, __LINE__);
    if (e.device_ptr == nullptr || e.codes != nullptr || e.scales != nullptr)
        return Status::invalid_state("bf16 gemv: bad weight slot", __FILE__, __LINE__);
    if (e.rows == 0 || e.cols == 0)
        return Status::invalid_state("bf16 gemv: weight dims == 0", __FILE__, __LINE__);
    auto in_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok())
        return in_res.status();
    const HostResolvedValue in = in_res.value();
    if (in.dtype != ValueDType::BF16)
        return PhysicalResolveStatus::NotApplicable;
    if (in.feature_count != e.cols)
        return Status::invalid_state("bf16 gemv: input features != weight cols", __FILE__, __LINE__);
    auto out_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok())
        return out_res.status();
    const HostResolvedValue dst = out_res.value();
    const uint64_t expected_out_features =
        e.output_col_total != 0u ? static_cast<uint64_t>(e.output_col_total)
                                 : static_cast<uint64_t>(e.rows);
    if (static_cast<uint64_t>(dst.feature_count) != expected_out_features)
        return Status::invalid_state("bf16 gemv: output features != weight column width",
                                     __FILE__, __LINE__);
    if (e.output_col_total != 0u &&
        static_cast<uint64_t>(e.output_col_offset) + e.rows >
            static_cast<uint64_t>(e.output_col_total))
        return Status::invalid_state("bf16 gemv: column offset out of range", __FILE__,
                                     __LINE__);
    const bool out_bf16 = dst.dtype == ValueDType::BF16;
    const bool out_f32 = dst.dtype == ValueDType::F32;
    if (!out_bf16 && !out_f32)
        return Status::invalid_state("bf16 gemv: output dtype not BF16/F32", __FILE__, __LINE__);
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0)
        return PhysicalResolveStatus::Skipped;
    const ::shelf::bf16_t* weight = static_cast<const ::shelf::bf16_t*>(e.device_ptr);
    const ::shelf::bf16_t* input = static_cast<const ::shelf::bf16_t*>(in.ptr);
    const uint64_t output_offset_bytes =
        static_cast<uint64_t>(e.output_col_offset) * value_dtype_bytes(dst.dtype);
    Bf16GemmSelectorInput lin;
    lin.rows = rows;
    lin.out_features = e.rows;
    lin.k = e.cols;
    lin.input_row_stride = in.row_stride;
    lin.weight_aligned16 =
        (reinterpret_cast<uintptr_t>(weight) & 15u) == 0u;
    lin.input_aligned16 =
        (reinterpret_cast<uintptr_t>(input) & 15u) == 0u;
    const auto config = select_bf16_gemm_config(lin);
    if (!config) {
        return PhysicalResolveStatus::NotApplicable;
    }
    if (std::getenv("SHELF_BF16_DISPATCH_DEBUG") != nullptr) {
        std::fprintf(stderr, "BF16_DISPATCH out=%u k=%u rows=%u id=%u\n", e.rows, e.cols,
                     rows, static_cast<unsigned>(config->id));
    }

    out.weight = reinterpret_cast<uint64_t>(e.device_ptr);
    out.input = reinterpret_cast<uint64_t>(in.ptr);
    out.output = reinterpret_cast<uint64_t>(dst.ptr) + output_offset_bytes;
    out.rows = rows;
    out.out_features = e.rows;
    out.k = e.cols;
    out.input_row_stride = in.row_stride;
    out.output_row_stride = dst.row_stride;
    out.output_dtype = out_bf16 ? ::shelf::kernel::Bf16GemmOutputDType::BF16
                                : ::shelf::kernel::Bf16GemmOutputDType::F32;
    out.config = *config;
    out.output_rows_domain =
        dst.row_domain == ValueRowDomain::OUTPUT_ROWS;
    switch (config->id) {
        case ::shelf::kernel::Bf16GemmConfigId::ExactRows:
            out.variant = PhysicalBf16Variant::ExactRows;
            out.grid_x = ::shelf::kernel::bf16_exact_rows_grid_x(e.rows);
            out.grid_y = 1u;
            out.workgroup = ::shelf::kernel::kBf16GemmExactRowsThreads;
            break;
        case ::shelf::kernel::Bf16GemmConfigId::ExactRowsDot2:
            out.variant = PhysicalBf16Variant::ExactRowsDot2;
            out.grid_x = ::shelf::kernel::bf16_exact_rows_grid_x(e.rows);
            out.grid_y = 1u;
            out.workgroup = ::shelf::kernel::kBf16GemmExactRowsThreads;
            break;
        case ::shelf::kernel::Bf16GemmConfigId::SmallRows:
            out.variant = PhysicalBf16Variant::SmallRows;
            out.grid_x = e.rows;
            out.grid_y = 1u;
            out.workgroup = ::shelf::kernel::kBf16GemmExactRowsThreads;
            break;
        case ::shelf::kernel::Bf16GemmConfigId::WmmaKPartition:
            out.variant = PhysicalBf16Variant::WmmaKPartition;
            out.grid_x = ::shelf::kernel::bf16_wmma_grid_x(e.rows);
            out.grid_y = ::shelf::kernel::bf16_wmma_grid_y(rows, 1u);
            out.workgroup = ::shelf::kernel::kBf16SplitKWmmaThreads;
            break;
        case ::shelf::kernel::Bf16GemmConfigId::WmmaWide:
            out.variant = PhysicalBf16Variant::WmmaWide;
            out.grid_x = rows >= 512u
                             ? ::shelf::kernel::bf16_wmma_grid_x_wide(e.rows)
                             : ::shelf::kernel::bf16_wmma_grid_x(e.rows);
            out.grid_y = ::shelf::kernel::bf16_wmma_grid_y(
                rows, ::shelf::kernel::kBf16WmmaWideRowTiles);
            out.workgroup = ::shelf::kernel::kBf16WmmaThreads;
            break;
        case ::shelf::kernel::Bf16GemmConfigId::Wmma:
            out.variant = PhysicalBf16Variant::Wmma;
            out.grid_x = ::shelf::kernel::bf16_wmma_grid_x(e.rows);
            out.grid_y = ::shelf::kernel::bf16_wmma_grid_y(rows, 1u);
            out.workgroup = ::shelf::kernel::kBf16WmmaThreads;
            break;
    }
    return PhysicalResolveStatus::Applicable;
}

bool exl3_mul24_use_a() {
    static const bool requested = [] {
        const char* env = std::getenv("SHELF_EXL3_MUL24_PREFILL_AB");
        return env == nullptr || ((env[0] == 'A' || env[0] == 'a') && env[1] == '\0');
    }();
    if (!requested) return false;
    const char* rocroller = std::getenv("HIPBLASLT_USE_ROCROLLER");
    if (rocroller != nullptr && std::strcmp(rocroller, "1") == 0) {
        static std::atomic_flag reported = ATOMIC_FLAG_INIT;
        if (!reported.test_and_set())
            std::fprintf(stderr, "EXL3 INT8 prefill: rocRoller selected; using B kernel\n");
        return false;
    }
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess) return false;
    static thread_local int checked_device = -1;
    static thread_local bool supported = false;
    if (checked_device != device) {
        hipDeviceProp_t props{};
        supported = hipGetDeviceProperties(&props, device) == hipSuccess &&
                    std::strncmp(props.gcnArchName, "gfx1201", 7u) == 0;
        checked_device = device;
    }
    return supported;
}

bool exl3_mul24_use_dense(bool dense_ok, uint32_t n_rows, uint32_t dense_min_rows,
                          uint64_t rowsum_bytes) {
    return dense_ok && n_rows >= dense_min_rows &&
           static_cast<uint64_t>(n_rows) * sizeof(float) <= rowsum_bytes;
}

bool exl3_mul24_use_a_prefill(bool a_selected, uint32_t num_prefill_requests) {
    return a_selected && num_prefill_requests != 0u;
}

// EXL3-INT8 (mul24): rotate + int8-quantize the activations, run the int8 WMMA GEMM that decodes the resident
// trellis on every read, then rotate the output back. The three launches share the EXL3 workspace of the executor:
// the xrot region holds the int8 codes and their scales, the accum region the F32 pre-rotation output.
Result<OptimizedLaunchResult> try_launch_linear_exl3_mul24(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream, const ::shelf::kernel::Exl3SwigluProducer* producer,
    const ::shelf::kernel::Exl3GdnProducer* gdn_producer,
    const ::shelf::kernel::Exl3AttnProducer* attn_producer, BoundaryDeferred* deferred,
    const FinalProducer* final_producer) {
    if (deferred) *deferred = {};
    if (b.output_count != 1 || b.input_count != 1)
        return Status::invalid_state("exl3 int8 linear: bad io count", __FILE__, __LINE__);
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return Status::invalid_state("exl3 int8 linear: weight_index invalid", __FILE__, __LINE__);
    const ::shelf::runtime::WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 4u || e.device_ptr != nullptr || e.codes == nullptr || e.exl3_suh == nullptr ||
        e.exl3_svh == nullptr || e.preshuffled || e.exl3_half) {
        return Status::invalid_state("exl3 int8 linear: bad weight slot", __FILE__, __LINE__);
    }
    if (e.rows == 0 || e.cols == 0 || e.k_padded != e.cols || (e.cols % 128u) != 0u || (e.rows % 128u) != 0u)
        return Status::invalid_state("exl3 int8 linear: weight dims invalid", __FILE__, __LINE__);
    auto in_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    if (in.dtype != ValueDType::BF16) return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (in.feature_count != e.cols)
        return Status::invalid_state("exl3 int8 linear: input features != weight cols", __FILE__, __LINE__);
    auto out_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue dst = out_res.value();
    const uint64_t expected_out_features =
        e.output_col_total != 0u ? static_cast<uint64_t>(e.output_col_total) : static_cast<uint64_t>(e.rows);
    if (static_cast<uint64_t>(dst.feature_count) != expected_out_features)
        return Status::invalid_state("exl3 int8 linear: output features mismatch", __FILE__, __LINE__);
    if (e.output_col_total != 0u &&
        static_cast<uint64_t>(e.output_col_offset) + e.rows > static_cast<uint64_t>(e.output_col_total))
        return Status::invalid_state("exl3 int8 linear: column offset out of range", __FILE__, __LINE__);
    const bool out_bf16 = dst.dtype == ValueDType::BF16;
    if (!out_bf16 && dst.dtype != ValueDType::F32)
        return Status::invalid_state("exl3 int8 linear: output dtype not BF16/F32", __FILE__, __LINE__);
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0u) return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
    if (ctx.exl3_xrot == nullptr || ctx.exl3_accum == nullptr)
        return Status::invalid_state("exl3 int8 linear: no EXL3 workspace", __FILE__, __LINE__);

    const uint32_t n = e.rows, k = e.cols;
    const auto layout = ::shelf::kernel::exl3_detail::mul24::make_workspace_layout(
        rows, n, k, ctx.exl3_xrot_bytes, ctx.exl3_accum_bytes, ctx.exl3_dense_bytes);
    if (layout.chunk_rows == 0u)
        return Status::insufficient_memory(
            "exl3 int8 linear: workspace smaller than one row (n=" +
                std::to_string(n) + " k=" + std::to_string(k) + ")", __FILE__, __LINE__);
    const uint32_t chunk = layout.chunk_rows;
    const uint32_t scratch_rows = layout.scratch_rows;
    const uint32_t code_stride = layout.code_stride;
    const uint32_t scale_stride_bytes = layout.scale_stride_bytes;
    int8_t* codes = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(ctx.exl3_xrot) + layout.codes_offset);
    float* scales = reinterpret_cast<float*>(static_cast<uint8_t*>(ctx.exl3_xrot) + layout.scales_offset);
    float* yprime = reinterpret_cast<float*>(static_cast<uint8_t*>(ctx.exl3_accum) + layout.yprime_offset);
    const uint32_t out_elem_bytes = out_bf16 ? 2u : 4u;
    // That GEMM takes one activation scale per row (decode keeps one per 128-K group).
    static const uint32_t dense_min_rows = [] {
        const char* env = std::getenv("SHELF_EXL3_INT8_DENSE");
        if (env == nullptr) return 256u;
        const unsigned long v = std::strtoul(env, nullptr, 10);
        return v == 0ul ? 0xFFFFFFFFu : static_cast<uint32_t>(v);
    }();
    // A: hipBLASLt row-scale dense prefill + flagged-row correction.
    // B: hand-written gfx12 iu8 WMMA, K128 scale/FMA and no row re-quantization.
    const bool prefill_wmma_b = (e.plamo_exl3_linear && std::getenv("SHELF_EXL3_MUL24_PREFILL_AB") == nullptr) || !exl3_mul24_use_a();
    static const float flag_share = [] {
        const char* env = std::getenv("SHELF_EXL3_INT8_FLAG");
        return env != nullptr ? std::strtof(env, nullptr) : 0.5f;
    }();
    // Workspace layout of the dense path: slots[chunk] (+ the slot count), then the compact
    // codes / scales of the flagged rows in the xrot region; the accum region holds
    // acc[chunk][n] followed by the flagged rows' F32 output yprime2[cap][n].
    static const bool plamo_enabled = [] {
        const char *env = std::getenv("SHELF_PLAMO_EXL3_LINEAR");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    const bool plamo_prefill = e.plamo_exl3_linear && plamo_enabled &&
                               std::getenv("SHELF_EXL3_INT8_DENSE") == nullptr &&
                               std::getenv("SHELF_EXL3_MUL24_PREFILL_AB") == nullptr;
    const bool dense_ok = ctx.exl3_dense != nullptr && ctx.exl3_rowsum != nullptr && layout.fits;
    uint32_t launches = 0;
    for (uint32_t r0 = 0; r0 < rows; r0 += chunk) {
        const uint32_t n_rows = std::min(chunk, rows - r0);
        const bool use_dense =
            exl3_mul24_use_dense(dense_ok, n_rows, dense_min_rows, ctx.exl3_rowsum_bytes);
        const ::shelf::bf16_t* x = reinterpret_cast<const ::shelf::bf16_t*>(in.ptr) + static_cast<size_t>(r0) * in.row_stride;
        const uint16_t* suh = static_cast<const uint16_t*>(e.exl3_suh);
        const uint16_t* svh = static_cast<const uint16_t*>(e.exl3_svh);
        void* out_ptr = static_cast<uint8_t*>(dst.ptr) +
            (static_cast<uint64_t>(e.output_col_offset) + static_cast<uint64_t>(r0) * dst.row_stride) * out_elem_bytes;
        const auto out_dtype =
            out_bf16 ? ::shelf::kernel::Exl3OutputDType::BF16 : ::shelf::kernel::Exl3OutputDType::F32;
        hipError_t herr = hipSuccess;
        bool done = false;
        bool fallback_b = false;
        if (use_dense && exl3_mul24_use_a_prefill(!prefill_wmma_b, ctx.num_prefill_requests)) {
            float* rowscale = static_cast<float*>(ctx.exl3_rowsum);
            int8_t* dense = static_cast<int8_t*>(ctx.exl3_dense);
            uint8_t* xr = static_cast<uint8_t*>(ctx.exl3_xrot);
            int32_t* slots = reinterpret_cast<int32_t*>(xr + layout.slots_offset);
            uint32_t* count = reinterpret_cast<uint32_t*>(xr + layout.count_offset);
            int8_t* codes_f = reinterpret_cast<int8_t*>(xr + layout.flagged_codes_offset);
            float* scales_f = reinterpret_cast<float*>(xr + layout.flagged_scales_offset);
            float* yprime2 = reinterpret_cast<float*>(static_cast<uint8_t*>(ctx.exl3_accum) + layout.flagged_yprime_offset);
            const uint32_t cap = static_cast<uint32_t>(::shelf::kernel::exl3_mul24_flag_cap(n_rows));
            {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                herr = ::shelf::kernel::launch_exl3_mul24_rotate_quantize(
                    x, in.row_stride, suh, codes, scales, n_rows, k, code_stride,
                    scale_stride_bytes, stream, count);
                if (semantic_timing_enabled())
                    semantic_timing_finish(stream, timing, 0xffff0000u,
                                           static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            if (herr == hipSuccess) {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                static const bool row_fuse = [] {
                    if (std::getenv("SHELF_DISABLE_EXL3_ROW_FUSE") != nullptr) return false;
                    const char* v = std::getenv("SHELF_EXL3_ROW_FUSE");
                    return v == nullptr || (v[0] != '\0' && v[0] != '0');
                }();
                herr = row_fuse ? ::shelf::kernel::launch_exl3_mul24_rowscale_fused(
                    codes, scales, rowscale, slots, count, codes_f, scales_f, n_rows, cap, k, code_stride,
                    scale_stride_bytes, flag_share, stream)
                                : ::shelf::kernel::launch_exl3_mul24_rowscale(
                    codes, scales, rowscale, slots, count, codes_f, scales_f, n_rows, cap, k, code_stride,
                    scale_stride_bytes, flag_share, stream);
                if (semantic_timing_enabled())
                    semantic_timing_finish(stream, timing, 0xffff0001u,
                                           static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            if (herr == hipSuccess) {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                herr = ::shelf::kernel::launch_exl3_mul24_recon(
                    static_cast<const uint16_t*>(e.codes), e.exl3_ka, dense, n, k, stream);
                if (semantic_timing_enabled())
                    semantic_timing_finish(stream, timing, 0xffff0002u,
                                           static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            if (herr == hipSuccess) {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                herr = ::shelf::execution::exl3_dense_gemm_int8(n_rows, n, k, dense, codes, yprime, stream);
                if (herr == hipErrorNotSupported || herr == hipErrorInitializationError ||
                    herr == hipErrorOutOfMemory) {
                    static std::atomic_flag reported = ATOMIC_FLAG_INIT;
                    if (!reported.test_and_set())
                        std::fprintf(stderr, "EXL3 INT8 prefill: hipBLASLt unavailable; using B kernel\n");
                    herr = hipErrorNotSupported;
                }
                if (semantic_timing_enabled())
                    semantic_timing_finish(stream, timing, 0xffff0003u,
                                           static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            // The flagged rows (count of them, at most cap) go through the fused per-group kernel.
            if (herr == hipSuccess) {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                if (plamo_prefill)
                    herr = ::shelf::kernel::launch_plamo_exl3_flagged_gemm(
                        codes_f, scales_f, code_stride, scale_stride_bytes,
                        static_cast<const uint16_t *>(e.codes), e.exl3_ka, yprime2, cap, count, n, k,
                        stream);
                else
                    herr = ::shelf::kernel::launch_exl3_mul24_gemm(
                        codes_f, scales_f, code_stride, scale_stride_bytes,
                        static_cast<const uint16_t *>(e.codes), e.exl3_ka, yprime2, cap, n, k, 1u, stream,
                        count);
                if (semantic_timing_enabled())
                    semantic_timing_finish(stream, timing, 0xffff0004u,
                                           static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            if (herr == hipSuccess) {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                if (plamo_prefill)
                    herr = ::shelf::kernel::launch_plamo_exl3_rotate_out_i32(
                        static_cast<const int32_t *>(ctx.exl3_accum), rowscale, slots, yprime2, svh,
                        out_ptr, out_bf16, n_rows, n, dst.row_stride, stream);
                else
                    herr = ::shelf::kernel::launch_exl3_mul24_rotate_out_i32(
                        static_cast<const int32_t *>(ctx.exl3_accum), rowscale, slots, yprime2, svh,
                        out_ptr, out_dtype, n_rows, n, dst.row_stride, stream);
                if (semantic_timing_enabled())
                    semantic_timing_finish(stream, timing, 0xffff0005u,
                                           static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            done = herr == hipSuccess;
            launches += done ? 7u : 0u;
            if (herr == hipErrorNotSupported) {
                fallback_b = true;
                herr = hipSuccess;
            }
        }
        if (use_dense && (prefill_wmma_b || fallback_b)) {
            // B uses the exact original group codes/scales and the same dense
            // int8 weight reconstruction as A. No rowscale, slots or flagged GEMM.
            const uint32_t timing0 = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
            herr = ::shelf::kernel::launch_exl3_mul24_rotate_quantize(
                x, in.row_stride, suh, codes, scales, n_rows, k, code_stride, scale_stride_bytes, stream);
            if (semantic_timing_enabled()) semantic_timing_finish(
                stream, timing0, 0xffff0010u, static_cast<uint32_t>(SemPath::Optimized), 0);
            if (herr == hipSuccess) {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                herr = ::shelf::kernel::launch_exl3_mul24_recon_wmma(
                    static_cast<const uint16_t*>(e.codes), e.exl3_ka,
                    static_cast<int8_t*>(ctx.exl3_dense), n, k, stream);
                if (semantic_timing_enabled()) semantic_timing_finish(
                    stream, timing, 0xffff0011u, static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            if (herr == hipSuccess) {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                herr = ::shelf::kernel::launch_exl3_mul24_prefill_wmma(
                    codes, scales, code_stride, scale_stride_bytes,
                    static_cast<const int8_t*>(ctx.exl3_dense), yprime, n_rows, n, k, stream);
                if (semantic_timing_enabled()) semantic_timing_finish(
                    stream, timing, 0xffff0012u, static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            if (herr == hipSuccess) {
                const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
                herr = ::shelf::kernel::launch_exl3_mul24_rotate_out(
                    yprime, svh, out_ptr, out_dtype, n_rows, n, dst.row_stride, 1u, stream);
                if (semantic_timing_enabled()) semantic_timing_finish(
                    stream, timing, 0xffff0013u, static_cast<uint32_t>(SemPath::Optimized), 0);
            }
            done = herr == hipSuccess;
            launches += done ? 4u : 0u;
        }
        if (!done && herr == hipSuccess) {
            const uint32_t split_rows =
                ctx.splitk_tuning_rows != 0u ? ctx.splitk_tuning_rows : n_rows;
            uint32_t splits = ::shelf::kernel::exl3_mul24_gemm_splits(split_rows, n, k, scratch_rows, e.exl3_ka);
            if (e.plamo_exl3_split4 && ctx.num_prefill_requests == 0u && ctx.splitk_tuning_rows == 0u &&
                n_rows > 16u && n_rows <= 64u && scratch_rows >= 4u * n_rows)
                splits = 4u;
            herr = final_producer ? launch_final_gather_norm_rotate(
                *final_producer, suh, codes, scales, n_rows, code_stride, scale_stride_bytes, stream) :
                attn_producer ? ::shelf::kernel::launch_exl3_attn_rotate_rows(
                *attn_producer, suh, codes, scales, n_rows, k, code_stride, scale_stride_bytes, stream) :
                gdn_producer ? ::shelf::kernel::launch_exl3_gdn_rotate_rows(
                *gdn_producer, suh, codes, scales, n_rows, k, code_stride, scale_stride_bytes, stream) :
                producer ? ::shelf::kernel::launch_exl3_swiglu_rotate_rows(
                *producer, suh, codes, scales, n_rows, k, code_stride, scale_stride_bytes, stream) :
                ::shelf::kernel::launch_exl3_mul24_rotate_quantize(
                x, in.row_stride, suh, codes, scales, n_rows, k, code_stride, scale_stride_bytes, stream);
            if (herr == hipSuccess)
                herr = ::shelf::kernel::launch_exl3_mul24_gemm(
                    codes, scales, code_stride, scale_stride_bytes, static_cast<const uint16_t*>(e.codes), e.exl3_ka,
                    yprime, n_rows, n, k, splits, stream);
            const bool can_defer = deferred && r0 == 0u && n_rows == rows && n == 5120u &&
                out_bf16 && e.output_col_total == 0u && !e.plamo_exl3_linear && rows <= 16u;
            if (herr == hipSuccess && can_defer) {
                *deferred = {yprime, svh, out_ptr, rows, n, dst.row_stride, splits};
            } else if (herr == hipSuccess) {
                herr = ::shelf::kernel::launch_exl3_mul24_rotate_out(
                    yprime, svh, out_ptr, out_dtype, n_rows, n, dst.row_stride, splits, stream);
            }
            launches += can_defer ? 2u : 3u;
        }
        if (herr != hipSuccess)
            return Status::hip_error("exl3 int8 launch", hipGetErrorString(herr), __FILE__, __LINE__);
    }
    record_physical_launch(launches);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_linear_bf16(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    PhysicalBf16Launch launch{};
    const Result<PhysicalResolveStatus> resolved =
        resolve_bf16_physical(program, b, ctx, launch);
    if (!resolved.ok())
        return resolved.status();
    if (resolved.value() == PhysicalResolveStatus::NotApplicable)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (resolved.value() == PhysicalResolveStatus::Skipped)
        return OptimizedLaunchResult{OptimizedLaunchKind::Skipped, 1u};
    constexpr uint32_t kLtMinRows = 128u;
    hipError_t herr = hipErrorNotSupported;
    if (launch.rows >= kLtMinRows) {
        herr = shelf::execution::gemm_bf16_lt(
            reinterpret_cast<const ::shelf::bf16_t*>(launch.weight),
            reinterpret_cast<const ::shelf::bf16_t*>(launch.input),
            reinterpret_cast<void*>(launch.output), launch.rows, launch.out_features,
            launch.k, launch.input_row_stride, launch.output_row_stride,
            launch.output_dtype == ::shelf::kernel::Bf16GemmOutputDType::BF16, stream);
    }
    if (herr == hipErrorNotSupported) {
        herr = ::shelf::kernel::launch_gemm_bf16(
            launch.config, reinterpret_cast<const ::shelf::bf16_t*>(launch.weight),
            reinterpret_cast<const ::shelf::bf16_t*>(launch.input),
            reinterpret_cast<void*>(launch.output), launch.output_dtype,
            launch.rows, launch.out_features, launch.k, launch.input_row_stride,
            launch.output_row_stride, stream);
    }
    if (herr == hipErrorNotSupported)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (herr != hipSuccess)
        return Status::hip_error("bf16 gemm launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}


Result<OptimizedLaunchResult> try_launch_linear_bf16_pair(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    const auto miss = OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_BF16_PAIR") == nullptr;
    if (!enabled) return miss;
    if (dispatch_index + 1u >= program.dispatches.size()) return miss;
    const auto& b0 = program.dispatches[dispatch_index];
    const auto& b1 = program.dispatches[dispatch_index + 1u];
    if (b0.kernel_id != ::shelf::runtime::KernelId::LINEAR_BF16 ||
        b1.kernel_id != ::shelf::runtime::KernelId::LINEAR_BF16 ||
        b0.input_count != 1u || b0.output_count != 1u || b1.input_count != 1u ||
        b1.output_count != 1u)
        return miss;
    if (b0.input_slots[0] != b1.input_slots[0]) return miss;
    PhysicalBf16Launch l0{};
    PhysicalBf16Launch l1{};
    const Result<PhysicalResolveStatus> r0 = resolve_bf16_physical(program, b0, ctx, l0);
    if (!r0.ok()) return r0.status();
    if (r0.value() != PhysicalResolveStatus::Applicable) return miss;
    const Result<PhysicalResolveStatus> r1 = resolve_bf16_physical(program, b1, ctx, l1);
    if (!r1.ok()) return r1.status();
    if (r1.value() != PhysicalResolveStatus::Applicable) return miss;
    if (l0.variant != PhysicalBf16Variant::ExactRows ||
        l1.variant != PhysicalBf16Variant::ExactRows)
        return miss;
    if (l0.rows == 0u || l0.rows != l1.rows ||
        l0.rows > ::shelf::kernel::kBf16GemmExactRowsKernelMax)
        return miss;
    if (l0.k != l1.k || l0.input != l1.input ||
        l0.input_row_stride != l1.input_row_stride)
        return miss;
    const hipError_t herr = ::shelf::kernel::launch_gemm_bf16_exact_rows_pair(
        reinterpret_cast<const ::shelf::bf16_t*>(l0.weight),
        reinterpret_cast<void*>(l0.output), l0.output_dtype, l0.out_features,
        l0.output_row_stride,
        reinterpret_cast<const ::shelf::bf16_t*>(l1.weight),
        reinterpret_cast<void*>(l1.output), l1.output_dtype, l1.out_features,
        l1.output_row_stride,
        reinterpret_cast<const ::shelf::bf16_t*>(l0.input), l0.input_row_stride, l0.rows,
        l0.k, false, stream);
    if (herr == hipErrorNotSupported) return miss;
    if (herr != hipSuccess)
        return Status::hip_error("bf16 pair gemm launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 2u};
}


Result<OptimizedLaunchResult> try_launch_linear_hyper_up_mix(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    const auto miss = OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_HYPER_UP_MIX") == nullptr;
    if (!enabled) return miss;
    if (dispatch_index + 1u >= program.dispatches.size()) return miss;
    const auto& b0 = program.dispatches[dispatch_index];
    const auto& b1 = program.dispatches[dispatch_index + 1u];
    if (b0.kernel_id != ::shelf::runtime::KernelId::LINEAR_BF16 ||
        b1.kernel_id != ::shelf::runtime::KernelId::HYPER_MIX)
        return miss;
    if (b0.input_count != 1u || b0.output_count != 1u || b1.input_count != 2u ||
        b1.output_count != 1u)
        return miss;
    if (b1.input_slots[0] != b0.output_slots[0]) return miss;
    const uint32_t hc = b1.group_size;
    const uint32_t hidden = b1.flags;
    if (hc == 0u || hidden == 0u || hc > ::shelf::kernel::kHyperUpMixMaxHc) return miss;
    PhysicalBf16Launch l{};
    const Result<PhysicalResolveStatus> resolved = resolve_bf16_physical(program, b0, ctx, l);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable) return miss;
    if (l.variant != PhysicalBf16Variant::ExactRows) return miss;
    if (l.rows == 0u || l.rows > ::shelf::kernel::kBf16GemmExactRowsKernelMax) return miss;
    if (l.out_features != hc * hidden) return miss;
    auto gate = resolve_host_value(program, ::shelf::runtime::ValueId{b1.input_slots[0]}, ctx);
    if (!gate.ok()) return gate.status();
    auto normed = resolve_host_value(program, ::shelf::runtime::ValueId{b1.input_slots[1]}, ctx);
    if (!normed.ok()) return normed.status();
    auto out = resolve_host_value(program, ::shelf::runtime::ValueId{b1.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    if (gate.value().dtype != ValueDType::BF16 || normed.value().dtype != ValueDType::BF16 ||
        out.value().dtype != ValueDType::BF16)
        return miss;
    if (gate.value().feature_count != hc * hidden || normed.value().feature_count != hc * hidden ||
        out.value().feature_count != hidden)
        return miss;
    if (gate.value().row_domain != normed.value().row_domain ||
        gate.value().row_domain != out.value().row_domain)
        return miss;
    if (gate.value().ptr != reinterpret_cast<const void*>(l.output) ||
        normed.value().ptr == nullptr || out.value().ptr == nullptr)
        return miss;
    if (rows_for(out.value(), ctx) != l.rows) return miss;
    ::shelf::kernel::HyperUpMixArgs args{};
    args.up_weight = reinterpret_cast<const void*>(l.weight);
    args.input = reinterpret_cast<const void*>(l.input);
    args.normed = normed.value().ptr;
    args.output = out.value().ptr;
    args.rows = l.rows;
    args.hc = hc;
    args.hidden = hidden;
    args.k = l.k;
    args.input_stride = l.input_row_stride;
    args.normed_stride = normed.value().row_stride;
    args.output_stride = out.value().row_stride;
    args.sigmoid_gate = b1.compute_spec != 0u ? 1u : 0u;
    const hipError_t herr = ::shelf::kernel::launch_hyper_up_mix(args, stream);
    if (herr == hipErrorNotSupported) return miss;
    if (herr != hipSuccess)
        return Status::hip_error("hyper up+mix launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 2u};
}


Result<OptimizedLaunchResult> try_launch_linear_int8_w32(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    const auto miss = OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (b.input_count != 1u || b.output_count != 1u) return miss;
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return miss;
    const auto& e = ctx.weights[b.weight_index];
    if (e.encoding != static_cast<uint8_t>(::shelf::weights::MatrixEncoding::Int8W32)) return miss;
    if (e.codes == nullptr || e.scales == nullptr || e.rows == 0u || e.cols == 0u) return miss;
    if ((e.cols % 128u) != 0u || e.k_padded != e.cols) return miss;
    auto input = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!input.ok()) return input.status();
    auto output = resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!output.ok()) return output.status();
    const HostResolvedValue& inv = input.value();
    const HostResolvedValue& outv = output.value();
    const uint32_t rows = rows_for(outv, ctx);
    // A zero-row context still walks the head dispatch; there is nothing to compute,
    // and the device interpreter has no int8 w32 opcode to fall back to.
    if (rows == 0u) return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
    if (rows > 16u) return miss;
    if (inv.ptr == nullptr || outv.ptr == nullptr) return miss;
    if (inv.dtype != ::shelf::runtime::ValueDType::BF16 ||
        outv.dtype != ::shelf::runtime::ValueDType::F32)
        return miss;
    if (inv.feature_count != e.cols || outv.feature_count < e.rows) return miss;
    if (inv.row_stride < e.cols || outv.row_stride < outv.feature_count) return miss;
    if (static_cast<uint64_t>(e.output_col_offset) + e.rows > outv.feature_count) return miss;
    auto* logits = static_cast<float*>(outv.ptr) + e.output_col_offset;
    const hipError_t herr = ::shelf::kernel::launch_lm_head_int8_w32(
        static_cast<const ::shelf::bf16_t*>(inv.ptr), inv.row_stride,
        static_cast<const uint8_t*>(e.codes),
        static_cast<const ::shelf::fp16_t*>(e.scales), logits, outv.row_stride, rows,
        e.rows, e.cols, stream);
    if (herr != hipSuccess)
        return Status::hip_error("int8 w32 lm_head launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_boundary_i8(
    const ::shelf::runtime::Program&, size_t, size_t, HostExecutionContext&, hipStream_t,
    const ::shelf::kernel::PlamoExl3ProjectionGroup&);
Result<OptimizedLaunchResult> try_launch_boundary_i9(
    const ::shelf::runtime::Program&, size_t, size_t, HostExecutionContext&, hipStream_t,
    const ::shelf::kernel::PlamoExl3ProjectionGroup&, const BoundaryDeferred&);

Result<OptimizedLaunchResult> try_launch_exl3_projection(
    const ::shelf::runtime::Program& program, size_t first, size_t end,
    HostExecutionContext& ctx, hipStream_t stream,
    const ::shelf::runtime::DispatchBinding* producer = nullptr,
    const BoundaryDeferred* deferred = nullptr) {
    const auto miss = OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    const auto& binding = program.dispatches[first];
    if (binding.weight_index >= ctx.weight_count || ctx.weights == nullptr) return miss;
    const auto& weight = ctx.weights[binding.weight_index];
    const bool is_plamo = weight.plamo_exl3_linear;
    static const bool generic_enabled = std::getenv("SHELF_EXL3_INT8_ROTATE_F4") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr;
    const bool grouped_3bit = weight.exl3_ka == 3u && ctx.capturing_graph;
    if (!is_plamo && ((!grouped_3bit && weight.exl3_ka != 4u) ||
                      ctx.layer_override_count != 0u || !generic_enabled))
        return miss;
    static const bool enabled = [] {
        const char* env = std::getenv("SHELF_PLAMO_EXL3_LINEAR");
        return env == nullptr || std::strcmp(env, "0") != 0;
    }();
    if ((is_plamo && !enabled) || binding.input_count != 1u || binding.output_count != 1u) return miss;
    auto input = resolve_host_value(program, ::shelf::runtime::ValueId{binding.input_slots[0]}, ctx);
    if (!input.ok()) return input.status();
    auto output = resolve_host_value(program, ::shelf::runtime::ValueId{binding.output_slots[0]}, ctx);
    if (!output.ok()) return output.status();
    const uint32_t rows = rows_for(output.value(), ctx);
    static const uint32_t prefill_mode = [] {
        const char* env = std::getenv("SHELF_PLAMO_EXL3_PREFILL");
        return env ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10)) : 1u;
    }();
    static const uint32_t prefill_rows = [] {
        const char* env = std::getenv("SHELF_PLAMO_EXL3_PREFILL_ROWS");
        return env ? std::max(65u, static_cast<uint32_t>(std::strtoul(env, nullptr, 10))) : 65u;
    }();
    const bool use_prefill = is_plamo && prefill_mode != 0u && rows >= prefill_rows &&
        (weight.exl3_ka == 4u || weight.exl3_ka == 5u);
    if (rows == 0u || (!is_plamo && rows > (grouped_3bit ? 16u : 64u)) || (rows >= 256u && !use_prefill) || input.value().dtype != ValueDType::BF16 ||
        output.value().dtype != ValueDType::BF16 ||
        ctx.exl3_xrot == nullptr || ctx.exl3_accum == nullptr)
        return miss;
    const bool custom_prefill = std::getenv("SHELF_EXL3_INT8_DENSE") != nullptr ||
        (is_plamo && std::getenv("SHELF_EXL3_MUL24_PREFILL_AB") != nullptr) ||
        (!is_plamo && rows > 16u && exl3_mul24_use_a_prefill(exl3_mul24_use_a(),
                                                             ctx.num_prefill_requests));
    if (custom_prefill)
        return miss;
    uint32_t count = 0u;
    if (is_plamo) {
        count = weight.rows == 3072u && weight.cols == 2560u ? 3u :
                weight.rows == 10240u && weight.cols == 2560u ? 2u : 1u;
        if (count > end - first) return miss;
    } else {
        while (count < 3u && first + count < end) {
            const auto& b = program.dispatches[first + count];
            if (b.kernel_id != KernelId::LINEAR_EXL3_INT8 || b.input_count != 1u ||
                b.output_count != 1u || b.input_slots[0] != binding.input_slots[0] ||
                b.weight_index >= ctx.weight_count) break;
            const auto& w = ctx.weights[b.weight_index];
            if (w.plamo_exl3_linear || w.encoding != 4u || w.device_ptr != nullptr ||
                w.exl3_ka != weight.exl3_ka || w.cols != weight.cols || w.k_padded != w.cols || w.experts != 1u ||
                w.rows == 0u || w.rows % 128u != 0u || w.output_col_total != 0u || w.output_col_offset != 0u ||
                w.preshuffled || w.exl3_half || !w.codes || !w.exl3_suh || !w.exl3_svh) break;
            ++count;
        }
        if (count < 2u) return miss;
    }
    ::shelf::kernel::PlamoExl3ProjectionGroup group{};
    group.input = static_cast<const ::shelf::bf16_t*>(input.value().ptr);
    group.rows = rows;
    group.k = weight.cols;
    group.input_stride = input.value().row_stride;
    group.scale_stride_bytes = (group.k / 128u * 4u + 15u) & ~15u;
    group.count = count;
    uint64_t xoffset = 0u, yoffset = 0u;
    auto align = [](uint64_t offset) { return (offset + 255u) & ~uint64_t{255u}; };
    for (uint32_t j = 0u; j < count; ++j) {
        const auto& b = program.dispatches[first + j];
        if (b.kernel_id != KernelId::LINEAR_EXL3_INT8 || b.input_count != 1u ||
            b.output_count != 1u || b.input_slots[0] != binding.input_slots[0] ||
            b.weight_index >= ctx.weight_count) return miss;
        const auto& w = ctx.weights[b.weight_index];
        const uint32_t expected_n = count == 3u && j != 0u ? 512u : weight.rows;
        if (w.plamo_exl3_linear != is_plamo || w.encoding != 4u || w.device_ptr != nullptr ||
            w.exl3_ka != weight.exl3_ka || w.cols != group.k || (is_plamo && w.rows != expected_n) ||
            w.output_col_total != 0u || w.preshuffled || w.exl3_half ||
            w.codes == nullptr || w.exl3_suh == nullptr || w.exl3_svh == nullptr)
            return miss;
        auto dst = resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
        if (!dst.ok()) return dst.status();
        if (dst.value().dtype != ValueDType::BF16 || dst.value().row_stride % 4u != 0u ||
            dst.value().feature_count != w.rows || rows_for(dst.value(), ctx) != rows ||
            input.value().feature_count != group.k)
            return miss;
        if (!is_plamo) {
            const auto layout = ::shelf::kernel::exl3_detail::mul24::make_workspace_layout(
                rows, w.rows, w.cols, ctx.exl3_xrot_bytes, ctx.exl3_accum_bytes, ctx.exl3_dense_bytes);
            if (layout.chunk_rows != rows) return miss;
            const uintptr_t begin = reinterpret_cast<uintptr_t>(dst.value().ptr);
            const uintptr_t end_out = begin + (static_cast<uint64_t>(rows - 1u) * dst.value().row_stride + w.rows) * 2u;
            const uintptr_t in_begin = reinterpret_cast<uintptr_t>(input.value().ptr);
            const uintptr_t in_end = in_begin + (static_cast<uint64_t>(rows - 1u) * input.value().row_stride + group.k) * 2u;
            if (begin < in_end && in_begin < end_out) return miss;
            for (uint32_t i = 0; i < j; ++i) {
                const auto& prev = group.p[i];
                const uintptr_t prev_begin = reinterpret_cast<uintptr_t>(prev.out);
                const uintptr_t prev_end = prev_begin + (static_cast<uint64_t>(rows - 1u) * prev.out_stride + prev.n) * 2u;
                if (begin < prev_end && prev_begin < end_out) return miss;
            }
        }
        auto& p = group.p[j];
        p.n = w.rows;
        p.out_stride = dst.value().row_stride;
        p.out = static_cast<::shelf::bf16_t*>(dst.value().ptr);
        p.trellis = static_cast<const uint16_t*>(w.codes);
        p.suh = static_cast<const uint16_t*>(w.exl3_suh);
        p.svh = static_cast<const uint16_t*>(w.exl3_svh);
        const uint32_t scratch_rows = static_cast<uint32_t>(std::min<uint64_t>(
            ctx.exl3_accum_bytes / (static_cast<uint64_t>(p.n) * 4u), UINT32_MAX));
        const uint32_t split_rows =
            ctx.splitk_tuning_rows != 0u ? ctx.splitk_tuning_rows : rows;
        p.splits = ::shelf::kernel::exl3_mul24_gemm_splits(split_rows, p.n, group.k, scratch_rows, w.exl3_ka);
        if (w.plamo_exl3_split4 && ctx.num_prefill_requests == 0u && ctx.splitk_tuning_rows == 0u &&
            rows > 16u && rows <= 64u && scratch_rows >= 4u * rows) p.splits = 4u;
        xoffset = align(xoffset);
        p.codes = reinterpret_cast<int8_t*>(static_cast<uint8_t*>(ctx.exl3_xrot) + xoffset);
        xoffset += static_cast<uint64_t>(rows) * group.k;
        xoffset = align(xoffset);
        p.scales = reinterpret_cast<float*>(static_cast<uint8_t*>(ctx.exl3_xrot) + xoffset);
        xoffset += static_cast<uint64_t>(rows) * group.scale_stride_bytes;
        yoffset = align(yoffset);
        p.yprime = reinterpret_cast<float*>(static_cast<uint8_t*>(ctx.exl3_accum) + yoffset);
        yoffset += static_cast<uint64_t>(rows) * p.n * p.splits * 4u;
        if (xoffset > ctx.exl3_xrot_bytes || yoffset > ctx.exl3_accum_bytes) return miss;
    }
    uint32_t launch_count = 0u;
    hipError_t err = hipSuccess;
    if (producer != nullptr && producer->kernel_id == KernelId::SWIGLU) {
        const auto* swiglu = producer;
        if (!is_plamo || rows != 1u || count != 1u || weight.cols != 10240u ||
            weight.rows != 2560u || swiglu->input_count != 2u || swiglu->output_count != 1u ||
            swiglu->output_slots[0] != binding.input_slots[0]) return miss;
        auto gate = resolve_host_value(program, ::shelf::runtime::ValueId{swiglu->input_slots[0]}, ctx);
        auto up = resolve_host_value(program, ::shelf::runtime::ValueId{swiglu->input_slots[1]}, ctx);
        if (!gate.ok()) return gate.status();
        if (!up.ok()) return up.status();
        if (gate.value().dtype != ValueDType::BF16 || up.value().dtype != ValueDType::BF16 ||
            gate.value().feature_count != group.k || up.value().feature_count != group.k ||
            rows_for(gate.value(), ctx) != 1u || rows_for(up.value(), ctx) != 1u) return miss;
        const uintptr_t out = reinterpret_cast<uintptr_t>(input.value().ptr);
        const auto safe = [&](const void* p) {
            const uintptr_t x = reinterpret_cast<uintptr_t>(p);
            return x == out || x + group.k * 2u <= out || out + group.k * 2u <= x;
        };
        if (!safe(gate.value().ptr) || !safe(up.value().ptr)) return miss;
        err = ::shelf::kernel::launch_plamo_exl3_swiglu_rotate(
            group, static_cast<const ::shelf::bf16_t*>(gate.value().ptr),
            static_cast<const ::shelf::bf16_t*>(up.value().ptr),
            static_cast<::shelf::bf16_t*>(input.value().ptr), swiglu->flags != 0u, stream);
        if (err != hipSuccess)
            return Status::hip_error("plamo swiglu rotate", hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (producer != nullptr && producer->kernel_id == KernelId::RMS_NORM) {
        if (!is_plamo || rows != 1u || group.k != 2560u ||
            producer->output_count != 1u || producer->output_slots[0] != binding.input_slots[0] ||
            producer->flags != static_cast<uint32_t>(::shelf::runtime::RmsNormWeightMode::OFFSET_BF16)) return miss;
        PhysicalRmsNormLaunch norm{};
        auto resolved = resolve_rmsnorm_physical(program, *producer, ctx, norm);
        if (!resolved.ok()) return resolved.status();
        if (resolved.value() != PhysicalResolveStatus::Applicable || norm.rows != 1u ||
            norm.features != group.k || norm.group_size != group.k ||
            norm.weight_layout != ::shelf::kernel::RmsNormWeightLayout::PER_FEATURE ||
            norm.input_dtype != ::shelf::kernel::RmsNormDataType::BF16 ||
            norm.output_dtype != ::shelf::kernel::RmsNormDataType::BF16 ||
            (norm.input + group.k * 2u > norm.output && norm.output + group.k * 2u > norm.input))
            return miss;
        err = ::shelf::kernel::launch_plamo_exl3_norm_rotate(
            group, reinterpret_cast<const ::shelf::bf16_t*>(norm.input),
            reinterpret_cast<const ::shelf::bf16_t*>(norm.weight),
            reinterpret_cast<::shelf::bf16_t*>(norm.output), norm.eps, norm.weight_offset, stream);
        if (err != hipSuccess)
            return Status::hip_error("plamo norm rotate", hipGetErrorString(err), __FILE__, __LINE__);
    }
    if (producer != nullptr && producer->kernel_id == KernelId::RESIDUAL_ADD) {
        if (is_plamo || first < 2u || rows > 16u || group.k != 5120u) return miss;
        auto fused = deferred ? try_launch_boundary_i9(program, first - 2u, end, ctx, stream, group, *deferred)
                              : try_launch_boundary_i8(program, first - 2u, end, ctx, stream, group);
        if (!fused.ok()) return fused.status();
        if (fused.value().kind != OptimizedLaunchKind::Launched) return miss;
    }
    if (use_prefill) {
        ::shelf::kernel::PlamoExl3PrefillGroup prefill{};
        prefill.projection = group;
        prefill.cached = prefill_mode == 3u || (prefill_mode == 1u && weight.exl3_ka == 5u);
        prefill.rowscaled = rows >= 256u;
        static const float flag_share = [] {
            const char* env = std::getenv("SHELF_EXL3_INT8_FLAG");
            return env ? std::strtof(env, nullptr) : 0.5f;
        }();
        prefill.flag_share = flag_share;
        for (uint32_t j = 0u; j < count; ++j) {
            prefill.rowscale[j] = group.p[j].yprime;
            prefill.row_flags[j] = reinterpret_cast<int32_t*>(group.p[j].yprime + rows);
        }
        if (prefill.cached) {
            if (ctx.exl3_dense == nullptr) return miss;
            uint64_t offset = 0u;
            for (uint32_t j = 0u; j < count; ++j) {
                offset = align(offset);
                prefill.packed[j] = static_cast<int8_t*>(ctx.exl3_dense) + offset;
                offset += static_cast<uint64_t>(group.p[j].n) * group.k;
            }
            if (offset > ctx.exl3_dense_bytes) return miss;
        }
        auto stage = [&](uint32_t id, auto launch) {
            const uint32_t timing = semantic_timing_enabled() ? semantic_timing_start(stream) : 0xffffffffu;
            const hipError_t result = launch();
            if (semantic_timing_enabled()) semantic_timing_finish(
                stream, timing, id, static_cast<uint32_t>(SemPath::Optimized), 0);
            return result;
        };
        err = stage(0xffff0010u, [&] {
            return ::shelf::kernel::launch_plamo_exl3_projection_rotate(group, stream);
        });
        launch_count = 1u;
        if (err == hipSuccess && prefill.rowscaled) {
            err = stage(0xffff0001u, [&] {
                return ::shelf::kernel::launch_plamo_exl3_prefill_rowscale(prefill, stream);
            });
            launch_count += 2u;
        }
        if (err == hipSuccess && prefill.cached) {
            err = stage(0xffff0011u, [&] {
                return ::shelf::kernel::launch_plamo_exl3_prefill_recon(prefill, weight.exl3_ka, stream);
            });
            ++launch_count;
        }
        if (err == hipSuccess) {
            err = stage(0xffff0012u, [&] {
                return ::shelf::kernel::launch_plamo_exl3_prefill_gemm(prefill, weight.exl3_ka, stream);
            });
            ++launch_count;
        }
    } else {
        err = ::shelf::kernel::launch_plamo_exl3_projection(group, weight.exl3_ka, stream, &launch_count,
                                                            is_plamo, producer != nullptr);
        if (producer != nullptr) ++launch_count;
    }
    if (err != hipSuccess)
        return Status::hip_error(is_plamo ? "plamo exl3 projection" : "exl3 grouped projection",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    record_physical_launch(launch_count);
    for (uint32_t j = 0u; j < count; ++j) record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, count};
}

Result<KernelMode> read_kernel_mode() {
    const char* e = std::getenv("SHELF_QWEN35_KERNEL_MODE");
    if (e == nullptr || e[0] == '\0')
        return KernelMode::Auto;
    if (std::strcmp(e, "auto") == 0)
        return KernelMode::Auto;
    if (std::strcmp(e, "correctness") == 0)
        return KernelMode::Correctness;
    return Status::invalid_argument("invalid SHELF_QWEN35_KERNEL_MODE", __FILE__, __LINE__);
}

Result<OptimizedLaunchResult> try_launch_boundary_i1(
    const ::shelf::runtime::Program&, size_t, size_t, HostExecutionContext&, hipStream_t);

Result<OptimizedLaunchResult> try_fixedcost_lucebox(
    const ::shelf::runtime::Program&, size_t, size_t, HostExecutionContext&, hipStream_t);

Result<OptimizedLaunchResult> try_fixedcost_lucebox_qk(
    const ::shelf::runtime::Program&, size_t, size_t, HostExecutionContext&, hipStream_t);

Result<OptimizedLaunchResult> try_launch_attn_q(
    const ::shelf::runtime::Program&, size_t, size_t, HostExecutionContext&, hipStream_t);

Result<OptimizedLaunchResult> try_launch_attn_kv(
    const ::shelf::runtime::Program&, size_t, size_t, HostExecutionContext&, hipStream_t);

Result<OptimizedLaunchResult> try_launch_fp8_swiglu_quantize(
    const ::shelf::runtime::Program& program, size_t i, size_t end,
    HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_FP8_SWIGLU") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
        std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count ||
        i + 2u >= end || ctx.workspace == nullptr || ctx.weights == nullptr) return miss;
    const auto& swiglu = program.dispatches[i];
    const auto& quant = program.dispatches[i + 1u];
    const auto& linear = program.dispatches[i + 2u];
    if (swiglu.kernel_id != KernelId::SWIGLU ||
        quant.kernel_id != KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED ||
        linear.kernel_id != KernelId::LINEAR_FP8_FINEGRAINED) return miss;
    if (swiglu.input_count != 2u || swiglu.output_count != 1u ||
        quant.input_count != 1u || linear.input_count != 0u ||
        linear.output_count != 1u ||
        swiglu.output_slots[0] != quant.input_slots[0] ||
        quant.workspace_slot == ::shelf::runtime::kNoWorkspace ||
        quant.workspace_slot != linear.workspace_slot) return miss;
    if (linear.weight_index >= ctx.weight_count) return miss;
    const auto& w = ctx.weights[linear.weight_index];
    if (w.encoding != 1u) return miss;
    const auto* out_binding =
        program.find_value(::shelf::runtime::ValueId{swiglu.output_slots[0]});
    if (out_binding == nullptr ||
        (out_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         out_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW))
        return miss;
    for (size_t j = i + 2u; j < program.dispatches.size(); ++j)
        for (uint32_t k = 0; k < program.dispatches[j].input_count; ++k)
            if (program.dispatches[j].input_slots[k] == swiglu.output_slots[0]) return miss;
    PhysicalElementwiseLaunch elem{};
    auto resolved_elem = resolve_elementwise_physical(program, swiglu, ctx, elem);
    if (!resolved_elem.ok()) return resolved_elem.status();
    if (resolved_elem.value() != PhysicalResolveStatus::Applicable ||
        elem.profile != ElementwiseProfile::SwigluBf16 ||
        elem.rows == 0u || elem.features == 0u || !elem.in0 || !elem.in1 ||
        elem.in0_stride < elem.features || elem.in1_stride < elem.features)
        return miss;
    PhysicalActivationQuantizeLaunch q{};
    auto resolved_quant = resolve_activation_quantize_physical(program, quant, ctx, q);
    if (!resolved_quant.ok()) return resolved_quant.status();
    if (resolved_quant.value() != PhysicalResolveStatus::Applicable ||
        q.k != elem.features || q.rows != elem.rows || q.k_padded < q.k ||
        w.cols != q.k || w.k_padded != q.k_padded ||
        q.code_row_stride == 0u || q.scale_row_stride == 0u ||
        q.codes == 0u || q.scales == 0u) return miss;
    ::shelf::execution::Fp8SwigluArgs args{};
    args.gate = reinterpret_cast<const ::shelf::bf16_t*>(elem.in0);
    args.up = reinterpret_cast<const ::shelf::bf16_t*>(elem.in1);
    args.codes = reinterpret_cast<uint8_t*>(q.codes);
    args.scales = reinterpret_cast<float*>(q.scales);
    args.rows = elem.rows;
    args.k = q.k;
    args.k_padded = q.k_padded;
    args.gate_stride = elem.in0_stride;
    args.up_stride = elem.in1_stride;
    args.code_stride = q.code_row_stride;
    args.scale_stride = q.scale_row_stride;
    args.bf16_intermediate = elem.aux != 0u;
    const hipError_t herr = ::shelf::execution::launch_fp8_swiglu_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fp8 swiglu quantize launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 2u};
}

Result<OptimizedLaunchResult> try_launch_fp8_norm_quantize(
    const ::shelf::runtime::Program& program, size_t i, size_t end,
    HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_FP8_BOUNDARY") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
        std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count ||
        i + 2u >= end || ctx.workspace == nullptr || ctx.weights == nullptr) return miss;
    const auto& norm = program.dispatches[i];
    const auto& quant = program.dispatches[i + 1u];
    const auto& linear = program.dispatches[i + 2u];
    if (norm.kernel_id != KernelId::RMS_NORM ||
        quant.kernel_id != KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED ||
        linear.kernel_id != KernelId::LINEAR_FP8_FINEGRAINED) return miss;
    if (norm.input_count != 1u || norm.output_count != 1u ||
        quant.input_count != 1u || linear.input_count != 0u ||
        linear.output_count != 1u ||
        norm.output_slots[0] != quant.input_slots[0] ||
        quant.workspace_slot == ::shelf::runtime::kNoWorkspace ||
        quant.workspace_slot != linear.workspace_slot) return miss;
    if (linear.weight_index >= ctx.weight_count) return miss;
    const auto& w = ctx.weights[linear.weight_index];
    if (w.encoding != 1u || w.cols != 5120u || w.k_padded != 5120u) return miss;
    const auto* norm_binding =
        program.find_value(::shelf::runtime::ValueId{norm.output_slots[0]});
    if (norm_binding == nullptr ||
        (norm_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         norm_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW))
        return miss;
    for (size_t j = i + 2u; j < program.dispatches.size(); ++j)
        for (uint32_t k = 0; k < program.dispatches[j].input_count; ++k)
            if (program.dispatches[j].input_slots[k] == norm.output_slots[0]) return miss;
    PhysicalRmsNormLaunch n{};
    auto resolved_norm = resolve_rmsnorm_physical(program, norm, ctx, n);
    if (!resolved_norm.ok()) return resolved_norm.status();
    if (resolved_norm.value() != PhysicalResolveStatus::Applicable ||
        n.variant != PhysicalRmsNormVariant::Bf16PfOnePlus ||
        n.features != 5120u || n.group_size != 5120u || n.rows == 0u || !n.weight)
        return miss;
    PhysicalActivationQuantizeLaunch q{};
    auto resolved_quant = resolve_activation_quantize_physical(program, quant, ctx, q);
    if (!resolved_quant.ok()) return resolved_quant.status();
    if (resolved_quant.value() != PhysicalResolveStatus::Applicable ||
        q.k != 5120u || q.k_padded != 5120u || q.rows != n.rows ||
        q.code_row_stride == 0u || q.scale_row_stride == 0u ||
        q.codes == 0u || q.scales == 0u) return miss;
    auto input = resolve_host_value(program, ::shelf::runtime::ValueId{norm.input_slots[0]}, ctx);
    if (!input.ok()) return input.status();
    if (input.value().dtype != ValueDType::BF16 ||
        input.value().feature_count != 5120u || input.value().ptr == nullptr)
        return miss;
    ::shelf::execution::Fp8NormArgs args{};
    args.input = static_cast<const ::shelf::bf16_t*>(input.value().ptr);
    args.weight = reinterpret_cast<const ::shelf::bf16_t*>(n.weight);
    args.codes = reinterpret_cast<uint8_t*>(q.codes);
    args.scales = reinterpret_cast<float*>(q.scales);
    args.rows = q.rows;
    args.k = q.k;
    args.k_padded = q.k_padded;
    args.input_stride = input.value().row_stride;
    args.code_stride = q.code_row_stride;
    args.scale_stride = q.scale_row_stride;
    args.eps = n.eps;
    const hipError_t herr = ::shelf::execution::launch_fp8_norm_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fp8 norm quantize launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 2u};
}

Result<OptimizedLaunchResult> try_launch_fp8_residual_norm_quantize(
    const ::shelf::runtime::Program& program, size_t i, size_t end,
    HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_FP8_BOUNDARY") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
        std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count ||
        i + 3u >= end || ctx.workspace == nullptr || ctx.weights == nullptr) return miss;
    const auto& add = program.dispatches[i];
    const auto& norm = program.dispatches[i + 1u];
    const auto& quant = program.dispatches[i + 2u];
    const auto& linear = program.dispatches[i + 3u];
    if (add.kernel_id != KernelId::RESIDUAL_ADD || norm.kernel_id != KernelId::RMS_NORM ||
        quant.kernel_id != KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED ||
        linear.kernel_id != KernelId::LINEAR_FP8_FINEGRAINED) return miss;
    if (add.input_count != 2u || add.output_count != 1u ||
        norm.input_count != 1u || norm.output_count != 1u ||
        quant.input_count != 1u || linear.input_count != 0u ||
        linear.output_count != 1u ||
        add.output_slots[0] != norm.input_slots[0] ||
        norm.output_slots[0] != quant.input_slots[0] ||
        quant.workspace_slot == ::shelf::runtime::kNoWorkspace ||
        quant.workspace_slot != linear.workspace_slot) return miss;
    if (linear.weight_index >= ctx.weight_count) return miss;
    const auto& w = ctx.weights[linear.weight_index];
    if (w.encoding != 1u || w.cols != 5120u || w.k_padded != 5120u) return miss;
    const auto* norm_binding =
        program.find_value(::shelf::runtime::ValueId{norm.output_slots[0]});
    if (norm_binding == nullptr ||
        (norm_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         norm_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW))
        return miss;
    for (size_t j = i + 3u; j < program.dispatches.size(); ++j)
        for (uint32_t k = 0; k < program.dispatches[j].input_count; ++k)
            if (program.dispatches[j].input_slots[k] == norm.output_slots[0]) return miss;
    PhysicalRmsNormLaunch n{};
    auto resolved_norm = resolve_rmsnorm_physical(program, norm, ctx, n);
    if (!resolved_norm.ok()) return resolved_norm.status();
    if (resolved_norm.value() != PhysicalResolveStatus::Applicable ||
        n.variant != PhysicalRmsNormVariant::Bf16PfOnePlus ||
        n.features != 5120u || n.group_size != 5120u || n.rows == 0u || !n.weight)
        return miss;
    PhysicalActivationQuantizeLaunch q{};
    auto resolved_quant = resolve_activation_quantize_physical(program, quant, ctx, q);
    if (!resolved_quant.ok()) return resolved_quant.status();
    if (resolved_quant.value() != PhysicalResolveStatus::Applicable ||
        q.k != 5120u || q.k_padded != 5120u || q.rows != n.rows ||
        q.code_row_stride == 0u || q.scale_row_stride == 0u ||
        q.codes == 0u || q.scales == 0u) return miss;
    auto a = resolve_host_value(program, ::shelf::runtime::ValueId{add.input_slots[0]}, ctx);
    auto b = resolve_host_value(program, ::shelf::runtime::ValueId{add.input_slots[1]}, ctx);
    auto r = resolve_host_value(program, ::shelf::runtime::ValueId{add.output_slots[0]}, ctx);
    if (!a.ok()) return a.status();
    if (!b.ok()) return b.status();
    if (!r.ok()) return r.status();
    if (a.value().dtype != ValueDType::BF16 || b.value().dtype != ValueDType::BF16 ||
        r.value().dtype != ValueDType::BF16 ||
        a.value().feature_count != 5120u || b.value().feature_count != 5120u ||
        r.value().feature_count != 5120u || a.value().ptr == nullptr ||
        b.value().ptr == nullptr || r.value().ptr == nullptr ||
        rows_for(r.value(), ctx) != q.rows) return miss;
    ::shelf::execution::Fp8ResidualNormArgs args{};
    args.a = static_cast<const ::shelf::bf16_t*>(a.value().ptr);
    args.b = static_cast<const ::shelf::bf16_t*>(b.value().ptr);
    args.residual = static_cast<::shelf::bf16_t*>(r.value().ptr);
    args.weight = reinterpret_cast<const ::shelf::bf16_t*>(n.weight);
    args.codes = reinterpret_cast<uint8_t*>(q.codes);
    args.scales = reinterpret_cast<float*>(q.scales);
    args.rows = q.rows;
    args.k = q.k;
    args.k_padded = q.k_padded;
    args.a_stride = a.value().row_stride;
    args.b_stride = b.value().row_stride;
    args.residual_stride = r.value().row_stride;
    args.code_stride = q.code_row_stride;
    args.scale_stride = q.scale_row_stride;
    args.eps = n.eps;
    const hipError_t herr =
        ::shelf::execution::launch_fp8_residual_norm_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fp8 residual norm quantize launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3u};
}

bool mxfp4_producer_enabled() {
    static const bool enabled = std::getenv("SHELF_DISABLE_MXFP4_PRODUCER") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
        std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
    return enabled;
}

bool mxfp4_quant_consumer_matches(const ::shelf::runtime::Program& program, size_t qi,
                                  size_t li, const HostExecutionContext& ctx,
                                  PhysicalActivationQuantizeLaunch& q) {
    const auto& quant = program.dispatches[qi];
    const auto& linear = program.dispatches[li];
    if (quant.kernel_id != KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF ||
        linear.kernel_id != KernelId::LINEAR_MXFP4_W4A8)
        return false;
    if (quant.input_count != 1u || quant.workspace_slot == ::shelf::runtime::kNoWorkspace ||
        quant.workspace_slot != linear.workspace_slot || linear.input_count != 0u ||
        linear.output_count != 1u)
        return false;
    if (linear.weight_index >= ctx.weight_count) return false;
    const auto& w = ctx.weights[linear.weight_index];
    if (w.encoding != static_cast<uint8_t>(::shelf::weights::MatrixEncoding::Mxfp4))
        return false;
    auto resolved = resolve_activation_quantize_physical(program, quant, ctx, q);
    if (!resolved.ok() || resolved.value() != PhysicalResolveStatus::Applicable)
        return false;
    if (q.row_index != 0u || q.codes == 0u || q.scales == 0u || q.rows == 0u ||
        q.k == 0u || q.k_padded < q.k || q.code_row_stride < q.k_padded ||
        q.scale_row_stride == 0u)
        return false;
    {
        const uint32_t groups = (q.k_padded + 127u) / 128u;
        if (q.scale_row_stride < groups * static_cast<uint32_t>(sizeof(float)))
            return false;
    }
    return w.cols == q.k && w.k_padded == q.k_padded;
}

bool mxfp4_value_is_exclusive(const ::shelf::runtime::Program& program, uint32_t slot, size_t from,
                              size_t until) {
    for (size_t j = from; j < until; ++j)
        for (uint32_t n = 0; n < program.dispatches[j].input_count; ++n)
            if (program.dispatches[j].input_slots[n] == slot) return false;
    return true;
}

Result<OptimizedLaunchResult> try_launch_mxfp4_swiglu_quantize(
    const ::shelf::runtime::Program& program, size_t i, size_t end, HostExecutionContext& ctx,
    hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    if (!mxfp4_producer_enabled() || ctx.value_trace || ctx.layer_override_count ||
        i + 2u >= end || ctx.workspace == nullptr || ctx.weights == nullptr) return miss;
    const auto& swiglu = program.dispatches[i];
    const auto& quant = program.dispatches[i + 1u];
    if (swiglu.kernel_id != KernelId::SWIGLU || swiglu.input_count != 2u ||
        swiglu.output_count != 1u)
        return miss;
    PhysicalActivationQuantizeLaunch q{};
    if (!mxfp4_quant_consumer_matches(program, i + 1u, i + 2u, ctx, q)) return miss;
    if (swiglu.output_slots[0] != quant.input_slots[0]) return miss;
    if (!mxfp4_value_is_exclusive(program, swiglu.output_slots[0], i + 2u,
                                  program.dispatches.size()))
        return miss;
    PhysicalElementwiseLaunch elem{};
    auto resolved = resolve_elementwise_physical(program, swiglu, ctx, elem);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable ||
        elem.profile != ElementwiseProfile::SwigluBf16 || elem.rows != q.rows ||
        elem.features != q.k || !elem.in0 || !elem.in1 ||
        elem.in0_stride < elem.features || elem.in1_stride < elem.features)
        return miss;
    ::shelf::execution::Mxfp4SwigluArgs args{};
    args.gate = reinterpret_cast<const ::shelf::bf16_t*>(elem.in0);
    args.up = reinterpret_cast<const ::shelf::bf16_t*>(elem.in1);
    args.codes = reinterpret_cast<uint8_t*>(q.codes);
    args.scales = reinterpret_cast<float*>(q.scales);
    args.rows = q.rows;
    args.k = q.k;
    args.k_padded = q.k_padded;
    args.gate_stride = elem.in0_stride;
    args.up_stride = elem.in1_stride;
    args.code_stride = q.code_row_stride;
    args.scale_stride = q.scale_row_stride;
    args.bf16_intermediate = elem.aux != 0u;
    const hipError_t herr = ::shelf::execution::launch_mxfp4_swiglu_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("mxfp4 swiglu quantize launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 2u};
}

Result<OptimizedLaunchResult> try_launch_mxfp4_norm_quantize(
    const ::shelf::runtime::Program& program, size_t i, size_t end, HostExecutionContext& ctx,
    hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    if (!mxfp4_producer_enabled() || ctx.value_trace || ctx.layer_override_count ||
        i + 2u >= end || ctx.workspace == nullptr || ctx.weights == nullptr) return miss;
    const auto& norm = program.dispatches[i];
    const auto& quant = program.dispatches[i + 1u];
    if (norm.kernel_id != KernelId::RMS_NORM || norm.input_count != 1u ||
        norm.output_count != 1u)
        return miss;
    PhysicalActivationQuantizeLaunch q{};
    if (!mxfp4_quant_consumer_matches(program, i + 1u, i + 2u, ctx, q)) return miss;
    if (norm.output_slots[0] != quant.input_slots[0]) return miss;
    const auto* binding = program.find_value(::shelf::runtime::ValueId{norm.output_slots[0]});
    if (binding == nullptr ||
        (binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW))
        return miss;
    if (!mxfp4_value_is_exclusive(program, norm.output_slots[0], i + 2u,
                                  program.dispatches.size()))
        return miss;
    PhysicalRmsNormLaunch n{};
    auto resolved_norm = resolve_rmsnorm_physical(program, norm, ctx, n);
    if (!resolved_norm.ok()) return resolved_norm.status();
    if (resolved_norm.value() != PhysicalResolveStatus::Applicable ||
        n.variant != PhysicalRmsNormVariant::Bf16PfOnePlus || n.rows != q.rows ||
        n.features != q.k || n.group_size != q.k || !n.weight)
        return miss;
    auto input = resolve_host_value(program, ::shelf::runtime::ValueId{norm.input_slots[0]}, ctx);
    if (!input.ok()) return input.status();
    if (input.value().dtype != ValueDType::BF16 || input.value().feature_count != q.k ||
        input.value().row_stride < q.k || input.value().ptr == nullptr)
        return miss;
    ::shelf::execution::Mxfp4NormArgs args{};
    args.input = static_cast<const ::shelf::bf16_t*>(input.value().ptr);
    args.weight = reinterpret_cast<const ::shelf::bf16_t*>(n.weight);
    args.codes = reinterpret_cast<uint8_t*>(q.codes);
    args.scales = reinterpret_cast<float*>(q.scales);
    args.rows = q.rows;
    args.k = q.k;
    args.k_padded = q.k_padded;
    args.input_stride = input.value().row_stride;
    args.code_stride = q.code_row_stride;
    args.scale_stride = q.scale_row_stride;
    args.eps = n.eps;
    const hipError_t herr = ::shelf::execution::launch_mxfp4_norm_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("mxfp4 norm quantize launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 2u};
}

Result<OptimizedLaunchResult> try_launch_mxfp4_residual_norm_quantize(
    const ::shelf::runtime::Program& program, size_t i, size_t end, HostExecutionContext& ctx,
    hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    if (!mxfp4_producer_enabled() || ctx.value_trace || ctx.layer_override_count ||
        i + 3u >= end || ctx.workspace == nullptr || ctx.weights == nullptr) return miss;
    const auto& add = program.dispatches[i];
    const auto& norm = program.dispatches[i + 1u];
    const auto& quant = program.dispatches[i + 2u];
    if (add.kernel_id != KernelId::RESIDUAL_ADD || add.input_count != 2u ||
        add.output_count != 1u || norm.kernel_id != KernelId::RMS_NORM ||
        norm.input_count != 1u || norm.output_count != 1u)
        return miss;
    PhysicalActivationQuantizeLaunch q{};
    if (!mxfp4_quant_consumer_matches(program, i + 2u, i + 3u, ctx, q)) return miss;
    if (add.output_slots[0] != norm.input_slots[0] ||
        norm.output_slots[0] != quant.input_slots[0])
        return miss;
    const auto* binding = program.find_value(::shelf::runtime::ValueId{norm.output_slots[0]});
    if (binding == nullptr ||
        (binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW))
        return miss;
    if (!mxfp4_value_is_exclusive(program, norm.output_slots[0], i + 3u,
                                  program.dispatches.size()))
        return miss;
    PhysicalRmsNormLaunch n{};
    auto resolved_norm = resolve_rmsnorm_physical(program, norm, ctx, n);
    if (!resolved_norm.ok()) return resolved_norm.status();
    if (resolved_norm.value() != PhysicalResolveStatus::Applicable ||
        n.variant != PhysicalRmsNormVariant::Bf16PfOnePlus || n.rows != q.rows ||
        n.features != q.k || n.group_size != q.k || !n.weight)
        return miss;
    auto a = resolve_host_value(program, ::shelf::runtime::ValueId{add.input_slots[0]}, ctx);
    auto b = resolve_host_value(program, ::shelf::runtime::ValueId{add.input_slots[1]}, ctx);
    auto r = resolve_host_value(program, ::shelf::runtime::ValueId{add.output_slots[0]}, ctx);
    if (!a.ok()) return a.status();
    if (!b.ok()) return b.status();
    if (!r.ok()) return r.status();
    if (a.value().dtype != ValueDType::BF16 || b.value().dtype != ValueDType::BF16 ||
        r.value().dtype != ValueDType::BF16 || a.value().feature_count != q.k ||
        b.value().feature_count != q.k || r.value().feature_count != q.k ||
        a.value().row_stride < q.k || b.value().row_stride < q.k ||
        r.value().row_stride < q.k ||
        a.value().ptr == nullptr || b.value().ptr == nullptr || r.value().ptr == nullptr ||
        rows_for(r.value(), ctx) != q.rows)
        return miss;
    ::shelf::execution::Mxfp4ResidualNormArgs args{};
    args.a = static_cast<const ::shelf::bf16_t*>(a.value().ptr);
    args.b = static_cast<const ::shelf::bf16_t*>(b.value().ptr);
    args.residual = static_cast<::shelf::bf16_t*>(r.value().ptr);
    args.weight = reinterpret_cast<const ::shelf::bf16_t*>(n.weight);
    args.codes = reinterpret_cast<uint8_t*>(q.codes);
    args.scales = reinterpret_cast<float*>(q.scales);
    args.rows = q.rows;
    args.k = q.k;
    args.k_padded = q.k_padded;
    args.a_stride = a.value().row_stride;
    args.b_stride = b.value().row_stride;
    args.residual_stride = r.value().row_stride;
    args.code_stride = q.code_row_stride;
    args.scale_stride = q.scale_row_stride;
    args.eps = n.eps;
    const hipError_t herr =
        ::shelf::execution::launch_mxfp4_residual_norm_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("mxfp4 residual norm quantize launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3u};
}

Result<OptimizedLaunchResult> try_launch_mxfp4_gdn_gated_quantize(
    const ::shelf::runtime::Program& program, size_t i, size_t end, HostExecutionContext& ctx,
    hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    if (!mxfp4_producer_enabled() || ctx.value_trace || ctx.layer_override_count ||
        i + 4u >= end || ctx.workspace == nullptr || ctx.weights == nullptr) return miss;
    const auto& norm = program.dispatches[i];
    const auto& silu = program.dispatches[i + 1u];
    const auto& mul = program.dispatches[i + 2u];
    const auto& quant = program.dispatches[i + 3u];
    if (norm.kernel_id != KernelId::RMS_NORM || silu.kernel_id != KernelId::SILU ||
        mul.kernel_id != KernelId::MUL || norm.output_count != 1u ||
        silu.output_count != 1u || mul.input_count != 2u || mul.output_count != 1u)
        return miss;
    PhysicalActivationQuantizeLaunch q{};
    if (!mxfp4_quant_consumer_matches(program, i + 3u, i + 4u, ctx, q)) return miss;
    if (norm.output_slots[0] != mul.input_slots[0] ||
        silu.output_slots[0] != mul.input_slots[1] ||
        mul.output_slots[0] != quant.input_slots[0]) {
        return miss;
    }
    if (!mxfp4_value_is_exclusive(program, norm.output_slots[0], i + 4u,
                                  program.dispatches.size()) ||
        !mxfp4_value_is_exclusive(program, silu.output_slots[0], i + 4u,
                                  program.dispatches.size()) ||
        !mxfp4_value_is_exclusive(program, mul.output_slots[0], i + 4u,
                                  program.dispatches.size()))
        return miss;
    PhysicalRmsNormLaunch n{};
    auto resolved_norm = resolve_rmsnorm_physical(program, norm, ctx, n);
    if (!resolved_norm.ok()) return resolved_norm.status();
    if (resolved_norm.value() != PhysicalResolveStatus::Applicable ||
        n.variant != PhysicalRmsNormVariant::F32PgDirect || n.rows != q.rows ||
        n.features != q.k || n.group_size != 128u || !n.weight) {
        return miss;
    }
    PhysicalElementwiseLaunch z{};
    auto resolved_silu = resolve_elementwise_physical(program, silu, ctx, z);
    if (!resolved_silu.ok()) return resolved_silu.status();
    if (resolved_silu.value() != PhysicalResolveStatus::Applicable ||
        z.profile != ElementwiseProfile::SiluBf16ToF32 || z.rows != q.rows ||
        z.features != q.k)
        return miss;
    PhysicalElementwiseLaunch m{};
    auto resolved_mul = resolve_elementwise_physical(program, mul, ctx, m);
    if (!resolved_mul.ok()) return resolved_mul.status();
    if (resolved_mul.value() != PhysicalResolveStatus::Applicable ||
        m.profile != ElementwiseProfile::MulF32F32ToBf16 || m.rows != q.rows ||
        m.features != q.k || z.in0_stride < q.k)
        return miss;
    auto rec = resolve_host_value(program, ::shelf::runtime::ValueId{norm.input_slots[0]}, ctx);
    if (!rec.ok()) return rec.status();
    if (rec.value().dtype != ValueDType::F32 || rec.value().feature_count != q.k ||
        rec.value().row_stride < q.k || rec.value().ptr == nullptr)
        return miss;
    ::shelf::execution::Mxfp4GdnGatedArgs args{};
    args.rec = static_cast<const float*>(rec.value().ptr);
    args.z = reinterpret_cast<const ::shelf::bf16_t*>(z.in0);
    args.weight = reinterpret_cast<const ::shelf::bf16_t*>(n.weight);
    args.codes = reinterpret_cast<uint8_t*>(q.codes);
    args.scales = reinterpret_cast<float*>(q.scales);
    args.rows = q.rows;
    args.features = q.k;
    args.group_size = n.group_size;
    args.rec_stride = rec.value().row_stride;
    args.z_stride = z.in0_stride;
    args.code_stride = q.code_row_stride;
    args.scale_stride = q.scale_row_stride;
    args.eps = n.eps;
    const hipError_t herr =
        ::shelf::execution::launch_mxfp4_gdn_gated_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("mxfp4 gdn gated quantize launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 4u};
}

Result<OptimizedLaunchResult> try_launch_fp8_final_gather(
    const ::shelf::runtime::Program& program, size_t i, size_t end,
    HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_FP8_FINAL") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
        std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count ||
        ctx.num_prefill_requests || ctx.num_requests > 1u || ctx.batch_context == nullptr ||
        ctx.actual_rows == 0u || ctx.actual_outputs == 0u || ctx.actual_outputs > 16u ||
        i + 3u >= end || ctx.workspace == nullptr || ctx.weights == nullptr) return miss;
    const auto& gather = program.dispatches[i];
    const auto& norm = program.dispatches[i + 1u];
    const auto& quant = program.dispatches[i + 2u];
    const auto& linear = program.dispatches[i + 3u];
    if (gather.kernel_id != KernelId::OUTPUT_GATHER || norm.kernel_id != KernelId::RMS_NORM ||
        quant.kernel_id != KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED ||
        linear.kernel_id != KernelId::LINEAR_FP8_FINEGRAINED) return miss;
    if (gather.input_count != 1u || gather.output_count != 1u ||
        norm.input_count != 1u || norm.output_count != 1u ||
        quant.input_count != 1u || linear.input_count != 0u ||
        linear.output_count != 1u ||
        gather.output_slots[0] != norm.input_slots[0] ||
        norm.output_slots[0] != quant.input_slots[0] ||
        quant.workspace_slot == ::shelf::runtime::kNoWorkspace ||
        quant.workspace_slot != linear.workspace_slot) return miss;
    if (linear.weight_index >= ctx.weight_count) return miss;
    const auto& w = ctx.weights[linear.weight_index];
    if (w.encoding != 1u || w.cols != 5120u || w.k_padded != 5120u) return miss;
    const auto* gathered_binding =
        program.find_value(::shelf::runtime::ValueId{gather.output_slots[0]});
    const auto* norm_binding =
        program.find_value(::shelf::runtime::ValueId{norm.output_slots[0]});
    if (gathered_binding == nullptr || norm_binding == nullptr ||
        (gathered_binding->storage != ::shelf::runtime::ValueStorage::EXTERNAL_OUTPUT &&
         gathered_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         gathered_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW) ||
        (norm_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         norm_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW))
        return miss;
    for (size_t j = i + 3u; j < program.dispatches.size(); ++j)
        for (uint32_t k = 0; k < program.dispatches[j].input_count; ++k)
            if (program.dispatches[j].input_slots[k] == norm.output_slots[0]) return miss;
    PhysicalRmsNormLaunch n{};
    auto resolved_norm = resolve_rmsnorm_physical(program, norm, ctx, n);
    if (!resolved_norm.ok()) return resolved_norm.status();
    if (resolved_norm.value() != PhysicalResolveStatus::Applicable ||
        n.variant != PhysicalRmsNormVariant::Bf16PfOnePlus ||
        n.features != 5120u || n.group_size != 5120u || !n.weight) return miss;
    PhysicalActivationQuantizeLaunch q{};
    auto resolved_quant = resolve_activation_quantize_physical(program, quant, ctx, q);
    if (!resolved_quant.ok()) return resolved_quant.status();
    if (resolved_quant.value() != PhysicalResolveStatus::Applicable ||
        q.k != 5120u || q.k_padded != 5120u || q.rows != ctx.actual_outputs ||
        q.code_row_stride == 0u || q.scale_row_stride == 0u ||
        q.codes == 0u || q.scales == 0u) return miss;
    auto input = resolve_host_value(program, ::shelf::runtime::ValueId{gather.input_slots[0]}, ctx);
    auto gathered = resolve_host_value(program, ::shelf::runtime::ValueId{gather.output_slots[0]}, ctx);
    if (!input.ok()) return input.status();
    if (!gathered.ok()) return gathered.status();
    if (input.value().dtype != ValueDType::BF16 || gathered.value().dtype != ValueDType::BF16 ||
        input.value().feature_count != 5120u || gathered.value().feature_count != 5120u ||
        input.value().ptr == nullptr || gathered.value().ptr == nullptr ||
        input.value().row_stride < 5120u || gathered.value().row_stride < 5120u ||
        input.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        gathered.value().row_domain != ValueRowDomain::OUTPUT_ROWS) return miss;
    const auto src_start = reinterpret_cast<uintptr_t>(input.value().ptr);
    const auto dst_start = reinterpret_cast<uintptr_t>(gathered.value().ptr);
    const size_t src_size =
        (size_t(ctx.actual_rows - 1u) * input.value().row_stride + 5120u) * 2u;
    const size_t dst_size =
        (size_t(ctx.actual_outputs - 1u) * gathered.value().row_stride + 5120u) * 2u;
    if (src_start < dst_start + dst_size && dst_start < src_start + src_size) return miss;
    ::shelf::execution::Fp8FinalProducer args{};
    args.input = static_cast<const ::shelf::bf16_t*>(input.value().ptr);
    args.weight = reinterpret_cast<const ::shelf::bf16_t*>(n.weight);
    args.gathered = static_cast<::shelf::bf16_t*>(gathered.value().ptr);
    args.codes = reinterpret_cast<uint8_t*>(q.codes);
    args.scales = reinterpret_cast<float*>(q.scales);
    args.batch = ctx.batch_context;
    args.rows = ctx.actual_outputs;
    args.k = q.k;
    args.k_padded = q.k_padded;
    args.input_stride = input.value().row_stride;
    args.gathered_stride = gathered.value().row_stride;
    args.code_stride = q.code_row_stride;
    args.scale_stride = q.scale_row_stride;
    args.eps = n.eps;
    const hipError_t herr =
        ::shelf::execution::launch_fp8_final_gather_norm_quantize(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("fp8 final gather norm quantize launch",
                                 hipGetErrorString(herr), __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3u};
}

Result<OptimizedLaunchResult> try_launch_final_gather(
    const ::shelf::runtime::Program& program, size_t i, size_t end,
    HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_FINAL_RQ") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
        std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr && std::getenv("SHELF_EXL3_INT8_DENSE") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count || ctx.tp_active ||
        ctx.num_prefill_requests || ctx.num_requests > 1u || !ctx.batch_context ||
        ctx.actual_rows == 0u || ctx.actual_rows > 16u || ctx.actual_outputs == 0u ||
        ctx.actual_outputs > 16u || i + 2u >= end || !ctx.weights) return miss;
    const auto& gather = program.dispatches[i];
    const auto& norm = program.dispatches[i + 1u];
    const auto& linear = program.dispatches[i + 2u];
    if (gather.input_count != 1u || gather.output_count != 1u ||
        norm.kernel_id != KernelId::RMS_NORM || norm.input_count != 1u || norm.output_count != 1u ||
        linear.kernel_id != KernelId::LINEAR_EXL3_INT8 || linear.input_count != 1u || linear.output_count != 1u ||
        gather.output_slots[0] != norm.input_slots[0] || norm.output_slots[0] != linear.input_slots[0] ||
        linear.weight_index >= ctx.weight_count) return miss;
    const auto& w = ctx.weights[linear.weight_index];
    if (w.rows != 248320u || w.cols != 5120u || w.k_padded != w.cols || w.exl3_ka != 6u ||
        w.encoding != 4u || w.experts != 1u || w.plamo_exl3_linear || w.device_ptr ||
        !w.codes || !w.exl3_suh || !w.exl3_svh || w.preshuffled || w.exl3_half ||
        w.output_col_total || w.output_col_offset) return miss;
    auto input = resolve_host_value(program, ::shelf::runtime::ValueId{gather.input_slots[0]}, ctx);
    auto gathered = resolve_host_value(program, ::shelf::runtime::ValueId{gather.output_slots[0]}, ctx);
    auto normalized = resolve_host_value(program, ::shelf::runtime::ValueId{norm.output_slots[0]}, ctx);
    auto output = resolve_host_value(program, ::shelf::runtime::ValueId{linear.output_slots[0]}, ctx);
    if (!input.ok() || !gathered.ok() || !normalized.ok() || !output.ok()) return miss;
    if (input.value().dtype != ValueDType::BF16 || gathered.value().dtype != ValueDType::BF16 ||
        normalized.value().dtype != ValueDType::BF16 || input.value().feature_count != 5120u ||
        gathered.value().feature_count != 5120u || normalized.value().feature_count != 5120u ||
        input.value().row_stride < 5120u || (input.value().row_stride % 4u) != 0u ||
        input.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        gathered.value().row_domain != ValueRowDomain::OUTPUT_ROWS ||
        output.value().row_domain != ValueRowDomain::OUTPUT_ROWS || !input.value().ptr) return miss;
    const auto* gathered_binding = program.find_value(::shelf::runtime::ValueId{gather.output_slots[0]});
    const auto* norm_binding = program.find_value(::shelf::runtime::ValueId{norm.output_slots[0]});
    if (!gathered_binding || !norm_binding ||
        (gathered_binding->storage != ::shelf::runtime::ValueStorage::EXTERNAL_OUTPUT &&
         gathered_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         gathered_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW) ||
        (norm_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         norm_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW)) return miss;
    if (!gathered.value().ptr || gathered.value().row_stride < 5120u) return miss;
    const auto src_start = reinterpret_cast<uintptr_t>(input.value().ptr);
    const auto dst_start = reinterpret_cast<uintptr_t>(gathered.value().ptr);
    const size_t src_size = (size_t(ctx.actual_rows - 1u) * input.value().row_stride + 5120u) * 2u;
    const size_t dst_size = (size_t(ctx.actual_outputs - 1u) * gathered.value().row_stride + 5120u) * 2u;
    if (src_start < dst_start + dst_size && dst_start < src_start + src_size) return miss;
    for (size_t j = i + 3u; j < program.dispatches.size(); ++j)
        for (uint32_t k = 0; k < program.dispatches[j].input_count; ++k)
            if (program.dispatches[j].input_slots[k] == norm.output_slots[0]) return miss;
    PhysicalRmsNormLaunch n{};
    auto resolved = resolve_rmsnorm_physical(program, norm, ctx, n);
    if (!resolved.ok()) return resolved.status();
    if (resolved.value() != PhysicalResolveStatus::Applicable ||
        n.variant != PhysicalRmsNormVariant::Bf16PfOnePlus || n.rows != ctx.actual_outputs ||
        n.features != 5120u || n.group_size != 5120u || n.workgroup != 256u || !n.weight) return miss;
    const auto layout = ::shelf::kernel::exl3_detail::mul24::make_workspace_layout(
        ctx.actual_outputs, w.rows, w.cols, ctx.exl3_xrot_bytes, ctx.exl3_accum_bytes, ctx.exl3_dense_bytes);
    if (layout.chunk_rows != ctx.actual_outputs) return miss;
    FinalProducer producer{static_cast<const ::shelf::bf16_t*>(input.value().ptr),
        reinterpret_cast<const ::shelf::bf16_t*>(n.weight),
        static_cast<::shelf::bf16_t*>(gathered.value().ptr), ctx.batch_context,
        input.value().row_stride, gathered.value().row_stride, n.eps};
    auto result = try_launch_linear_exl3_mul24(program, linear, ctx, stream, nullptr, nullptr, nullptr, nullptr, &producer);
    if (!result.ok()) return result;
    if (result.value().kind == OptimizedLaunchKind::Launched)
        return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 3u};
    return miss;
}

Result<OptimizedLaunchResult> try_launch_linear_exl3_with_boundary(
    const ::shelf::runtime::Program& program, size_t index, size_t dispatch_end,
    HostExecutionContext& ctx, hipStream_t stream,
    const ::shelf::kernel::Exl3SwigluProducer* producer,
    const ::shelf::kernel::Exl3GdnProducer* gdn_producer,
    const ::shelf::kernel::Exl3AttnProducer* attn_producer) {
    const auto& b = program.dispatches[index];
    static const bool i9_enabled = std::getenv("SHELF_DISABLE_BOUNDARY_I9") == nullptr &&
        std::getenv("SHELF_DISABLE_BOUNDARY_I8") == nullptr && std::getenv("SHELF_DISABLE_BOUNDARY_I1") == nullptr;
    if (i9_enabled && !ctx.tp_active && !ctx.value_trace && !ctx.layer_override_count &&
        ctx.actual_rows > 0 && ctx.actual_rows <= 16 && index + 4u < dispatch_end &&
        program.dispatches[index + 1u].kernel_id == KernelId::RESIDUAL_ADD &&
        program.dispatches[index + 2u].kernel_id == KernelId::RMS_NORM &&
        program.dispatches[index + 3u].kernel_id == KernelId::LINEAR_EXL3_INT8 &&
        program.dispatches[index + 4u].kernel_id == KernelId::LINEAR_EXL3_INT8) {
        BoundaryDeferred deferred{};
        auto linear = try_launch_linear_exl3_mul24(program, b, ctx, stream, producer, gdn_producer, attn_producer, &deferred);
        if (!linear.ok() || !deferred.yprime) return linear;
        auto fused = try_launch_exl3_projection(program, index + 3u, dispatch_end, ctx, stream,
                          &program.dispatches[index + 1u], &deferred);
        if (!fused.ok()) return fused;
        if (fused.value().kind == OptimizedLaunchKind::Launched)
            return OptimizedLaunchResult{OptimizedLaunchKind::Launched, fused.value().dispatch_count + 3u};
        const auto err = ::shelf::kernel::launch_exl3_mul24_rotate_out(
            deferred.yprime, deferred.svh, deferred.output, ::shelf::kernel::Exl3OutputDType::BF16,
            deferred.rows, deferred.features, deferred.output_stride, deferred.splits, stream);
        if (err != hipSuccess) return Status::hip_error("boundary rotate fallback", hipGetErrorString(err), __FILE__, __LINE__);
        record_physical_launch(1);
        return linear;
    }
    return try_launch_linear_exl3_mul24(program, b, ctx, stream, producer, gdn_producer, attn_producer);
}

Result<OptimizedLaunchResult> try_launch_swiglu_moe_down(
    const ::shelf::runtime::Program& program, size_t i, size_t end,
    HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_SWIGLU_MOE_DOWN") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
        std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count || i + 1u >= end) return miss;
    const auto& swiglu = program.dispatches[i];
    const auto& down = program.dispatches[i + 1u];
    if (swiglu.kernel_id != ::shelf::runtime::KernelId::SWIGLU ||
        down.kernel_id != ::shelf::runtime::KernelId::LINEAR_EXL3_INT8_GROUPED)
        return miss;
    if (swiglu.input_count != 2u || swiglu.output_count != 1u ||
        down.input_count != 1u || down.output_count != 1u ||
        down.input_slots[0] != swiglu.output_slots[0])
        return miss;
    if (!moe_swiglu_fusion_available(program, down, ctx)) return miss;
    return OptimizedLaunchResult{OptimizedLaunchKind::Skipped, 1u};
}

Result<OptimizedLaunchResult> try_launch_moe_combine_sigmoid(
    const ::shelf::runtime::Program& program, size_t i, size_t end,
    HostExecutionContext& ctx, hipStream_t stream) {
    const OptimizedLaunchResult miss{OptimizedLaunchKind::NotApplicable};
    static const bool enabled = std::getenv("SHELF_DISABLE_MOE_COMBINE_SIGMOID") == nullptr &&
        std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
        std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
    if (!enabled || ctx.value_trace || ctx.layer_override_count || i + 1u >= end) return miss;
    const auto& sig = program.dispatches[i];
    const auto& comb = program.dispatches[i + 1u];
    if (sig.kernel_id != ::shelf::runtime::KernelId::SIGMOID ||
        comb.kernel_id != ::shelf::runtime::KernelId::MOE_COMBINE)
        return miss;
    if (sig.input_count != 1u || sig.output_count != 1u || comb.input_count != 3u ||
        comb.output_count != 1u || comb.input_slots[2] != sig.output_slots[0])
        return miss;
    const ::shelf::runtime::ValueBinding* out_binding =
        program.find_value(::shelf::runtime::ValueId{sig.output_slots[0]});
    if (out_binding == nullptr ||
        out_binding->dtype != ::shelf::runtime::ValueDType::F32 ||
        (out_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         out_binding->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW))
        return miss;
    for (size_t j = i + 2u; j < program.dispatches.size(); ++j)
        for (uint32_t k = 0; k < program.dispatches[j].input_count; ++k)
            if (program.dispatches[j].input_slots[k] == sig.output_slots[0]) return miss;
    const auto logits =
        resolve_host_value(program, ::shelf::runtime::ValueId{sig.input_slots[0]}, ctx);
    if (!logits.ok() || logits.value().dtype != ::shelf::runtime::ValueDType::F32 ||
        logits.value().feature_count != 1u || logits.value().row_stride == 0u ||
        logits.value().ptr == nullptr)
        return miss;
    auto result = try_launch_moe_combine_gate(program, comb, ctx, stream, logits.value().ptr,
                                              logits.value().row_stride);
    if (!result.ok()) return result;
    if (result.value().kind == OptimizedLaunchKind::Launched)
        return OptimizedLaunchResult{OptimizedLaunchKind::Launched, 2u};
    return miss;
}

Result<OptimizedLaunchResult> try_launch_optimized(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream,
    size_t dispatch_end) {
    if (dispatch_index >= program.dispatches.size())
        return Status::invalid_state("dispatch_index out of range", __FILE__, __LINE__);
    const auto& b = program.dispatches[dispatch_index];
    switch (b.kernel_id) {
        case KernelId::EMBEDDING_LOOKUP:
            return try_launch_embedding(program, dispatch_index, ctx, stream);
        case KernelId::SAMPLING:
            return try_launch_sampling(program, dispatch_index, ctx, stream);
        case KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED:
        case KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF:
            return try_launch_activation_quantize(program, b, ctx, stream);
        case KernelId::ACTIVATION_QUANTIZE_FP8_PERTOKEN:
            return try_launch_activation_quantize_fp8_pertoken(program, b, ctx, stream);
        case KernelId::LINEAR_FP8_FINEGRAINED:
            return try_launch_linear_fp8(program, b, ctx, stream);
        case KernelId::LINEAR_FP8_PTPC:
            return try_launch_linear_fp8_ptpc(program, b, ctx, stream);
        case KernelId::LINEAR_MXFP4_W4A8:
            return try_launch_linear_mxfp4(program, b, ctx, stream);
        case KernelId::MOE_ROUTER:
            return try_launch_moe_router(program, b, ctx, stream);
        case KernelId::MOE_PLAN_SCAN:
            return try_launch_moe_plan_scan(program, b, ctx, stream);
        case KernelId::MOE_PLAN_SCATTER:
            return try_launch_moe_plan_scatter(program, b, ctx, stream);
        case KernelId::LINEAR_MXFP4_W4A8_GROUPED:
            return try_launch_moe_grouped_linear(program, b, ctx, stream);
        case KernelId::LINEAR_EXL3_INT8_GROUPED:
            return try_launch_moe_grouped_exl3(program, b, ctx, stream);
        case KernelId::MOE_COMBINE:
            return try_launch_moe_combine(program, b, ctx, stream);
        case KernelId::LINEAR_EXL3_INT8: {
            auto projected = try_launch_exl3_projection(program, dispatch_index, dispatch_end, ctx, stream);
            if (!projected.ok() || projected.value().kind == OptimizedLaunchKind::Launched) return projected;
            return try_launch_linear_exl3_with_boundary(program, dispatch_index, dispatch_end, ctx, stream);
        }
        case KernelId::LINEAR_BF16: {
            auto pair = try_launch_linear_bf16_pair(program, dispatch_index, ctx, stream);
            if (!pair.ok() || pair.value().kind == OptimizedLaunchKind::Launched)
                return pair;
            auto up_mix = try_launch_linear_hyper_up_mix(program, dispatch_index, ctx, stream);
            if (!up_mix.ok() || up_mix.value().kind == OptimizedLaunchKind::Launched)
                return up_mix;
            return try_launch_linear_bf16(program, b, ctx, stream);
        }
        case KernelId::LINEAR_INT8_W32:
            return try_launch_linear_int8_w32(program, b, ctx, stream);
        case KernelId::RMS_NORM: {
            auto fp8 = try_launch_fp8_norm_quantize(program, dispatch_index, dispatch_end, ctx, stream);
            if (!fp8.ok() || fp8.value().kind == OptimizedLaunchKind::Launched) return fp8;
            auto gdn_mxfp4 = try_launch_mxfp4_gdn_gated_quantize(
                program, dispatch_index, dispatch_end, ctx, stream);
            if (!gdn_mxfp4.ok() || gdn_mxfp4.value().kind == OptimizedLaunchKind::Launched)
                return gdn_mxfp4;
            auto mxfp4 = try_launch_mxfp4_norm_quantize(
                program, dispatch_index, dispatch_end, ctx, stream);
            if (!mxfp4.ok() || mxfp4.value().kind == OptimizedLaunchKind::Launched)
                return mxfp4;
            static const bool enabled = [] {
                const char* env = std::getenv("SHELF_PLAMO_EXL3_NORM_ROTATE");
                return (env == nullptr || std::strcmp(env, "0") != 0) &&
                    std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
                    std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
            }();
            if (enabled && ctx.value_trace == nullptr && ctx.layer_override_count == 0u &&
                ctx.num_prefill_requests == 0u && ctx.num_verify_requests == 0u &&
                dispatch_index + 1u < dispatch_end &&
                program.dispatches[dispatch_index + 1u].kernel_id == KernelId::LINEAR_EXL3_INT8) {
                auto projected = try_launch_exl3_projection(program, dispatch_index + 1u, dispatch_end,
                                                             ctx, stream, &b);
                if (!projected.ok()) return projected;
                if (projected.value().kind == OptimizedLaunchKind::Launched)
                    return OptimizedLaunchResult{OptimizedLaunchKind::Launched,
                                                 projected.value().dispatch_count + 1u};
            }
            auto attn = try_launch_attn_kv(program, dispatch_index, dispatch_end, ctx, stream);
            if (!attn.ok() || attn.value().kind == OptimizedLaunchKind::Launched) return attn;
            auto fused = try_fixedcost_lucebox(program, dispatch_index, dispatch_end, ctx, stream);
            if (!fused.ok() || fused.value().kind == OptimizedLaunchKind::Launched) return fused;
            return try_launch_rmsnorm(program, b, ctx, stream);
        }
        case KernelId::ROPE:
            return try_launch_rope(program, dispatch_index, ctx, stream);
        case KernelId::OUTPUT_GATHER: {
            auto fp8 = try_launch_fp8_final_gather(program, dispatch_index, dispatch_end, ctx, stream);
            if (!fp8.ok() || fp8.value().kind == OptimizedLaunchKind::Launched) return fp8;
            auto fused = try_launch_final_gather(program, dispatch_index, dispatch_end, ctx, stream);
            if (!fused.ok() || fused.value().kind == OptimizedLaunchKind::Launched) return fused;
            return try_launch_output_gather(program, b, ctx, stream);
        }
        case KernelId::L2_NORMALIZE: {
            auto fused = try_fixedcost_lucebox_qk(program, dispatch_index, dispatch_end, ctx, stream);
            if (!fused.ok() || fused.value().kind == OptimizedLaunchKind::Launched) return fused;
            return try_launch_l2_normalize(program, dispatch_index, ctx, stream);
        }
        case KernelId::SWIGLU: {
            auto fp8 = try_launch_fp8_swiglu_quantize(program, dispatch_index, dispatch_end, ctx, stream);
            if (!fp8.ok() || fp8.value().kind == OptimizedLaunchKind::Launched) return fp8;
            auto mxfp4 = try_launch_mxfp4_swiglu_quantize(
                program, dispatch_index, dispatch_end, ctx, stream);
            if (!mxfp4.ok() || mxfp4.value().kind == OptimizedLaunchKind::Launched)
                return mxfp4;
            auto moe_down = try_launch_swiglu_moe_down(program, dispatch_index,
                                                       dispatch_end, ctx, stream);
            if (!moe_down.ok() || moe_down.value().kind == OptimizedLaunchKind::Launched)
                return moe_down;
            static const bool rq_enabled = std::getenv("SHELF_DISABLE_SWIGLU_RQ") == nullptr &&
                std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
                std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr &&
                std::getenv("SHELF_EXL3_INT8_DENSE") == nullptr;
            if (rq_enabled && ctx.value_trace == nullptr && ctx.layer_override_count == 0u &&
                ::shelf::runtime::is_target_execution_class(program.execution_class) &&
                program.execution_class != ::shelf::runtime::ExecutionClass::PREFILL &&
                ctx.num_prefill_requests == 0u && dispatch_index >= 2u && dispatch_index + 1u < dispatch_end &&
                program.dispatches[dispatch_index - 1u].kernel_id == KernelId::LINEAR_EXL3_INT8 &&
                program.dispatches[dispatch_index - 2u].kernel_id == KernelId::LINEAR_EXL3_INT8 &&
                b.input_count == 2u && b.output_count == 1u) {
                const auto& linear = program.dispatches[dispatch_index + 1u];
                const auto& gate_linear = program.dispatches[dispatch_index - 2u];
                const auto& up_linear = program.dispatches[dispatch_index - 1u];
                bool exclusive = linear.kernel_id == KernelId::LINEAR_EXL3_INT8 && linear.input_count == 1u &&
                    linear.output_count == 1u && linear.input_slots[0] == b.output_slots[0] &&
                    gate_linear.input_count == 1u && up_linear.input_count == 1u &&
                    gate_linear.output_count == 1u && up_linear.output_count == 1u &&
                    gate_linear.output_slots[0] == b.input_slots[0] && up_linear.output_slots[0] == b.input_slots[1] &&
                    gate_linear.input_slots[0] == up_linear.input_slots[0] &&
                    gate_linear.weight_index < ctx.weight_count && up_linear.weight_index < ctx.weight_count &&
                    linear.weight_index < ctx.weight_count && ctx.weights;
                for (size_t j = dispatch_index + 2u; exclusive && j < program.dispatches.size(); ++j)
                    for (uint32_t n = 0; n < program.dispatches[j].input_count; ++n)
                        if (program.dispatches[j].input_slots[n] == b.output_slots[0] ||
                            program.dispatches[j].input_slots[n] == b.input_slots[0] ||
                            program.dispatches[j].input_slots[n] == b.input_slots[1]) exclusive = false;
                if (exclusive) {
                    const auto& w = ctx.weights[linear.weight_index];
                    const auto& gw = ctx.weights[gate_linear.weight_index];
                    const auto& uw = ctx.weights[up_linear.weight_index];
                    auto linear_out = resolve_host_value(program, ::shelf::runtime::ValueId{linear.output_slots[0]}, ctx);
                    auto gate = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
                    PhysicalElementwiseLaunch elem{};
                    auto resolved = resolve_elementwise_physical(program, b, ctx, elem);
                    if (!resolved.ok()) return resolved.status();
                    const auto* binding = program.find_value(::shelf::runtime::ValueId{b.output_slots[0]});
                    const auto layout = ::shelf::kernel::exl3_detail::mul24::make_workspace_layout(
                        elem.rows, w.rows, w.cols, ctx.exl3_xrot_bytes, ctx.exl3_accum_bytes, ctx.exl3_dense_bytes);
                    if (resolved.value() == PhysicalResolveStatus::Applicable && elem.rows >= 1u && elem.rows <= 16u &&
                        gw.rows == 17408u && gw.cols == 5120u && uw.rows == 17408u && uw.cols == 5120u &&
                        gw.experts == 1u && uw.experts == 1u && !gw.plamo_exl3_linear && !uw.plamo_exl3_linear &&
                        w.rows == 5120u && w.cols == 17408u && w.k_padded == w.cols && w.exl3_ka == 4u &&
                        w.encoding == 4u && w.experts == 1u && !w.plamo_exl3_linear && !w.device_ptr &&
                        w.codes && w.exl3_suh && w.exl3_svh && !w.preshuffled && !w.exl3_half &&
                        !w.output_col_total && !w.output_col_offset && binding &&
                        binding->storage == ::shelf::runtime::ValueStorage::WORKSPACE &&
                        linear_out.ok() && gate.ok() && linear_out.value().row_domain == gate.value().row_domain &&
                        elem.features == w.cols && layout.chunk_rows == elem.rows) {
                        ::shelf::kernel::Exl3SwigluProducer producer{
                            reinterpret_cast<const ::shelf::bf16_t*>(elem.in0),
                            reinterpret_cast<const ::shelf::bf16_t*>(elem.in1), elem.in0_stride, elem.in1_stride,
                            elem.aux != 0u};
                        auto result = try_launch_linear_exl3_with_boundary(program, dispatch_index + 1u, dispatch_end, ctx, stream, &producer);
                        if (!result.ok()) return result;
                        if (result.value().kind == OptimizedLaunchKind::Launched)
                            return OptimizedLaunchResult{OptimizedLaunchKind::Launched, result.value().dispatch_count + 1u};
                    }
                }
            }
            static const bool producer_enabled = [] {
                const char* env = std::getenv("SHELF_PLAMO_EXL3_PRODUCER_ROTATE");
                return (env == nullptr || std::strcmp(env, "0") != 0) &&
                    std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
                    std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr;
            }();
            if (producer_enabled && ctx.value_trace == nullptr && ctx.layer_override_count == 0u &&
                ctx.num_prefill_requests == 0u && ctx.num_verify_requests == 0u &&
                dispatch_index + 1u < dispatch_end &&
                program.dispatches[dispatch_index + 1u].kernel_id == KernelId::LINEAR_EXL3_INT8) {
                auto projected = try_launch_exl3_projection(program, dispatch_index + 1u, dispatch_end,
                                                             ctx, stream, &b);
                if (!projected.ok()) return projected;
                if (projected.value().kind == OptimizedLaunchKind::Launched)
                    return OptimizedLaunchResult{OptimizedLaunchKind::Launched,
                                                 projected.value().dispatch_count + 1u};
            }
            return try_launch_elementwise(program, dispatch_index, ctx, stream);
        }
        case KernelId::SPLIT: {
            auto fused = try_launch_attn_q(program, dispatch_index, dispatch_end, ctx, stream);
            if (!fused.ok() || fused.value().kind == OptimizedLaunchKind::Launched) return fused;
            return try_launch_elementwise(program, dispatch_index, ctx, stream);
        }
        case KernelId::SIGMOID: {
            auto combine = try_launch_moe_combine_sigmoid(program, dispatch_index,
                                                          dispatch_end, ctx, stream);
            if (!combine.ok() || combine.value().kind == OptimizedLaunchKind::Launched)
                return combine;
            static const bool rq_enabled = std::getenv("SHELF_DISABLE_ATTN_RQ") == nullptr &&
                std::getenv("SHELF_OP_DUMP") == nullptr && std::getenv("SHELF_OP_SYNC") == nullptr &&
                std::getenv("SHELF_LINEAR_INPUT_DUMP") == nullptr && std::getenv("SHELF_EXL3_INT8_DENSE") == nullptr;
            if (rq_enabled && ctx.value_trace == nullptr && ctx.layer_override_count == 0u &&
                ::shelf::runtime::is_target_execution_class(program.execution_class) &&
                program.execution_class != ::shelf::runtime::ExecutionClass::PREFILL &&
                ctx.num_prefill_requests == 0u && dispatch_index > 0u && dispatch_index + 2u < dispatch_end &&
                program.dispatches[dispatch_index - 1u].kernel_id == KernelId::PAGED_ATTENTION &&
                b.input_count == 1u && b.output_count == 1u) {
                const auto& attn = program.dispatches[dispatch_index - 1u];
                const auto& mul = program.dispatches[dispatch_index + 1u];
                const auto& linear = program.dispatches[dispatch_index + 2u];
                bool exclusive = attn.output_count == 1u && mul.kernel_id == KernelId::MUL &&
                    mul.input_count == 2u && mul.output_count == 1u && mul.input_slots[0] == attn.output_slots[0] &&
                    mul.input_slots[1] == b.output_slots[0] && linear.kernel_id == KernelId::LINEAR_EXL3_INT8 &&
                    linear.input_count == 1u && linear.output_count == 1u && linear.input_slots[0] == mul.output_slots[0] &&
                    linear.weight_index < ctx.weight_count && ctx.weights;
                for (size_t j = dispatch_index + 2u; exclusive && j < program.dispatches.size(); ++j)
                    for (uint32_t n = 0; n < program.dispatches[j].input_count; ++n)
                        if (program.dispatches[j].input_slots[n] == b.output_slots[0] ||
                            (j > dispatch_index + 2u && program.dispatches[j].input_slots[n] == mul.output_slots[0])) exclusive = false;
                if (exclusive) {
                    PhysicalElementwiseLaunch sigmoid{}, product{};
                    auto rs = resolve_elementwise_physical(program, b, ctx, sigmoid);
                    auto rm = resolve_elementwise_physical(program, mul, ctx, product);
                    if (!rs.ok()) return rs.status();
                    if (!rm.ok()) return rm.status();
                    auto linear_out = resolve_host_value(program, ::shelf::runtime::ValueId{linear.output_slots[0]}, ctx);
                    auto gate = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
                    const auto* sb = program.find_value(::shelf::runtime::ValueId{b.output_slots[0]});
                    const auto* mb = program.find_value(::shelf::runtime::ValueId{mul.output_slots[0]});
                    const auto& w = ctx.weights[linear.weight_index];
                    const auto layout = ::shelf::kernel::exl3_detail::mul24::make_workspace_layout(
                        product.rows, w.rows, w.cols, ctx.exl3_xrot_bytes, ctx.exl3_accum_bytes, ctx.exl3_dense_bytes);
                    const auto disjoint = [](uint64_t a, size_t an, uint64_t x, size_t xn) {
                        return a + an <= x || x + xn <= a;
                    };
                    const size_t sig_bytes = ((size_t(sigmoid.rows - 1u) * sigmoid.out0_stride) + sigmoid.features) * 4u;
                    const size_t attn_bytes = ((size_t(product.rows - 1u) * product.in0_stride) + product.features) * 4u;
                    if (rs.value() == PhysicalResolveStatus::Applicable && rm.value() == PhysicalResolveStatus::Applicable &&
                        sigmoid.profile == ElementwiseProfile::SigmoidBf16ToF32 &&
                        product.profile == ElementwiseProfile::MulF32F32ToBf16 &&
                        product.rows >= 1u && product.rows <= 16u && sigmoid.rows == product.rows &&
                        sigmoid.features == 6144u && product.features == 6144u &&
                        disjoint(sigmoid.out0, sig_bytes, product.in0, attn_bytes) &&
                        gate.ok() && linear_out.ok() && gate.value().row_domain == linear_out.value().row_domain &&
                        sb && mb && sb->storage == ::shelf::runtime::ValueStorage::WORKSPACE &&
                        mb->storage == ::shelf::runtime::ValueStorage::WORKSPACE &&
                        w.rows == 5120u && w.cols == 6144u && w.k_padded == w.cols && w.exl3_ka == 4u &&
                        w.encoding == 4u && w.experts == 1u && !w.plamo_exl3_linear && !w.device_ptr &&
                        w.codes && w.exl3_suh && w.exl3_svh && !w.preshuffled && !w.exl3_half &&
                        !w.output_col_total && !w.output_col_offset && layout.chunk_rows == product.rows) {
                        ::shelf::kernel::Exl3AttnProducer producer{reinterpret_cast<const float*>(product.in0),
                            reinterpret_cast<const ::shelf::bf16_t*>(sigmoid.in0), product.in0_stride, sigmoid.in0_stride};
                        auto result = try_launch_linear_exl3_with_boundary(program, dispatch_index + 2u, dispatch_end, ctx, stream, nullptr, nullptr, &producer);
                        if (!result.ok()) return result;
                        if (result.value().kind == OptimizedLaunchKind::Launched)
                            return OptimizedLaunchResult{OptimizedLaunchKind::Launched, result.value().dispatch_count + 2u};
                    }
                }
            }
            return try_launch_elementwise(program, dispatch_index, ctx, stream);
        }
        case KernelId::RESIDUAL_ADD: {
            auto fp8 = try_launch_fp8_residual_norm_quantize(
                program, dispatch_index, dispatch_end, ctx, stream);
            if (!fp8.ok() || fp8.value().kind == OptimizedLaunchKind::Launched) return fp8;
            auto mxfp4 = try_launch_mxfp4_residual_norm_quantize(
                program, dispatch_index, dispatch_end, ctx, stream);
            if (!mxfp4.ok() || mxfp4.value().kind == OptimizedLaunchKind::Launched)
                return mxfp4;
            if (dispatch_index + 2u < dispatch_end &&
                program.dispatches[dispatch_index + 1u].kernel_id == KernelId::RMS_NORM &&
                program.dispatches[dispatch_index + 2u].kernel_id == KernelId::LINEAR_EXL3_INT8) {
                auto projected = try_launch_exl3_projection(program, dispatch_index + 2u, dispatch_end, ctx, stream, &b);
                if (!projected.ok()) return projected;
                if (projected.value().kind == OptimizedLaunchKind::Launched)
                    return OptimizedLaunchResult{OptimizedLaunchKind::Launched, projected.value().dispatch_count + 2u};
            }
            auto fused = try_launch_boundary_i1(program, dispatch_index, dispatch_end, ctx, stream);
            if (!fused.ok() || fused.value().kind == OptimizedLaunchKind::Launched) return fused;
            return try_launch_elementwise(program, dispatch_index, ctx, stream);
        }
        case KernelId::SILU:
        case KernelId::MUL:
        case KernelId::SCALE:
        case KernelId::CONCAT:
            return try_launch_elementwise(program, dispatch_index, ctx, stream);
        case KernelId::KV_APPEND:
            return try_launch_kv_append(program, dispatch_index, ctx, stream);
        case KernelId::PAGED_ATTENTION:
            return try_launch_paged_attention(program, dispatch_index, ctx, stream);
        case KernelId::STATEFUL_CAUSAL_CONV1D:
            return try_launch_gdn_conv1d(program, dispatch_index, ctx, stream);
        case KernelId::GDN_RECURRENCE:
            return try_launch_gdn_recurrence(program, dispatch_index, ctx, stream);
        case KernelId::HYPER_MIX:
            return try_launch_hyper_mix(program, b, ctx, stream);
        case KernelId::HYPER_INJECT:
            return try_launch_hyper_inject(program, b, ctx, stream);
        case KernelId::HYPER_APPLY:
            return try_launch_hyper_apply(program, b, ctx, stream);
        case KernelId::MEAN_STREAMS:
            return try_launch_mean_streams(program, b, ctx, stream);
        case KernelId::PLE_STREAM_DOT:
            return try_launch_ple_stream_dot(program, b, ctx, stream);
        case KernelId::REPEAT_STREAMS:
            return try_launch_repeat_streams(program, b, ctx, stream);
        case KernelId::PLE_GATE:
            return try_launch_ple_gate(program, b, ctx, stream);
        case KernelId::PLE_BROADCAST_MUL:
            return try_launch_ple_broadcast_mul(program, b, ctx, stream);
        case KernelId::PLE_CONV:
            return try_launch_ple_conv(program, b, ctx, stream);
        case KernelId::QSA_KEY_CACHE:
            return try_launch_qsa_key_cache(program, b, ctx, stream);
        case KernelId::QSA_SELECT:
            return try_launch_qsa_select(program, b, ctx, stream);
        case KernelId::QSA_ROPE:
            return try_launch_qsa_rope(program, b, ctx, stream);
        default:
            return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    }
}

}
