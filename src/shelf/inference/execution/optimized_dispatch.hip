#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/quantization/quant_contract.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/optimized/rmsnorm.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/linear/fp8.h>
#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/inference/execution/decode_perf_stats.h>
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
    const hipError_t herr = ::shelf::kernel::launch_rmsnorm(
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
    lin.verify_exact = ctx.role == ::shelf::runtime::ExecutionRole::Verify;
    const auto config = select_bf16_gemm_config(lin);
    if (!config) {
        return PhysicalResolveStatus::NotApplicable;
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
        case ::shelf::kernel::Bf16GemmConfigId::WmmaKPartition:
            out.variant = PhysicalBf16Variant::WmmaKPartition;
            out.grid_x = ::shelf::kernel::bf16_wmma_grid_x(e.rows);
            out.grid_y = ::shelf::kernel::bf16_wmma_grid_y(rows, 1u);
            out.workgroup = ::shelf::kernel::kBf16SplitKWmmaThreads;
            break;
        case ::shelf::kernel::Bf16GemmConfigId::WmmaWide:
            out.variant = PhysicalBf16Variant::WmmaWide;
            out.grid_x = ::shelf::kernel::bf16_wmma_grid_x(e.rows);
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

Result<OptimizedLaunchResult> try_launch_linear_exl3(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.output_count != 1 || b.input_count != 1)
        return Status::invalid_state("exl3 linear: bad io count", __FILE__, __LINE__);
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return Status::invalid_state("exl3 linear: weight_index invalid", __FILE__, __LINE__);
    const ::shelf::runtime::WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 4u)
        return Status::invalid_state("exl3 linear: encoding != exl3", __FILE__, __LINE__);
    if (e.device_ptr != nullptr || e.codes == nullptr || e.exl3_suh == nullptr ||
        e.exl3_svh == nullptr) {
        return Status::invalid_state("exl3 linear: bad weight slot", __FILE__, __LINE__);
    }
    if (e.preshuffled)
        return Status::invalid_state("exl3 linear: trellis must not be preshuffled", __FILE__,
                                     __LINE__);
    if (e.rows == 0 || e.cols == 0 || e.k_padded != e.cols)
        return Status::invalid_state("exl3 linear: weight dims invalid", __FILE__, __LINE__);
    auto in_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    if (in.dtype != ValueDType::BF16) return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (in.feature_count != e.cols)
        return Status::invalid_state("exl3 linear: input features != weight cols", __FILE__,
                                     __LINE__);
    auto out_res = resolve_host_value(program, ::shelf::runtime::ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue dst = out_res.value();
    const uint64_t expected_out_features =
        e.output_col_total != 0u ? static_cast<uint64_t>(e.output_col_total)
                                 : static_cast<uint64_t>(e.rows);
    if (static_cast<uint64_t>(dst.feature_count) != expected_out_features)
        return Status::invalid_state("exl3 linear: output features mismatch", __FILE__,
                                     __LINE__);
    if (e.output_col_total != 0u &&
        static_cast<uint64_t>(e.output_col_offset) + e.rows >
            static_cast<uint64_t>(e.output_col_total)) {
        return Status::invalid_state("exl3 linear: column offset out of range", __FILE__,
                                     __LINE__);
    }
    const bool out_bf16 = dst.dtype == ValueDType::BF16;
    const bool out_f32 = dst.dtype == ValueDType::F32;
    if (!out_bf16 && !out_f32)
        return Status::invalid_state("exl3 linear: output dtype not BF16/F32", __FILE__,
                                     __LINE__);
    const uint32_t rows = rows_for(dst, ctx);
    if (rows == 0) return OptimizedLaunchResult{OptimizedLaunchKind::Launched};

    ::shelf::kernel::Exl3GemmArgs args;
    args.trellis = e.codes;
    args.suh = e.exl3_suh;
    args.svh = e.exl3_svh;
    args.input = in.ptr;
    args.output = static_cast<void*>(
        static_cast<uint8_t*>(dst.ptr) +
        static_cast<uint64_t>(e.output_col_offset) * value_dtype_bytes(dst.dtype));
    args.rows = rows;
    args.out_features = e.rows;
    args.k = e.cols;
    args.input_row_stride = in.row_stride;
    args.output_row_stride = dst.row_stride;
    args.output_dtype = out_bf16 ? 0u : 1u;
    args.ka = e.exl3_ka;
    args.half = e.exl3_half ? 1u : 0u;
    args.splitk_scratch = ctx.scratch;
    args.splitk_floats = ctx.scratch_floats;
    static const uint32_t split_override = [] {
        const char* e = std::getenv("SHELF_EXL3_SPLIT");
        return e != nullptr ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    args.splitk_splits = split_override;
    static const uint32_t groups_override = [] {
        const char* e = std::getenv("SHELF_EXL3_GROUPS");
        return e != nullptr ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    args.groups = groups_override;
    const uint64_t xrot_bytes = static_cast<uint64_t>(args.rows) * args.k * 2u;
    static const bool pre_override = [] {
        const char* e = std::getenv("SHELF_EXL3_PRE");
        return e != nullptr && std::strtoul(e, nullptr, 10) == 0u;
    }();
    const bool use_pre = !pre_override && ctx.exl3_xrot != nullptr && args.rows >= 4u &&
                         xrot_bytes <= ctx.exl3_xrot_bytes;
    hipError_t herr = hipSuccess;
    if (use_pre) {
        herr = ::shelf::kernel::launch_rotate_exl3(args, ctx.exl3_xrot, stream);
        if (herr == hipSuccess) {
            herr = ::shelf::kernel::launch_gemm_exl3_w16a16_pre(args, ctx.exl3_xrot, stream);
        }
    } else {
        herr = ::shelf::kernel::launch_gemm_exl3_w16a16(args, stream);
    }
    if (herr == hipErrorNotSupported)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (herr != hipSuccess)
        return Status::hip_error("exl3 gemm launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);
    record_physical_launch(use_pre ? 2u : 1u);
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
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    const hipError_t herr = ::shelf::kernel::launch_gemm_bf16(
        launch.config, reinterpret_cast<const ::shelf::bf16_t*>(launch.weight),
        reinterpret_cast<const ::shelf::bf16_t*>(launch.input),
        reinterpret_cast<void*>(launch.output), launch.output_dtype,
        launch.rows, launch.out_features, launch.k, launch.input_row_stride,
        launch.output_row_stride, stream);
    if (herr == hipErrorNotSupported)
        return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    if (herr != hipSuccess)
        return Status::hip_error("bf16 gemm launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
    record_physical_launch(1);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
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

Result<OptimizedLaunchResult> try_launch_optimized(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream) {
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
        case KernelId::LINEAR_FP8_FINEGRAINED:
            return try_launch_linear_fp8(program, b, ctx, stream);
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
        case KernelId::MOE_COMBINE:
            return try_launch_moe_combine(program, b, ctx, stream);
        case KernelId::LINEAR_EXL3_W16A16:
            return try_launch_linear_exl3(program, b, ctx, stream);
        case KernelId::LINEAR_BF16:
            return try_launch_linear_bf16(program, b, ctx, stream);
        case KernelId::RMS_NORM:
            return try_launch_rmsnorm(program, b, ctx, stream);
        case KernelId::ROPE:
            return try_launch_rope(program, dispatch_index, ctx, stream);
        case KernelId::OUTPUT_GATHER:
            return try_launch_output_gather(program, b, ctx, stream);
        case KernelId::L2_NORMALIZE:
            return try_launch_l2_normalize(program, dispatch_index, ctx, stream);
        case KernelId::SPLIT:
        case KernelId::SWIGLU:
        case KernelId::SIGMOID:
        case KernelId::RESIDUAL_ADD:
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
        default:
            return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    }
}

}
