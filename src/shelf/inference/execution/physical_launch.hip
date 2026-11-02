#include <shelf/inference/execution/physical_launch.h>
#include <shelf/quantization/quant_contract.h>

#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/program_executor.h>
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

namespace ps::execution {

using ::ps::runtime::DispatchBinding;
using ::ps::runtime::Int8ActivationWorkspaceLayout;
using ::ps::runtime::KernelId;
using ::ps::runtime::Program;
using ::ps::runtime::ValueDType;

namespace {

uint32_t rows_for(const HostResolvedValue& v, const HostExecutionContext& ctx) {
    if (v.row_domain == ::ps::runtime::ValueRowDomain::OUTPUT_ROWS)
        return ctx.actual_outputs;
    return ctx.actual_rows != 0 ? ctx.actual_rows : 1u;
}

}  // namespace

::ps::kernel::RmsNormDataType rmsnorm_kd(ValueDType d) {
    return d == ValueDType::F32 ? ::ps::kernel::RmsNormDataType::F32
                                : ::ps::kernel::RmsNormDataType::BF16;
}

Result<PhysicalResolveStatus> resolve_rmsnorm_physical(
    const Program& program,
    const DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalRmsNormLaunch& out) {
    out = PhysicalRmsNormLaunch{};
    if (b.input_count != 1 || b.output_count != 1)
        return Status::invalid_state("rmsnorm optimized: io count != 1", __FILE__,
                                     __LINE__);
    auto in_res =
        resolve_host_value(program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    auto out_res = resolve_host_value(program,
                                      ::ps::runtime::ValueId{b.output_slots[0]},
                                      ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue dst = out_res.value();
    const uint32_t F = in.feature_count;
    if (F == 0 || in.feature_count != dst.feature_count)
        return Status::invalid_state("rmsnorm optimized: features mismatch",
                                     __FILE__, __LINE__);
    const uint32_t G = (b.group_size != 0 && b.group_size <= F) ? b.group_size : F;
    if (F % G != 0)
        return Status::invalid_state("rmsnorm optimized: F % G != 0", __FILE__,
                                     __LINE__);
    if (in.dtype != ValueDType::BF16 && in.dtype != ValueDType::F32)
        return PhysicalResolveStatus::NotApplicable;
    if (dst.dtype != ValueDType::BF16 && dst.dtype != ValueDType::F32)
        return PhysicalResolveStatus::NotApplicable;
    ::ps::kernel::RmsNormWeightLayout wlayout =
        ::ps::kernel::RmsNormWeightLayout::NONE;
    const void* wptr = nullptr;
    ValueDType wdt = ValueDType::BF16;
    if (b.parameter_index0 != ::ps::runtime::kNoParameter) {
        if (ctx.host_parameters == nullptr ||
            b.parameter_index0 >= ctx.host_parameter_count)
            return PhysicalResolveStatus::NotApplicable;
        const auto slot = ctx.host_parameters[b.parameter_index0];
        if (slot.dtype != ValueDType::BF16)
            return PhysicalResolveStatus::NotApplicable;
        wptr = slot.device_ptr;
        wdt = slot.dtype;
        if (slot.elements == F)
            wlayout = ::ps::kernel::RmsNormWeightLayout::PER_FEATURE;
        else if (slot.elements == G)
            wlayout = ::ps::kernel::RmsNormWeightLayout::PER_GROUP;
        else
            return PhysicalResolveStatus::NotApplicable;
    }
    const uint32_t rows = rows_for(in, ctx);
    if (rows == 0) return PhysicalResolveStatus::Skipped;
    RmsNormSelectorInput si;
    si.input_dtype = in.dtype;
    si.output_dtype = dst.dtype;
    si.weight_dtype = wdt;
    si.weight_layout = wlayout == ::ps::kernel::RmsNormWeightLayout::PER_FEATURE
                           ? RmsNormSelectorWeightLayout::PerFeature
                       : wlayout == ::ps::kernel::RmsNormWeightLayout::PER_GROUP
                           ? RmsNormSelectorWeightLayout::PerGroup
                           : RmsNormSelectorWeightLayout::None;
    si.weight_mode = b.flags;
    si.features = F;
    si.group_size = G;
    si.rows = rows;
    if (select_rmsnorm_implementation(si) != RmsNormImplementation::Optimized)
        return PhysicalResolveStatus::NotApplicable;

    out.input = reinterpret_cast<uint64_t>(in.ptr);
    out.weight = reinterpret_cast<uint64_t>(wptr);
    out.output = reinterpret_cast<uint64_t>(dst.ptr);
    out.rows = rows;
    out.features = F;
    out.group_size = G;
    out.input_row_stride = in.row_stride;
    out.output_row_stride = dst.row_stride;
    out.eps = b.scalar_a;
    out.input_dtype = rmsnorm_kd(in.dtype);
    out.weight_dtype = rmsnorm_kd(wdt);
    out.output_dtype = rmsnorm_kd(dst.dtype);
    out.weight_layout = wlayout;
    out.weight_mode = b.flags == 0u
                          ? ::ps::kernel::RmsNormWeightMode::ONE_PLUS
                          : ::ps::kernel::RmsNormWeightMode::DIRECT;
    const bool bf16_pf_one_plus =
        in.dtype == ValueDType::BF16 && dst.dtype == ValueDType::BF16 &&
        wdt == ValueDType::BF16 &&
        wlayout == ::ps::kernel::RmsNormWeightLayout::PER_FEATURE &&
        out.weight_mode == ::ps::kernel::RmsNormWeightMode::ONE_PLUS;
    const bool f32_pg_direct =
        in.dtype == ValueDType::F32 && dst.dtype == ValueDType::F32 &&
        wdt == ValueDType::BF16 &&
        wlayout == ::ps::kernel::RmsNormWeightLayout::PER_GROUP &&
        out.weight_mode == ::ps::kernel::RmsNormWeightMode::DIRECT;
    const bool bf16_f32_pg_one_plus =
        in.dtype == ValueDType::BF16 && dst.dtype == ValueDType::F32 &&
        wdt == ValueDType::BF16 &&
        wlayout == ::ps::kernel::RmsNormWeightLayout::PER_GROUP &&
        out.weight_mode == ::ps::kernel::RmsNormWeightMode::ONE_PLUS;
    out.variant = bf16_pf_one_plus
                      ? PhysicalRmsNormVariant::Bf16PfOnePlus
                      : (f32_pg_direct
                             ? PhysicalRmsNormVariant::F32PgDirect
                             : (bf16_f32_pg_one_plus
                                    ? PhysicalRmsNormVariant::Bf16F32PgOnePlus
                                    : PhysicalRmsNormVariant::Unsupported));
    if (out.variant != PhysicalRmsNormVariant::Unsupported) {
        out.grid = ::ps::kernel::rmsnorm_bf16_pf_oneplus_grid(rows, F, G);
        out.workgroup = ::ps::kernel::rmsnorm_bf16_pf_oneplus_block_size(G);
    }
    return PhysicalResolveStatus::Applicable;
}

Result<PhysicalResolveStatus> resolve_activation_quantize_physical(
    const Program& program,
    const DispatchBinding& b,
    const HostExecutionContext& ctx,
    PhysicalActivationQuantizeLaunch& out) {
    out = PhysicalActivationQuantizeLaunch{};
    if (b.input_count != 1)
        return Status::invalid_state("act-quant optimized: input_count != 1",
                                     __FILE__, __LINE__);
    if (b.group_size != 32)
        return Status::invalid_state("act-quant optimized: group_size != 32",
                                     __FILE__, __LINE__);
    if (b.workspace_slot >= program.workspace.ranges.size())
        return Status::invalid_state("act-quant optimized: workspace_slot invalid",
                                     __FILE__, __LINE__);
    if (b.activation_max_rows == 0)
        return Status::invalid_state("act-quant optimized: activation_max_rows == 0",
                                     __FILE__, __LINE__);
    auto in_res =
        resolve_host_value(program, ::ps::runtime::ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    if (in.dtype != ValueDType::BF16)
        return Status::invalid_state("act-quant optimized: input dtype != BF16",
                                     __FILE__, __LINE__);
    if (in.feature_count == 0)
        return Status::invalid_state("act-quant optimized: input feature_count == 0",
                                     __FILE__, __LINE__);
    if (b.activation_kp < in.feature_count || (b.activation_kp & 31u) != 0u)
        return Status::invalid_state("act-quant optimized: activation_kp invalid",
                                     __FILE__, __LINE__);
    const uint32_t rows = rows_for(in, ctx);
    if (rows == 0) return PhysicalResolveStatus::NotApplicable;
    if (rows > b.activation_max_rows)
        return Status::invalid_state("act-quant optimized: rows out of range",
                                     __FILE__, __LINE__);
    const Int8ActivationWorkspaceLayout layout =
        Int8ActivationWorkspaceLayout::make(in.feature_count,
                                            b.activation_max_rows);
    if (layout.k_padded != b.activation_kp ||
        layout.code_row_stride_bytes != b.activation_code_stride ||
        layout.scale_row_stride_bytes != b.activation_scale_stride)
        return Status::invalid_state("act-quant optimized: workspace layout mismatch",
                                     __FILE__, __LINE__);
    if (ctx.workspace == nullptr)
        return Status::invalid_state("act-quant optimized: workspace not bound",
                                     __FILE__, __LINE__);
    const uint64_t range_off = program.workspace.ranges[b.workspace_slot].offset;
    uint8_t* range_base = ctx.workspace + range_off;

    out.kind = b.kernel_id == KernelId::ACTIVATION_QUANTIZE_W4A8
                   ? PhysicalActivationQuantizeKind::E4M3
                   : PhysicalActivationQuantizeKind::I8Row;
    out.input = reinterpret_cast<uint64_t>(in.ptr);
    out.codes = reinterpret_cast<uint64_t>(range_base + layout.codes_offset);
    out.scales = reinterpret_cast<uint64_t>(range_base + layout.scales_offset);
    out.rows = rows;
    out.k = in.feature_count;
    out.k_padded = b.activation_kp;
    out.input_row_stride = in.row_stride;
    out.code_row_stride = layout.code_row_stride_bytes;
    out.scale_row_stride = layout.scale_row_stride_bytes;
    out.grid = rows;
    out.workgroup = 256u;
    return PhysicalResolveStatus::Applicable;
}

}  // namespace ps::execution
