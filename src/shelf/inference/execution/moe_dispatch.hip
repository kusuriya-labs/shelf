#include <shelf/kernels/fused/exl3/decode.h>
#include <shelf/kernels/fused/exl3/grouped.h>
#include <shelf/inference/execution/moe_dispatch.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/physical_launch.h>
#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/fused/moe/combine.h>
#include <shelf/kernels/fused/moe/router.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/quantization/quantization_types.h>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace shelf::execution {

using ::shelf::runtime::DispatchBinding;
using ::shelf::runtime::Int8ActivationWorkspaceLayout;
using ::shelf::runtime::KernelId;
using ::shelf::runtime::MoePlanScratch;
using ::shelf::runtime::Program;
using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueId;
using ::shelf::runtime::ValueRowDomain;
using ::shelf::runtime::grouped_row_capacity;
using ::shelf::runtime::WeightSlot;

namespace {

constexpr OptimizedLaunchResult kNone{OptimizedLaunchKind::NotApplicable};

Status bad(const char* msg) {
    return Status::invalid_state(msg, __FILE__, __LINE__);
}

}

uint32_t moe_group_rows(const DispatchBinding& b, const HostExecutionContext& ctx) noexcept {
    if (b.scratch_experts == 0u || b.scratch_top_k == 0u) return 0u;
    const uint32_t tokens = ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
    return grouped_row_capacity(tokens, {b.scratch_experts, b.scratch_top_k});
}

Result<MoePlanView> resolve_moe_plan(const Program& program, const DispatchBinding& b,
                                     const HostExecutionContext& ctx) {
    if (b.scratch_slot == ::shelf::runtime::kNoWorkspace)
        return bad("moe plan: binding has no plan scratch");
    if (ctx.workspace == nullptr) return bad("moe plan: workspace not bound");
    if (b.scratch_slot >= program.workspace.ranges.size())
        return bad("moe plan: scratch slot out of range");
    const MoePlanScratch capacity =
        MoePlanScratch::make(b.scratch_tokens, b.scratch_experts, b.scratch_top_k);
    if (!capacity.valid()) return bad("moe plan: scratch layout invalid");
    const auto& range = program.workspace.ranges[b.scratch_slot];
    if (range.bytes < capacity.bytes()) return bad("moe plan: scratch range too small");

    const uint32_t tokens = ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
    if (tokens > b.scratch_tokens) return bad("moe plan: tokens exceed the scratch capacity");

    MoePlanView view;
    view.tokens = tokens;
    view.experts = b.scratch_experts;
    view.logit_experts = b.scratch_logit_experts != 0u ? b.scratch_logit_experts : b.scratch_experts;
    view.expert_begin = b.scratch_expert_begin;
    view.top_k = b.scratch_top_k;
    view.group_capacity =
        ::shelf::runtime::grouped_row_tile_count(tokens, {b.scratch_experts, b.scratch_top_k});
    view.row_capacity =
        ::shelf::runtime::grouped_row_capacity(tokens, {b.scratch_experts, b.scratch_top_k});

    uint32_t* base = reinterpret_cast<uint32_t*>(ctx.workspace + range.offset);
    view.arrays.sel_expert = reinterpret_cast<int32_t*>(base + capacity.sel_expert_offset);
    view.arrays.sel_weight = reinterpret_cast<float*>(base + capacity.sel_weight_offset);
    view.arrays.tile_counts = base + capacity.tile_counts_offset;
    view.arrays.expert_row_begin = base + capacity.expert_row_begin_offset;
    view.arrays.tile_base = base + capacity.tile_base_offset;
    view.arrays.expert_group_begin = base + capacity.expert_group_begin_offset;
    view.arrays.group_expert = base + capacity.group_expert_offset;
    view.arrays.group_rows = base + capacity.group_rows_offset;
    view.arrays.group_count = base + capacity.group_count_offset;
    view.arrays.row_token = reinterpret_cast<int32_t*>(base + capacity.row_token_offset);
    view.arrays.slot_row_of = reinterpret_cast<int32_t*>(base + capacity.slot_row_of_offset);
    view.arrays.hit_slot = base + capacity.hit_slot_offset;
    view.arrays.hit_rows = base + capacity.hit_rows_offset;
    view.arrays.hit_base = base + capacity.hit_base_offset;
    view.arrays.hit_count = base + capacity.hit_count_offset;
    view.arrays.miss_slot = base + capacity.miss_slot_offset;
    view.arrays.miss_rows = base + capacity.miss_rows_offset;
    view.arrays.miss_base = base + capacity.miss_base_offset;
    view.arrays.miss_count = base + capacity.miss_count_offset;
    view.arrays.hit_expert_group_begin = base + capacity.hit_expert_group_begin_offset;
    view.arrays.miss_expert_group_begin = base + capacity.miss_expert_group_begin_offset;
    return view;
}

Result<OptimizedLaunchResult> try_launch_moe_router(const Program& program,
                                                    const DispatchBinding& b,
                                                    HostExecutionContext& ctx,
                                                    hipStream_t stream) {
    const Result<MoePlanView> plan = resolve_moe_plan(program, b, ctx);
    if (!plan.ok()) return plan.status();
    if (b.input_count != 1u) return bad("moe router: input_count != 1");
    const Result<HostResolvedValue> in_res =
        resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    if (in.dtype != ValueDType::F32) return bad("moe router: logits are not f32");
    if (in.row_domain != ValueRowDomain::TOKEN_ROWS) return bad("moe router: logits row domain");
    if (in.feature_count != plan.value().logit_experts)
        return bad("moe router: logits are not one column per expert");
    if (in.ptr == nullptr) return bad("moe router: logits unbound");
    if (in.row_stride < in.feature_count) return bad("moe router: logits row stride");

    uint32_t* route_freq = nullptr;
    if (b.parameter_index1 != ::shelf::runtime::kNoParameter) {
        if (ctx.host_parameters == nullptr ||
            b.parameter_index1 >= ctx.host_parameter_count) {
            return bad("moe router: route frequency parameter out of range");
        }
        route_freq = reinterpret_cast<uint32_t*>(
            const_cast<void*>(ctx.host_parameters[b.parameter_index1].device_ptr));
    }
    const hipError_t herr = ::shelf::kernel::launch_moe_router(
        static_cast<const float*>(in.ptr), in.row_stride, plan.value().arrays.sel_expert,
        plan.value().arrays.sel_weight, plan.value().arrays.tile_counts, plan.value().tokens,
        plan.value().logit_experts, plan.value().top_k, plan.value().expert_begin,
        plan.value().experts, ctx.error_word, stream, route_freq);
    if (herr != hipSuccess)
        return Status::hip_error("moe router launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_moe_plan_scan(const Program& program,
                                                       const DispatchBinding& b,
                                                       HostExecutionContext& ctx,
                                                       hipStream_t stream) {
    const Result<MoePlanView> plan = resolve_moe_plan(program, b, ctx);
    if (!plan.ok()) return plan.status();
    if (b.input_count != 0u) return bad("moe plan scan: input_count != 0");
    const auto& a = plan.value().arrays;
    const int32_t* expert_slot = nullptr;
    if (b.parameter_index0 != ::shelf::runtime::kNoParameter) {
        if (ctx.host_parameters == nullptr ||
            b.parameter_index0 >= ctx.host_parameter_count) {
            return bad("moe plan scan: slot map parameter out of range");
        }
        expert_slot = static_cast<const int32_t*>(
            ctx.host_parameters[b.parameter_index0].device_ptr);
        if (expert_slot == nullptr) return bad("moe plan scan: slot map is unbound");
    }
    const hipError_t herr = ::shelf::kernel::launch_moe_slot_scan_split(
        a.tile_counts, a.expert_row_begin, a.tile_base, a.expert_group_begin, a.group_expert,
        a.group_rows, a.group_count, a.row_token, expert_slot, a.hit_slot, a.hit_rows,
        a.hit_base, a.hit_count, a.miss_slot, a.miss_rows, a.miss_base, a.miss_count,
        a.hit_expert_group_begin, a.miss_expert_group_begin,
        plan.value().tokens, plan.value().experts, plan.value().top_k, stream);
    if (herr != hipSuccess)
        return Status::hip_error("moe plan scan launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_moe_plan_scatter(const Program& program,
                                                          const DispatchBinding& b,
                                                          HostExecutionContext& ctx,
                                                          hipStream_t stream) {
    const Result<MoePlanView> plan = resolve_moe_plan(program, b, ctx);
    if (!plan.ok()) return plan.status();
    if (b.input_count != 0u) return bad("moe plan scatter: input_count != 0");
    const auto& a = plan.value().arrays;
    const hipError_t herr = ::shelf::kernel::launch_moe_slot_scatter(
        a.sel_expert, a.tile_base, a.expert_row_begin, a.expert_group_begin, a.row_token,
        a.slot_row_of, plan.value().tokens, plan.value().experts, plan.value().top_k, stream);
    if (herr != hipSuccess)
        return Status::hip_error("moe plan scatter launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_moe_grouped_linear(const Program& program,
                                                            const DispatchBinding& b,
                                                            HostExecutionContext& ctx,
                                                            hipStream_t stream) {
    const Result<MoePlanView> plan = resolve_moe_plan(program, b, ctx);
    if (!plan.ok()) return plan.status();
    if (b.output_count != 1u || b.input_count != 0u)
        return bad("moe grouped linear: bad io count");
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return bad("moe grouped linear: weight_index invalid");
    if (b.workspace_slot == ::shelf::runtime::kNoWorkspace)
        return bad("moe grouped linear: no activation plane");
    if (b.workspace_slot >= program.workspace.ranges.size())
        return bad("moe grouped linear: workspace_slot invalid");

    const WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 3u) return bad("moe grouped linear: encoding != Mxfp4");
    if (e.codes == nullptr || e.scales == nullptr)
        return bad("moe grouped linear: weight planes missing");
    if (e.experts < 2u) return bad("moe grouped linear: weight is not a rank-3 expert plane");
    if (!e.preshuffled) return bad("moe grouped linear: weight not preshuffled");
    if (e.weight_scale_group != 32u)
        return bad("moe grouped linear: weight_scale_group must be 32");
    const uint32_t wkp = e.k_padded;
    if (wkp == 0u || (wkp & 127u) != 0u) return bad("moe grouped linear: k_padded invalid");
    if (e.rows == 0u || (e.rows & 15u) != 0u)
        return bad("moe grouped linear: expert rows are not 16 aligned");
    const uint32_t nb = wkp >> 5;
    if (e.codes_row_stride_bytes != nb * ::shelf::quantization::mxfp4::mxfp4_native_block_bytes ||
        e.scale_row_stride_bytes != nb || e.storage_scale_stride_bytes != nb)
        return bad("moe grouped linear: expert row strides are not the native layout");
    if (e.codes_batch_stride_bytes !=
            ::shelf::quantization::mxfp4::mxfp4_native_codes_batch_stride_bytes(e.rows, wkp) ||
        e.scales_batch_stride_bytes !=
            ::shelf::quantization::mxfp4::mxfp4_native_scale_batch_stride_bytes(e.rows, wkp))
        return bad("moe grouped linear: expert batch strides are not the native layout");

    const uint32_t akp = b.activation_kp;
    if (akp != wkp) return bad("moe grouped linear: activation_kp != weight k_padded");
    if (b.activation_max_rows == 0u) return bad("moe grouped linear: activation_max_rows == 0");
    const Int8ActivationWorkspaceLayout layout =
        Int8ActivationWorkspaceLayout::make(akp, b.activation_max_rows);
    if (layout.k_padded != akp || layout.code_row_stride_bytes != b.activation_code_stride ||
        layout.scale_row_stride_bytes != b.activation_scale_stride)
        return bad("moe grouped linear: activation plane layout mismatch");
    if (b.activation_max_rows < plan.value().row_capacity)
        return bad("moe grouped linear: activation plane smaller than the group rows");

    const Result<HostResolvedValue> out_res =
        resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue out = out_res.value();
    if (out.row_domain != ValueRowDomain::GROUPED_ROWS)
        return bad("moe grouped linear: output is not in group rows");
    if (out.feature_count != e.rows)
        return bad("moe grouped linear: output features != expert rows");
    if (out.row_stride < e.rows) return bad("moe grouped linear: output row stride too small");
    if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
        return bad("moe grouped linear: output dtype not BF16/F32");

    uint8_t* range_base = ctx.workspace + program.workspace.ranges[b.workspace_slot].offset;
    const uint8_t* acodes = reinterpret_cast<const uint8_t*>(range_base + layout.codes_offset);
    const float* ascales = reinterpret_cast<const float*>(range_base + layout.scales_offset);

    auto launch_plane = [&](const WeightSlot& plane, const uint32_t* slot,
                            const uint32_t* rows, const uint32_t* base,
                            const uint32_t* count) -> hipError_t {
        return ::shelf::kernel::launch_gemm_mxfp4_w4a8_grouped_split(
            static_cast<const uint8_t*>(plane.codes),
            static_cast<const uint8_t*>(plane.scales),
            plane.row_ref_exact && plane.row_ref != nullptr
                ? static_cast<const uint8_t*>(plane.row_ref)
                : nullptr,
            reinterpret_cast<const int8_t*>(acodes), ascales, out.ptr,
            out.dtype == ValueDType::BF16 ? ::shelf::kernel::Mxfp4GemmOutputDType::BF16
                                         : ::shelf::kernel::Mxfp4GemmOutputDType::F32,
            plane.rows, wkp, plane.rows, plane.codes_batch_stride_bytes,
            plane.scales_batch_stride_bytes, plane.scale_row_stride_bytes,
            b.activation_code_stride, b.activation_scale_stride, out.row_stride, slot, rows,
            base, count, plan.value().group_capacity, stream);
    };

    hipError_t herr = launch_plane(e, plan.value().arrays.hit_slot, plan.value().arrays.hit_rows,
                                   plan.value().arrays.hit_base,
                                   plan.value().arrays.hit_count);
    if (herr == hipErrorNotSupported) return kNone;
    if (herr != hipSuccess)
        return Status::hip_error("moe grouped gemm launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);

    if (b.host_weight_index != ::shelf::runtime::kNoWeight) {
        if (b.host_weight_index >= ctx.weight_count)
            return bad("moe grouped linear: host weight_index invalid");
        const WeightSlot host = ctx.weights[b.host_weight_index];
        if (host.encoding != 3u || host.codes == nullptr || host.scales == nullptr ||
            host.experts < 2u || !host.preshuffled || host.weight_scale_group != 32u ||
            host.rows != e.rows || host.k_padded != wkp ||
            host.codes_row_stride_bytes != e.codes_row_stride_bytes ||
            host.scale_row_stride_bytes != e.scale_row_stride_bytes ||
            host.storage_scale_stride_bytes != e.storage_scale_stride_bytes ||
            host.codes_batch_stride_bytes != e.codes_batch_stride_bytes ||
            host.scales_batch_stride_bytes != e.scales_batch_stride_bytes)
            return bad("moe grouped linear: host expert plane mismatch");
        herr = launch_plane(host, plan.value().arrays.miss_slot,
                            plan.value().arrays.miss_rows, plan.value().arrays.miss_base,
                            plan.value().arrays.miss_count);
        if (herr == hipErrorNotSupported) return kNone;
        if (herr != hipSuccess)
            return Status::hip_error("moe grouped host gemm launch",
                                     hipGetErrorString(herr), __FILE__, __LINE__);
    }
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

namespace {
bool span_overlap(uintptr_t a0, uintptr_t a1, uintptr_t b0, uintptr_t b1) {
    return a0 < b1 && b0 < a1;
}
}

bool moe_swiglu_fusion_available(const ::shelf::runtime::Program& program,
                                 const ::shelf::runtime::DispatchBinding& b,
                                 HostExecutionContext& ctx) {
    const MoeSwigluProducer producer = moe_find_swiglu_producer(program, b, ctx);
    if (!producer.found) return false;
    MoePreRotatePlane plane;
    if (!moe_pre_rotate_plane(program, b, ctx, plane)) return false;
    const uint64_t plane_elems = static_cast<uint64_t>(b.activation_max_rows);
    const uintptr_t c0 = reinterpret_cast<uintptr_t>(plane.codes);
    const uintptr_t c1 = c0 + plane_elems * b.activation_kp;
    const uintptr_t g0 = reinterpret_cast<uintptr_t>(producer.gate);
    const uintptr_t g1 = g0 + plane_elems * producer.gate_row_stride * 2u;
    const uintptr_t u0 = reinterpret_cast<uintptr_t>(producer.up);
    const uintptr_t u1 = u0 + plane_elems * producer.up_row_stride * 2u;
    return !span_overlap(c0, c1, g0, g1) && !span_overlap(c0, c1, u0, u1);
}

bool moe_pre_rotate_plane(const ::shelf::runtime::Program& program,
                          const ::shelf::runtime::DispatchBinding& b,
                          HostExecutionContext& ctx, MoePreRotatePlane& out) {
    out = MoePreRotatePlane{};
    static const bool disabled = [] {
        const char* v = std::getenv("SHELF_DISABLE_EXL3_GROUPED_PREROTATE");
        return v != nullptr && v[0] != '\0';
    }();
    if (disabled || ctx.workspace == nullptr || ctx.weights == nullptr) return false;
    if (b.weight_index >= ctx.weight_count) return false;
    const WeightSlot e = ctx.weights[b.weight_index];
    if (b.workspace_slot == ::shelf::runtime::kNoWorkspace ||
        b.workspace_slot >= program.workspace.ranges.size() ||
        b.activation_max_rows == 0u || b.activation_kp != e.cols)
        return false;
    const Result<MoePlanView> plan = resolve_moe_plan(program, b, ctx);
    if (!plan.ok()) return false;
    if (plan.value().tokens * plan.value().top_k < 128u) return false;
    const uint64_t code_bytes =
        static_cast<uint64_t>(b.activation_max_rows) * b.activation_kp;
    const uint64_t scale_bytes = static_cast<uint64_t>(b.activation_max_rows) *
                                 (b.activation_kp / 128u) * sizeof(float);
    const uint64_t aligned = (code_bytes + 255ull) & ~std::uint64_t{255ull};
    const auto& range = program.workspace.ranges[b.workspace_slot];
    if (range.bytes < aligned + scale_bytes) return false;
    out.codes = ctx.workspace + range.offset;
    out.scales = reinterpret_cast<float*>(out.codes + aligned);
    return true;
}

MoeSwigluProducer moe_find_swiglu_producer(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx) {
    MoeSwigluProducer out;
    if (b.input_count != 1u) return out;
    const uint32_t slot = b.input_slots[0];
    const Result<HostResolvedValue> dst = resolve_host_value(program, ValueId{slot}, ctx);
    if (!dst.ok() || dst.value().ptr == nullptr) return out;
    size_t b_index = program.dispatches.size();
    for (size_t idx = 0; idx < program.dispatches.size(); ++idx)
        if (&program.dispatches[idx] == &b) { b_index = idx; break; }
    const DispatchBinding* src = nullptr;
    size_t src_index = 0;
    for (size_t idx = 0; idx < b_index; ++idx) {
        const auto& d = program.dispatches[idx];
        if (d.kernel_id == KernelId::SWIGLU && d.output_count == 1u &&
            d.output_slots[0] == slot) {
            src = &d;
            src_index = idx;
        }
    }
    if (src == nullptr) return out;
    for (size_t idx = src_index + 1u; idx < program.dispatches.size(); ++idx) {
        const auto& d = program.dispatches[idx];
        if (&d == &b) continue;
        for (uint32_t k = 0; k < d.input_count; ++k)
            if (d.input_slots[k] == slot) return out;
    }
    const ::shelf::runtime::ValueBinding* vb = program.find_value(ValueId{slot});
    if (vb == nullptr ||
        (vb->storage != ::shelf::runtime::ValueStorage::WORKSPACE &&
         vb->storage != ::shelf::runtime::ValueStorage::WORKSPACE_VIEW))
        return out;
    PhysicalElementwiseLaunch elem{};
    const auto resolved = resolve_elementwise_physical(program, *src, ctx, elem);
    if (!resolved.ok() || resolved.value() != PhysicalResolveStatus::Applicable ||
        elem.profile != ElementwiseProfile::SwigluBf16 || elem.rows == 0u ||
        elem.features == 0u || !elem.in0 || !elem.in1 || !elem.out0 ||
        elem.in0_stride < elem.features || elem.in1_stride < elem.features ||
        elem.out0_stride < elem.features)
        return out;
    if (reinterpret_cast<const void*>(elem.out0) != dst.value().ptr) return out;
    out.found = true;
    out.gate = reinterpret_cast<const void*>(elem.in0);
    out.up = reinterpret_cast<const void*>(elem.in1);
    out.out = reinterpret_cast<void*>(elem.out0);
    out.gate_row_stride = elem.in0_stride;
    out.up_row_stride = elem.in1_stride;
    out.out_row_stride = elem.out0_stride;
    out.rows = elem.rows;
    out.features = elem.features;
    out.bf16_intermediate = elem.aux != 0u;
    return out;
}

Result<OptimizedLaunchResult> try_launch_moe_grouped_exl3(const Program& program,
                                                          const DispatchBinding& b,
                                                          HostExecutionContext& ctx,
                                                          hipStream_t stream) {
    const MoeSwigluProducer swiglu = moe_find_swiglu_producer(program, b, ctx);
    const Result<MoePlanView> plan = resolve_moe_plan(program, b, ctx);
    if (!plan.ok()) return plan.status();
    if (b.output_count != 1u || b.input_count != 1u)
        return bad("moe grouped exl3: bad io count");
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return bad("moe grouped exl3: weight_index invalid");

    const WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 4u) return bad("moe grouped exl3: encoding != Exl3");
    if (e.compute_spec != static_cast<uint8_t>(
            ::shelf::quantization::ComputeSpecId::EXL3_MUL24_INT8_W8A8_F32_BF16))
        return Status::unsupported("moe grouped exl3 requires mul24 INT8", __FILE__, __LINE__);
    if (e.codes == nullptr || e.exl3_suh == nullptr || e.exl3_svh == nullptr)
        return bad("moe grouped exl3: weight planes missing");
    if (e.experts < 2u) return bad("moe grouped exl3: weight is not an expert plane");
    if (e.k_padded != e.cols || (e.cols & 127u) != 0u || (e.rows & 127u) != 0u)
        return bad("moe grouped exl3: shape is not 128-aligned");

    const Result<HostResolvedValue> in_res =
        resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in_res.ok()) return in_res.status();
    const HostResolvedValue in = in_res.value();
    const bool token_rows = in.row_domain == ValueRowDomain::TOKEN_ROWS;
    if (in.dtype != ValueDType::BF16 ||
        (!token_rows && in.row_domain != ValueRowDomain::GROUPED_ROWS))
        return bad("moe grouped exl3: input is not BF16 token/group rows");
    if (in.feature_count != e.cols || in.row_stride < in.feature_count)
        return bad("moe grouped exl3: input features mismatch");
    if (in.ptr == nullptr) return bad("moe grouped exl3: input unbound");

    const Result<HostResolvedValue> out_res =
        resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue out = out_res.value();
    if (out.row_domain != ValueRowDomain::GROUPED_ROWS)
        return bad("moe grouped exl3: output is not in group rows");
    if (out.feature_count != e.rows || out.row_stride < out.feature_count)
        return bad("moe grouped exl3: output features mismatch");
    if (out.dtype != ValueDType::BF16 && out.dtype != ValueDType::F32)
        return bad("moe grouped exl3: output dtype not BF16/F32");
    if (out.ptr == nullptr) return bad("moe grouped exl3: output unbound");

    ::shelf::kernel::Exl3GroupedArgs args;
    args.trellis = e.codes;
    args.suh = e.exl3_suh;
    args.svh = e.exl3_svh;
    args.input = in.ptr;
    args.output = out.ptr;
    args.out_features = e.rows;
    args.k = e.cols;
    args.input_row_stride = in.row_stride;
    args.output_row_stride = out.row_stride;
    args.output_dtype = out.dtype == ValueDType::BF16 ? 0u : 1u;
    args.ka = e.exl3_ka;
    args.half = e.exl3_half;
    args.trellis_batch_stride = e.codes_batch_stride_bytes;
    args.suh_batch_stride = e.exl3_suh_batch_stride_bytes;
    args.svh_batch_stride = e.exl3_svh_batch_stride_bytes;
    args.group_expert = plan.value().arrays.group_expert;
    args.group_rows = plan.value().arrays.group_rows;
    args.group_count = plan.value().arrays.group_count;
    args.expert_group_begin = plan.value().arrays.expert_group_begin;
    args.row_token = token_rows ? plan.value().arrays.row_token : nullptr;
    args.group_capacity = plan.value().group_capacity;

    // One INT8 activation plane per projection: rotating once here replaces the rotation
    // every N-tile block would otherwise repeat. Decode-sized routing keeps the fused
    // rotation because a second launch costs more than it saves.
    MoePreRotatePlane pre_plane;
    (void)moe_pre_rotate_plane(program, b, ctx, pre_plane);
    uint8_t* const pre_codes = pre_plane.codes;
    float* const pre_scales = pre_plane.scales;

    static const bool swiglu_fusion_enabled =
        std::getenv("SHELF_DISABLE_SWIGLU_MOE_DOWN") == nullptr;
    const bool fused_swiglu = swiglu_fusion_enabled && swiglu.found &&
                               pre_codes != nullptr &&
                               moe_swiglu_fusion_available(program, b, ctx);
    if (swiglu.found && !fused_swiglu) {
        const hipError_t werr = ::shelf::kernel::launch_swiglu_bf16(
            static_cast<const shelf::bf16_t*>(swiglu.gate), swiglu.gate_row_stride,
            static_cast<const shelf::bf16_t*>(swiglu.up), swiglu.up_row_stride,
            static_cast<shelf::bf16_t*>(swiglu.out), swiglu.out_row_stride,
            swiglu.rows, swiglu.features, stream, swiglu.bf16_intermediate);
        if (werr != hipSuccess)
            return Status::hip_error("fused swiglu write", hipGetErrorString(werr),
                                     __FILE__, __LINE__);
    }
    auto launch_plane = [&](const WeightSlot& plane, const uint32_t* group_expert,
                            const uint32_t* group_rows, const uint32_t* group_count,
                            const uint32_t* group_row_base,
                            const uint32_t* expert_group_begin) -> hipError_t {
        ::shelf::kernel::Exl3GroupedArgs a = args;
        a.trellis = plane.codes;
        a.suh = plane.exl3_suh;
        a.svh = plane.exl3_svh;
        a.group_expert = group_expert;
        a.group_rows = group_rows;
        a.group_count = group_count;
        a.group_row_base = group_row_base;
        a.expert_group_begin = expert_group_begin;
        if (pre_codes != nullptr) {
            const hipError_t perr = fused_swiglu
                ? ::shelf::kernel::launch_exl3_grouped_swiglu_activate(
                      swiglu.gate, swiglu.up, plane.exl3_suh, pre_codes,
                      pre_scales, group_expert, group_rows, group_count, group_row_base,
                      args.row_token, b.activation_max_rows, e.cols,
                      swiglu.gate_row_stride, swiglu.up_row_stride,
                      swiglu.bf16_intermediate ? 1u : 0u,
                      plane.exl3_suh_batch_stride_bytes, stream)
                : ::shelf::kernel::launch_exl3_grouped_activate(
                      in.ptr, plane.exl3_suh, pre_codes, pre_scales, group_expert, group_rows,
                      group_count, group_row_base, args.row_token, b.activation_max_rows,
                      e.cols, in.row_stride, plane.exl3_suh_batch_stride_bytes, stream);
            if (perr != hipSuccess) return perr;
            a.pre_rotated_codes = pre_codes;
            a.pre_rotated_scales = pre_scales;
        }
        return ::shelf::kernel::launch_gemm_exl3_int8_grouped(a, stream);
    };

    const uint32_t* hit_expert_group_begin =
        b.host_weight_index == ::shelf::runtime::kNoWeight
            ? plan.value().arrays.expert_group_begin
            : plan.value().arrays.hit_expert_group_begin;
    hipError_t herr = launch_plane(e, plan.value().arrays.hit_slot,
                                   plan.value().arrays.hit_rows,
                                   plan.value().arrays.hit_count,
                                   plan.value().arrays.hit_base,
                                   hit_expert_group_begin);
    if (herr != hipSuccess)
        return Status::hip_error("moe grouped exl3 launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);

    if (b.host_weight_index != ::shelf::runtime::kNoWeight) {
        if (b.host_weight_index >= ctx.weight_count)
            return bad("moe grouped exl3: host weight_index invalid");
        const WeightSlot host = ctx.weights[b.host_weight_index];
        if (host.encoding != 4u || host.codes == nullptr || host.exl3_suh == nullptr ||
            host.exl3_svh == nullptr || host.experts < 2u || host.rows != e.rows ||
            host.k_padded != e.k_padded || host.exl3_ka != e.exl3_ka ||
            host.exl3_half != e.exl3_half || host.compute_spec != e.compute_spec ||
            host.codes_row_stride_bytes != e.codes_row_stride_bytes ||
            host.codes_batch_stride_bytes != e.codes_batch_stride_bytes ||
            host.exl3_suh_batch_stride_bytes != e.exl3_suh_batch_stride_bytes ||
            host.exl3_svh_batch_stride_bytes != e.exl3_svh_batch_stride_bytes)
            return bad("moe grouped exl3: host expert plane mismatch");
        herr = launch_plane(host, plan.value().arrays.miss_slot,
                            plan.value().arrays.miss_rows,
                            plan.value().arrays.miss_count,
                            plan.value().arrays.miss_base,
                            plan.value().arrays.miss_expert_group_begin);
        if (herr != hipSuccess)
            return Status::hip_error("moe grouped exl3 host launch", hipGetErrorString(herr),
                                     __FILE__, __LINE__);
    }
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_moe_combine(const Program& program,
                                                     const DispatchBinding& b,
                                                     HostExecutionContext& ctx,
                                                     hipStream_t stream) {
    const Result<MoePlanView> plan = resolve_moe_plan(program, b, ctx);
    if (!plan.ok()) return plan.status();
    if (b.input_count != 3u || b.output_count != 1u)
        return bad("moe combine: bad io count");

    const bool include_experts = (b.flags & 1u) != 0u;
    const bool include_shared = (b.flags & 2u) != 0u;
    const Result<HostResolvedValue> expert_res =
        resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!expert_res.ok()) return expert_res.status();
    const HostResolvedValue expert = expert_res.value();
    const Result<HostResolvedValue> shared_res =
        resolve_host_value(program, ValueId{b.input_slots[1]}, ctx);
    if (!shared_res.ok()) return shared_res.status();
    const HostResolvedValue shared = shared_res.value();
    const Result<HostResolvedValue> gate_res =
        resolve_host_value(program, ValueId{b.input_slots[2]}, ctx);
    if (!gate_res.ok()) return gate_res.status();
    const HostResolvedValue gate = gate_res.value();
    const Result<HostResolvedValue> out_res =
        resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out_res.ok()) return out_res.status();
    const HostResolvedValue out = out_res.value();

    if (out.dtype != ValueDType::BF16)
        return bad("moe combine: output must be bf16");
    if (include_experts && expert.dtype != ValueDType::BF16)
        return bad("moe combine: expert output must be bf16");
    if (include_shared && (shared.dtype != ValueDType::BF16 ||
                           gate.dtype != ValueDType::F32))
        return bad("moe combine: shared value / gate dtype");
    if (expert.row_domain != ValueRowDomain::GROUPED_ROWS)
        return bad("moe combine: expert output is not in group rows");
    if (shared.row_domain != ValueRowDomain::TOKEN_ROWS ||
        out.row_domain != ValueRowDomain::TOKEN_ROWS)
        return bad("moe combine: shared or output is not one row per token");
    if (include_experts && expert.feature_count != out.feature_count)
        return bad("moe combine: expert feature count mismatch");
    if (include_shared && shared.feature_count != out.feature_count)
        return bad("moe combine: shared feature count mismatch");
    if (include_shared && gate.feature_count != 1u)
        return bad("moe combine: shared gate is not one column");
    if (include_shared && gate.row_stride == 0u)
        return bad("moe combine: shared gate row stride is zero");
    if (out.ptr == nullptr) return bad("moe combine: output unbound");
    if (include_experts && expert.ptr == nullptr) return bad("moe combine: expert unbound");
    if (include_shared && (shared.ptr == nullptr || gate.ptr == nullptr))
        return bad("moe combine: shared value unbound");
    if (include_experts && expert.row_stride < expert.feature_count)
        return bad("moe combine: expert row stride too small");
    if (include_shared && shared.row_stride < shared.feature_count)
        return bad("moe combine: shared row stride too small");
    if (include_experts) {
        const ::shelf::runtime::ValueBinding* expert_binding =
            program.find_value(ValueId{b.input_slots[0]});
        if (expert_binding == nullptr) return bad("moe combine: expert output unbound");
        const uint32_t elem = ::shelf::runtime::value_dtype_bytes(expert.dtype);
        if (elem == 0u ||
            expert_binding->bytes <
                static_cast<uint64_t>(plan.value().row_capacity) * expert.row_stride * elem)
            return bad("moe combine: group rows exceed the expert output plane");
    }

    const hipError_t herr = ::shelf::kernel::launch_moe_combine(
        static_cast<::shelf::bf16_t*>(out.ptr),
        include_experts ? static_cast<const ::shelf::bf16_t*>(expert.ptr) : nullptr,
        plan.value().arrays.sel_weight, plan.value().arrays.slot_row_of,
        include_shared ? static_cast<const ::shelf::bf16_t*>(shared.ptr) : nullptr,
        include_shared ? static_cast<const float*>(gate.ptr) : nullptr,
        plan.value().tokens, plan.value().top_k, out.feature_count,
        include_experts ? expert.row_stride : out.feature_count,
        include_shared ? shared.row_stride : out.feature_count,
        include_shared ? gate.row_stride : 0u, include_experts, stream);
    if (herr != hipSuccess)
        return Status::hip_error("moe combine launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}
