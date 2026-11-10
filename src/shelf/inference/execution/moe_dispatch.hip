#include <shelf/inference/execution/moe_dispatch.h>
#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/kernels/optimized/moe/combine.h>
#include <shelf/kernels/optimized/moe/router.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/quantization/quantization_types.h>

#include <cstdint>
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

    const hipError_t herr = ::shelf::kernel::launch_moe_router(
        static_cast<const float*>(in.ptr), in.row_stride, plan.value().arrays.sel_expert,
        plan.value().arrays.sel_weight, plan.value().arrays.tile_counts, plan.value().tokens,
        plan.value().logit_experts, plan.value().top_k, plan.value().expert_begin,
        plan.value().experts, ctx.error_word, stream);
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
    const hipError_t herr = ::shelf::kernel::launch_moe_slot_scan(
        a.tile_counts, a.expert_row_begin, a.tile_base, a.expert_group_begin, a.group_expert,
        a.group_rows, a.group_count, a.row_token, plan.value().tokens, plan.value().experts,
        plan.value().top_k, stream);
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

    const hipError_t herr = ::shelf::kernel::launch_gemm_mxfp4_w4a8_grouped(
        static_cast<const uint8_t*>(e.codes), static_cast<const uint8_t*>(e.scales),
        e.row_ref_exact && e.row_ref != nullptr ? static_cast<const uint8_t*>(e.row_ref)
                                               : nullptr,
        reinterpret_cast<const int8_t*>(acodes), ascales, out.ptr,
        out.dtype == ValueDType::BF16 ? ::shelf::kernel::Mxfp4GemmOutputDType::BF16
                                     : ::shelf::kernel::Mxfp4GemmOutputDType::F32,
        e.rows, wkp, e.rows, e.codes_batch_stride_bytes, e.scales_batch_stride_bytes,
        e.scale_row_stride_bytes, b.activation_code_stride, b.activation_scale_stride,
        out.row_stride, plan.value().arrays.group_expert, plan.value().arrays.group_rows,
        plan.value().arrays.group_count, plan.value().group_capacity, stream);
    if (herr == hipErrorNotSupported) return kNone;
    if (herr != hipSuccess)
        return Status::hip_error("moe grouped gemm launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

// Default for mul24 expert activations; SHELF_EXL3_INT8_MOE overrides it.
constexpr bool kExl3Int8MoeDefault = false;

Result<OptimizedLaunchResult> try_launch_moe_grouped_exl3(const Program& program,
                                                          const DispatchBinding& b,
                                                          HostExecutionContext& ctx,
                                                          hipStream_t stream) {
    const Result<MoePlanView> plan = resolve_moe_plan(program, b, ctx);
    if (!plan.ok()) return plan.status();
    if (b.output_count != 1u || b.input_count != 1u)
        return bad("moe grouped exl3: bad io count");
    if (b.weight_index >= ctx.weight_count || ctx.weights == nullptr)
        return bad("moe grouped exl3: weight_index invalid");

    const WeightSlot e = ctx.weights[b.weight_index];
    if (e.encoding != 4u) return bad("moe grouped exl3: encoding != Exl3");
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
    args.mul24 = e.compute_spec == static_cast<uint8_t>(
                                       ::shelf::quantization::ComputeSpecId::
                                           EXL3_MUL24_INT8_W8A8_F32_BF16)
                     ? 1u
                     : 0u;
    // SHELF_EXL3_INT8_MOE=1 runs mul24 experts with INT8 activations (iu8 WMMA); 0 keeps the FP16 GEMM.
    static const bool int8_act = [] {
        const char* v = std::getenv("SHELF_EXL3_INT8_MOE");
        return v != nullptr ? std::atoi(v) != 0 : kExl3Int8MoeDefault;
    }();
    args.int8_act = args.mul24 != 0u && int8_act ? 1u : 0u;
    args.trellis_batch_stride = e.codes_batch_stride_bytes;
    args.suh_batch_stride = e.exl3_suh_batch_stride_bytes;
    args.svh_batch_stride = e.exl3_svh_batch_stride_bytes;
    args.group_expert = plan.value().arrays.group_expert;
    args.group_rows = plan.value().arrays.group_rows;
    args.group_count = plan.value().arrays.group_count;
    args.expert_group_begin = plan.value().arrays.expert_group_begin;
    args.row_token = token_rows ? plan.value().arrays.row_token : nullptr;
    args.group_capacity = plan.value().group_capacity;

    const hipError_t herr =
        ::shelf::kernel::launch_gemm_exl3_w16a16_grouped(args, stream);
    if (herr != hipSuccess)
        return Status::hip_error("moe grouped exl3 launch", hipGetErrorString(herr),
                                 __FILE__, __LINE__);
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

    if (expert.dtype != ValueDType::BF16 || shared.dtype != ValueDType::BF16 ||
        out.dtype != ValueDType::BF16)
        return bad("moe combine: expert / shared / output must be bf16");
    if (gate.dtype != ValueDType::F32) return bad("moe combine: shared gate is not f32");
    if (expert.row_domain != ValueRowDomain::GROUPED_ROWS)
        return bad("moe combine: expert output is not in group rows");
    if (shared.row_domain != ValueRowDomain::TOKEN_ROWS ||
        out.row_domain != ValueRowDomain::TOKEN_ROWS)
        return bad("moe combine: shared or output is not one row per token");
    if (expert.feature_count != out.feature_count || shared.feature_count != out.feature_count)
        return bad("moe combine: feature count mismatch");
    if (gate.feature_count != 1u) return bad("moe combine: shared gate is not one column");
    if (gate.row_stride == 0u) return bad("moe combine: shared gate row stride is zero");
    if (expert.ptr == nullptr || shared.ptr == nullptr || gate.ptr == nullptr ||
        out.ptr == nullptr)
        return bad("moe combine: value unbound");
    if (expert.row_stride < expert.feature_count || shared.row_stride < shared.feature_count)
        return bad("moe combine: row stride too small");
    const ::shelf::runtime::ValueBinding* expert_binding =
        program.find_value(ValueId{b.input_slots[0]});
    if (expert_binding == nullptr) return bad("moe combine: expert output unbound");
    const uint32_t elem = ::shelf::runtime::value_dtype_bytes(expert.dtype);
    if (elem == 0u ||
        expert_binding->bytes <
            static_cast<uint64_t>(plan.value().row_capacity) * expert.row_stride * elem)
        return bad("moe combine: group rows exceed the expert output plane");

    const hipError_t herr = ::shelf::kernel::launch_moe_combine(
        static_cast<::shelf::bf16_t*>(out.ptr), static_cast<const ::shelf::bf16_t*>(expert.ptr),
        plan.value().arrays.sel_weight, plan.value().arrays.slot_row_of,
        static_cast<const ::shelf::bf16_t*>(shared.ptr),
        static_cast<const float*>(gate.ptr), plan.value().tokens, plan.value().top_k,
        out.feature_count, expert.row_stride, shared.row_stride, gate.row_stride, stream);
    if (herr != hipSuccess)
        return Status::hip_error("moe combine launch", hipGetErrorString(herr), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}
