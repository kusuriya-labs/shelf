#include <shelf/inference/execution/qsa_dispatch.h>

#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/model_dispatch_state.h>
#include <shelf/kernels/optimized/qsa.h>

namespace shelf::execution {

using ::shelf::runtime::DispatchBinding;
using ::shelf::runtime::KernelId;
using ::shelf::runtime::PrimitiveStateKind;
using ::shelf::runtime::Program;
using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueId;
using ::shelf::runtime::ValueRowDomain;

namespace {

Status bad(const char* msg) {
    return Status::invalid_state(msg, __FILE__, __LINE__);
}

const ::shelf::runtime::StateBinding* find_state_binding(const Program& program,
                                                         uint32_t descriptor_slot) {
    if (descriptor_slot == 0u) return nullptr;
    for (const auto& state : program.states) {
        if (state.slot + 1u == descriptor_slot) return &state;
    }
    return nullptr;
}

bool qsa_view_ready(const ::shelf::state::QsaKeyStatePoolDeviceView& view) {
    return view.pooled_base != nullptr && view.partial_base != nullptr &&
           view.pool_count_base != nullptr && view.partial_count_base != nullptr &&
           view.max_sequences != 0u && view.max_blocks != 0u &&
           view.compress_ratio != 0u && view.head_dim != 0u;
}

}  // namespace

Result<OptimizedLaunchResult> try_launch_qsa_key_cache(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 0u)
        return bad("qsa key cache: bad io count");
    const auto* state = find_state_binding(program, b.descriptor_slot);
    if (state == nullptr || state->kind != PrimitiveStateKind::QSA_KEY_STATE)
        return bad("qsa key cache: state binding missing");
    const auto& view = ctx.model_state.qsa;
    if (!qsa_view_ready(view)) return bad("qsa key cache: state pool missing");
    if (b.group_size != view.compress_ratio || b.flags != view.max_blocks)
        return bad("qsa key cache: geometry differs from the state pool");
    auto in = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in.ok()) return in.status();
    if (in.value().dtype != ValueDType::BF16 ||
        in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        in.value().feature_count != view.head_dim || in.value().ptr == nullptr) {
        return bad("qsa key cache: value contract mismatch");
    }
    if (b.parameter_index0 >= ctx.host_parameter_count || ctx.host_parameters == nullptr)
        return bad("qsa key cache: norm parameter missing");
    const auto& weight = ctx.host_parameters[b.parameter_index0];
    if (weight.device_ptr == nullptr || weight.dtype != ValueDType::BF16 ||
        weight.elements < view.head_dim) {
        return bad("qsa key cache: norm parameter invalid");
    }
    const auto* requests =
        ctx.staged_requests != nullptr ? ctx.staged_requests : ctx.device_requests;
    const uint32_t num_requests = ctx.staged_request_count != 0u ? ctx.staged_request_count
                                                                : ctx.num_requests;
    if (requests == nullptr || num_requests == 0u || ctx.actual_rows == 0u ||
        ctx.row_positions == nullptr) {
        return bad("qsa key cache: requests missing");
    }

    ::shelf::kernel::QsaKeyCacheBatchArgs args;
    args.raw_keys = static_cast<const ::shelf::bf16_t*>(in.value().ptr);
    args.norm_weight = static_cast<const ::shelf::bf16_t*>(weight.device_ptr);
    args.positions = ctx.row_positions;
    args.requests = requests;
    args.pooled_base = view.pooled_base;
    args.partial_base = view.partial_base;
    args.pool_count_base = view.pool_count_base;
    args.partial_count_base = view.partial_count_base;
    args.partial_history = ctx.spec.qsa_spec_history.partial;
    args.count_history = ctx.spec.qsa_spec_history.counts;
    args.history_rows = ctx.spec.qsa_spec_history.rows;
    args.history_partial_state_stride =
        ctx.spec.qsa_spec_history.partial_state_stride;
    args.history_count_state_stride = ctx.spec.qsa_spec_history.count_state_stride;
    args.num_requests = num_requests;
    args.actual_rows = ctx.actual_rows;
    args.max_sequences = view.max_sequences;
    args.state_index = state->state_index;
    args.max_blocks = view.max_blocks;
    args.head_dim = view.head_dim;
    args.compress_ratio = b.group_size;
    args.rotary_dim = b.rotary_dim;
    args.raw_row_stride = in.value().row_stride;
    args.pooled_slot_stride = view.pooled_slot_stride;
    args.pooled_layer_stride = view.pooled_layer_stride;
    args.partial_slot_stride = view.partial_slot_stride;
    args.partial_layer_stride = view.partial_layer_stride;
    args.count_slot_stride = view.count_slot_stride;
    args.count_layer_stride = view.count_layer_stride;
    args.theta = b.scalar_a;
    args.position_scale = b.scalar_b;
    args.inv_freq = ctx.rope_inv_freq;
    const hipError_t error = ::shelf::kernel::launch_qsa_key_cache_batch(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("qsa key cache launch", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_qsa_select(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u)
        return bad("qsa select: bad io count");
    const auto* state = find_state_binding(program, b.descriptor_slot);
    if (state == nullptr || state->kind != PrimitiveStateKind::QSA_KEY_STATE)
        return bad("qsa select: state binding missing");
    const auto& view = ctx.model_state.qsa;
    if (!qsa_view_ready(view)) return bad("qsa select: state pool missing");
    if (b.flags != view.max_blocks ||
        static_cast<uint32_t>(b.scalar_b) != view.compress_ratio)
        return bad("qsa select: geometry differs from the state pool");
    auto q = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!q.ok()) return q.status();
    auto mask = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!mask.ok()) return mask.status();
    if (q.value().dtype != ValueDType::BF16 ||
        q.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        q.value().feature_count !=
            static_cast<uint64_t>(b.group_size) * b.rotary_dim ||
        mask.value().dtype != ValueDType::U8 ||
        mask.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        mask.value().feature_count * 8u < b.flags ||
        q.value().ptr == nullptr || mask.value().ptr == nullptr) {
        return bad("qsa select: value contract mismatch");
    }
    if (ctx.row_sequence_slots == nullptr || ctx.row_positions == nullptr)
        return bad("qsa select: row metadata missing");
    const uint32_t rows = ctx.actual_rows;
    if (rows == 0u) return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};

    ::shelf::kernel::QsaSelectBatchArgs args;
    args.q = static_cast<const ::shelf::bf16_t*>(q.value().ptr);
    args.pooled_base = view.pooled_base;
    args.block_mask = static_cast<uint8_t*>(mask.value().ptr);
    args.row_sequence_slots = ctx.row_sequence_slots;
    args.positions = ctx.row_positions;
    args.rows = rows;
    args.blocks = b.flags;
    args.n_heads = b.group_size;
    args.head_dim = b.rotary_dim;
    args.compress_ratio = static_cast<uint32_t>(b.scalar_b);
    args.state_index = state->state_index;
    args.max_sequences = view.max_sequences;
    args.top_k = b.compute_spec;
    args.q_row_stride = q.value().row_stride;
    args.mask_row_bytes = mask.value().row_stride;
    args.pooled_slot_stride = view.pooled_slot_stride;
    args.pooled_layer_stride = view.pooled_layer_stride;
    args.scale = b.scalar_a;
    const hipError_t error = ::shelf::kernel::launch_qsa_select_batch(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("qsa select launch", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_qsa_rope(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u)
        return bad("qsa rope: bad io count");
    auto in = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in.ok()) return in.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    if (in.value().dtype != ValueDType::BF16 || out.value().dtype != ValueDType::BF16 ||
        in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        out.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        in.value().feature_count != static_cast<uint64_t>(b.group_size) * b.flags ||
        out.value().feature_count != in.value().feature_count ||
        in.value().ptr == nullptr || out.value().ptr == nullptr) {
        return bad("qsa rope: value contract mismatch");
    }
    if (ctx.row_positions == nullptr) return bad("qsa rope: positions missing");
    const uint32_t rows = ctx.actual_rows;
    if (rows == 0u) return OptimizedLaunchResult{OptimizedLaunchKind::NotApplicable};
    // The kernel rotates in place; the graph gives the rope its own output buffer, so the
    // caller hands the source over by copying it first. The two rows can have different
    // aligned strides.
    if (in.value().ptr != out.value().ptr) {
        const std::size_t width = static_cast<std::size_t>(in.value().feature_count) *
                                  sizeof(::shelf::bf16_t);
        const hipError_t copy_error = hipMemcpy2DAsync(
            out.value().ptr, static_cast<std::size_t>(out.value().row_stride) *
                                 sizeof(::shelf::bf16_t),
            in.value().ptr,
            static_cast<std::size_t>(in.value().row_stride) * sizeof(::shelf::bf16_t),
            width, rows, hipMemcpyDeviceToDevice, stream);
        if (copy_error != hipSuccess)
            return Status::hip_error("qsa rope copy", hipGetErrorString(copy_error),
                                     __FILE__, __LINE__);
    }

    ::shelf::kernel::QsaRopeArgs args;
    args.x = static_cast<::shelf::bf16_t*>(out.value().ptr);
    args.positions = ctx.row_positions;
    args.inv_freq = ctx.rope_inv_freq;
    args.rows = rows;
    args.heads = b.group_size;
    args.head_dim = b.flags;
    args.rotary_dim = b.rotary_dim;
    args.theta = b.scalar_a;
    args.position_scale = b.scalar_b;
    const hipError_t error = ::shelf::kernel::launch_qsa_rope(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("qsa rope launch", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
