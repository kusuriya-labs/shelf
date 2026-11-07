#include <shelf/inference/execution/ple_dispatch.h>

#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/model_dispatch_state.h>
#include <shelf/kernels/optimized/ple.h>
#include <shelf/kernels/optimized/ple_conv.h>
#include <shelf/kernels/optimized/repeat_streams.h>

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

}  // namespace

Result<OptimizedLaunchResult> try_launch_repeat_streams(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u)
        return bad("repeat streams: bad io count");
    const uint32_t hc = b.group_size;
    const uint32_t hidden = b.flags;
    if (hc == 0u || hidden == 0u) return bad("repeat streams: empty geometry");
    auto in = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in.ok()) return in.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    if (in.value().dtype != ValueDType::BF16 || out.value().dtype != ValueDType::BF16 ||
        in.value().feature_count != hidden ||
        out.value().feature_count != static_cast<uint64_t>(hc) * hidden ||
        in.value().ptr == nullptr || out.value().ptr == nullptr) {
        return bad("repeat streams: value contract mismatch");
    }
    ::shelf::kernel::RepeatStreamsArgs args;
    args.input = static_cast<const ::shelf::bf16_t*>(in.value().ptr);
    args.output = static_cast<::shelf::bf16_t*>(out.value().ptr);
    args.rows = ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
    args.hc = hc;
    args.hidden = hidden;
    args.input_stride = in.value().row_stride;
    args.output_stride = out.value().row_stride;
    const hipError_t error = ::shelf::kernel::launch_repeat_streams(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("repeat streams launch", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_ple_stream_dot(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 2u || b.output_count != 1u)
        return bad("ple stream dot: bad io count");
    const uint32_t hc = b.group_size;
    const uint32_t hidden = b.flags;
    if (hc == 0u || hidden == 0u) return bad("ple stream dot: empty geometry");
    auto key = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!key.ok()) return key.status();
    auto query = resolve_host_value(program, ValueId{b.input_slots[1]}, ctx);
    if (!query.ok()) return query.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    const uint64_t width = static_cast<uint64_t>(hc) * hidden;
    if (key.value().dtype != ValueDType::BF16 ||
        query.value().dtype != ValueDType::BF16 ||
        out.value().dtype != ValueDType::BF16 ||
        key.value().feature_count != width ||
        query.value().feature_count != width || out.value().feature_count != hc ||
        key.value().ptr == nullptr || query.value().ptr == nullptr ||
        out.value().ptr == nullptr) {
        return bad("ple stream dot: value contract mismatch");
    }
    ::shelf::kernel::PleStreamDotArgs args;
    args.key = static_cast<const ::shelf::bf16_t*>(key.value().ptr);
    args.query = static_cast<const ::shelf::bf16_t*>(query.value().ptr);
    args.output = static_cast<::shelf::bf16_t*>(out.value().ptr);
    args.rows = ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
    args.hc = hc;
    args.hidden = hidden;
    args.key_stride = key.value().row_stride;
    args.query_stride = query.value().row_stride;
    args.output_stride = out.value().row_stride;
    const hipError_t error = ::shelf::kernel::launch_ple_stream_dot(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("ple stream dot launch", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_ple_gate(const Program& program,
                                                  const DispatchBinding& b,
                                                  HostExecutionContext& ctx,
                                                  hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u)
        return bad("ple gate: bad io count");
    auto in = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in.ok()) return in.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    if (in.value().dtype != ValueDType::BF16 || out.value().dtype != ValueDType::BF16 ||
        in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        out.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        in.value().feature_count != out.value().feature_count ||
        in.value().ptr == nullptr || out.value().ptr == nullptr) {
        return bad("ple gate: value contract mismatch");
    }
    ::shelf::kernel::PleGateArgs args;
    args.input = static_cast<const ::shelf::bf16_t*>(in.value().ptr);
    args.output = static_cast<::shelf::bf16_t*>(out.value().ptr);
    args.rows = ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
    args.hidden = in.value().feature_count;
    args.input_stride = in.value().row_stride;
    args.output_stride = out.value().row_stride;
    const hipError_t error = ::shelf::kernel::launch_ple_gate(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("ple gate launch", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_ple_broadcast_mul(
    const Program& program, const DispatchBinding& b, HostExecutionContext& ctx,
    hipStream_t stream) {
    if (b.input_count != 2u || b.output_count != 1u)
        return bad("ple broadcast: bad io count");
    const uint32_t hc = b.group_size;
    const uint32_t hidden = b.flags;
    if (hc == 0u || hidden == 0u) return bad("ple broadcast: empty geometry");
    auto gate = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!gate.ok()) return gate.status();
    auto value = resolve_host_value(program, ValueId{b.input_slots[1]}, ctx);
    if (!value.ok()) return value.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    if (gate.value().dtype != ValueDType::BF16 ||
        value.value().dtype != ValueDType::BF16 || out.value().dtype != ValueDType::BF16 ||
        gate.value().feature_count != hc || value.value().feature_count != hidden ||
        out.value().feature_count != hc * hidden || gate.value().ptr == nullptr ||
        value.value().ptr == nullptr || out.value().ptr == nullptr) {
        return bad("ple broadcast: value contract mismatch");
    }
    ::shelf::kernel::PleBroadcastMulArgs args;
    args.gate = static_cast<const ::shelf::bf16_t*>(gate.value().ptr);
    args.value = static_cast<const ::shelf::bf16_t*>(value.value().ptr);
    args.output = static_cast<::shelf::bf16_t*>(out.value().ptr);
    args.rows = ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
    args.hc = hc;
    args.hidden = hidden;
    args.gate_stride = gate.value().row_stride;
    args.value_stride = value.value().row_stride;
    args.output_stride = out.value().row_stride;
    const hipError_t error = ::shelf::kernel::launch_ple_broadcast_mul(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("ple broadcast launch", hipGetErrorString(error),
                                 __FILE__, __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_ple_conv(const Program& program,
                                                  const DispatchBinding& b,
                                                  HostExecutionContext& ctx,
                                                  hipStream_t stream) {
    if (b.input_count != 1u || b.output_count != 1u)
        return bad("ple conv: bad io count");
    const auto* state = find_state_binding(program, b.descriptor_slot);
    if (state == nullptr || state->kind != PrimitiveStateKind::PLE_CONV_STATE)
        return bad("ple conv: state binding missing");
    const auto& view = ctx.model_state.ple_conv;
    if (view.base == nullptr || view.max_sequences == 0u || view.channels == 0u ||
        view.slot_stride == 0u || view.layer_stride == 0u ||
        view.history_stride < view.channels) {
        return bad("ple conv: state pool missing");
    }
    auto in = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!in.ok()) return in.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();
    if (in.value().dtype != ValueDType::BF16 || out.value().dtype != ValueDType::BF16 ||
        in.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        out.value().row_domain != ValueRowDomain::TOKEN_ROWS ||
        in.value().feature_count != view.channels ||
        out.value().feature_count != view.channels || in.value().ptr == nullptr ||
        out.value().ptr == nullptr) {
        return bad("ple conv: value contract mismatch");
    }
    if (b.parameter_index0 >= ctx.host_parameter_count || ctx.host_parameters == nullptr)
        return bad("ple conv: weight parameter missing");
    const auto& weight = ctx.host_parameters[b.parameter_index0];
    if (weight.device_ptr == nullptr || weight.dtype != ValueDType::BF16 ||
        weight.elements < static_cast<std::uint64_t>(view.channels) * 4u) {
        return bad("ple conv: weight parameter invalid");
    }
    const auto* requests = ctx.device_requests;
    const uint32_t num_requests = ctx.staged_request_count != 0u
                                      ? ctx.staged_request_count
                                      : ctx.num_requests;
    if (requests == nullptr || num_requests == 0u || ctx.actual_rows == 0u)
        return bad("ple conv: requests missing");

    ::shelf::kernel::PleConvArgs args;
    args.input = static_cast<const ::shelf::bf16_t*>(in.value().ptr);
    args.output = static_cast<::shelf::bf16_t*>(out.value().ptr);
    args.weight = static_cast<const ::shelf::bf16_t*>(weight.device_ptr);
    args.state = view.base;
    args.requests = requests;
    args.num_requests = num_requests;
    args.actual_rows = ctx.actual_rows;
    args.channels = view.channels;
    args.max_sequences = view.max_sequences;
    args.state_index = state->state_index;
    args.input_row_stride = in.value().row_stride;
    args.output_row_stride = out.value().row_stride;
    args.state_slot_stride = view.slot_stride;
    args.state_layer_stride = view.layer_stride;
    args.state_history_stride = view.history_stride;
    args.state_history_store = ctx.spec.ple_conv_history.store;
    args.history_rows = ctx.spec.ple_conv_history.rows;
    args.history_state_stride = ctx.spec.ple_conv_history.state_stride;
    const hipError_t error = ::shelf::kernel::launch_ple_conv(args, stream);
    if (error != hipSuccess)
        return Status::hip_error("ple conv launch", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
