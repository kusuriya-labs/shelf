#include <shelf/inference/execution/hyper_connection_dispatch.h>

#include <shelf/inference/execution/decode_perf_stats.h>
#include <shelf/inference/execution/program_executor.h>
#include <shelf/kernels/optimized/hyper_connection.h>

namespace shelf::execution {

using ::shelf::runtime::DispatchBinding;
using ::shelf::runtime::Program;
using ::shelf::runtime::ValueDType;
using ::shelf::runtime::ValueId;
using ::shelf::runtime::ValueRowDomain;

namespace {

Status bad(const char* msg) {
    return Status::invalid_state(msg, __FILE__, __LINE__);
}

uint32_t token_rows(const HostExecutionContext& ctx) noexcept {
    return ctx.actual_rows != 0u ? ctx.actual_rows : 1u;
}

}  // namespace

Result<OptimizedLaunchResult> try_launch_hyper_mix(const Program& program,
                                                   const DispatchBinding& b,
                                                   HostExecutionContext& ctx,
                                                   hipStream_t stream) {
    (void)program;
    if (b.input_count != 2u) return bad("hyper mix: input_count != 2");
    const uint32_t hc = b.group_size;
    const uint32_t hidden = b.flags;
    if (hc == 0u || hidden == 0u) return bad("hyper mix: geometry is empty");
    const uint32_t width = hc * hidden;

    auto gate = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!gate.ok()) return gate.status();
    auto normed = resolve_host_value(program, ValueId{b.input_slots[1]}, ctx);
    if (!normed.ok()) return normed.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();

    if (gate.value().dtype != ValueDType::BF16 ||
        normed.value().dtype != ValueDType::BF16 ||
        out.value().dtype != ValueDType::BF16) {
        return bad("hyper mix: dtypes must be BF16");
    }
    if (gate.value().feature_count != width || normed.value().feature_count != width ||
        out.value().feature_count != hidden) {
        return bad("hyper mix: feature counts are invalid");
    }
    if (gate.value().row_domain != normed.value().row_domain ||
        gate.value().row_domain != out.value().row_domain) {
        return bad("hyper mix: row domains disagree");
    }
    if (gate.value().ptr == nullptr || normed.value().ptr == nullptr ||
        out.value().ptr == nullptr) {
        return bad("hyper mix: value is unbound");
    }

    ::shelf::kernel::HyperMixArgs args;
    args.gate = gate.value().ptr;
    args.normed = normed.value().ptr;
    args.output = out.value().ptr;
    args.rows = token_rows(ctx);
    args.hc = hc;
    args.hidden = hidden;
    args.gate_stride = gate.value().row_stride;
    args.normed_stride = normed.value().row_stride;
    args.output_stride = out.value().row_stride;
    const hipError_t error = ::shelf::kernel::launch_hyper_mix(args, stream);
    if (error != hipSuccess) {
        return Status::hip_error("hyper mix launch", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    }
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

Result<OptimizedLaunchResult> try_launch_hyper_apply(const Program& program,
                                                     const DispatchBinding& b,
                                                     HostExecutionContext& ctx,
                                                     hipStream_t stream) {
    (void)program;
    if (b.input_count != 3u) return bad("hyper apply: input_count != 3");
    const uint32_t hc = b.group_size;
    const uint32_t hidden = b.flags;
    if (hc == 0u || hidden == 0u) return bad("hyper apply: geometry is empty");
    const uint32_t width = hc * hidden;

    auto raw = resolve_host_value(program, ValueId{b.input_slots[0]}, ctx);
    if (!raw.ok()) return raw.status();
    auto y = resolve_host_value(program, ValueId{b.input_slots[1]}, ctx);
    if (!y.ok()) return y.status();
    auto injection = resolve_host_value(program, ValueId{b.input_slots[2]}, ctx);
    if (!injection.ok()) return injection.status();
    auto out = resolve_host_value(program, ValueId{b.output_slots[0]}, ctx);
    if (!out.ok()) return out.status();

    if (raw.value().dtype != ValueDType::BF16 || y.value().dtype != ValueDType::BF16 ||
        injection.value().dtype != ValueDType::BF16 ||
        out.value().dtype != ValueDType::BF16) {
        return bad("hyper apply: dtypes must be BF16");
    }
    if (raw.value().feature_count != width || y.value().feature_count != hidden ||
        injection.value().feature_count != hc || out.value().feature_count != width) {
        return bad("hyper apply: feature counts are invalid");
    }
    if (raw.value().row_domain != y.value().row_domain ||
        raw.value().row_domain != injection.value().row_domain ||
        raw.value().row_domain != out.value().row_domain) {
        return bad("hyper apply: row domains disagree");
    }
    if (raw.value().ptr == nullptr || y.value().ptr == nullptr ||
        injection.value().ptr == nullptr || out.value().ptr == nullptr) {
        return bad("hyper apply: value is unbound");
    }

    ::shelf::kernel::HyperApplyArgs args;
    args.raw = raw.value().ptr;
    args.y = y.value().ptr;
    args.injection = injection.value().ptr;
    args.output = out.value().ptr;
    args.rows = token_rows(ctx);
    args.hc = hc;
    args.hidden = hidden;
    args.raw_stride = raw.value().row_stride;
    args.y_stride = y.value().row_stride;
    args.injection_stride = injection.value().row_stride;
    args.output_stride = out.value().row_stride;
    const hipError_t error = ::shelf::kernel::launch_hyper_apply(args, stream);
    if (error != hipSuccess) {
        return Status::hip_error("hyper apply launch", hipGetErrorString(error), __FILE__,
                                 __LINE__);
    }
    record_physical_launch(1u);
    record_optimized_op();
    return OptimizedLaunchResult{OptimizedLaunchKind::Launched};
}

}  // namespace shelf::execution
