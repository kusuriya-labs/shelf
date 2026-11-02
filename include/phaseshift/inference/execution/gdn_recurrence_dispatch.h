#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/kernels/optimized/gdn/recurrence.h>
#include <phaseshift/inference/execution/optimized_dispatch.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::execution {

struct HostExecutionContext;

struct GdnRecurrenceResolvedArgs {
    ::ps::kernel::GdnRecurrenceArgs kernel;
    uint32_t rows = 0;
    uint32_t num_requests = 0;
    uint32_t max_request_rows = 0;
};

Result<GdnRecurrenceResolvedArgs>
resolve_gdn_recurrence_args(
    const ::ps::runtime::Program& program,
    const ::ps::runtime::DispatchBinding& b,
    const HostExecutionContext& ctx);

Result<OptimizedLaunchResult>
try_launch_gdn_recurrence(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace ps::execution
