#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/inference/execution/optimized_dispatch.h>
#include <phaseshift/runtime/program/program.h>
#include <hip/hip_runtime.h>

#include <cstddef>

namespace ps::execution {

struct HostExecutionContext;

Result<OptimizedLaunchResult> try_launch_gdn_conv1d(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace ps::execution
