#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/inference/execution/optimized_dispatch.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ps::execution {

struct HostExecutionContext;

Result<OptimizedLaunchResult> try_launch_elementwise(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace ps::execution
