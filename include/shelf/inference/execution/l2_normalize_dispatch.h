#pragma once
#include <shelf/core/status.h>
#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>

#include <cstddef>

namespace ps::execution {

struct HostExecutionContext;

Result<OptimizedLaunchResult> try_launch_l2_normalize(
    const ::ps::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace ps::execution
