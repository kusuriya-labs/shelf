#pragma once

#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/runtime/program/program.h>

#include <hip/hip_runtime.h>

namespace shelf::execution {

struct HostExecutionContext;

Result<OptimizedLaunchResult> try_launch_hyper_mix(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_hyper_inject(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_hyper_apply(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace shelf::execution
