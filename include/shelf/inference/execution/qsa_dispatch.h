#pragma once

#include <shelf/inference/execution/optimized_dispatch.h>
#include <shelf/runtime/program/program.h>

#include <hip/hip_runtime.h>

namespace shelf::execution {

struct HostExecutionContext;

Result<OptimizedLaunchResult> try_launch_qsa_key_cache(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_qsa_scores(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_qsa_rope(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_qsa_topk(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_qsa_expand_mask(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

}  // namespace shelf::execution
