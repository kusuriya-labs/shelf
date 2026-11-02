#pragma once
#include <shelf/core/status.h>
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace shelf::execution {


struct HostExecutionContext;

enum class KernelMode : uint8_t {
    Correctness,
    Auto,
};

Result<KernelMode> read_kernel_mode();

enum class OptimizedLaunchKind : uint8_t {
    NotApplicable,
    Launched,
};

struct OptimizedLaunchResult {
    OptimizedLaunchKind kind = OptimizedLaunchKind::NotApplicable;
};
Result<OptimizedLaunchResult> try_launch_rmsnorm(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);

Result<OptimizedLaunchResult> try_launch_activation_quantize(
    const ::shelf::runtime::Program& program,
    const ::shelf::runtime::DispatchBinding& b,
    HostExecutionContext& ctx,
    hipStream_t stream);
Result<OptimizedLaunchResult> try_launch_optimized(
    const ::shelf::runtime::Program& program,
    size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);


void record_correctness_fallback(::shelf::runtime::KernelId id);

void rope_optimized_hit();
void gdn_recurrence_optimized_hit();

uint32_t embedding_optimized_count() noexcept;

}
