#pragma once
#include <shelf/core/status.h>
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel { struct Exl3SwigluProducer; struct Exl3GdnProducer; struct Exl3AttnProducer; }
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
    uint32_t dispatch_count = 1;
};
Result<OptimizedLaunchResult> try_launch_linear_exl3_mul24(
    const ::shelf::runtime::Program&, const ::shelf::runtime::DispatchBinding&,
    HostExecutionContext&, hipStream_t, const ::shelf::kernel::Exl3SwigluProducer* = nullptr,
    const ::shelf::kernel::Exl3GdnProducer* = nullptr,
    const ::shelf::kernel::Exl3AttnProducer* = nullptr);
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
    hipStream_t stream,
    size_t dispatch_end);

}
