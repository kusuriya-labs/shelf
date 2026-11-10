#pragma once
#include <shelf/core/status.h>
#include <shelf/kernels/optimized/linear/detail/types.h>
#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace shelf::execution {


struct HostExecutionContext;

// Host-side description of the path a dispatch will take. Selection is separated from
// execution so the whole table can be tested on the CPU; it is not passed to the GPU.
struct Exl3DispatchDecision {
    ::shelf::kernel::Exl3Phase phase = ::shelf::kernel::Exl3Phase::Decode;
    ::shelf::kernel::Exl3Codebook codebook = ::shelf::kernel::Exl3Codebook::Mul1;
    ::shelf::kernel::exl3_detail::Exl3Backend backend =
        ::shelf::kernel::exl3_detail::Exl3Backend::Direct;
    // nullptr when the direct path is used; otherwise why the dense path was not taken.
    const char* fallback_reason = nullptr;
};

// mul24 prefill backend choice. The recon+dense path needs a usable dense workspace
// (`dense_ok`), at least `dense_min_rows` active rows, and `n_rows * sizeof(float)`
// bytes of rowsum capacity. This is the existing condition, moved to a pure function.
bool exl3_mul24_use_dense(bool dense_ok, uint32_t n_rows, uint32_t dense_min_rows,
                          uint64_t rowsum_bytes);

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

}
