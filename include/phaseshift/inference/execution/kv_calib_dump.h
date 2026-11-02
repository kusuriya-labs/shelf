#pragma once
#include <phaseshift/runtime/program/program.h>
#include <hip/hip_runtime.h>
#include <cstddef>

namespace ps::execution {

struct HostExecutionContext;

void kv_calib_dump_maybe(
    const ::ps::runtime::Program& program,
    std::size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}
