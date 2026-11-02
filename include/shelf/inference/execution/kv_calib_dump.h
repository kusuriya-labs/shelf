#pragma once
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>
#include <cstddef>

namespace shelf::execution {

struct HostExecutionContext;

void kv_calib_dump_maybe(
    const ::shelf::runtime::Program& program,
    std::size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}
