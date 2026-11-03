#pragma once
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>
#include <cstddef>

namespace shelf::execution {

struct HostExecutionContext;

// Captures the BF16 activations that enter the FP8 activation quantizer, together
// with the weight slot of the linear that consumes them. Used to evaluate
// alternative activation quantizations offline against real activation data.
// Enabled by SHELF_LINEAR_INPUT_DUMP=<directory>.
void linear_input_dump_maybe(
    const ::shelf::runtime::Program& program,
    std::size_t dispatch_index,
    HostExecutionContext& ctx,
    hipStream_t stream);

}
