#pragma once
#include <shelf/core/status.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/program.h>
#include <shelf/kernels/model_dispatch_state.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

Status launch_model_dispatch_correctness(
    const ::shelf::runtime::DeviceProgramView& program,
    const ::shelf::runtime::DispatchBinding& binding,
    ::shelf::runtime::DeviceBatchContext* batch_context,
    const ModelDispatchStateView& state,
    float* scratch,
    uint32_t* error_word,
    hipStream_t stream);

}
