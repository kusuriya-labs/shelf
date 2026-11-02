#pragma once
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/program/device_program.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/kernels/model_dispatch_state.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

Status launch_model_dispatch_correctness(
    const ::ps::runtime::DeviceProgramView& program,
    const ::ps::runtime::DispatchBinding& binding,
    ::ps::runtime::DeviceBatchContext* batch_context,
    const ModelDispatchStateView& state,
    float* scratch,
    uint32_t* error_word,
    hipStream_t stream);

}
