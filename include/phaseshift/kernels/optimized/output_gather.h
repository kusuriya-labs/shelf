#pragma once

#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/runtime/graph/value_type.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

hipError_t launch_output_gather(
    const void* input, uint32_t input_row_stride,
    void* output, uint32_t output_row_stride,
    const ::ps::runtime::DeviceBatchContext* ctx,
    uint32_t num_outputs,
    uint32_t features, ::ps::runtime::ValueDType dtype, hipStream_t stream);

}  // namespace ps::kernel

