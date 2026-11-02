#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <shelf/runtime/graph/value_type.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_output_gather(
    const void* input, uint32_t input_row_stride,
    void* output, uint32_t output_row_stride,
    const ::shelf::runtime::DeviceBatchContext* ctx,
    uint32_t num_outputs,
    uint32_t features, ::shelf::runtime::ValueDType dtype, hipStream_t stream);

}  // namespace shelf::kernel

