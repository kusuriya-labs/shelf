#pragma once
#include <shelf/core/memory/types.h>
#include <shelf/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::execution {
struct FinalProducer {
    const ::shelf::bf16_t* input;
    const ::shelf::bf16_t* weight;
    ::shelf::bf16_t* gathered;
    const ::shelf::runtime::DeviceBatchContext* batch;
    uint32_t input_stride;
    uint32_t gathered_stride;
    float eps;
};
hipError_t launch_final_gather_norm_rotate(const FinalProducer&, const uint16_t*, int8_t*, float*,
                                          uint32_t, uint32_t, uint32_t, hipStream_t);
}
