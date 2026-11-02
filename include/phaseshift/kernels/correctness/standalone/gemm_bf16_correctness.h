#pragma once

#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps {
namespace kernel {
namespace detail {

hipError_t launch_gemm_bf16_baseline(
    const bf16_t* weight,
    const bf16_t* input,
    bf16_t* output,
    uint32_t out_features,
    uint32_t actual_rows,
    uint32_t k,
    hipStream_t stream);

hipError_t launch_gemm_bf16_baseline_graph_ready(
    const bf16_t* weight,
    const bf16_t* input,
    bf16_t* output,
    uint32_t output_features,
    uint32_t bucket_rows,
    uint32_t k,
    const ps::runtime::DeviceBatchContext* ctx,
    hipStream_t stream);

}
}
}
