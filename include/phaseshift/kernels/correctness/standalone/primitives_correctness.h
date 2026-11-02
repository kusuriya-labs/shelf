#pragma once

#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/status.h>
#include <hip/hip_runtime.h>

namespace ps {
namespace kernel {

Status rmsnorm(
    const gpu::Tensor& input,
    const gpu::Tensor& weight,
    gpu::Tensor& output,
    double eps,
    hipStream_t stream);

Status silu(
    const gpu::Tensor& input,
    gpu::Tensor& output,
    hipStream_t stream);

Status swiglu(
    const gpu::Tensor& gate,
    const gpu::Tensor& up,
    gpu::Tensor& output,
    hipStream_t stream);

}
}
