#include <phaseshift/kernels/correctness/standalone/primitives_correctness.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace ps {
namespace kernel {

__device__ __forceinline__ ps::bf16_t bf16_pack(float f) {
    return ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(f))};
}

__device__ __forceinline__ float bf16_unpack(const ps::bf16_t& x) {
    return __bfloat162float(__ushort_as_bfloat16(x.data));
}
__global__ void swiglu_kernel(
    const ps::bf16_t* __restrict__ gate,
    const ps::bf16_t* __restrict__ up,
    ps::bf16_t* __restrict__ output,
    std::size_t n)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float gx = bf16_unpack(gate[idx]);
    float silu_g = gx / (1.0f + expf(-gx));
    float ux = bf16_unpack(up[idx]);
    output[idx] = bf16_pack(silu_g * ux);
}
Status swiglu(
    const gpu::Tensor& gate,
    const gpu::Tensor& up,
    gpu::Tensor& output,
    hipStream_t stream)
{
    if (!gate.same_shape(up)) {
        return Status::invalid_argument("gate and up shape mismatch", __FILE__, __LINE__);
    }

    if (!output.same_shape(gate)) {
        return Status::invalid_argument("output shape mismatch", __FILE__, __LINE__);
    }

    std::size_t total = 1;
    for (std::size_t i = 0; i < gate.ndim(); ++i) {
        total *= gate.dim(i);
    }

    const std::size_t threads_per_block = 256;
    const std::size_t blocks = (total + threads_per_block - 1) / threads_per_block;

    swiglu_kernel<<<static_cast<unsigned>(blocks), threads_per_block, 0, stream>>>(
        gate.data<ps::bf16_t>(),
        up.data<ps::bf16_t>(),
        output.data<ps::bf16_t>(),
        total);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("swiglu_kernel launch", hipGetErrorString(err), __FILE__, __LINE__);
    }

    return Status::make_ok();
}

} // namespace kernel
} // namespace ps
