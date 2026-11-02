#include <shelf/kernels/correctness/standalone/primitives_correctness.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace shelf {
namespace kernel {

__device__ __forceinline__ shelf::bf16_t bf16_pack(float f) {
    return shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(f))};
}

__device__ __forceinline__ float bf16_unpack(const shelf::bf16_t& x) {
    return __bfloat162float(__ushort_as_bfloat16(x.data));
}
__global__ void silu_kernel(
    const shelf::bf16_t* __restrict__ input,
    shelf::bf16_t* __restrict__ output,
    std::size_t n)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = bf16_unpack(input[idx]);
    float y = x / (1.0f + expf(-x));
    output[idx] = bf16_pack(y);
}
Status silu(
    const gpu::Tensor& input,
    gpu::Tensor& output,
    hipStream_t stream)
{
    if (!output.same_shape(input)) {
        return Status::invalid_argument("output shape mismatch", __FILE__, __LINE__);
    }

    std::size_t total = 1;
    for (std::size_t i = 0; i < input.ndim(); ++i) {
        total *= input.dim(i);
    }

    const std::size_t threads_per_block = 256;
    const std::size_t blocks = (total + threads_per_block - 1) / threads_per_block;

    silu_kernel<<<static_cast<unsigned>(blocks), threads_per_block, 0, stream>>>(
        input.data<shelf::bf16_t>(),
        output.data<shelf::bf16_t>(),
        total);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("silu_kernel launch", hipGetErrorString(err), __FILE__, __LINE__);
    }

    return Status::make_ok();
}

} // namespace kernel
} // namespace shelf
