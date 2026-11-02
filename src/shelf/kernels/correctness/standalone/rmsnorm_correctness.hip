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
__global__ void rmsnorm_kernel(
    const shelf::bf16_t* __restrict__ input,
    const shelf::bf16_t* __restrict__ weight,
    shelf::bf16_t* __restrict__ output,
    std::size_t seq_len,
    std::size_t hidden,
    double eps)
{
    std::size_t row = static_cast<std::size_t>(blockIdx.x);
    if (row >= seq_len) return;

    const shelf::bf16_t* __restrict__ inp = input + row * hidden;
    const shelf::bf16_t* __restrict__ wgt = weight;
    shelf::bf16_t* __restrict__ out = output + row * hidden;

    extern __shared__ float shared_mem[];

    float* shared_vals = shared_mem;
    float* sum_ptr = shared_mem + hidden;

    std::size_t tid = static_cast<std::size_t>(threadIdx.x);
    std::size_t n_threads = static_cast<std::size_t>(blockDim.x);

    for (std::size_t i = tid; i < hidden; i += n_threads) {
        shared_vals[i] = bf16_unpack(inp[i]);
    }
    __syncthreads();

    float sum_sq = 0.0f;
    for (std::size_t i = tid; i < hidden; i += n_threads) {
        float v = shared_vals[i];
        sum_sq += v * v;
    }

    if (tid == 0) sum_ptr[0] = 0.0f;
    __syncthreads();

    atomicAdd(sum_ptr, sum_sq);
    __syncthreads();

    float total_sum = sum_ptr[0];
    float mean_sq = total_sum / (float)hidden;
    float rs = 1.0f / sqrtf(mean_sq + eps);

    for (std::size_t i = tid; i < hidden; i += n_threads) {
        float v = shared_vals[i];
        float w = bf16_unpack(wgt[i]);
        out[i] = bf16_pack(v * rs * w);
    }
}
Status rmsnorm(
    const gpu::Tensor& input,
    const gpu::Tensor& weight,
    gpu::Tensor& output,
    double eps,
    hipStream_t stream)
{
    if (input.ndim() != 2) {
        return Status::invalid_argument("input must be 2D", __FILE__, __LINE__);
    }

    std::size_t seq_len = input.dim(0);
    std::size_t hidden = input.dim(1);

    if (weight.ndim() != 1 || weight.dim(0) != hidden) {
        return Status::invalid_argument("weight shape mismatch", __FILE__, __LINE__);
    }

    if (!output.same_shape(input)) {
        return Status::invalid_argument("output shape mismatch", __FILE__, __LINE__);
    }

    const std::size_t threads_per_block = 256;
    const std::size_t shared_mem_size = (hidden + 1) * sizeof(float);

    rmsnorm_kernel<<<static_cast<unsigned>(seq_len), threads_per_block, shared_mem_size, stream>>>(
        input.data<shelf::bf16_t>(),
        weight.data<shelf::bf16_t>(),
        output.data<shelf::bf16_t>(),
        seq_len,
        hidden,
        eps);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("rmsnorm_kernel launch", hipGetErrorString(err), __FILE__, __LINE__);
    }

    return Status::make_ok();
}

} // namespace kernel
} // namespace shelf
