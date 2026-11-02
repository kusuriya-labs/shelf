#include <phaseshift/kernels/correctness/standalone/gemm_bf16_correctness.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstddef>
#include <cstdint>

namespace ps {
namespace kernel {
namespace detail {

__device__ __forceinline__ float gemm_bf16_unpack(const bf16_t& x) {
    return __bfloat162float(__ushort_as_bfloat16(x.data));
}

__device__ __forceinline__ bf16_t gemm_bf16_pack(float x) {
    return bf16_t{__bfloat16_as_ushort(__float2bfloat16(x))};
}
__global__ void gemm_bf16_baseline_kernel(
    const bf16_t* __restrict__ weight,
    const bf16_t* __restrict__ input,
    bf16_t* __restrict__ output,
    uint32_t out_features,
    uint32_t actual_rows,
    uint32_t k)
{
    uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= out_features || t >= actual_rows) return;

    const bf16_t* wrow = weight + static_cast<std::size_t>(m) * k;
    const bf16_t* xt = input + static_cast<std::size_t>(t) * k;

    float acc = 0.0f;
    for (uint32_t ki = 0; ki < k; ++ki) {
        acc = fmaf(gemm_bf16_unpack(wrow[ki]), gemm_bf16_unpack(xt[ki]), acc);
    }
    output[static_cast<std::size_t>(t) * out_features + m] = gemm_bf16_pack(acc);
}

hipError_t launch_gemm_bf16_baseline(
    const bf16_t* weight,
    const bf16_t* input,
    bf16_t* output,
    uint32_t out_features,
    uint32_t actual_rows,
    uint32_t k,
    hipStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x, (actual_rows + block.y - 1) / block.y);

    gemm_bf16_baseline_kernel<<<grid, block, 0, stream>>>(
        weight, input, output, out_features, actual_rows, k);

    return hipGetLastError();
}

__global__ void gemm_bf16_baseline_graph_ready_kernel(
    const bf16_t* __restrict__ weight,
    const bf16_t* __restrict__ input,
    bf16_t* __restrict__ output,
    uint32_t output_features,
    uint32_t bucket_rows,
    uint32_t k,
    const ps::runtime::DeviceBatchContext* __restrict__ ctx)
{
    uint32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= output_features) return;
    uint32_t actual_rows = ctx->actual_rows;
    if (t >= actual_rows) return;
    if (t >= bucket_rows) return;
    const bf16_t* wrow = weight + static_cast<std::size_t>(m) * k;
    const bf16_t* xt = input + static_cast<std::size_t>(t) * k;
    float acc = 0.0f;
    for (uint32_t ki = 0; ki < k; ++ki) {
        acc = fmaf(gemm_bf16_unpack(wrow[ki]), gemm_bf16_unpack(xt[ki]), acc);
    }
    output[static_cast<std::size_t>(t) * output_features + m] = gemm_bf16_pack(acc);
}

hipError_t launch_gemm_bf16_baseline_graph_ready(
    const bf16_t* weight,
    const bf16_t* input,
    bf16_t* output,
    uint32_t output_features,
    uint32_t bucket_rows,
    uint32_t k,
    const ps::runtime::DeviceBatchContext* ctx,
    hipStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((output_features + block.x - 1) / block.x, (bucket_rows + block.y - 1) / block.y);
    gemm_bf16_baseline_graph_ready_kernel<<<grid, block, 0, stream>>>(
        weight, input, output, output_features, bucket_rows, k, ctx);
    return hipGetLastError();
}

}
}
}
