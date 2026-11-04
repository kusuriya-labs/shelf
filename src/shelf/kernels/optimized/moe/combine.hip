#include <shelf/kernels/optimized/moe/combine.h>
#include <shelf/kernels/optimized/moe/router.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>

namespace shelf::kernel {
namespace {

__device__ __forceinline__ float bf16_f32(shelf::bf16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value.data) << 16u);
}

constexpr uint32_t kCombineThreads = 128u;
constexpr uint32_t kCombineMaxTopK = kMoeRouterMaxTopK;

__global__ void __launch_bounds__(kCombineThreads) moe_combine_kernel(
    shelf::bf16_t* __restrict__ output,
    const shelf::bf16_t* __restrict__ expert_output,
    const float* __restrict__ sel_weight,
    const int32_t* __restrict__ slot_row_of,
    const shelf::bf16_t* __restrict__ shared_output,
    const float* __restrict__ shared_gate_weight,
    uint32_t rows,
    uint32_t top_k,
    uint32_t features,
    uint32_t expert_row_stride,
    uint32_t shared_row_stride) {
    const uint32_t t = blockIdx.x;
    if (t >= rows)
        return;
    const uint32_t tid = threadIdx.x;
    const uint32_t f = blockIdx.y * kCombineThreads + tid;

    __shared__ int32_t s_row[kCombineMaxTopK];
    __shared__ float s_weight[kCombineMaxTopK];
    if (tid < top_k) {
        const uint32_t at = static_cast<size_t>(t) * top_k + tid;
        s_row[tid] = slot_row_of[at];
        s_weight[tid] = sel_weight[at];
    }
    __syncthreads();

    if (tid == 0u) {
        for (uint32_t i = 1u; i < top_k; ++i) {
            const int32_t key_row = s_row[i];
            const float key_weight = s_weight[i];
            int32_t j = static_cast<int32_t>(i) - 1;
            while (j >= 0 && s_row[j] > key_row) {
                s_row[j + 1] = s_row[j];
                s_weight[j + 1] = s_weight[j];
                --j;
            }
            s_row[j + 1] = key_row;
            s_weight[j + 1] = key_weight;
        }
    }
    __syncthreads();

    float gate = 0.0f;
    if (shared_output != nullptr)
        gate = shared_gate_weight[t];

    const size_t out_base = static_cast<size_t>(t) * features;
    const size_t shared_base = static_cast<size_t>(t) * shared_row_stride;
    if (f >= features)
        return;
    float acc = 0.0f;
    for (uint32_t j = 0; j < top_k; ++j) {
        const int32_t row = s_row[j];
        if (row < 0)
            continue;
        const float value =
            bf16_f32(expert_output[static_cast<size_t>(row) * expert_row_stride + f]);
        acc = __fmaf_rn(s_weight[j], value, acc);
    }
    if (shared_output != nullptr) {
        const float shared_value = bf16_f32(shared_output[shared_base + f]);
        acc = __fmaf_rn(gate, shared_value, acc);
    }
    output[out_base + f] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(acc))};
}

}  // namespace

hipError_t launch_moe_combine(
    shelf::bf16_t* output,
    const shelf::bf16_t* expert_output,
    const float* sel_weight,
    const int32_t* slot_row_of,
    const shelf::bf16_t* shared_output,
    const float* shared_gate_weight,
    uint32_t rows,
    uint32_t top_k,
    uint32_t features,
    uint32_t expert_row_stride,
    uint32_t shared_row_stride,
    hipStream_t stream) {
    if (output == nullptr || expert_output == nullptr || sel_weight == nullptr ||
        slot_row_of == nullptr)
        return hipErrorInvalidValue;
    if ((shared_output == nullptr) != (shared_gate_weight == nullptr))
        return hipErrorInvalidValue;
    if (rows == 0u || features == 0u)
        return hipErrorInvalidValue;
    if (top_k == 0u || top_k > kCombineMaxTopK)
        return hipErrorInvalidValue;
    if (expert_row_stride < features)
        return hipErrorInvalidValue;
    if (shared_output != nullptr && shared_row_stride < features)
        return hipErrorInvalidValue;
    const dim3 grid(rows, (features + kCombineThreads - 1u) / kCombineThreads);
    moe_combine_kernel<<<grid, kCombineThreads, 0, stream>>>(
        output, expert_output, sel_weight, slot_row_of, shared_output, shared_gate_weight, rows, top_k,
        features, expert_row_stride, shared_row_stride);
    return hipGetLastError();
}

}  // namespace shelf::kernel
