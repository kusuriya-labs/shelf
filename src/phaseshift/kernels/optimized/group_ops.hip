#include <phaseshift/kernels/optimized/group_ops.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreads = kGroupOpsBlock;

__global__ void __launch_bounds__(kThreads)
group_mean_f32_impl(const float* __restrict__ input, uint32_t input_row_stride,
                    float* __restrict__ output, uint32_t output_row_stride,
                    uint32_t features, uint32_t group) {
    const uint32_t row = blockIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= features) return;
    const size_t base = static_cast<size_t>(row) * input_row_stride + col;
    float acc = 0.0f;
    for (uint32_t g = 0; g < group; ++g) {
        acc += input[base + static_cast<size_t>(g) * features];
    }
    output[static_cast<size_t>(row) * output_row_stride + col] =
        acc / static_cast<float>(group);
}

__global__ void __launch_bounds__(kThreads)
group_broadcast_mul_f32_impl(const float* __restrict__ a, uint32_t a_row_stride,
                             const float* __restrict__ b, uint32_t b_row_stride,
                             float* __restrict__ output, uint32_t output_row_stride,
                             uint32_t features, uint32_t group) {
    const uint32_t row = blockIdx.y;
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t out_features = features * group;
    if (j >= out_features) return;
    const uint32_t g = j / features;
    const uint32_t h = j % features;
    const float av = a[static_cast<size_t>(row) * a_row_stride + h];
    const float bv = b[static_cast<size_t>(row) * b_row_stride + g];
    output[static_cast<size_t>(row) * output_row_stride + j] = av * bv;
}

}  // namespace

hipError_t launch_group_mean_f32(
    const float* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, uint32_t group,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr) return hipErrorInvalidValue;
    if (rows == 0u || features == 0u || group == 0u) return hipErrorInvalidValue;
    if (input_row_stride < features * group || output_row_stride < features)
        return hipErrorInvalidValue;
    const uint32_t gx = (features + kThreads - 1u) / kThreads;
    dim3 grid(gx, rows);
    group_mean_f32_impl<<<grid, kThreads, 0, stream>>>(
        input, input_row_stride, output, output_row_stride, features, group);
    return hipGetLastError();
}

hipError_t launch_group_broadcast_mul_f32(
    const float* a, uint32_t a_row_stride,
    const float* b, uint32_t b_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, uint32_t group,
    hipStream_t stream) {
    if (a == nullptr || b == nullptr || output == nullptr) return hipErrorInvalidValue;
    if (rows == 0u || features == 0u || group == 0u) return hipErrorInvalidValue;
    if (a_row_stride < features || b_row_stride < group ||
        output_row_stride < features * group)
        return hipErrorInvalidValue;
    const uint32_t out_features = features * group;
    const uint32_t gx = (out_features + kThreads - 1u) / kThreads;
    dim3 grid(gx, rows);
    group_broadcast_mul_f32_impl<<<grid, kThreads, 0, stream>>>(
        a, a_row_stride, b, b_row_stride, output, output_row_stride, features, group);
    return hipGetLastError();
}

}
