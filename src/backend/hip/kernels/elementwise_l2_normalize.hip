// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/l2_normalize.h>
#include <shelf/backend/hip/kernels/detail/l2_normalize_device.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk {

namespace {

using bf16 = __hip_bfloat16;

constexpr uint32_t kL2Threads = detail::kL2Threads;
static_assert(kL2Threads == 32u);
static_assert(kL2Threads == kL2NormalizeThreads);
constexpr uint32_t kL2ElemsPerLane = detail::kL2ElemsPerLane;

__device__ __forceinline__ void l2_normalize_bf16_f32_body(
    const bf16* __restrict__ input, uint32_t input_row_stride,
    float* __restrict__ output, uint32_t output_row_stride,
    uint32_t group_size, float eps) {
    const uint32_t row = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    const size_t in_off = static_cast<size_t>(row) * input_row_stride
                        + static_cast<size_t>(blockIdx.x) * group_size;
    const size_t out_off = static_cast<size_t>(row) * output_row_stride
                         + static_cast<size_t>(blockIdx.x) * group_size;
    const bool aligned = (in_off & 3u) == 0u;

    detail::l2_normalize_head_body<false>(
        input + in_off, output + out_off, nullptr, 0.0f, lane, group_size, eps, aligned);
}

}  // namespace

extern "C" __global__ __launch_bounds__(32)
void phaseshift_l2_normalize_bf16_f32(L2NormalizeArgs args) {
    l2_normalize_bf16_f32_body(static_cast<const bf16*>(args.input),
                               args.input_row_stride,
                               static_cast<float*>(args.output),
                               args.output_row_stride, args.group_size, args.eps);
}

hipError_t launch_l2_normalize_bf16_f32(
    const bf16_t* input, uint32_t input_row_stride,
    float* output, uint32_t output_row_stride,
    uint32_t rows, uint32_t features, uint32_t group_size,
    float eps, hipStream_t stream) {
    if (input == nullptr || output == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || features == 0u || group_size == 0u)
        return hipErrorInvalidValue;
    if (features % group_size != 0u)
        return hipErrorInvalidValue;
    const uint32_t groups = features / group_size;
    L2NormalizeArgs args{};
    args.input = input;
    args.output = output;
    args.input_row_stride = input_row_stride;
    args.output_row_stride = output_row_stride;
    args.group_size = group_size;
    args.eps = eps;
    dim3 grid(groups, rows);
    phaseshift_l2_normalize_bf16_f32<<<grid, kL2Threads, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::hipk
