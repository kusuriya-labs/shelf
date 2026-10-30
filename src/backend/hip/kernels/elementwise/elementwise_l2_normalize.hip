// vendored from PhaseShift (own work); namespace/type adapted only.
#include <shelf/backend/hip/kernels/elementwise/l2_normalize.h>
#include <shelf/backend/hip/kernels/detail/row_grid.h>
#include <shelf/backend/hip/kernels/detail/l2_normalize_device.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <algorithm>
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
    // 行方向は grid.y 上限（65535）で分割発射する（基底ポインタを進める）。
    for (uint32_t begin = 0u; begin < rows; begin += detail::kMaxGridY) {
        L2NormalizeArgs chunk = args;
        const uint32_t n = std::min(rows - begin, detail::kMaxGridY);
        chunk.input = static_cast<const bf16*>(args.input) +
                      static_cast<size_t>(begin) * args.input_row_stride;
        chunk.output = static_cast<float*>(args.output) +
                       static_cast<size_t>(begin) * args.output_row_stride;
        phaseshift_l2_normalize_bf16_f32<<<dim3(groups, n), kL2Threads, 0, stream>>>(chunk);
        const hipError_t e = hipGetLastError();
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}

}  // namespace shelf::hipk
