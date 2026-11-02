#include <phaseshift/kernels/optimized/rmsnorm_quantize.h>
#include "detail/rmsnorm_device.h"
#include "detail/activation_quantize_device.h"
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreads = 256u;
constexpr uint32_t kK = 5120u;

template <uint32_t K>
__global__ __launch_bounds__(kThreads)
void fused_rmsnorm_quantize_kernel(
    const void* input, const void* weight, void* output, uint8_t* codes,
    float* scales, uint32_t, uint32_t, uint32_t group_size,
    uint32_t input_row_stride, uint32_t output_row_stride,
    uint32_t code_row_stride, uint32_t scale_row_stride, float eps) {
    const uint32_t row = blockIdx.x;
    detail::rmsnorm_row_body<RmsNormDataType::BF16, RmsNormDataType::BF16,
                             RmsNormDataType::BF16, RmsNormWeightLayout::PER_FEATURE,
                             RmsNormWeightMode::ONE_PLUS>(
        input, weight, output, row, 0u, group_size, input_row_stride,
        output_row_stride, eps);
    __syncthreads();
    detail::activation_quantize_e4m3_vec_row_body<K, K>(
        reinterpret_cast<const ::ps::bf16_t*>(output), codes, scales, row,
        output_row_stride, code_row_stride, scale_row_stride);
}

}

hipError_t launch_fused_rmsnorm_quantize(const FusedRmsNormQuantizeArgs& a,
                                         hipStream_t stream) {
    if (a.rows == 0u) return hipSuccess;
    if (a.input == nullptr || a.weight == nullptr || a.output == nullptr ||
        a.codes == nullptr || a.scales == nullptr) {
        return hipErrorInvalidValue;
    }
    if (a.features != kK || a.group_size != a.features) return hipErrorInvalidValue;
    if (a.input_row_stride < a.features || a.output_row_stride < a.features) {
        return hipErrorInvalidValue;
    }
    fused_rmsnorm_quantize_kernel<kK><<<a.rows, kThreads, 0, stream>>>(
        a.input, a.weight, a.output, a.codes, a.scales, a.rows, a.features,
        a.group_size, a.input_row_stride, a.output_row_stride, a.code_row_stride,
        a.scale_row_stride, a.eps);
    return hipGetLastError();
}

}
