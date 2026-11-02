#include <shelf/kernels/optimized/swiglu_quantize.h>
#include "detail/target_swiglu_device.h"
#include "detail/activation_quantize_device.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kThreads = detail::kQuantThreadsPerRow;

template <uint32_t K, uint32_t KP>
__global__ __launch_bounds__(kThreads)
void fused_swiglu_quantize_kernel(
    const ::ps::bf16_t* __restrict__ gate, uint32_t gate_row_stride,
    const ::ps::bf16_t* __restrict__ up, uint32_t up_row_stride,
    ::ps::bf16_t* __restrict__ output, uint32_t output_row_stride,
    uint8_t* __restrict__ codes, float* __restrict__ scales,
    uint32_t code_row_stride, uint32_t scale_row_stride) {
    const uint32_t row = blockIdx.x;
    const __hip_bfloat16* gb = reinterpret_cast<const __hip_bfloat16*>(gate) +
                               static_cast<size_t>(row) * gate_row_stride;
    const __hip_bfloat16* ub = reinterpret_cast<const __hip_bfloat16*>(up) +
                               static_cast<size_t>(row) * up_row_stride;
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(output) +
                         static_cast<size_t>(row) * output_row_stride;
    for (uint32_t c = threadIdx.x; c < K; c += kThreads) {
        const float gv = __bfloat162float(gb[c]);
        const float uv = __bfloat162float(ub[c]);
        ob[c] = __float2bfloat16(detail::target_swiglu_f32(gv, uv));
    }
    __syncthreads();
    detail::activation_quantize_fp8_finegrained_vec_row_body<K, KP>(
        reinterpret_cast<const ::ps::bf16_t*>(output), codes, scales, row,
        output_row_stride, code_row_stride, scale_row_stride);
}

}

hipError_t launch_fused_swiglu_quantize(const FusedSwiGluQuantizeArgs& a,
                                        hipStream_t stream) {
    if (a.rows == 0u) return hipSuccess;
    if (a.gate == nullptr || a.up == nullptr || a.output == nullptr ||
        a.codes == nullptr || a.scales == nullptr)
        return hipErrorInvalidValue;
    if (a.k == 0u || a.gate_row_stride < a.k || a.up_row_stride < a.k ||
        a.output_row_stride < a.k)
        return hipErrorInvalidValue;
    const ::ps::bf16_t* gate = static_cast<const ::ps::bf16_t*>(a.gate);
    const ::ps::bf16_t* up = static_cast<const ::ps::bf16_t*>(a.up);
    ::ps::bf16_t* output = static_cast<::ps::bf16_t*>(a.output);
    switch (a.k) {
        case 5120u:
            fused_swiglu_quantize_kernel<5120u, 5120u>
                <<<a.rows, kThreads, 0, stream>>>(
                    gate, a.gate_row_stride, up, a.up_row_stride, output,
                    a.output_row_stride, a.codes, a.scales, a.code_row_stride,
                    a.scale_row_stride);
            break;
        case 6144u:
            fused_swiglu_quantize_kernel<6144u, 6144u>
                <<<a.rows, kThreads, 0, stream>>>(
                    gate, a.gate_row_stride, up, a.up_row_stride, output,
                    a.output_row_stride, a.codes, a.scales, a.code_row_stride,
                    a.scale_row_stride);
            break;
        case 17408u:
            fused_swiglu_quantize_kernel<17408u, 17408u>
                <<<a.rows, kThreads, 0, stream>>>(
                    gate, a.gate_row_stride, up, a.up_row_stride, output,
                    a.output_row_stride, a.codes, a.scales, a.code_row_stride,
                    a.scale_row_stride);
            break;
        case 8704u:
            fused_swiglu_quantize_kernel<8704u, 8704u>
                <<<a.rows, kThreads, 0, stream>>>(
                    gate, a.gate_row_stride, up, a.up_row_stride, output,
                    a.output_row_stride, a.codes, a.scales, a.code_row_stride,
                    a.scale_row_stride);
            break;
        case 3072u:
            fused_swiglu_quantize_kernel<3072u, 3072u>
                <<<a.rows, kThreads, 0, stream>>>(
                    gate, a.gate_row_stride, up, a.up_row_stride, output,
                    a.output_row_stride, a.codes, a.scales, a.code_row_stride,
                    a.scale_row_stride);
            break;
        default:
            return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

}
