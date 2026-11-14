#include <shelf/inference/execution/final_producer.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include "../../kernels/optimized/detail/rmsnorm_device.h"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace shelf::execution {
namespace {
using bf16 = __hip_bfloat16;
using namespace ::shelf::kernel::exl3_detail;
using namespace ::shelf::kernel::exl3_detail::mul24;

__global__ void final_gather_norm_rotate(FinalProducer p, const uint16_t* suh,
                                         int8_t* codes, float* scales,
                                         uint32_t code_stride, uint32_t scale_stride) {
    __shared__ float sums[8];
    __shared__ float smem[256];
    __shared__ float peaks[8];
    const uint32_t row = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const uint32_t lane = tid & 31u, wave = tid >> 5u;
    const auto* input = reinterpret_cast<const bf16*>(p.input) +
        static_cast<size_t>(p.batch->output_rows[row]) * p.input_stride;
    if (g == 0u) {
        auto* gathered = reinterpret_cast<bf16*>(p.gathered) + static_cast<size_t>(row) * p.gathered_stride;
        for (uint32_t i = tid; i < 5120u; i += 256u) gathered[i] = input[i];
    }
    float sum0 = 0.f, sum1 = 0.f;
    for (uint32_t base = tid * 4u; base < 5120u; base += 1024u) {
        const auto v = ::shelf::kernel::detail::rms_load_bf16_x4(input + base);
        sum0 += v.x0 * v.x0; sum1 += v.x1 * v.x1;
        sum0 += v.x2 * v.x2; sum1 += v.x3 * v.x3;
    }
    float sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum0 + sum1);
    if (lane == 0u) sums[wave] = sum;
    __syncthreads();
    if (wave == 0u) {
        sum = lane < 8u ? sums[lane] : 0.f;
        sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum);
        if (lane == 0u) sums[0] = rsqrtf(sum / 5120.f + p.eps);
    }
    __syncthreads();
    const uint32_t t = tid & 127u, col = g * 128u + t;
    const auto* weight = reinterpret_cast<const bf16*>(p.weight);
    float norm = __bfloat162float(input[col]) * sums[0];
    norm *= 1.f + __bfloat162float(weight[col]);
    const bf16 rounded = __float2bfloat16(norm);
    float v = __bfloat162float(rounded) * f16_bits_to_f32(suh[col]);
    v = fwht128_smem(v, smem + (tid / 128u) * 128u, t) * kInvSqrt128;
    float peak = fabsf(v);
    for (uint32_t off = 16u; off; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off));
    if (lane == 0u) peaks[wave] = peak;
    __syncthreads();
    const uint32_t base = (tid / 128u) * 4u;
    peak = fmaxf(fmaxf(peaks[base], peaks[base + 1u]), fmaxf(peaks[base + 2u], peaks[base + 3u]));
    const float scale = peak / 127.f;
    const float q = scale > 0.f ? fminf(fmaxf(rintf(v / scale), -128.f), 127.f) : 0.f;
    if (tid < 128u) {
        codes[static_cast<size_t>(row) * code_stride + col] = static_cast<int8_t>(static_cast<int>(q));
        if (tid == 0u) scales[static_cast<size_t>(row) * scale_stride + g] = scale;
    }
}
}
hipError_t launch_final_gather_norm_rotate(const FinalProducer& producer, const uint16_t* suh,
                                          int8_t* codes, float* scales, uint32_t rows,
                                          uint32_t code_stride, uint32_t scale_stride_bytes, hipStream_t stream) {
    final_gather_norm_rotate<<<dim3(rows, 40u), 256, 0, stream>>>(
        producer, suh, codes, scales, code_stride, scale_stride_bytes / 4u);
    return hipGetLastError();
}
}
