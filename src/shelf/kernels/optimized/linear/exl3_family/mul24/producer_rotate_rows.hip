#include <shelf/kernels/optimized/linear/exl3_family/mul24/producer_rotate_rows.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_bf16.h>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;
__global__ void __launch_bounds__(128) swiglu_rotate_rows(
    Exl3SwigluProducer p, const uint16_t* suh, int8_t* codes, float* scales,
    uint32_t code_stride, uint32_t scale_stride) {
    __shared__ float smem[128];
    __shared__ float wave_peak[4];
    const uint32_t t = threadIdx.x, row = blockIdx.x, g = blockIdx.y, col = g * 128u + t;
    const float a = bf16_to_f32(p.gate[static_cast<size_t>(row) * p.gate_stride + col]);
    const float b = bf16_to_f32(p.up[static_cast<size_t>(row) * p.up_stride + col]);
    float activated = p.intermediate ? a / (1.0f + __expf(-a)) : a / (1.0f + expf(-a));
    if (p.intermediate) activated = __bfloat162float(__float2bfloat16(activated));
    const float product = __bfloat162float(__float2bfloat16(b * activated));
    float v = product * f16_bits_to_f32(suh[col]);
    v = fwht128_smem(v, smem, t) * kInvSqrt128;
    float peak = fabsf(v);
    for (uint32_t off = 16u; off > 0u; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off));
    if ((t & 31u) == 0u) wave_peak[t >> 5u] = peak;
    __syncthreads();
    peak = fmaxf(fmaxf(wave_peak[0], wave_peak[1]), fmaxf(wave_peak[2], wave_peak[3]));
    const float scale = peak / 127.0f;
    float q = 0.0f;
    if (scale > 0.0f) q = fminf(fmaxf(rintf(v / scale), -128.0f), 127.0f);
    codes[static_cast<size_t>(row) * code_stride + col] = static_cast<int8_t>(static_cast<int>(q));
    if (t == 0u) scales[static_cast<size_t>(row) * scale_stride + g] = scale;
}
__global__ void __launch_bounds__(128) gdn_rotate_rows(
    Exl3GdnProducer p, const uint16_t* suh, int8_t* codes, float* scales,
    uint32_t code_stride, uint32_t scale_stride) {
    __shared__ float smem[128];
    __shared__ float wave_peak[4];
    const uint32_t t = threadIdx.x, lane = t & 31u, row = blockIdx.x, g = blockIdx.y, col = g * 128u + t;
    const float* input = p.input + static_cast<size_t>(row) * p.input_stride + g * 128u;
    const bf16_t* gate = p.gate + static_cast<size_t>(row) * p.gate_stride;
    float sum = 0.f;
    for (uint32_t i = lane; i < 128u; i += 32u) {
        const float x = input[i];
        sum += x * x;
    }
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_xor(sum, offset, 32);
    const float inv = rsqrtf(sum / 128u + p.eps);
    const float z = bf16_to_f32(gate[col]);
    const float rounded = __bfloat162float(__float2bfloat16(input[t] * inv));
    const float norm = rounded * bf16_to_f32(p.weight[t]);
    const float product = __bfloat162float(__float2bfloat16(norm * (z / (1.f + expf(-z)))));
    float v = product * f16_bits_to_f32(suh[col]);
    v = fwht128_smem(v, smem, t) * kInvSqrt128;
    float peak = fabsf(v);
    for (uint32_t off = 16u; off > 0u; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off));
    if ((t & 31u) == 0u) wave_peak[t >> 5u] = peak;
    __syncthreads();
    peak = fmaxf(fmaxf(wave_peak[0], wave_peak[1]), fmaxf(wave_peak[2], wave_peak[3]));
    const float scale = peak / 127.0f;
    float q = 0.0f;
    if (scale > 0.0f) q = fminf(fmaxf(rintf(v / scale), -128.0f), 127.0f);
    codes[static_cast<size_t>(row) * code_stride + col] = static_cast<int8_t>(static_cast<int>(q));
    if (t == 0u) scales[static_cast<size_t>(row) * scale_stride + g] = scale;
}
}
hipError_t launch_exl3_gdn_rotate_rows(const Exl3GdnProducer& p, const uint16_t* suh,
    int8_t* codes, float* scales, uint32_t rows, uint32_t k, uint32_t code_stride,
    uint32_t scale_stride_bytes, hipStream_t stream) {
    if (!rows || !k || k % 128u || scale_stride_bytes % 4u) return hipErrorInvalidValue;
    gdn_rotate_rows<<<dim3(rows, k / 128u), 128, 0, stream>>>(p, suh, codes, scales, code_stride, scale_stride_bytes / 4u);
    return hipGetLastError();
}
hipError_t launch_exl3_swiglu_rotate_rows(const Exl3SwigluProducer& p, const uint16_t* suh,
    int8_t* codes, float* scales, uint32_t rows, uint32_t k, uint32_t code_stride,
    uint32_t scale_stride_bytes, hipStream_t stream) {
    if (!rows || !k || k % 128u || scale_stride_bytes % 4u) return hipErrorInvalidValue;
    swiglu_rotate_rows<<<dim3(rows, k / 128u), 128, 0, stream>>>(p, suh, codes, scales, code_stride, scale_stride_bytes / 4u);
    return hipGetLastError();
}
}
