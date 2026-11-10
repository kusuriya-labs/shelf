#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace shelf::kernel {
namespace {
using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));
// B: One 128-K group is accumulated exactly in int32, then scaled and
// accumulated in FP32. Weight is the reconstructed signed INT8 in the trellis
// n-major tile order (launch_exl3_mul24_recon_wmma), so one WMMA B fragment is
// a contiguous 256-byte read. Activation scales are NOT collapsed per row.
// Wave layout matches exl3_mul24_gemm_kernel: 16 output channels per wave,
// lanes 0..15/16..31 own the lower/upper 8 output rows respectively.
// Four waves per CTA, four 16-row tiles per wave for prefill throughput.
template <uint32_t MT>
__global__ void __launch_bounds__(128) exl3_mul24_prefill_wmma_kernel(
    const int8_t* __restrict__ codes,
    const float* __restrict__ scales,
    uint32_t code_stride,
    uint32_t scale_stride_floats,
    const int8_t* __restrict__ weights,
    float* __restrict__ yprime,
    uint32_t rows,
    uint32_t n,
    uint32_t k) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t col = lane & 15u;
    const uint32_t half = lane >> 4u;
    const uint32_t n0 = (blockIdx.x * 4u + wave) * 16u;
    const uint32_t nt = blockIdx.x * 4u + wave;
    const uint32_t m0 = blockIdx.y * (16u * MT);
    const uint32_t groups = k / 128u;

    float accum[MT][8];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) accum[mt][i] = 0.0f;

    for (uint32_t g = 0; g < groups; ++g) {
        int32x8_t acc[MT];
#pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) acc[mt][i] = 0;

#pragma unroll
        for (uint32_t s = 0; s < 8u; ++s) {
            const int32x2_t b = *reinterpret_cast<const int32x2_t*>(
                weights + (static_cast<size_t>(nt) * (k >> 4u) + g * 8u + s) * 256u +
                    static_cast<size_t>(lane) * 8u);
#pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt) {
                const uint32_t m = m0 + mt * 16u + col;
                int32x2_t a;
                if (m < rows) {
                    a = *reinterpret_cast<const int32x2_t*>(
                        codes + static_cast<size_t>(m) * code_stride + g * 128u + s * 16u + half * 8u);
                } else { a[0] = 0; a[1] = 0; }
                acc[mt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    true, a, true, b, acc[mt], false);
            }
        }
#pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const uint32_t m = m0 + mt * 16u + half * 8u + i;
                const float scale = m < rows ? scales[static_cast<size_t>(m) * scale_stride_floats + g] : 0.0f;
                accum[mt][i] = fmaf(static_cast<float>(acc[mt][i]), scale, accum[mt][i]);
            }
        }
    }
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) {
            const uint32_t m = m0 + mt * 16u + half * 8u + i;
            if (m < rows) {
                yprime[static_cast<size_t>(m) * n + n0 + col] =
                    accum[mt][i] * ::shelf::quantization::exl3::kExl3Int8GridStep;
            }
        }
    }
}
}  // namespace

hipError_t launch_exl3_mul24_prefill_wmma(
    const int8_t* codes, const float* scales,
    uint32_t code_stride, uint32_t scale_stride_bytes,
    const int8_t* dense_weights, float* yprime,
    uint32_t rows, uint32_t n, uint32_t k, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (!codes || !scales || !dense_weights || !yprime ||
        (n % 64u) != 0u || (k % 128u) != 0u ||
        (scale_stride_bytes % sizeof(float)) != 0u ||
        code_stride < k || scale_stride_bytes < (k / 128u) * sizeof(float))
        return hipErrorInvalidValue;
    constexpr uint32_t MT = 4u;
    exl3_mul24_prefill_wmma_kernel<MT><<<dim3(n / 64u, (rows + 16u * MT - 1u) / (16u * MT)), 128u, 0, stream>>>(
        codes, scales, code_stride, scale_stride_bytes / 4u, dense_weights, yprime, rows, n, k);
    return hipGetLastError();
}
}  // namespace shelf::kernel
