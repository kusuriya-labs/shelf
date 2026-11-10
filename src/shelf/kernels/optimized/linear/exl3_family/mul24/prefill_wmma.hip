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
//
// Wave layout follows gemm_fp8_finegrained_w8a8_wmma_prefill_2d_kernel: 8 waves
// split into 4 row groups (wave_m) and 2 output groups (wave_n), so one CTA
// covers 4*MT*16 rows and 2*NT*16 outputs. A wave owns MT row tiles and NT
// output tiles; one activation fragment feeds NT WMMA calls and one weight
// fragment feeds MT calls, which cuts both re-read counts by the tile heights.
// Lanes 0..15 / 16..31 own the lower / upper 8 output rows of a tile.
template <uint32_t MT, uint32_t NT, uint32_t WM>
__global__ void __launch_bounds__(256) exl3_mul24_prefill_wmma_kernel(
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
    const uint32_t wave_m = wave % WM;
    const uint32_t wave_n = wave / WM;
    const uint32_t m0 = blockIdx.y * (WM * MT * 16u) + wave_m * (MT * 16u);
    const uint32_t n_base = blockIdx.x * ((8u / WM) * NT * 16u) + wave_n * (NT * 16u);
    const uint32_t n_tile = n_base >> 4u;
    const uint32_t groups = k / 128u;

    float accum[MT][NT][8];
    #pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (uint32_t nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (uint32_t i = 0; i < 8; ++i) accum[mt][nt][i] = 0.0f;

    for (uint32_t g = 0; g < groups; ++g) {
        int32x8_t acc[MT][NT];
        #pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt)
                #pragma unroll
                for (uint32_t i = 0; i < 8; ++i) acc[mt][nt][i] = 0;

        #pragma unroll
        for (uint32_t s = 0; s < 8u; ++s) {
            int32x2_t b[NT];
            #pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt) {
                b[nt] = *reinterpret_cast<const int32x2_t*>(
                    weights + (static_cast<size_t>(n_tile + nt) * (k >> 4u) + g * 8u + s) * 256u +
                        static_cast<size_t>(lane) * 8u);
            }
            #pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt) {
                const uint32_t m = m0 + mt * 16u + col;
                int32x2_t a;
                if (m < rows) {
                    a = *reinterpret_cast<const int32x2_t*>(
                        codes + static_cast<size_t>(m) * code_stride + g * 128u + s * 16u + half * 8u);
                } else { a[0] = 0; a[1] = 0; }
                #pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                        true, a, true, b[nt], acc[mt][nt], false);
            }
        }

        #pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt) {
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) {
                const uint32_t m = m0 + mt * 16u + half * 8u + i;
                const float scale = m < rows ? scales[static_cast<size_t>(m) * scale_stride_floats + g] : 0.0f;
                #pragma unroll
                for (uint32_t nt = 0; nt < NT; ++nt)
                    accum[mt][nt][i] = fmaf(static_cast<float>(acc[mt][nt][i]), scale, accum[mt][nt][i]);
            }
        }
    }

    #pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (uint32_t i = 0; i < 8u; ++i) {
            const uint32_t m = m0 + mt * 16u + half * 8u + i;
            if (m >= rows) continue;
            #pragma unroll
            for (uint32_t nt = 0; nt < NT; ++nt)
                yprime[static_cast<size_t>(m) * n + n_base + nt * 16u + col] =
                    accum[mt][nt][i] * ::shelf::quantization::exl3::kExl3Int8GridStep;
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
    constexpr uint32_t kMT = 2u;
    constexpr uint32_t kNT = 2u;
    constexpr uint32_t kWm = 4u;
    constexpr uint32_t kRowTile = kWm * kMT * 16u;
    constexpr uint32_t kOutTile = (8u / kWm) * kNT * 16u;
    exl3_mul24_prefill_wmma_kernel<kMT, kNT, kWm>
        <<<dim3(n / kOutTile, (rows + kRowTile - 1u) / kRowTile), 256u, 0, stream>>>(
            codes, scales, code_stride, scale_stride_bytes / 4u, dense_weights, yprime, rows, n, k);
    return hipGetLastError();
}
}  // namespace shelf::kernel
