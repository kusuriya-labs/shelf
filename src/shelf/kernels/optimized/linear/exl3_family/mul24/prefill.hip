#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul1/decode.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;

__global__ void __launch_bounds__(256) exl3_mul24_rowstat_kernel(
    const float* __restrict__ scales,
    float* __restrict__ rowscale,
    int32_t* __restrict__ slots,
    const uint32_t scale_row_stride_floats,
    const uint32_t groups,
    const float share) {
    __shared__ float red[2][8];
    const uint32_t m = blockIdx.x;
    const uint32_t t = threadIdx.x;
    float peak = 0.0f, sq = 0.0f;
    for (uint32_t g = t; g < groups; g += 256u) {
        const float sg = scales[static_cast<size_t>(m) * scale_row_stride_floats + g];
        peak = fmaxf(peak, sg);
        sq += sg * sg;
    }
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        peak = fmaxf(peak, __shfl_xor(peak, off));
        sq += __shfl_xor(sq, off);
    }
    if ((t & 31u) == 0u) { red[0][t >> 5u] = peak; red[1][t >> 5u] = sq; }
    __syncthreads();
    if (t == 0u) {
        float pk = red[0][0], sm = red[1][0];
        for (uint32_t i = 1u; i < 8u; ++i) { pk = fmaxf(pk, red[0][i]); sm += red[1][i]; }
        rowscale[m] = pk;
        slots[m] = (pk > 0.0f && pk * pk > share * sm) ? 1 : -1;
    }
}

__global__ void __launch_bounds__(1024) exl3_mul24_rowslot_kernel(
    int32_t* __restrict__ slots,
    uint32_t* __restrict__ count,
    const uint32_t rows,
    const uint32_t cap) {
    __shared__ uint32_t partial[1024];
    const uint32_t t = threadIdx.x;
    const uint32_t per = (rows + 1023u) / 1024u;
    const uint32_t lo = min(rows, t * per), hi = min(rows, lo + per);
    uint32_t c = 0u;
    for (uint32_t m = lo; m < hi; ++m) c += slots[m] >= 0 ? 1u : 0u;
    partial[t] = c;
    __syncthreads();
    if (t == 0u) {
        uint32_t run = 0u;
        for (uint32_t i = 0u; i < 1024u; ++i) { const uint32_t v = partial[i]; partial[i] = run; run += v; }
        count[0] = min(run, cap);
    }
    __syncthreads();
    uint32_t next = partial[t];
    for (uint32_t m = lo; m < hi; ++m) {
        if (slots[m] < 0) continue;
        slots[m] = next < cap ? static_cast<int32_t>(next) : -1;
        ++next;
    }
}

__global__ void __launch_bounds__(256) exl3_mul24_rowfinish_kernel(
    int8_t* __restrict__ codes,
    const float* __restrict__ scales,
    const float* __restrict__ rowscale,
    const int32_t* __restrict__ slots,
    int8_t* __restrict__ codes_f,
    float* __restrict__ scales_f,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride_floats,
    const uint32_t k) {
    const uint32_t m = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const int32_t slot = slots[m];
    int8_t* row = codes + static_cast<size_t>(m) * code_row_stride;
    const float* srow = scales + static_cast<size_t>(m) * scale_row_stride_floats;
    if (slot >= 0) {
        int8_t* dst = codes_f + static_cast<size_t>(slot) * code_row_stride;
        for (uint32_t i = t; i < k; i += 256u) dst[i] = row[i];
        float* sdst = scales_f + static_cast<size_t>(slot) * scale_row_stride_floats;
        for (uint32_t g = t; g < k / kBlock; g += 256u) sdst[g] = srow[g];
        return;
    }
    const float peak = rowscale[m];
    if (peak <= 0.0f) return;
    for (uint32_t i = t; i < k; i += 256u) {
        const float ratio = srow[i / kBlock] / peak;
        if (ratio == 1.0f) continue;
        row[i] = static_cast<int8_t>(static_cast<int>(rintf(static_cast<float>(row[i]) * ratio)));
    }
}

// trellis -> dense int8 weights. kTileMajor keeps the trellis layout (one 256-byte 16x16 tile per
// (n tile, k tile), lane l owning its eight weights), which makes a WMMA B fragment one contiguous
// 256-byte read; otherwise the output is [n][k] (k contiguous) for a hipBLASLt int8 GEMM with A
// transposed. One wave per 16x16 tile, the same lane / window / decode as the gemm kernel: lane l
// holds column n = l % 16 and the eight k rows 8 * (l / 16) .. +7 of the tile.
template <int BITS, bool kTileMajor>
__global__ void __launch_bounds__(256) exl3_mul24_recon_kernel(
    const uint32_t* __restrict__ trellis,
    int8_t* __restrict__ dense,
    const uint32_t k) {
    constexpr uint32_t kWords = 8u * BITS;
    constexpr uint32_t kBits = 256u * BITS;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t kt = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t nt = blockIdx.y;
    const uint32_t base_bit = (lane * 8u * BITS + BITS + kBits - 16u) % kBits;
    const uint32_t off = base_bit & 31u;
    const uint32_t i0 = base_bit >> 5u;
    const uint32_t* tile = trellis + (static_cast<size_t>(nt) * (k >> 4u) + kt) * kWords;
    const uint32_t w0 = tile[i0];
    const uint32_t w1 = tile[(i0 + 1u) % kWords];
    const uint32_t w2 = tile[(i0 + 2u) % kWords];
    uint32_t packed[2] = {0u, 0u};
    #pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        const uint32_t e = off + i * BITS + 16u;
        const uint32_t wl = (e - 1u) >> 5u;
        const uint32_t r = 32u - (e - 32u * wl);
        const uint32_t lo = wl == 0u ? w0 : (wl == 1u ? w1 : w2);
        const uint32_t hi = wl == 0u ? 0u : (wl == 1u ? w0 : w1);
        const uint32_t state = __builtin_amdgcn_alignbit(hi, lo, r) & 0xFFFFu;
        const uint32_t b3 = __builtin_amdgcn_udot4(__umul24(state, 0xDCD12Du), 0x01010101u, 3u, false);
        packed[i >> 2u] |= (b3 >> 2u) << ((i & 3u) * 8u);
    }
    uint2 v;
    v.x = packed[0] ^ 0x80808080u;
    v.y = packed[1] ^ 0x80808080u;
    int8_t* dst = kTileMajor
        ? dense + (static_cast<size_t>(nt) * (k >> 4u) + kt) * 256u + static_cast<size_t>(lane) * 8u
        : dense + static_cast<size_t>(nt * 16u + (lane & 15u)) * k + kt * 16u + (lane >> 4u) * 8u;
    *reinterpret_cast<uint2*>(dst) = v;
}

// Same as rotate_out but the input is the int32 hipBLASLt accumulator: yprime = acc * row scale * S_FIX.
// Rows with a slot take the F32 result of the fused per-group kernel (yprime2[slot]) instead.
template <bool BF16>
__global__ void __launch_bounds__(kBlock) exl3_mul24_rotate_out_i32_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ rowscale,
    const int32_t* __restrict__ slots,
    const float* __restrict__ yprime2,
    const uint16_t* __restrict__ svh,
    void* __restrict__ out,
    const uint32_t n,
    const uint32_t out_row_stride) {
    __shared__ float smem[kBlock];
    const uint32_t t = threadIdx.x;
    const uint32_t m = blockIdx.x;
    const uint32_t col = blockIdx.y * kBlock + t;
    const int32_t slot = slots[m];
    float v = slot >= 0 ? yprime2[static_cast<size_t>(slot) * n + col]
                        : static_cast<float>(acc[static_cast<size_t>(m) * n + col]) * rowscale[m] *
                              ::shelf::quantization::exl3::kExl3Int8GridStep;
    v = fwht128_smem(v, smem, t) * kInvSqrt128 * f16_bits_to_f32(svh[col]);
    if constexpr (BF16) {
        reinterpret_cast<::shelf::bf16_t*>(out)[static_cast<size_t>(m) * out_row_stride + col] =
            ::shelf::bf16_t{f32_to_bf16_rne_device(v)};
    } else {
        reinterpret_cast<float*>(out)[static_cast<size_t>(m) * out_row_stride + col] = v;
    }
}

// Same math, float4 form: one CTA per 8 rows, 8 waves, one wave per row. int4 activation load /
// float4 flagged-row load, H128 and svh inside one wave, one 64-bit output store.
// Each wave walks the whole output row.
template <bool BF16>
__global__ void __launch_bounds__(256) exl3_mul24_rotate_out_i32_f4_kernel(
    const int32_t* __restrict__ acc,
    const float* __restrict__ rowscale,
    const int32_t* __restrict__ slots,
    const float* __restrict__ yprime2,
    const uint16_t* __restrict__ svh,
    void* __restrict__ out,
    const uint32_t n,
    const uint32_t out_row_stride,
    const uint32_t rows) {
    const uint32_t m = blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (m >= rows) return;
    const uint32_t lane = threadIdx.x & 31u;
    const int32_t slot = slots[m];
    const float rs = rowscale[m];
    for (uint32_t cb = 0u; cb < n / kBlock; ++cb) {
        const uint32_t col = cb * kBlock + lane * 4u;
        float v0, v1, v2, v3;
        if (slot >= 0) {
            const float4 y = *reinterpret_cast<const float4*>(yprime2 + static_cast<size_t>(slot) * n + col);
            v0 = y.x;
            v1 = y.y;
            v2 = y.z;
            v3 = y.w;
        } else {
            const int4 a = *reinterpret_cast<const int4*>(acc + static_cast<size_t>(m) * n + col);
            v0 = (static_cast<float>(a.x) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v1 = (static_cast<float>(a.y) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v2 = (static_cast<float>(a.z) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v3 = (static_cast<float>(a.w) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
        }
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        v0 = (v0 * kInvSqrt128) * f16_bits_to_f32(svh[col + 0u]);
        v1 = (v1 * kInvSqrt128) * f16_bits_to_f32(svh[col + 1u]);
        v2 = (v2 * kInvSqrt128) * f16_bits_to_f32(svh[col + 2u]);
        v3 = (v3 * kInvSqrt128) * f16_bits_to_f32(svh[col + 3u]);
        const size_t index = static_cast<size_t>(m) * out_row_stride + col;
        if constexpr (BF16) {
            uint2 packed;
            packed.x = static_cast<uint32_t>(f32_to_bf16_rne_device(v0)) |
                       (static_cast<uint32_t>(f32_to_bf16_rne_device(v1)) << 16u);
            packed.y = static_cast<uint32_t>(f32_to_bf16_rne_device(v2)) |
                       (static_cast<uint32_t>(f32_to_bf16_rne_device(v3)) << 16u);
            *reinterpret_cast<uint2*>(reinterpret_cast<::shelf::bf16_t*>(out) + index) = packed;
        } else {
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(out) + index) = float4{v0, v1, v2, v3};
        }
    }
}
}  // namespace

hipError_t launch_exl3_mul24_rowscale(
    int8_t* codes, const float* scales, float* rowscale, int32_t* slots, uint32_t* count, int8_t* codes_f,
    float* scales_f, uint32_t rows, uint32_t cap, uint32_t k, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, float share, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (k == 0u || (k % kBlock) != 0u || (scale_row_stride_bytes % 4u) != 0u) return hipErrorInvalidValue;
    exl3_mul24_rowstat_kernel<<<rows, 256, 0, stream>>>(scales, rowscale, slots, scale_row_stride_bytes / 4u, k / kBlock, share);
    exl3_mul24_rowslot_kernel<<<1, 1024, 0, stream>>>(slots, count, rows, cap);
    exl3_mul24_rowfinish_kernel<<<rows, 256, 0, stream>>>(
        codes, scales, rowscale, slots, codes_f, scales_f, code_row_stride_bytes, scale_row_stride_bytes / 4u, k);
    return hipGetLastError();
}

template <bool kTileMajor>
hipError_t launch_recon(const uint16_t* trellis, uint32_t bits, int8_t* dense, uint32_t n, uint32_t k,
                        hipStream_t stream) {
    if ((n % 16u) != 0u || (k % kBlock) != 0u) return hipErrorInvalidValue;
    const dim3 grid(k / 128u, n / 16u);
    const uint32_t* t32 = reinterpret_cast<const uint32_t*>(trellis);
    switch (bits) {
        case 1: exl3_mul24_recon_kernel<1, kTileMajor><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 2: exl3_mul24_recon_kernel<2, kTileMajor><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 3: exl3_mul24_recon_kernel<3, kTileMajor><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 4: exl3_mul24_recon_kernel<4, kTileMajor><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 5: exl3_mul24_recon_kernel<5, kTileMajor><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 6: exl3_mul24_recon_kernel<6, kTileMajor><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 7: exl3_mul24_recon_kernel<7, kTileMajor><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 8: exl3_mul24_recon_kernel<8, kTileMajor><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

hipError_t launch_exl3_mul24_recon(const uint16_t* trellis, uint32_t bits, int8_t* dense, uint32_t n, uint32_t k,
                                  hipStream_t stream) {
    return launch_recon<false>(trellis, bits, dense, n, k, stream);
}

hipError_t launch_exl3_mul24_recon_wmma(const uint16_t* trellis, uint32_t bits, int8_t* dense, uint32_t n, uint32_t k,
                                        hipStream_t stream) {
    return launch_recon<true>(trellis, bits, dense, n, k, stream);
}

template <bool kF4, bool kBF16>
hipError_t launch_rotate_out_i32_impl(const int32_t* acc, const float* rowscale, const int32_t* slots,
                                      const float* yprime2, const uint16_t* svh, void* out, uint32_t rows,
                                      uint32_t n, uint32_t out_row_stride, hipStream_t stream) {
    if constexpr (kF4) {
        const dim3 grid((rows + 7u) / 8u);
        exl3_mul24_rotate_out_i32_f4_kernel<kBF16>
            <<<grid, 256, 0, stream>>>(acc, rowscale, slots, yprime2, svh, out, n, out_row_stride, rows);
    } else {
        const dim3 grid(rows, n / kBlock);
        exl3_mul24_rotate_out_i32_kernel<kBF16>
            <<<grid, kBlock, 0, stream>>>(acc, rowscale, slots, yprime2, svh, out, n, out_row_stride);
    }
    return hipGetLastError();
}

hipError_t launch_exl3_mul24_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (n == 0u || (n % kBlock) != 0u) return hipErrorInvalidValue;
    const bool f4 = rotate_f4_enabled() && rows >= kRotateOutF4MinRows && (out_row_stride % 4u) == 0u;
    const bool bf16 = out_dtype == Exl3OutputDType::BF16;
    if (f4)
        return bf16 ? launch_rotate_out_i32_impl<true, true>(acc, rowscale, slots, yprime2, svh, out, rows, n,
                                                             out_row_stride, stream)
                    : launch_rotate_out_i32_impl<true, false>(acc, rowscale, slots, yprime2, svh, out, rows, n,
                                                              out_row_stride, stream);
    return bf16 ? launch_rotate_out_i32_impl<false, true>(acc, rowscale, slots, yprime2, svh, out, rows, n,
                                                          out_row_stride, stream)
                : launch_rotate_out_i32_impl<false, false>(acc, rowscale, slots, yprime2, svh, out, rows, n,
                                                           out_row_stride, stream);
}

}  // namespace shelf::kernel
