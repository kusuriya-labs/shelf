#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul1/decode.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;

template <int BITS, int MT>
__global__ void __launch_bounds__(128) exl3_mul24_gemm_kernel(
    const int8_t* __restrict__ codes,
    const float* __restrict__ scales,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride_floats,
    const uint32_t* __restrict__ trellis,
    float* __restrict__ yprime,
    const uint32_t rows_cap,
    const uint32_t* __restrict__ active_rows,
    const uint32_t n,
    const uint32_t k,
    const uint32_t groups_per_split) {
    // active_rows (device) lets a launch sized for an upper bound skip the row tiles that hold nothing.
    const uint32_t rows = active_rows != nullptr ? min(rows_cap, *active_rows) : rows_cap;
    if (blockIdx.y * (16u * MT) >= rows) return;
    constexpr uint32_t kWords = 8u * BITS;
    constexpr uint32_t kBits = 256u * BITS;
    constexpr bool kAlignedWindows = (BITS % 4) == 0;
    // The third word is needed when a lane's windows reach past 64 bits of its stream.
    constexpr bool kNeedW2 = kAlignedWindows ? ((((BITS + 16u) % 32u) + 7u * BITS + 16u) > 64u) : true;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t nt = blockIdx.x * 4u + wave;
    const uint32_t n0 = nt * 16u;
    const uint32_t m0 = blockIdx.y * (16u * MT);
    const uint32_t col = lane & 15u;
    const uint32_t half = lane >> 4u;

    // The lane's decode window (constant over all tiles).
    const uint32_t base_bit = (lane * 8u * BITS + BITS + kBits - 16u) % kBits;
    const uint32_t off = base_bit & 31u;
    const uint32_t i0 = base_bit >> 5u;
    const uint32_t i1 = (i0 + 1u) % kWords;
    const uint32_t i2 = (i0 + 2u) % kWords;

    const int8_t* arow[MT];
    uint32_t arow_ok[MT];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const uint32_t m = m0 + mt * 16u + col;
        arow_ok[mt] = m < rows;
        arow[mt] = codes + static_cast<size_t>(arow_ok[mt] ? m : 0u) * code_row_stride + half * 8u;
    }

    float accf[MT][8];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int i = 0; i < 8; ++i) accf[mt][i] = 0.0f;

    // split-K: blockIdx.z owns the 128-K groups [g0, g1) and writes its own yprime slice (rotate_out adds them).
    const uint32_t g0 = blockIdx.z * groups_per_split;
    const uint32_t g1 = min(k >> 7u, g0 + groups_per_split);
    yprime += static_cast<size_t>(blockIdx.z) * rows * n;
    // The resident trellis is n-major ([nt][kt] tiles, see exl3_tiles_to_nmajor): one wave walks one contiguous stream.
    const uint32_t* tile_base = trellis + static_cast<size_t>(nt) * (k >> 4u) * kWords;
    constexpr size_t kt_stride = kWords;
    uint32_t cur[8][3];
    uint32_t nxt[8][3];
    #pragma unroll
    for (uint32_t s = 0u; s < 8u; ++s) {
        const uint32_t* tile = tile_base + static_cast<size_t>(g0 * 8u + s) * kt_stride;
        cur[s][0] = tile[i0];
        cur[s][1] = tile[i1];
        cur[s][2] = kNeedW2 ? tile[i2] : 0u;
    }
    for (uint32_t g = g0; g < g1; ++g) {
        if (g + 1u < g1) {
            #pragma unroll
            for (uint32_t s = 0u; s < 8u; ++s) {
                const uint32_t* tile = tile_base + static_cast<size_t>((g + 1u) * 8u + s) * kt_stride;
                nxt[s][0] = tile[i0];
                nxt[s][1] = tile[i1];
                nxt[s][2] = kNeedW2 ? tile[i2] : 0u;
            }
        }
        // The group's activation fragments are loaded up front so their latency hides behind the decode.
        int32x2_t afrag[MT][8];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (uint32_t s = 0u; s < 8u; ++s) {
                afrag[mt][s] = *reinterpret_cast<const int32x2_t*>(arow[mt] + g * kBlock + s * 16u);
                if (!arow_ok[mt]) { afrag[mt][s][0] = 0; afrag[mt][s][1] = 0; }
            }
        int32x8_t acc[MT];
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[mt][i] = 0;
        #pragma unroll
        for (uint32_t s = 0u; s < 8u; ++s) {
            const uint32_t w0 = cur[s][0];
            const uint32_t w1 = cur[s][1];
            const uint32_t w2 = cur[s][2];
            uint32_t packed[2] = {0u, 0u};
            #pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) {
                uint32_t state;
                if constexpr (kAlignedWindows) {
                    // BITS % 4 == 0: the window offset is the same for every lane, so the word pair and
                    // the shift are compile-time constants (one funnel shift per weight).
                    constexpr uint32_t off_c = (BITS + 32u - 16u) % 32u;
                    const uint32_t pos = off_c + i * BITS;
                    const uint32_t e = pos + 16u;
                    const uint32_t wl = (e - 1u) >> 5u;
                    const uint32_t r = 32u - (e - 32u * wl);
                    const uint32_t lo = wl == 0u ? w0 : (wl == 1u ? w1 : w2);
                    const uint32_t hi = wl == 0u ? 0u : (wl == 1u ? w0 : w1);
                    state = __builtin_amdgcn_alignbit(hi, lo, r) & 0xFFFFu;
                } else {
                    const uint32_t pos = off + i * BITS;
                    const uint32_t e = pos + 16u;
                    const uint32_t wl = (e - 1u) >> 5u;
                    const uint32_t r = 32u - (e - 32u * wl);
                    const uint32_t lo = wl == 0u ? w0 : (wl == 1u ? w1 : w2);
                    const uint32_t hi = wl == 0u ? 0u : (wl == 1u ? w0 : w1);
                    state = __builtin_amdgcn_alignbit(hi, lo, r) & 0xFFFFu;
                }
                const uint32_t b3 = __builtin_amdgcn_udot4(__umul24(state, 0xDCD12Du), 0x01010101u, 3u, false);
                packed[i >> 2u] |= (b3 >> 2u) << ((i & 3u) * 8u);
            }
            int32x2_t b;
            b[0] = static_cast<int32_t>(packed[0] ^ 0x80808080u);
            b[1] = static_cast<int32_t>(packed[1] ^ 0x80808080u);
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                acc[mt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, afrag[mt][s], true, b, acc[mt], false);
            }
        }

        #pragma unroll
        for (uint32_t s = 0u; s < 8u; ++s) {
            cur[s][0] = nxt[s][0];
            cur[s][1] = nxt[s][1];
            cur[s][2] = nxt[s][2];
        }

        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t m = m0 + mt * 16u + half * 8u + i;
                const float sa = m < rows ? scales[static_cast<size_t>(m) * scale_row_stride_floats + g] : 0.0f;
                accf[mt][i] += static_cast<float>(acc[mt][i]) * sa;
            }
        }
    }

    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t m = m0 + mt * 16u + half * 8u + i;
            if (m < rows)
                yprime[static_cast<size_t>(m) * n + n0 + col] = accf[mt][i] * ::shelf::quantization::exl3::kExl3Int8GridStep;
        }
    }
}

template <int BITS>
hipError_t launch_gemm_bits(
    const int8_t* codes, const float* scales, uint32_t code_stride, uint32_t sstride,
    const uint32_t* trellis, float* yprime, uint32_t rows, const uint32_t* active_rows, uint32_t n, uint32_t k, uint32_t splits, hipStream_t stream) {
    const dim3 block(128);
    // Only the decode-sized (MT=1) kernel splits K; wider row tiles already have enough waves.
    const uint32_t sp = rows <= 16u ? splits : 1u;
    const uint32_t gps = (k / kBlock + sp - 1u) / sp;
    if (rows <= 16u) {
        exl3_mul24_gemm_kernel<BITS, 1><<<dim3(n / 64u, 1, sp), block, 0, stream>>>(
            codes, scales, code_stride, sstride, trellis, yprime, rows, active_rows, n, k, gps);
    } else if (rows <= 32u) {
        exl3_mul24_gemm_kernel<BITS, 2><<<dim3(n / 64u, (rows + 31u) / 32u), block, 0, stream>>>(
            codes, scales, code_stride, sstride, trellis, yprime, rows, active_rows, n, k, gps);
    } else {
        exl3_mul24_gemm_kernel<BITS, 4><<<dim3(n / 64u, (rows + 63u) / 64u), block, 0, stream>>>(
            codes, scales, code_stride, sstride, trellis, yprime, rows, active_rows, n, k, gps);
    }
    return hipGetLastError();
}

template <bool BF16>
__global__ void __launch_bounds__(kBlock) exl3_mul24_rotate_out_kernel(
    const float* __restrict__ yprime,
    const uint16_t* __restrict__ svh,
    void* __restrict__ out,
    const uint32_t n,
    const uint32_t out_row_stride,
    const uint32_t splits) {
    __shared__ float smem[kBlock];
    const uint32_t t = threadIdx.x;
    const uint32_t m = blockIdx.x;
    const uint32_t col = blockIdx.y * kBlock + t;
    float v = 0.0f;
    for (uint32_t z = 0u; z < splits; ++z)
        v += yprime[(static_cast<size_t>(z) * gridDim.x + m) * n + col];
    v = fwht128_smem(v, smem, t) * kInvSqrt128 * f16_bits_to_f32(svh[col]);
    if constexpr (BF16) {
        reinterpret_cast<::shelf::bf16_t*>(out)[static_cast<size_t>(m) * out_row_stride + col] =
            ::shelf::bf16_t{f32_to_bf16_rne_device(v)};
    } else {
        reinterpret_cast<float*>(out)[static_cast<size_t>(m) * out_row_stride + col] = v;
    }
}

// Same math, float4 form: one CTA per 8 rows, 8 waves, one wave per row (the exl3_dense_epilogue_kernel
// structure). The K-split sum, H128 and svh stay inside one wave; no shared memory and no barrier.
// Each wave walks the whole output row.
template <bool BF16>
__global__ void __launch_bounds__(256) exl3_mul24_rotate_out_f4_kernel(
    const float* __restrict__ yprime,
    const uint16_t* __restrict__ svh,
    void* __restrict__ out,
    const uint32_t n,
    const uint32_t out_row_stride,
    const uint32_t splits,
    const uint32_t rows) {
    const uint32_t m = blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (m >= rows) return;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t cb = 0u; cb < n / kBlock; ++cb) {
        const uint32_t col = cb * kBlock + lane * 4u;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        for (uint32_t z = 0u; z < splits; ++z) {
            const float4 y = *reinterpret_cast<const float4*>(
                yprime + (static_cast<size_t>(z) * rows + m) * n + col);
            v0 += y.x;
            v1 += y.y;
            v2 += y.z;
            v3 += y.w;
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


// Prefill path (structure of the W16A16 dense path): a single hipBLASLt int8 GEMM over the whole K cannot apply
// per-128-K-group activation scales, so the activations get ONE scale per row (the max group scale) and the
// per-group codes are re-rounded onto it. A row whose group scales differ a lot (a massive-activation token) would lose
// most of its resolution that way, so such rows keep their group codes and are computed by the fused per-group
// kernel instead; the dense GEMM result for them is ignored.
//   rowstat: rowscale[m] = max_g s_g; flag[m] = (max^2 > share * sum_g s_g^2), i.e. one group holds more than `share`
//            of the row's scale energy (a massive-activation token).
//   rowslot: flagged rows get consecutive slots in row order (deterministic); beyond `cap` they stay unflagged.
//   rowfinish: flagged rows are copied (codes + scales) to the compact buffers; the others are re-rounded in place.
}  // namespace

hipError_t launch_exl3_mul24_gemm(
    const int8_t* codes, const float* scales, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, const uint16_t* trellis, uint32_t bits, float* yprime,
    uint32_t rows, uint32_t n, uint32_t k, uint32_t splits, hipStream_t stream, const uint32_t* active_rows) {
    if (rows == 0u) return hipSuccess;
    if (splits == 0u || splits > k / kBlock) return hipErrorInvalidValue;
    if ((n % 64u) != 0u || (k % kBlock) != 0u || (scale_row_stride_bytes % 4u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t sstride = scale_row_stride_bytes / 4u;
    const uint32_t* t32 = reinterpret_cast<const uint32_t*>(trellis);
    switch (bits) {
        case 1: return launch_gemm_bits<1>(codes, scales, code_row_stride_bytes, sstride, t32, yprime, rows, active_rows, n, k, splits, stream);
        case 2: return launch_gemm_bits<2>(codes, scales, code_row_stride_bytes, sstride, t32, yprime, rows, active_rows, n, k, splits, stream);
        case 3: return launch_gemm_bits<3>(codes, scales, code_row_stride_bytes, sstride, t32, yprime, rows, active_rows, n, k, splits, stream);
        case 4: return launch_gemm_bits<4>(codes, scales, code_row_stride_bytes, sstride, t32, yprime, rows, active_rows, n, k, splits, stream);
        case 5: return launch_gemm_bits<5>(codes, scales, code_row_stride_bytes, sstride, t32, yprime, rows, active_rows, n, k, splits, stream);
        case 6: return launch_gemm_bits<6>(codes, scales, code_row_stride_bytes, sstride, t32, yprime, rows, active_rows, n, k, splits, stream);
        case 7: return launch_gemm_bits<7>(codes, scales, code_row_stride_bytes, sstride, t32, yprime, rows, active_rows, n, k, splits, stream);
        case 8: return launch_gemm_bits<8>(codes, scales, code_row_stride_bytes, sstride, t32, yprime, rows, active_rows, n, k, splits, stream);
        default: return hipErrorInvalidValue;
    }
}

template <bool kF4, bool kBF16>
hipError_t launch_rotate_out_impl(const float* yprime, const uint16_t* svh, void* out, uint32_t rows, uint32_t n,
                                  uint32_t out_row_stride, uint32_t splits, hipStream_t stream) {
    if constexpr (kF4) {
        const dim3 grid((rows + 7u) / 8u);
        exl3_mul24_rotate_out_f4_kernel<kBF16>
            <<<grid, 256, 0, stream>>>(yprime, svh, out, n, out_row_stride, splits, rows);
    } else {
        const dim3 grid(rows, n / kBlock);
        exl3_mul24_rotate_out_kernel<kBF16>
            <<<grid, kBlock, 0, stream>>>(yprime, svh, out, n, out_row_stride, splits);
    }
    return hipGetLastError();
}

hipError_t launch_exl3_mul24_rotate_out(
    const float* yprime, const uint16_t* svh, void* out, Exl3OutputDType out_dtype, uint32_t rows,
    uint32_t n, uint32_t out_row_stride, uint32_t splits, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (n == 0u || (n % kBlock) != 0u) return hipErrorInvalidValue;
    const bool f4 = rotate_f4_enabled() && rows >= kRotateOutF4MinRows && (out_row_stride % 4u) == 0u;
    const bool bf16 = out_dtype == Exl3OutputDType::BF16;
    if (f4)
        return bf16 ? launch_rotate_out_impl<true, true>(yprime, svh, out, rows, n, out_row_stride, splits, stream)
                    : launch_rotate_out_impl<true, false>(yprime, svh, out, rows, n, out_row_stride, splits, stream);
    return bf16 ? launch_rotate_out_impl<false, true>(yprime, svh, out, rows, n, out_row_stride, splits, stream)
                : launch_rotate_out_impl<false, false>(yprime, svh, out, rows, n, out_row_stride, splits, stream);
}



}  // namespace shelf::kernel
