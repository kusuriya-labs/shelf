#include <shelf/kernels/optimized/linear/exl3_int8.h>

#include <shelf/quantization/exl3/exl3_int8.h>

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

constexpr uint32_t kBlock = 128u;
constexpr float kInvSqrt128 = 0.08838834764831845f;

// Sylvester Walsh-Hadamard transform of one 128-wide block, one element per thread (128
// threads). Result is unnormalised.
__device__ __forceinline__ float fwht128(float v, float* smem, uint32_t t) {
    smem[t] = v;
    __syncthreads();
    #pragma unroll
    for (uint32_t h = 1u; h < kBlock; h <<= 1u) {
        const float a = smem[t];
        const float b = smem[t ^ h];
        __syncthreads();
        smem[t] = (t & h) ? (b - a) : (a + b);
        __syncthreads();
    }
    return smem[t];
}

__device__ __forceinline__ uint16_t f32_to_bf16_rne_device(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7F800000u) == 0x7F800000u) return (u & 0x007FFFFFu) != 0u ? 0x7FC0u : static_cast<uint16_t>(u >> 16);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float f16_bits_to_f32(uint16_t bits) {
    return static_cast<float>(__builtin_bit_cast(_Float16, bits));
}

__device__ __forceinline__ float bf16_to_f32(::shelf::bf16_t x) {
    return __uint_as_float(static_cast<uint32_t>(x.data) << 16);
}

__global__ void __launch_bounds__(kBlock) exl3_int8_rotate_quantize_kernel(
    const ::shelf::bf16_t* __restrict__ input,
    const uint32_t input_row_stride,
    const uint16_t* __restrict__ suh,
    int8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride_floats) {
    __shared__ float smem[kBlock];
    __shared__ float wave_peak[kBlock / 32u];
    const uint32_t t = threadIdx.x;
    const uint32_t m = blockIdx.x;
    const uint32_t g = blockIdx.y;
    const uint32_t col = g * kBlock + t;
    float v = bf16_to_f32(input[static_cast<size_t>(m) * input_row_stride + col]) * f16_bits_to_f32(suh[col]);
    v = fwht128(v, smem, t) * kInvSqrt128;

    float peak = fabsf(v);
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off));
    if ((t & 31u) == 0u) wave_peak[t >> 5u] = peak;
    __syncthreads();
    peak = fmaxf(fmaxf(wave_peak[0], wave_peak[1]), fmaxf(wave_peak[2], wave_peak[3]));

    const float scale = peak / 127.0f;
    float q = 0.0f;
    if (scale > 0.0f) q = fminf(fmaxf(rintf(v / scale), -128.0f), 127.0f);
    codes[static_cast<size_t>(m) * code_row_stride + col] = static_cast<int8_t>(static_cast<int>(q));
    if (t == 0u) scales[static_cast<size_t>(m) * scale_row_stride_floats + g] = scale;
}

// Weights stay in the EXL3 trellis (mul24 format, rocm_exl3 research/codebook) and are decoded to int8 on
// every read:
//   state = 16-bit window of the tile's circular MSB-first bit stream,
//   x = state * 0xDCD12D (uint32, one v_mul_u32_u24), B = byte sum of x, q = (B - 509) >> 2.
// q + 128 = (B + 3) >> 2 is taken unsigned (the +3 rides on the dot4 accumulator) and four of them
// are flipped to signed with one xor.
//
// A 16x16 trellis tile holds 256 weights in the WMMA B-fragment order: lane l of 32 decodes
// t = 8l..8l+7, which are column n = l % 16 at the input rows 8 * (l / 16) .. +7. Those eight bytes are
// the lane's iu8 B fragment as decoded, so there is no LDS and no k reordering; the activation codes
// are in natural k order.
//
// One wave owns a 16-wide output tile and MT 16-row tiles; a block has 4 waves along N.
template <int BITS, int MT>
__global__ void __launch_bounds__(128) exl3_int8_gemm_kernel(
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
        exl3_int8_gemm_kernel<BITS, 1><<<dim3(n / 64u, 1, sp), block, 0, stream>>>(
            codes, scales, code_stride, sstride, trellis, yprime, rows, active_rows, n, k, gps);
    } else if (rows <= 32u) {
        exl3_int8_gemm_kernel<BITS, 2><<<dim3(n / 64u, (rows + 31u) / 32u), block, 0, stream>>>(
            codes, scales, code_stride, sstride, trellis, yprime, rows, active_rows, n, k, gps);
    } else {
        exl3_int8_gemm_kernel<BITS, 4><<<dim3(n / 64u, (rows + 63u) / 64u), block, 0, stream>>>(
            codes, scales, code_stride, sstride, trellis, yprime, rows, active_rows, n, k, gps);
    }
    return hipGetLastError();
}

template <bool BF16>
__global__ void __launch_bounds__(kBlock) exl3_int8_rotate_out_kernel(
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
    v = fwht128(v, smem, t) * kInvSqrt128 * f16_bits_to_f32(svh[col]);
    if constexpr (BF16) {
        reinterpret_cast<::shelf::bf16_t*>(out)[static_cast<size_t>(m) * out_row_stride + col] =
            ::shelf::bf16_t{f32_to_bf16_rne_device(v)};
    } else {
        reinterpret_cast<float*>(out)[static_cast<size_t>(m) * out_row_stride + col] = v;
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
__global__ void __launch_bounds__(256) exl3_int8_rowstat_kernel(
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

__global__ void __launch_bounds__(1024) exl3_int8_rowslot_kernel(
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

__global__ void __launch_bounds__(256) exl3_int8_rowfinish_kernel(
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

// trellis -> dense int8 weight matrix [n][k] (k contiguous), for a hipBLASLt int8 GEMM with A transposed.
// One wave per 16x16 tile, the same lane / window / decode as the gemm kernel: lane l holds column n = l % 16 and
// the eight k rows 8 * (l / 16) .. +7 of the tile. A block is 8 consecutive k tiles of one n tile, so a row of the
// output is written in 128 contiguous bytes.
template <int BITS>
__global__ void __launch_bounds__(256) exl3_int8_recon_kernel(
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
    *reinterpret_cast<uint2*>(dense + static_cast<size_t>(nt * 16u + (lane & 15u)) * k + kt * 16u + (lane >> 4u) * 8u) = v;
}

// Same as rotate_out but the input is the int32 hipBLASLt accumulator: yprime = acc * row scale * S_FIX.
// Rows with a slot take the F32 result of the fused per-group kernel (yprime2[slot]) instead.
template <bool BF16>
__global__ void __launch_bounds__(kBlock) exl3_int8_rotate_out_i32_kernel(
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
    v = fwht128(v, smem, t) * kInvSqrt128 * f16_bits_to_f32(svh[col]);
    if constexpr (BF16) {
        reinterpret_cast<::shelf::bf16_t*>(out)[static_cast<size_t>(m) * out_row_stride + col] =
            ::shelf::bf16_t{f32_to_bf16_rne_device(v)};
    } else {
        reinterpret_cast<float*>(out)[static_cast<size_t>(m) * out_row_stride + col] = v;
    }
}
}

hipError_t launch_exl3_int8_rotate_quantize(
    const ::shelf::bf16_t* input, uint32_t input_row_stride, const uint16_t* suh, int8_t* codes,
    float* scales, uint32_t rows, uint32_t k, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (k == 0u || (k % kBlock) != 0u || (scale_row_stride_bytes % 4u) != 0u) return hipErrorInvalidValue;
    exl3_int8_rotate_quantize_kernel<<<dim3(rows, k / kBlock), kBlock, 0, stream>>>(
        input, input_row_stride, suh, codes, scales, code_row_stride_bytes, scale_row_stride_bytes / 4u);
    return hipGetLastError();
}

uint32_t exl3_int8_gemm_splits(uint32_t rows, uint32_t n, uint32_t k, uint32_t scratch_rows) {
    if (rows > 16u || rows == 0u || scratch_rows < rows) return 1u;
    // Wave slots of the device (6 resident waves per SIMD at ~215 VGPRs), queried once.
    static const uint32_t slots = [] {
        int dev = 0, wgp = 0;
        if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetAttribute(&wgp, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || wgp <= 0)
            return 768u;
        return static_cast<uint32_t>(wgp) * 4u * 6u;
    }();
    const uint32_t groups = k / kBlock, tiles = n / 16u;
    uint32_t best = 1u;
    double best_fill = 0.0;
    // Fill = waves / (slots * rounds): the last round of a grid that is not a multiple of the slots runs partly empty.
    // Keep >= 8 groups per split so the weight prefetch pipeline still pays off.
    for (uint32_t s = 1u; s <= 16u && s * rows <= scratch_rows && groups / s >= 8u; ++s) {
        const uint64_t waves = static_cast<uint64_t>(tiles) * s;
        const uint64_t rounds = (waves + slots - 1u) / slots;
        const double fill = static_cast<double>(waves) / static_cast<double>(rounds * slots);
        if (fill > best_fill + 0.02) { best_fill = fill; best = s; }
    }
    return best;
}

hipError_t launch_exl3_int8_gemm(
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

hipError_t launch_exl3_int8_rotate_out(
    const float* yprime, const uint16_t* svh, void* out, Exl3Int8OutputDType out_dtype, uint32_t rows,
    uint32_t n, uint32_t out_row_stride, uint32_t splits, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (n == 0u || (n % kBlock) != 0u) return hipErrorInvalidValue;
    const dim3 grid(rows, n / kBlock);
    if (out_dtype == Exl3Int8OutputDType::BF16)
        exl3_int8_rotate_out_kernel<true><<<grid, kBlock, 0, stream>>>(yprime, svh, out, n, out_row_stride, splits);
    else
        exl3_int8_rotate_out_kernel<false><<<grid, kBlock, 0, stream>>>(yprime, svh, out, n, out_row_stride, splits);
    return hipGetLastError();
}


hipError_t launch_exl3_int8_rowscale(
    int8_t* codes, const float* scales, float* rowscale, int32_t* slots, uint32_t* count, int8_t* codes_f,
    float* scales_f, uint32_t rows, uint32_t cap, uint32_t k, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, float share, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (k == 0u || (k % kBlock) != 0u || (scale_row_stride_bytes % 4u) != 0u) return hipErrorInvalidValue;
    exl3_int8_rowstat_kernel<<<rows, 256, 0, stream>>>(scales, rowscale, slots, scale_row_stride_bytes / 4u, k / kBlock, share);
    exl3_int8_rowslot_kernel<<<1, 1024, 0, stream>>>(slots, count, rows, cap);
    exl3_int8_rowfinish_kernel<<<rows, 256, 0, stream>>>(
        codes, scales, rowscale, slots, codes_f, scales_f, code_row_stride_bytes, scale_row_stride_bytes / 4u, k);
    return hipGetLastError();
}

hipError_t launch_exl3_int8_recon(const uint16_t* trellis, uint32_t bits, int8_t* dense, uint32_t n, uint32_t k,
                                  hipStream_t stream) {
    if ((n % 16u) != 0u || (k % kBlock) != 0u) return hipErrorInvalidValue;
    const dim3 grid(k / 128u, n / 16u);
    const uint32_t* t32 = reinterpret_cast<const uint32_t*>(trellis);
    switch (bits) {
        case 1: exl3_int8_recon_kernel<1><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 2: exl3_int8_recon_kernel<2><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 3: exl3_int8_recon_kernel<3><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 4: exl3_int8_recon_kernel<4><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 5: exl3_int8_recon_kernel<5><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 6: exl3_int8_recon_kernel<6><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 7: exl3_int8_recon_kernel<7><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        case 8: exl3_int8_recon_kernel<8><<<grid, 256, 0, stream>>>(t32, dense, k); break;
        default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

hipError_t launch_exl3_int8_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3Int8OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (n == 0u || (n % kBlock) != 0u) return hipErrorInvalidValue;
    const dim3 grid(rows, n / kBlock);
    if (out_dtype == Exl3Int8OutputDType::BF16)
        exl3_int8_rotate_out_i32_kernel<true><<<grid, kBlock, 0, stream>>>(acc, rowscale, slots, yprime2, svh, out, n, out_row_stride);
    else
        exl3_int8_rotate_out_i32_kernel<false><<<grid, kBlock, 0, stream>>>(acc, rowscale, slots, yprime2, svh, out, n, out_row_stride);
    return hipGetLastError();
}

}
