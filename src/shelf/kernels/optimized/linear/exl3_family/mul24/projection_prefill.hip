#include <cstdint>
#include <hip/hip_runtime.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/quantization/exl3/exl3_int8.h>

namespace shelf::kernel {
namespace {
using namespace shelf::kernel::exl3_detail;
using namespace shelf::kernel::exl3_detail::mul24;
using I2 = int32_t __attribute__((ext_vector_type(2)));
using I4 = int32_t __attribute__((ext_vector_type(4)));
using I8 = int32_t __attribute__((ext_vector_type(8)));
struct Args {
    const int8_t *codes;
    const float *scales;
    const uint16_t *trellis;
    const int8_t *cache;
    const float *rowscale;
    const int32_t *row_flags;
    const uint16_t *svh;
    shelf::bf16_t *out;
    uint32_t rows, n, k, code_stride, scale_stride, out_stride;
};

template <int BITS> __device__ __forceinline__ I2 decode(const uint32_t *tile, uint32_t lane) {
    constexpr uint32_t words = 8u * BITS, bits = 256u * BITS;
    const uint32_t bit = (lane * 8u * BITS + BITS + bits - 16u) % bits;
    const uint32_t i0 = bit >> 5u, off = bit & 31u;
    const uint32_t w0 = tile[i0], w1 = tile[(i0 + 1u) % words], w2 = tile[(i0 + 2u) % words];
    uint32_t p0 = 0u, p1 = 0u;
#pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        const uint32_t e = off + i * BITS + 16u, wl = (e - 1u) >> 5u, r = 32u - (e - wl * 32u);
        const uint32_t lo = wl == 0u   ? w0
                            : wl == 1u ? w1
                                       : w2,
                       hi = wl == 0u   ? 0u
                            : wl == 1u ? w0
                                       : w1;
        const uint32_t state = __builtin_amdgcn_alignbit(hi, lo, r) & 0xffffu;
        const uint32_t q =
            __builtin_amdgcn_udot4(__umul24(state, 0xDCD12Du), 0x01010101u, 3u, false) >> 2u;
        if (i < 4u)
            p0 |= q << (i * 8u);
        else
            p1 |= q << ((i - 4u) * 8u);
    }
    return I2{int32_t(p0 ^ 0x80808080u), int32_t(p1 ^ 0x80808080u)};
}

template <int BITS, bool CACHE>
__device__ __forceinline__ I4 load_b(const Args &a, uint32_t nt, uint32_t group, uint32_t pair,
                                     uint32_t lane) {
    if constexpr (CACHE) {
        const I4 *p = reinterpret_cast<const I4 *>(
            a.cache + ((size_t(nt) * (a.k / 128u) + group) * 32u + lane) * 64u + pair * 16u);
        return *p;
    } else {
        const uint32_t *p = reinterpret_cast<const uint32_t *>(a.trellis) +
                            (size_t(nt) * (a.k / 16u) + group * 8u + pair * 2u) * (8u * BITS);
        const I2 x = decode<BITS>(p, lane), y = decode<BITS>(p + 8u * BITS, lane);
        return I4{x[0], x[1], y[0], y[1]};
    }
}

template <int MT> struct AStage {
    uint4 data[(MT * 16u * 8u + 255u) / 256u];
    float scale;
};
template <int MT>
__device__ __forceinline__ AStage<MT> prefetch_a(const Args &a, uint32_t m0, uint32_t g) {
    AStage<MT> s{};
#pragma unroll
    for (uint32_t j = 0u; j < (MT * 16u * 8u + 255u) / 256u; ++j) {
        const uint32_t t = threadIdx.x + j * 256u, r = t / 8u, chunk = t & 7u;
        if (r < MT * 16u && m0 + r < a.rows)
            s.data[j] = *reinterpret_cast<const uint4 *>(a.codes + size_t(m0 + r) * a.code_stride +
                                                         g * 128u + chunk * 16u);
    }
    if (threadIdx.x < MT * 16u && m0 + threadIdx.x < a.rows)
        s.scale = a.scales[size_t(m0 + threadIdx.x) * a.scale_stride + g];
    return s;
}
template <int MT> __device__ __forceinline__ void stage_a(uint8_t *shared, const AStage<MT> &s) {
#pragma unroll
    for (uint32_t j = 0u; j < (MT * 16u * 8u + 255u) / 256u; ++j) {
        const uint32_t t = threadIdx.x + j * 256u, r = t / 8u, chunk = t & 7u;
        if (r < MT * 16u) {
            const uint32_t sw = (r & 7u) * 16u;
            const uint32_t pos = (chunk / 2u) * 32u + (chunk & 1u) * 8u;
            const uint4 raw = s.data[j];
            *reinterpret_cast<uint2 *>(shared + r * 128u + (pos ^ sw)) = uint2{raw.x, raw.y};
            *reinterpret_cast<uint2 *>(shared + r * 128u + ((pos + 16u) ^ sw)) =
                uint2{raw.z, raw.w};
        }
    }
    if (threadIdx.x < MT * 16u)
        reinterpret_cast<float *>(shared + MT * 16u * 128u)[threadIdx.x] = s.scale;
}

template <int BITS, int MT, bool CACHE, bool PIPE, bool ROWSCALE>
__device__ __forceinline__ void body(Args a, uint32_t mb, uint32_t nb) {
    constexpr uint32_t rows_tile = MT * 16u, plane = rows_tile * 132u;
    alignas(16) __shared__ uint8_t shared[2u * plane];
    const uint32_t m0 = mb * rows_tile, nt = nb * 8u + (threadIdx.x >> 5u);
    if (m0 >= a.rows)
        return;
    const uint32_t lane = threadIdx.x & 31u, col = lane & 15u, half = lane >> 4u,
                   groups = a.k / 128u;
    float accf[MT][8]{};
    uint32_t flagged[MT]{};
    if constexpr (ROWSCALE) {
#pragma unroll
        for (uint32_t mt = 0u; mt < MT; ++mt)
#pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) {
                const uint32_t row = m0 + mt * 16u + half * 8u + i;
                if (row < a.rows && a.row_flags[row] >= 0)
                    flagged[mt] |= 1u << i;
            }
    }
    I8 acc[MT]{};
    I4 cur[4];
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p)
        cur[p] = load_b<BITS, CACHE>(a, nt, 0u, p, lane);
    stage_a<MT>(shared, prefetch_a<MT>(a, m0, 0u));
    __syncthreads();
    for (uint32_t g = 0u; g < groups; ++g) {
        AStage<MT> next{};
        I4 bnext[4];
        if (g + 1u < groups) {
            next = prefetch_a<MT>(a, m0, g + 1u);
            if constexpr (PIPE) {
#pragma unroll
                for (uint32_t p = 0u; p < 4u; ++p)
                    bnext[p] = load_b<BITS, CACHE>(a, nt, g + 1u, p, lane);
            }
        }
        uint8_t *current = shared + (g & 1u) * plane;
#pragma unroll
        for (uint32_t mt = 0u; mt < MT; ++mt) {
            if constexpr (ROWSCALE) {
                if (flagged[mt] != 0u) {
#pragma unroll
                    for (uint32_t i = 0u; i < 8u; ++i)
                        if (flagged[mt] & (1u << i))
                            acc[mt][i] = 0;
                }
            } else
                acc[mt] = I8{};
        }
#pragma unroll
        for (uint32_t p = 0u; p < 4u; ++p) {
            const I4 b = cur[p];
#pragma unroll
            for (uint32_t mt = 0u; mt < MT; ++mt) {
                const uint32_t r = mt * 16u + col, sw = (r & 7u) * 16u;
                const I4 x = *reinterpret_cast<const I4 *>(current + r * 128u +
                                                           ((p * 32u + half * 16u) ^ sw));
                acc[mt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    true, I2{x[0], x[1]}, true, I2{b[0], b[1]}, acc[mt], false);
                acc[mt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(
                    true, I2{x[2], x[3]}, true, I2{b[2], b[3]}, acc[mt], false);
            }
        }
#pragma unroll
        for (uint32_t mt = 0u; mt < MT; ++mt) {
            if (!ROWSCALE || flagged[mt] != 0u) {
#pragma unroll
                for (uint32_t i = 0u; i < 8u; ++i) {
                    if (!ROWSCALE || (flagged[mt] & (1u << i))) {
                        const uint32_t r = mt * 16u + half * 8u + i;
                        const float scale =
                            reinterpret_cast<const float *>(current + rows_tile * 128u)[r];
                        accf[mt][i] += static_cast<float>(acc[mt][i]) * scale;
                    }
                }
            }
        }
        if (g + 1u < groups) {
            stage_a<MT>(shared + ((g + 1u) & 1u) * plane, next);
#pragma unroll
            for (uint32_t p = 0u; p < 4u; ++p) {
                if constexpr (PIPE)
                    cur[p] = bnext[p];
                else
                    cur[p] = load_b<BITS, CACHE>(a, nt, g + 1u, p, lane);
            }
        }
        __syncthreads();
    }
    if constexpr (ROWSCALE) {
#pragma unroll
        for (uint32_t mt = 0u; mt < MT; ++mt)
#pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) {
                const uint32_t row = m0 + mt * 16u + half * 8u + i;
                if (row < a.rows && !(flagged[mt] & (1u << i)))
                    accf[mt][i] = static_cast<float>(acc[mt][i]) * a.rowscale[row];
            }
    }
    float *epi = reinterpret_cast<float *>(shared);
#pragma unroll
    for (uint32_t phase = 0u; phase < rows_tile / 8u; ++phase) {
#pragma unroll
        for (uint32_t mt = 0u; mt < MT; ++mt)
            if (mt * 2u + half == phase)
#pragma unroll
                for (uint32_t i = 0u; i < 8u; ++i)
                    epi[i * 128u + (threadIdx.x >> 5u) * 16u + col] =
                        accf[mt][i] * shelf::quantization::exl3::kExl3Int8GridStep;
        __syncthreads();
        const uint32_t row = m0 + phase * 8u + (threadIdx.x >> 5u);
        if (row < a.rows) {
            const uint32_t c = nb * 128u + lane * 4u;
            const float4 value =
                *reinterpret_cast<const float4 *>(epi + (threadIdx.x >> 5u) * 128u + lane * 4u);
            float v0 = ROWSCALE ? value.x : 0.0f + value.x;
            float v1 = ROWSCALE ? value.y : 0.0f + value.y;
            float v2 = ROWSCALE ? value.z : 0.0f + value.z;
            float v3 = ROWSCALE ? value.w : 0.0f + value.w;
            shelf::kernel::exl3_detail::fwht128(v0, v1, v2, v3, lane);
            v0 = (v0 * kInvSqrt128) * f16_bits_to_f32(a.svh[c]);
            v1 = (v1 * kInvSqrt128) * f16_bits_to_f32(a.svh[c + 1u]);
            v2 = (v2 * kInvSqrt128) * f16_bits_to_f32(a.svh[c + 2u]);
            v3 = (v3 * kInvSqrt128) * f16_bits_to_f32(a.svh[c + 3u]);
            const uint2 value_out = {uint32_t(f32_to_bf16_rne_device(v0)) |
                                         (uint32_t(f32_to_bf16_rne_device(v1)) << 16u),
                                     uint32_t(f32_to_bf16_rne_device(v2)) |
                                         (uint32_t(f32_to_bf16_rne_device(v3)) << 16u)};
            *reinterpret_cast<uint2 *>(a.out + size_t(row) * a.out_stride + c) = value_out;
        }
        __syncthreads();
    }
}

template <int BITS> __global__ void __launch_bounds__(128) recon_group(PlamoExl3PrefillGroup args) {
    const auto group = args.projection;
    uint32_t nt = blockIdx.y;
    for (uint32_t i = 0u; i < group.count; ++i) {
        const auto p = group.p[i];
        if (nt >= p.n / 16u) {
            nt -= p.n / 16u;
            continue;
        }
        const uint32_t g = blockIdx.x, lane = threadIdx.x >> 2u, pair = threadIdx.x & 3u;
        const uint32_t *source = reinterpret_cast<const uint32_t *>(p.trellis) +
                                 (size_t(nt) * (group.k / 16u) + g * 8u + pair * 2u) * (8u * BITS);
        const I2 b0 = decode<BITS>(source, lane), b1 = decode<BITS>(source + 8u * BITS, lane);
        *reinterpret_cast<I4 *>(args.packed[i] +
                                ((size_t(nt) * (group.k / 128u) + g) * 32u + lane) * 64u +
                                pair * 16u) = I4{b0[0], b0[1], b1[0], b1[1]};
        return;
    }
}

template <int BITS, int MT, bool CACHE, bool PIPE, bool ROWSCALE>
__global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(1)))
gemm_group(PlamoExl3PrefillGroup args) {
    const auto group = args.projection;
    uint32_t nb = blockIdx.y;
    for (uint32_t i = 0u; i < group.count; ++i) {
        const auto p = group.p[i];
        if (nb >= p.n / 128u) {
            nb -= p.n / 128u;
            continue;
        }
        Args one{p.codes,
                 p.scales,
                 p.trellis,
                 args.packed[i],
                 args.rowscale[i],
                 args.row_flags[i],
                 p.svh,
                 p.out,
                 group.rows,
                 p.n,
                 group.k,
                 group.k,
                 group.scale_stride_bytes / 4u,
                 p.out_stride};
        body<BITS, MT, CACHE, PIPE, ROWSCALE>(one, blockIdx.x, nb);
        return;
    }
}

__global__ void __launch_bounds__(256) rowstat(PlamoExl3PrefillGroup args) {
    const auto p = args.projection.p[blockIdx.z];
    const auto scales = p.scales;
    const auto rowscale = args.rowscale[blockIdx.z];
    const auto row_flags = args.row_flags[blockIdx.z];
    const uint32_t scale_row_stride_floats = args.projection.scale_stride_bytes / 4u;
    const uint32_t groups = args.projection.k / 128u;
    const float share = args.flag_share;
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
    if ((t & 31u) == 0u) {
        red[0][t >> 5u] = peak;
        red[1][t >> 5u] = sq;
    }
    __syncthreads();
    if (t == 0u) {
        float pk = red[0][0], sm = red[1][0];
        for (uint32_t i = 1u; i < 8u; ++i) {
            pk = fmaxf(pk, red[0][i]);
            sm += red[1][i];
        }
        rowscale[m] = pk;
        row_flags[m] = (pk > 0.0f && pk * pk > share * sm) ? 1 : -1;
    }
}

__global__ void __launch_bounds__(256) rowfinish(PlamoExl3PrefillGroup args) {
    const auto p = args.projection.p[blockIdx.z];
    const auto codes = p.codes;
    const auto scales = p.scales;
    const auto rowscale = args.rowscale[blockIdx.z];
    const auto row_flags = args.row_flags[blockIdx.z];
    const uint32_t code_row_stride = args.projection.k;
    const uint32_t scale_row_stride_floats = args.projection.scale_stride_bytes / 4u;
    const uint32_t k = args.projection.k;
    const uint32_t m = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const int32_t slot = row_flags[m];
    int8_t *row = codes + static_cast<size_t>(m) * code_row_stride;
    const float *srow = scales + static_cast<size_t>(m) * scale_row_stride_floats;
    if (slot >= 0)
        return;
    const float peak = rowscale[m];
    if (peak <= 0.0f)
        return;
    for (uint32_t i = t; i < k; i += 256u) {
        const float ratio = srow[i / kBlock] / peak;
        if (ratio == 1.0f)
            continue;
        row[i] = static_cast<int8_t>(static_cast<int>(rintf(static_cast<float>(row[i]) * ratio)));
    }
}

bool valid(const PlamoExl3PrefillGroup &args) {
    const auto &a = args.projection;
    if (!a.rows || !a.k || a.k % 128u || !a.count || a.count > 3u || a.scale_stride_bytes % 4u ||
        a.scale_stride_bytes < a.k / 128u * 4u)
        return false;
    for (uint32_t i = 0u; i < a.count; ++i) {
        const auto &p = a.p[i];
        if (!p.n || p.n % 128u || !p.codes || !p.scales || !p.trellis || !p.svh || !p.out ||
            p.out_stride < p.n || p.out_stride % 4u || (args.cached && !args.packed[i]) ||
            (args.rowscaled && (!args.rowscale[i] || !args.row_flags[i])))
            return false;
    }
    return true;
}
}

hipError_t launch_plamo_exl3_prefill_recon(const PlamoExl3PrefillGroup &args, uint32_t bits,
                                           hipStream_t stream) {
    if (!valid(args) || !args.cached)
        return hipErrorInvalidValue;
    uint32_t nt = 0u;
    for (uint32_t i = 0u; i < args.projection.count; ++i)
        nt += args.projection.p[i].n / 16u;
    const dim3 grid(args.projection.k / 128u, nt);
    if (bits == 4u)
        recon_group<4><<<grid, 128, 0, stream>>>(args);
    else if (bits == 5u)
        recon_group<5><<<grid, 128, 0, stream>>>(args);
    else
        return hipErrorInvalidValue;
    return hipGetLastError();
}

hipError_t launch_plamo_exl3_prefill_rowscale(const PlamoExl3PrefillGroup &args,
                                              hipStream_t stream) {
    if (!valid(args) || !args.rowscaled)
        return hipErrorInvalidValue;
    const auto &a = args.projection;
    rowstat<<<dim3(a.rows, 1u, a.count), 256, 0, stream>>>(args);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess)
        return e;
    rowfinish<<<dim3(a.rows, 1u, a.count), 256, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_plamo_exl3_prefill_gemm(const PlamoExl3PrefillGroup &args, uint32_t bits,
                                          hipStream_t stream) {
    if (!valid(args) || (bits != 4u && bits != 5u))
        return hipErrorInvalidValue;
    uint32_t nt = 0u;
    for (uint32_t i = 0u; i < args.projection.count; ++i)
        nt += args.projection.p[i].n / 128u;
    if (args.rowscaled) {
        if (args.cached && (args.projection.count == 3u || args.projection.p[0].n <= 512u)) {
            gemm_group<4, 2, true, true, true>
                <<<dim3((args.projection.rows + 31u) / 32u, nt), 256, 0, stream>>>(args);
        } else if (args.cached) {
            gemm_group<4, 4, true, false, true>
                <<<dim3((args.projection.rows + 63u) / 64u, nt), 256, 0, stream>>>(args);
        } else if (bits == 4u) {
            gemm_group<4, 4, false, false, true>
                <<<dim3((args.projection.rows + 63u) / 64u, nt), 256, 0, stream>>>(args);
        } else if (bits == 5u) {
            gemm_group<5, 4, false, false, true>
                <<<dim3((args.projection.rows + 63u) / 64u, nt), 256, 0, stream>>>(args);
        } else
            return hipErrorInvalidValue;
    } else {
        if (args.cached && (args.projection.count == 3u || args.projection.p[0].n <= 512u)) {
            gemm_group<4, 2, true, true, false>
                <<<dim3((args.projection.rows + 31u) / 32u, nt), 256, 0, stream>>>(args);
        } else if (args.cached) {
            gemm_group<4, 4, true, false, false>
                <<<dim3((args.projection.rows + 63u) / 64u, nt), 256, 0, stream>>>(args);
        } else if (bits == 4u) {
            gemm_group<4, 4, false, false, false>
                <<<dim3((args.projection.rows + 63u) / 64u, nt), 256, 0, stream>>>(args);
        } else if (bits == 5u) {
            gemm_group<5, 4, false, false, false>
                <<<dim3((args.projection.rows + 63u) / 64u, nt), 256, 0, stream>>>(args);
        } else
            return hipErrorInvalidValue;
    }
    return hipGetLastError();
}
}
