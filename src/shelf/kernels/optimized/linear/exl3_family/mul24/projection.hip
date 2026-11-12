#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <algorithm>
#include <cstddef>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;
__global__ void __launch_bounds__(128) group_rotate(PlamoExl3ProjectionGroup args) {
    const auto p = args.p[blockIdx.z];
    const auto input = args.input;
    const auto suh = p.suh;
    const auto codes = p.codes;
    const auto scales = p.scales;
    const uint32_t input_row_stride = args.input_stride, code_row_stride = args.k,
                   scale_row_stride_floats = args.scale_stride_bytes / 4;

    __shared__ float smem[kBlock];
    __shared__ float wave_peak[kBlock / 32u];
    const uint32_t t = threadIdx.x;
    const uint32_t m = blockIdx.x;
    const uint32_t g = blockIdx.y;
    const uint32_t col = g * kBlock + t;
    float v = bf16_to_f32(input[static_cast<size_t>(m) * input_row_stride + col]) * f16_bits_to_f32(suh[col]);
    v = fwht128_smem(v, smem, t) * kInvSqrt128;

    float peak = fabsf(v);
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        peak = fmaxf(peak, __shfl_xor(peak, off));
    if ((t & 31u) == 0u)
        wave_peak[t >> 5u] = peak;
    __syncthreads();
    peak = fmaxf(fmaxf(wave_peak[0], wave_peak[1]), fmaxf(wave_peak[2], wave_peak[3]));

    const float scale = peak / 127.0f;
    float q = 0.0f;
    if (scale > 0.0f)
        q = fminf(fmaxf(rintf(v / scale), -128.0f), 127.0f);
    codes[static_cast<size_t>(m) * code_row_stride + col] = static_cast<int8_t>(static_cast<int>(q));
    if (t == 0u)
        scales[static_cast<size_t>(m) * scale_row_stride_floats + g] = scale;
}
__global__ void __launch_bounds__(256) group_rotate_f4(PlamoExl3ProjectionGroup args) {
    const auto p = args.p[blockIdx.z];
    const auto input = args.input;
    const auto suh = p.suh;
    const auto codes = p.codes;
    const auto scales = p.scales;
    const uint32_t input_row_stride = args.input_stride, code_row_stride = args.k,
                   scale_row_stride_floats = args.scale_stride_bytes / 4, groups = args.k / 128u;

    const uint32_t m = blockIdx.x;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t g = wave; g < groups; g += 8u) {
        const uint32_t col = g * kBlock + lane * 4u;
        const uint2 raw =
            *reinterpret_cast<const uint2 *>(input + static_cast<size_t>(m) * input_row_stride + col);
        float v0 = bf16_bits_to_f32(static_cast<uint16_t>(raw.x & 0xFFFFu)) * f16_bits_to_f32(suh[col + 0u]);
        float v1 = bf16_bits_to_f32(static_cast<uint16_t>(raw.x >> 16u)) * f16_bits_to_f32(suh[col + 1u]);
        float v2 = bf16_bits_to_f32(static_cast<uint16_t>(raw.y & 0xFFFFu)) * f16_bits_to_f32(suh[col + 2u]);
        float v3 = bf16_bits_to_f32(static_cast<uint16_t>(raw.y >> 16u)) * f16_bits_to_f32(suh[col + 3u]);
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        v0 *= kInvSqrt128;
        v1 *= kInvSqrt128;
        v2 *= kInvSqrt128;
        v3 *= kInvSqrt128;
        float peak = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            peak = fmaxf(peak, __shfl_xor(peak, off, 32));
        const float scale = peak / 127.0f;
        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
        if (scale > 0.0f) {
            q0 = fminf(fmaxf(rintf(v0 / scale), -128.0f), 127.0f);
            q1 = fminf(fmaxf(rintf(v1 / scale), -128.0f), 127.0f);
            q2 = fminf(fmaxf(rintf(v2 / scale), -128.0f), 127.0f);
            q3 = fminf(fmaxf(rintf(v3 / scale), -128.0f), 127.0f);
        }
        const uint32_t packed = (static_cast<uint32_t>(static_cast<int32_t>(q0)) & 0xFFu) |
                                ((static_cast<uint32_t>(static_cast<int32_t>(q1)) & 0xFFu) << 8u) |
                                ((static_cast<uint32_t>(static_cast<int32_t>(q2)) & 0xFFu) << 16u) |
                                ((static_cast<uint32_t>(static_cast<int32_t>(q3)) & 0xFFu) << 24u);
        *reinterpret_cast<uint32_t *>(codes + static_cast<size_t>(m) * code_row_stride + col) = packed;
        if (lane == 0u)
            scales[static_cast<size_t>(m) * scale_row_stride_floats + g] = scale;
    }
}
__global__ void __launch_bounds__(128) group_out(PlamoExl3ProjectionGroup args) {
    const auto p = args.p[blockIdx.z];
    const auto yprime = p.yprime;
    const auto svh = p.svh;
    const auto out = p.out;
    const uint32_t n = p.n, rows = args.rows, splits = p.splits, out_row_stride = p.out_stride;
    constexpr bool BF16 = true;
    constexpr int WAVES = 4;

    const uint32_t m = blockIdx.x;
    if (m >= rows)
        return;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t cb = blockIdx.y * WAVES + (threadIdx.x >> 5u);
         cb < min(n / kBlock, (blockIdx.y + 1u) * WAVES); cb += WAVES) {
        const uint32_t col = cb * kBlock + lane * 4u;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        for (uint32_t z = 0u; z < splits; ++z) {
            const float4 y =
                *reinterpret_cast<const float4 *>(yprime + (static_cast<size_t>(z) * rows + m) * n + col);
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
            *reinterpret_cast<uint2 *>(reinterpret_cast<::shelf::bf16_t *>(out) + index) = packed;
        } else {
            *reinterpret_cast<float4 *>(reinterpret_cast<float *>(out) + index) = float4{v0, v1, v2, v3};
        }
    }
}
template <int BITS, int MT, bool PIPE = true>
__device__ __forceinline__ void
group_body(const int8_t *__restrict__ codes, const float *__restrict__ scales, const uint32_t code_row_stride,
           const uint32_t scale_row_stride_floats, const uint32_t *__restrict__ trellis,
           float *__restrict__ yprime, const uint32_t rows_cap, const uint32_t *__restrict__ active_rows,
           const uint32_t n, const uint32_t k, const uint32_t groups_per_split, const uint32_t row_blocks,
           uint32_t tile_x, uint32_t tile_z) {
    const uint32_t rb = tile_x % row_blocks;
    const uint32_t nb = tile_x / row_blocks;
    const uint32_t rows = active_rows != nullptr ? min(rows_cap, *active_rows) : rows_cap;
    if (rb * (16u * MT) >= rows)
        return;
    constexpr uint32_t kWords = 8u * BITS;
    constexpr uint32_t kBits = 256u * BITS;
    constexpr bool kAlignedWindows = (BITS % 4) == 0;
    constexpr bool kNeedW2 = kAlignedWindows ? ((((BITS + 16u) % 32u) + 7u * BITS + 16u) > 64u) : true;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t nt = nb * 4u + wave;
    const uint32_t n0 = nt * 16u;
    const uint32_t m0 = rb * (16u * MT);
    const uint32_t col = lane & 15u;
    const uint32_t half = lane >> 4u;

    const uint32_t base_bit = (lane * 8u * BITS + BITS + kBits - 16u) % kBits;
    const uint32_t off = base_bit & 31u;
    const uint32_t i0 = base_bit >> 5u;
    const uint32_t i1 = (i0 + 1u) % kWords;
    const uint32_t i2 = (i0 + 2u) % kWords;

    const int8_t *arow[MT];
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
        for (int i = 0; i < 8; ++i)
            accf[mt][i] = 0.0f;

    const uint32_t g0 = tile_z * groups_per_split;
    const uint32_t g1 = min(k >> 7u, g0 + groups_per_split);
    yprime += static_cast<size_t>(tile_z) * rows * n;
    const uint32_t *tile_base = trellis + static_cast<size_t>(nt) * (k >> 4u) * kWords;
    constexpr size_t kt_stride = kWords;
    uint32_t cur[8][3];
    uint32_t nxt[8][3];
#pragma unroll
    for (uint32_t s = 0u; s < 8u; ++s) {
        const uint32_t *tile = tile_base + static_cast<size_t>(g0 * 8u + s) * kt_stride;
        cur[s][0] = tile[i0];
        cur[s][1] = tile[i1];
        cur[s][2] = kNeedW2 ? tile[i2] : 0u;
    }
    for (uint32_t g = g0; g < g1; ++g) {
        if constexpr (PIPE)
            if (g + 1u < g1) {
#pragma unroll
                for (uint32_t s = 0u; s < 8u; ++s) {
                    const uint32_t *tile = tile_base + static_cast<size_t>((g + 1u) * 8u + s) * kt_stride;
                    nxt[s][0] = tile[i0];
                    nxt[s][1] = tile[i1];
                    nxt[s][2] = kNeedW2 ? tile[i2] : 0u;
                }
            }
        int32x2_t afrag[MT][8];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (uint32_t s = 0u; s < 8u; ++s) {
                afrag[mt][s] = *reinterpret_cast<const int32x2_t *>(arow[mt] + g * kBlock + s * 16u);
                if (!arow_ok[mt]) {
                    afrag[mt][s][0] = 0;
                    afrag[mt][s][1] = 0;
                }
            }
        int32x8_t acc[MT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[mt][i] = 0;
#pragma unroll
        for (uint32_t s = 0u; s < 8u; ++s) {
            int32x2_t b;
            const uint32_t *tile = tile_base + static_cast<size_t>(g * 8u + s) * kt_stride;
            const uint32_t w0 = PIPE ? cur[s][0] : tile[i0];
            const uint32_t w1 = PIPE ? cur[s][1] : tile[i1];
            const uint32_t w2 = PIPE ? cur[s][2] : (kNeedW2 ? tile[i2] : 0u);
            uint32_t packed[2] = {0u, 0u};
#pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) {
                uint32_t state;
                if constexpr (kAlignedWindows) {
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
                const uint32_t b3 =
                    __builtin_amdgcn_udot4(__umul24(state, 0xDCD12Du), 0x01010101u, 3u, false);
                packed[i >> 2u] |= (b3 >> 2u) << ((i & 3u) * 8u);
            }
            b[0] = static_cast<int32_t>(packed[0] ^ 0x80808080u);
            b[1] = static_cast<int32_t>(packed[1] ^ 0x80808080u);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                acc[mt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, afrag[mt][s], true, b,
                                                                           acc[mt], false);
            }
        }

#pragma unroll
        for (uint32_t s = 0u; s < ((PIPE) ? 8u : 0u); ++s) {
            cur[s][0] = nxt[s][0];
            cur[s][1] = nxt[s][1];
            cur[s][2] = nxt[s][2];
        }

#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t m = m0 + mt * 16u + half * 8u + i;
                const float sa =
                    m < rows ? scales[static_cast<size_t>(m) * scale_row_stride_floats + g] : 0.0f;
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
                yprime[static_cast<size_t>(m) * n + n0 + col] =
                    accf[mt][i] * ::shelf::quantization::exl3::kExl3Int8GridStep;
        }
    }
}

template <int B, int MT, bool PIPE>
__global__ void __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu((B == 4 && MT >= 3) ? 8 : 1)))
group_gemm(PlamoExl3ProjectionGroup args) {
    uint32_t tile = blockIdx.x;
    const uint32_t rb = (args.rows + 16u * MT - 1u) / (16u * MT);
    for (uint32_t i = 0; i < args.count; ++i) {
        const auto p = args.p[i];
        const uint32_t blocks = p.n / 64u * rb, total = blocks * p.splits;
        if (tile < total) {
            group_body<B, MT, PIPE>(p.codes, p.scales, args.k, args.scale_stride_bytes / 4u,
                                    reinterpret_cast<const uint32_t *>(p.trellis), p.yprime, args.rows,
                                    nullptr, p.n, args.k, (args.k / 128u + p.splits - 1u) / p.splits, rb,
                                    tile % blocks, tile / blocks);
            return;
        }
        tile -= total;
    }
}
template <int B, int MT, bool PIPE>
hipError_t launch_group_bits(const PlamoExl3ProjectionGroup &args, hipStream_t stream) {
    uint32_t blocks = 0;
    const uint32_t rb = (args.rows + 16u * MT - 1u) / (16u * MT);
    for (uint32_t i = 0; i < args.count; ++i)
        blocks += args.p[i].n / 64u * rb * args.p[i].splits;
    group_gemm<B, MT, PIPE><<<blocks, 128, 0, stream>>>(args);
    return hipGetLastError();
}
}
hipError_t launch_plamo_exl3_projection(const PlamoExl3ProjectionGroup &args, uint32_t bits,
                                        hipStream_t stream) {
    if (args.count == 0u || args.count > 3u || args.rows == 0u || args.rows >= 256u || args.k % 128u != 0u ||
        args.scale_stride_bytes % 4u != 0u || !args.input || args.input_stride < args.k ||
        args.scale_stride_bytes < args.k / 128u * 4u || (bits != 4u && bits != 5u && bits != 8u))
        return hipErrorInvalidValue;
    uint32_t n = 0u;
    for (uint32_t i = 0; i < args.count; ++i) {
        const auto &p = args.p[i];
        if (!p.trellis || !p.suh || !p.svh || !p.codes || !p.scales || !p.yprime || !p.out || p.n == 0u ||
            p.n % 128u != 0u || p.splits == 0u || p.splits > args.k / 128u || p.out_stride < p.n ||
            p.out_stride % 4u != 0u)
            return hipErrorInvalidValue;
        n = std::max(n, p.n);
    }
    if (args.rows >= 32u && args.input_stride % 4u == 0u)
        group_rotate_f4<<<dim3(args.rows, 1, args.count), 256, 0, stream>>>(args);
    else
        group_rotate<<<dim3(args.rows, args.k / 128u, args.count), 128, 0, stream>>>(args);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        return err;
    if (bits == 4u)
        err = args.rows <= 16u ? launch_group_bits<4, 1, true>(args, stream)
              : args.rows <= 128u ? launch_group_bits<4, 2, true>(args, stream)
                                 : launch_group_bits<4, 4, true>(args, stream);
    else if (bits == 5u)
        err = args.rows <= 16u ? launch_group_bits<5, 1, true>(args, stream)
                               : launch_group_bits<5, 4, false>(args, stream);
    else
        err = args.rows <= 16u ? launch_group_bits<8, 1, true>(args, stream)
                               : launch_group_bits<8, 2, true>(args, stream);
    if (err != hipSuccess)
        return err;
    group_out<<<dim3(args.rows, (n / 128u + 3u) / 4u, args.count), 128, 0, stream>>>(args);
    return hipGetLastError();
}
}
