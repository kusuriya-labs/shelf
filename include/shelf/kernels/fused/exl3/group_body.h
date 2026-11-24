#pragma once

#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>

#include <algorithm>
#include <cstdint>

namespace shelf::kernel::exl3_detail::mul24 {

template <int BITS, int MT, bool PIPE = true, int WAVES = 4, bool FUSED = false, bool K_MAJOR = false,
          bool SHUFFLE_PREVIOUS = false>
__device__ __forceinline__ void
group_body(const int8_t *__restrict__ codes, const float *__restrict__ scales,
           const uint32_t code_row_stride, const uint32_t scale_row_stride_floats,
           const uint32_t *__restrict__ trellis, float *__restrict__ yprime, const uint32_t rows_cap,
           const uint32_t *__restrict__ active_rows, const uint32_t n, const uint32_t k,
           const uint32_t groups_per_split, const uint32_t row_blocks, uint32_t tile_x, uint32_t tile_z,
           const PlamoExl3Projection *ep = nullptr) {
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
    const uint32_t nt = nb * WAVES + wave;
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
    if constexpr (!FUSED)
        yprime += static_cast<size_t>(tile_z) * rows * n;
    const uint32_t *tile_base = trellis + static_cast<size_t>(nt) * (K_MAJOR ? 1u : (k >> 4u)) * kWords;
    const size_t kt_stride = K_MAJOR ? static_cast<size_t>(n >> 4u) * kWords : kWords;
    const uint32_t *wp0 = tile_base + static_cast<size_t>(g0 * 8u) * kt_stride + i0;
    const uint32_t *wp1 = tile_base + static_cast<size_t>(g0 * 8u) * kt_stride + i1;
    const uint32_t *wp2 = tile_base + static_cast<size_t>(g0 * 8u) * kt_stride + i2;
    uint32_t cur[PIPE ? 8 : 1][PIPE ? 3 : 1];
    uint32_t nxt[PIPE ? 8 : 1][PIPE ? 3 : 1];
    if constexpr (PIPE) {
#pragma unroll
        for (uint32_t s = 0u; s < 8u; ++s) {
            cur[s][0] = wp0[s * kt_stride];
            cur[s][1] = wp1[s * kt_stride];
            cur[s][2] = kNeedW2 ? wp2[s * kt_stride] : 0u;
        }
    }
    for (uint32_t g = g0; g < g1; ++g) {
        if constexpr (PIPE)
            if (g + 1u < g1) {
#pragma unroll
                for (uint32_t s = 0u; s < 8u; ++s) {
                    nxt[s][0] = wp0[(8u + s) * kt_stride];
                    nxt[s][1] = wp1[(8u + s) * kt_stride];
                    nxt[s][2] = kNeedW2 ? wp2[(8u + s) * kt_stride] : 0u;
                }
            }
        int32x2_t afrag[MT <= 3 ? MT : 1][MT <= 3 ? 8 : 1];
        if constexpr (MT <= 3) {
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
        }
        int32x8_t acc[MT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 8; ++i)
                acc[mt][i] = 0;
#pragma unroll
        for (uint32_t s = 0u; s < 8u; ++s) {
            int32x2_t afrag_s[MT <= 3 ? 1 : MT];
            if constexpr (MT > 3) {
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    afrag_s[mt] = *reinterpret_cast<const int32x2_t *>(arow[mt] + g * kBlock + s * 16u);
                    if (!arow_ok[mt]) {
                        afrag_s[mt][0] = 0;
                        afrag_s[mt][1] = 0;
                    }
                }
            }
            int32x2_t b;
            uint32_t w0, w1, w2;
            if constexpr (PIPE) {
                w0 = cur[s][0];
                w1 = cur[s][1];
                w2 = cur[s][2];
            } else {
                w1 = wp1[s * kt_stride];
                if constexpr (SHUFFLE_PREVIOUS) {
                    static_assert(BITS == 4 && !PIPE);
                    w0 = __shfl(w1, (lane + 31u) % 32u, 32);
                } else {
                    w0 = wp0[s * kt_stride];
                }
                w2 = kNeedW2 ? wp2[s * kt_stride] : 0u;
            }
            uint32_t packed[2] = {0u, 0u};
#pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) {
                uint32_t state;
                if constexpr (BITS == 3 && MT == 1) {
                    state = trellis_state_3bit(w0, w1, off, i);
                } else if constexpr (kAlignedWindows) {
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
                acc[mt] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, afrag[mt][s], true, b, acc[mt], false);
            }
        }

        if constexpr (PIPE) {
#pragma unroll
            for (uint32_t s = 0u; s < 8u; ++s) {
                cur[s][0] = nxt[s][0];
                cur[s][1] = nxt[s][1];
                cur[s][2] = nxt[s][2];
            }
        }
        wp0 += 8u * kt_stride;
        wp1 += 8u * kt_stride;
        wp2 += 8u * kt_stride;

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

    __shared__ __align__(16) float epi[FUSED ? MT * 16 * 128 : 1];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t m = m0 + mt * 16u + half * 8u + i;
            if (m < rows) {
                const float v = accf[mt][i] * ::shelf::quantization::exl3::kExl3Int8GridStep;
                if constexpr (FUSED)
                    epi[(m - m0) * 128u + wave * 16u + col] = v;
                else
                    yprime[static_cast<size_t>(m) * n + n0 + col] = v;
            }
        }
    }
    if constexpr (FUSED) {
        __syncthreads();
        for (uint32_t lm = wave; lm < 16u * MT && m0 + lm < rows; lm += WAVES) {
            const uint32_t local_col = lane * 4u, col = nb * 128u + local_col;
            const float4 v = *reinterpret_cast<const float4 *>(epi + lm * 128u + local_col);
            float v0 = 0.0f + v.x, v1 = 0.0f + v.y, v2 = 0.0f + v.z, v3 = 0.0f + v.w;
            exl3_detail::fwht128(v0, v1, v2, v3, lane);
            v0 = (v0 * kInvSqrt128) * f16_bits_to_f32(ep->svh[col]);
            v1 = (v1 * kInvSqrt128) * f16_bits_to_f32(ep->svh[col + 1u]);
            v2 = (v2 * kInvSqrt128) * f16_bits_to_f32(ep->svh[col + 2u]);
            v3 = (v3 * kInvSqrt128) * f16_bits_to_f32(ep->svh[col + 3u]);
            const size_t index = static_cast<size_t>(m0 + lm) * ep->out_stride + col;
            if constexpr (true) {
                const uint2 packed = {
                    uint32_t(f32_to_bf16_rne_device(v0)) | (uint32_t(f32_to_bf16_rne_device(v1)) << 16u),
                    uint32_t(f32_to_bf16_rne_device(v2)) | (uint32_t(f32_to_bf16_rne_device(v3)) << 16u)};
                *reinterpret_cast<uint2 *>(reinterpret_cast<uint16_t *>(ep->out) + index) = packed;
            } else
                *reinterpret_cast<float4 *>(reinterpret_cast<float *>(ep->out) + index) = {v0, v1, v2, v3};
        }
    }
}

}  // namespace shelf::kernel::exl3_detail::mul24
