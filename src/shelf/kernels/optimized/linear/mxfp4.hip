#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/optimized/detail/lds_barrier_device.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

namespace shelf::kernel {

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = kMxfp4GemmRowTile;
constexpr uint32_t kThreadsPerBlock = 32u;

// One native weight block is 16 outputs x 32 K, packed 4-bit: 256 bytes where
// output row ol owns 16 contiguous bytes and byte b holds code[k = b] in the
// low nibble and code[k = 16 + b] in the high nibble.
constexpr uint32_t kGroupElements = 32u;
constexpr uint32_t kWeightBlockBytes = 256u;
constexpr uint32_t kFragBlockBytes = 1024u;
constexpr uint32_t kFragLaneBytes = 32u;

#define MXFP4_ZERO8 \
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }

// The eight E2M1 magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6} scaled by 2^-d and stored
// as packed E4M3 bytes (low word = indices 0-3, high word = 4-7). d is the
// distance from the row's reference exponent, so row 0 is the plain E2M1 value.
// The products are exact for d <= 8; from d >= 9 the E4M3 subnormal step rounds
// the .5 ties to nearest-even (matching the hardware conversion) and d >= 13 is
// zero. Measured on the Qwen3.8-27B MXFP4 checkpoint, 99.99996% of the scale
// blocks have d <= 8 and the maximum d is 10.
constexpr uint32_t kFoldMag[16][2] = {
    {0x3C383000u, 0x4C484440u}, {0x34302800u, 0x44403C38u},
    {0x2C282000u, 0x3C383430u}, {0x24201800u, 0x34302C28u},
    {0x1C181000u, 0x2C282420u}, {0x14100800u, 0x24201C18u},
    {0x0C080400u, 0x1C181410u}, {0x06040200u, 0x14100C08u},
    {0x03020100u, 0x0C080604u}, {0x02010000u, 0x06040302u},
    {0x01000000u, 0x03020201u}, {0x00000000u, 0x02010100u},
    {0x00000000u, 0x01000000u}, {0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u}, {0x00000000u, 0x00000000u},
};

// E2M1 -> E4M3 expansion for four packed bytes (eight codes), with the magnitude
// table row for the fold distance already in hand. The native layout pairs code
// k with code k + 16 in one byte, so the low nibbles and the high nibbles of a
// dword become the two operand registers of one 16 K WMMA fragment with no byte
// spread. The E4M3 sign bit is bit 7 of the byte, so the code sign (bit 3) is
// shifted straight into the permuted magnitude.
__device__ __forceinline__ int32x2_t mxfp4_expand(uint32_t q4, uint32_t t0, uint32_t t1) {
    const uint32_t even = q4;
    const uint32_t odd = q4 >> 4u;
    const int32_t lo = static_cast<int32_t>(
        __builtin_amdgcn_perm(t1, t0, even & 0x07070707u) | ((even & 0x08080808u) << 4u));
    const int32_t hi = static_cast<int32_t>(
        __builtin_amdgcn_perm(t1, t0, odd & 0x07070707u) | ((odd & 0x08080808u) << 4u));
    return int32x2_t{lo, hi};
}

__device__ __forceinline__ void mxfp4_mag_lds(uint2* dst, const uint32_t tid) {
    if (tid < 16u)
        dst[tid] = make_uint2(kFoldMag[tid][0], kFoldMag[tid][1]);
}

// E8M0 is a biased power of two, so the scale is the f32 bit pattern with the
// exponent field set. The loader rejects byte 255 (the MX spec NaN); byte 0 is
// 2^-127, which is not representable as a normal f32 exponent and is emulated.
__device__ __forceinline__ float mxfp4_e8m0_f32(uint8_t byte) {
    const uint32_t bits = static_cast<uint32_t>(byte) << 23;
    return __uint_as_float(bits != 0u ? bits : 0x00400000u);
}

__device__ __forceinline__ void mxfp4_weight_pair(const uint8_t* wp, uint32_t t0, uint32_t t1,
                                                  int32x2_t& w0, int32x2_t& w1) {
    const uint64_t packed = *reinterpret_cast<const uint64_t*>(wp);
    const int32x2_t lo = mxfp4_expand(static_cast<uint32_t>(packed), t0, t1);
    const int32x2_t hi = mxfp4_expand(static_cast<uint32_t>(packed >> 32u), t0, t1);
    w0 = int32x2_t{lo[0], hi[0]};
    w1 = int32x2_t{lo[1], hi[1]};
}

// Fold distance of one (output row, K block) pair. weight_ref holds the row's
// maximum E8M0 byte, so d >= 0; the table clamps at its zero tail.
__device__ __forceinline__ uint32_t mxfp4_fold_d(int ref, int e8m0) {
    const int d = ref - e8m0;
    return d < 0 ? 0u : (d > 15 ? 15u : static_cast<uint32_t>(d));
}

// MB activation row tiles x 16 outputs per wave. With kFold the E8M0 exponent of
// every K block is folded into the E4M3 weight operand relative to the row's
// reference exponent, so the four 32 K blocks of one 128 K activation group
// share one accumulator and the group scale is applied once; with kFold false the
// block scale is applied per 32 K block, which is the accumulation order the
// correctness path uses.
template <int MB, bool kFold>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_mxfp4_w4a8_wmma_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ weight_ref,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t output_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4u;

    const uint32_t out_start = blockIdx.x * kTileOut;
    const uint32_t row_start = blockIdx.y * (MB * kTileRow);
    if (out_start >= out_features || row_start >= rows)
        return;

    const uint32_t nb = k_padded >> 5;
    const uint32_t ng = k_padded >> 7;
    const uint32_t as_row_floats = activation_scale_stride >> 2u;

    const bool out_valid = out_start + m_lane < out_features;
    const uint32_t o_self = out_start + m_lane;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(nb) * kWeightBlockBytes) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const uint8_t* wsrow = weight_scales + static_cast<size_t>(o_self) * weight_scale_stride;

    const uint8_t* arow[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        arow[t] = activation_codes +
            (static_cast<size_t>(row_start >> 4) + static_cast<uint32_t>(t)) *
                (static_cast<size_t>(activation_code_stride) * 16u) +
            static_cast<size_t>(tid) * kFragLaneBytes;
    }

    __shared__ float smem[2u * MB * 8u];
    __shared__ uint2 mag_lds[16];
    int ref = 0;
    float epi = 1.0f;
    if constexpr (kFold) {
        mxfp4_mag_lds(mag_lds, tid);
        ref = out_valid ? static_cast<int>(weight_ref[o_self]) : 0;
        epi = mxfp4_e8m0_f32(static_cast<uint8_t>(ref));
        detail::lds_barrier();
    }

    float8_t acc[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t)
        acc[t] = float8_t MXFP4_ZERO8;

    const bool full_tile = (out_start + kTileOut <= out_features);

    auto run_k_loop = [&]<bool kGuarded, int kUnroll>() {
        #pragma unroll kUnroll
        for (uint32_t ag = 0; ag < ng; ++ag) {
            for (uint32_t s = tid; s < 2u * MB * 8u; s += kThreadsPerBlock) {
                const uint32_t kg = s / (MB * 8u);
                const uint32_t t = (s >> 3u) % MB;
                const uint32_t j = s & 7u;
                const uint32_t rr = row_start + t * kTileRow + 8u * kg + j;
                smem[s] = (rr < rows)
                    ? activation_scales[static_cast<size_t>(rr) * as_row_floats + ag]
                    : 0.0f;
            }
            detail::lds_barrier();

            float asv[MB][8];
            #pragma unroll
            for (int t = 0; t < MB; ++t) {
                const uint32_t slot = (k_group * MB + static_cast<uint32_t>(t)) * 8u;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    asv[t][j] = smem[slot + static_cast<uint32_t>(j)];
            }

            #pragma unroll
            for (uint32_t u = 0; u < 4u; ++u) {
                const uint32_t ib = ag * 4u + u;
                uint32_t t0 = kFoldMag[0][0];
                uint32_t t1 = kFoldMag[0][1];
                float ws = 0.0f;
                auto prep = [&]() {
                    if constexpr (kFold) {
                        const uint2 mg = mag_lds[mxfp4_fold_d(ref, wsrow[ib])];
                        t0 = mg.x;
                        t1 = mg.y;
                    } else {
                        ws = mxfp4_e8m0_f32(wsrow[ib]);
                    }
                };
                int32x2_t w0 = {};
                int32x2_t w1 = {};
                const uint32_t woff = static_cast<size_t>(ib) * kWeightBlockBytes;
                if constexpr (!kGuarded) {
                    prep();
                    mxfp4_weight_pair(wrow + woff, t0, t1, w0, w1);
                } else if (out_valid) {
                    prep();
                    mxfp4_weight_pair(wrow + woff, t0, t1, w0, w1);
                }
                const uint32_t aoff = (ib >> 1) * kFragBlockBytes + (ib & 1u) * 16u;
                #pragma unroll
                for (int t = 0; t < MB; ++t) {
                    int32x2_t a0 = {};
                    int32x2_t a1 = {};
                    const bool rv =
                        (row_start + static_cast<uint32_t>(t) * kTileRow + m_lane) < rows;
                    if constexpr (!kGuarded) {
                        const uint8_t* ap = arow[t] + aoff;
                        a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        a1 = *reinterpret_cast<const int32x2_t*>(ap + 8u);
                    }
                    float8_t c = float8_t MXFP4_ZERO8;
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if constexpr (kFold)
                            acc[t][j] = __builtin_fmaf(c[j], asv[t][j], acc[t][j]);
                        else
                            acc[t][j] = __builtin_fmaf(c[j], ws * asv[t][j], acc[t][j]);
                    }
                }
            }
            detail::lds_barrier();
        }
    };
    constexpr int kUnroll = (MB >= 4) ? 2 : 4;
    if (full_tile)
        run_k_loop.template operator()<false, kUnroll>();
    else
        run_k_loop.template operator()<true, kUnroll>();

    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        if (!out_valid)
            continue;
        const uint32_t r_base = row_start + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t r = r_base + static_cast<uint32_t>(j);
            if (r >= rows)
                continue;
            float v = acc[t][j];
            if constexpr (kFold)
                v *= epi;
            const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
            if (output_dtype == static_cast<uint32_t>(Mxfp4GemmOutputDType::BF16)) {
                static_cast<shelf::bf16_t*>(output)[out_idx] =
                    shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

template <int MB, int kSplit, bool kFold>
__global__ void __launch_bounds__(kThreadsPerBlock * kSplit) gemm_mxfp4_w4a8_splitk_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ weight_ref,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t output_row_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t m_lane = lane & 15u;
    const uint32_t k_group = lane >> 4u;

    const uint32_t out_start = blockIdx.x * kTileOut;
    const uint32_t row_start = blockIdx.y * (MB * kTileRow);
    if (out_start >= out_features || row_start >= rows)
        return;

    const uint32_t nb = k_padded >> 5;
    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const bool out_valid = out_start + m_lane < out_features;
    const uint32_t o_self = out_start + m_lane;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(nb) * kWeightBlockBytes) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const uint8_t* wsrow = weight_scales + static_cast<size_t>(o_self) * weight_scale_stride;

    const uint8_t* arow[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        arow[t] = activation_codes +
            (static_cast<size_t>(row_start >> 4) + static_cast<size_t>(t)) *
                (static_cast<size_t>(activation_code_stride) * 16u) +
            static_cast<size_t>(lane) * kFragLaneBytes;
    }

    __shared__ uint2 mag_lds[16];
    int ref = 0;
    float epi = 1.0f;
    if constexpr (kFold) {
        mxfp4_mag_lds(mag_lds, lane);
        ref = out_valid ? static_cast<int>(weight_ref[o_self]) : 0;
        epi = mxfp4_e8m0_f32(static_cast<uint8_t>(ref));
        detail::lds_barrier();
    }

    float8_t acc[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t)
        acc[t] = float8_t MXFP4_ZERO8;

    const uint32_t ng = k_padded >> 7;
    const uint32_t ag_start = (ng * warp) / static_cast<uint32_t>(kSplit);
    const uint32_t ag_stop = (ng * (warp + 1u)) / static_cast<uint32_t>(kSplit);

    for (uint32_t ag = ag_start; ag < ag_stop; ++ag) {
        float asv[MB][8];
        #pragma unroll
        for (int t = 0; t < MB; ++t) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t rr = row_start + static_cast<uint32_t>(t) * kTileRow +
                                    8u * k_group + static_cast<uint32_t>(j);
                asv[t][j] = (rr < rows)
                    ? activation_scales[static_cast<size_t>(rr) * as_row_floats + ag]
                    : 0.0f;
            }
        }
        #pragma unroll
        for (uint32_t u = 0; u < 4u; ++u) {
            const uint32_t ib = ag * 4u + u;
            if (ib >= nb)
                break;
            uint32_t t0 = kFoldMag[0][0];
            uint32_t t1 = kFoldMag[0][1];
            float ws = 0.0f;
            if (out_valid) {
                if constexpr (kFold) {
                    const uint2 mg = mag_lds[mxfp4_fold_d(ref, wsrow[ib])];
                    t0 = mg.x;
                    t1 = mg.y;
                } else {
                    ws = mxfp4_e8m0_f32(wsrow[ib]);
                }
            }
            int32x2_t w0 = {};
            int32x2_t w1 = {};
            if (out_valid)
                mxfp4_weight_pair(wrow + static_cast<size_t>(ib) * kWeightBlockBytes,
                                  t0, t1, w0, w1);
            const uint32_t aoff = (ib >> 1) * kFragBlockBytes + (ib & 1u) * 16u;
            #pragma unroll
            for (int t = 0; t < MB; ++t) {
                int32x2_t a0 = {};
                int32x2_t a1 = {};
                if ((row_start + static_cast<uint32_t>(t) * kTileRow + m_lane) < rows) {
                    const uint8_t* ap = arow[t] + aoff;
                    a0 = *reinterpret_cast<const int32x2_t*>(ap);
                    a1 = *reinterpret_cast<const int32x2_t*>(ap + 8u);
                }
                float8_t c = float8_t MXFP4_ZERO8;
                c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
                c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if constexpr (kFold)
                        acc[t][j] = __builtin_fmaf(c[j], asv[t][j], acc[t][j]);
                    else
                        acc[t][j] = __builtin_fmaf(c[j], ws * asv[t][j], acc[t][j]);
                }
            }
        }
    }

    __shared__ float red[kSplit][MB][8][32];
    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            red[warp][t][j][lane] = acc[t][j];
    }
    detail::lds_barrier();

    if (warp != 0u)
        return;

    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        float s[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            s[j] = 0.0f;
        #pragma unroll
        for (int w = 0; w < kSplit; ++w) {
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                s[j] += red[w][t][j][lane];
        }
        if (!out_valid)
            continue;
        const uint32_t r_base = row_start + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t r = r_base + static_cast<uint32_t>(j);
            if (r >= rows)
                continue;
            float v = s[j];
            if constexpr (kFold)
                v *= epi;
            const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
            if (output_dtype == static_cast<uint32_t>(Mxfp4GemmOutputDType::BF16)) {
                static_cast<shelf::bf16_t*>(output)[out_idx] =
                    shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

constexpr uint32_t k2dThreads = 256u;
constexpr uint32_t k2dRowsPerBlock = kMxfp4GemmPrefill2dRowsPerBlock;
constexpr uint32_t k2dKChunk = 128u;
constexpr uint32_t k2dOutBlockFixed = 64u;

// Prefill: one block evaluates 256 activation rows x kBn outputs with 8 waves.
// Each wave expands its share of the 16-output x 32-K weight blocks into LDS
// once per 128-K chunk and reuses them across 4 activation row tiles, so the
// E2M1 -> E4M3 expansion (the dominant ALU cost of this format) is paid once
// per weight fragment instead of once per row tile. E8M0 scales are staged as
// bytes and applied per 32-K block, exactly like the row-block kernels, so the
// accumulation order and the result are unchanged.
// kActRescale == false is the measurement ablation of the A-side scale: the
// activation group scale is not applied in the K loop at all and the WMMA
// fragments chain straight into the accumulator. It is numerically wrong and is
// only instantiated by launch_gemm_mxfp4_w4a8_ablated_noactrescale.
template <uint32_t kBn, bool kGuarded, bool kFold, bool kActRescale = true>
__global__ void __launch_bounds__(k2dThreads) gemm_mxfp4_w4a8_prefill_2d_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ weight_ref,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t activation_code_stride,
    const uint32_t activation_scale_stride,
    const uint32_t output_row_stride) {
    constexpr uint32_t kNtTotal = kBn / kTileOut;
    constexpr uint32_t kKb = k2dKChunk / kGroupElements;
    constexpr uint32_t kNtPerWave = kNtTotal / 2u;
    constexpr uint32_t kSlotsPerWave = (kNtTotal * kKb) / 8u;
    static_assert(kBn == 64u || kBn == 128u, "prefill 2D output block");

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t m_lane = lane & 15u;
    const uint32_t k_group = lane >> 4u;
    const uint32_t wave_m = wave & 3u;
    const uint32_t wave_n = wave >> 2u;

    const uint32_t out_block_start = blockIdx.x * kBn;
    const uint32_t row_start = blockIdx.y * k2dRowsPerBlock;
    if (out_block_start >= out_features || row_start >= rows)
        return;

    const uint32_t nb = k_padded >> 5;
    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const uint32_t row_base = row_start + wave_m * 4u * kTileRow;

    // The activation fragment block is addressed by lane slot, exactly as in the
    // row-block kernels: lane L carries the L&15 output row of the tile and the
    // L>>4 half of each 16-K fragment.
    const uint8_t* arow[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        arow[t] = activation_codes +
            (static_cast<size_t>(row_base >> 4) + static_cast<size_t>(t)) *
                (static_cast<size_t>(activation_code_stride) * 16u) +
            static_cast<size_t>(lane) * kFragLaneBytes;
    }

    __shared__ int32x2_t w0f_lds[kNtTotal][kKb][32];
    __shared__ int32x2_t w1f_lds[kNtTotal][kKb][32];
    __shared__ uint8_t ws_lds[kNtTotal][kKb][16];
    __shared__ uint2 mag_lds[16];
    uint8_t ref_lds[kNtTotal];
    if constexpr (kFold) {
        if (wave == 0u)
            mxfp4_mag_lds(mag_lds, lane);
        #pragma unroll
        for (uint32_t nt = 0; nt < kNtTotal; ++nt) {
            const uint32_t o_ref = out_block_start + nt * kTileOut + m_lane;
            ref_lds[nt] = (!kGuarded || o_ref < out_features) ? weight_ref[o_ref] : 0u;
        }
        detail::lds_barrier();
    }

    float8_t acc[4][kNtPerWave];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (uint32_t nt = 0; nt < kNtPerWave; ++nt)
            acc[t][nt] = float8_t MXFP4_ZERO8;

    for (uint32_t cb = 0; cb < k_padded / k2dKChunk; ++cb) {
        #pragma unroll
        for (uint32_t s = 0; s < kSlotsPerWave; ++s) {
            const uint32_t slot = wave + s * 8u;
            const uint32_t nt = slot % kNtTotal;
            const uint32_t kb = slot / kNtTotal;
            const uint32_t o_stage = out_block_start + nt * kTileOut + m_lane;
            const bool stage_valid = !kGuarded || (o_stage < out_features);
            if (stage_valid) {
                const uint32_t ib = cb * kKb + kb;
                const uint8_t* wrow = weight_codes +
                    static_cast<size_t>(o_stage >> 4) * (static_cast<size_t>(nb) * kWeightBlockBytes) +
                    static_cast<size_t>(o_stage & 15u) * 16u + k_group * 8u;
                int32x2_t w0 = {};
                int32x2_t w1 = {};
                if constexpr (kFold) {
                    const uint32_t d = mxfp4_fold_d(
                        ref_lds[nt], weight_scales[static_cast<size_t>(o_stage) *
                                                   weight_scale_stride + ib]);
                    const uint2 mg = mag_lds[d];
                    mxfp4_weight_pair(wrow + static_cast<size_t>(ib) * kWeightBlockBytes,
                                      mg.x, mg.y, w0, w1);
                } else {
                    mxfp4_weight_pair(wrow + static_cast<size_t>(ib) * kWeightBlockBytes,
                                      kFoldMag[0][0], kFoldMag[0][1], w0, w1);
                    if (k_group == 0u) {
                        ws_lds[nt][kb][m_lane] = weight_scales[
                            static_cast<size_t>(o_stage) * weight_scale_stride + ib];
                    }
                }
                w0f_lds[nt][kb][lane] = w0;
                w1f_lds[nt][kb][lane] = w1;
            }
        }
        detail::lds_barrier();

        float asv[4][8];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t rr = row_base + static_cast<uint32_t>(t) * kTileRow +
                                    8u * k_group + static_cast<uint32_t>(j);
                asv[t][j] = (!kGuarded || rr < rows)
                    ? activation_scales[static_cast<size_t>(rr) * as_row_floats + cb]
                    : 0.0f;
            }
        }

        #pragma unroll
        for (uint32_t kb = 0; kb < kKb; ++kb) {
            int32x2_t wf0[kNtPerWave];
            int32x2_t wf1[kNtPerWave];
            float wsv[kNtPerWave];
            #pragma unroll
            for (uint32_t nt = 0; nt < kNtPerWave; ++nt) {
                const uint32_t nt_glob = wave_n * kNtPerWave + nt;
                wf0[nt] = w0f_lds[nt_glob][kb][lane];
                wf1[nt] = w1f_lds[nt_glob][kb][lane];
                wsv[nt] = mxfp4_e8m0_f32(ws_lds[nt_glob][kb][m_lane]);
            }
            const uint32_t ib = cb * kKb + kb;
            const uint32_t aoff = (ib >> 1) * kFragBlockBytes + (ib & 1u) * 16u;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                int32x2_t a0 = {};
                int32x2_t a1 = {};
                if (!kGuarded ||
                    (row_base + static_cast<uint32_t>(t) * kTileRow + m_lane) < rows) {
                    const uint8_t* ap = arow[t] + aoff;
                    a0 = *reinterpret_cast<const int32x2_t*>(ap);
                    a1 = *reinterpret_cast<const int32x2_t*>(ap + 8u);
                }
                #pragma unroll
                for (uint32_t nt = 0; nt < kNtPerWave; ++nt) {
                    if constexpr (!kActRescale) {
                        float8_t c = acc[t][nt];
                        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, wf0[nt], c);
                        acc[t][nt] =
                            __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, wf1[nt], c);
                        continue;
                    }
                    float8_t c = float8_t MXFP4_ZERO8;
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, wf0[nt], c);
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, wf1[nt], c);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        if constexpr (kFold)
                            acc[t][nt][j] = __builtin_fmaf(c[j], asv[t][j], acc[t][nt][j]);
                        else
                            acc[t][nt][j] = __builtin_fmaf(
                                c[j], wsv[nt] * asv[t][j], acc[t][nt][j]);
                    }
                }
            }
        }
        detail::lds_barrier();
    }

    #pragma unroll
    for (uint32_t nt = 0; nt < kNtPerWave; ++nt) {
        const uint32_t o_self = out_block_start + (wave_n * kNtPerWave + nt) * kTileOut + m_lane;
        if (kGuarded && o_self >= out_features)
            continue;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const uint32_t r_base = row_base + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t r = r_base + static_cast<uint32_t>(j);
                if (r >= rows)
                    continue;
                float v = acc[t][nt][j];
                if constexpr (kFold)
                    v *= mxfp4_e8m0_f32(ref_lds[wave_n * kNtPerWave + nt]);
                const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
                if (output_dtype == static_cast<uint32_t>(Mxfp4GemmOutputDType::BF16)) {
                    static_cast<shelf::bf16_t*>(output)[out_idx] =
                        shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
                } else {
                    static_cast<float*>(output)[out_idx] = v;
                }
            }
        }
    }
}

#define MXFP4_MB_LAUNCH(MB)                                                      \
    do {                                                                         \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,               \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1);   \
        if (weight_ref != nullptr) {                                             \
            gemm_mxfp4_w4a8_wmma_kernel<(MB), true>                              \
                <<<grid, kThreadsPerBlock, 0, stream>>>(                         \
                    weight_codes, weight_scales, weight_ref, acodes,             \
                    activation_scales, output,                                   \
                    static_cast<uint32_t>(output_dtype), rows, out_features,     \
                    k_padded, weight_scale_stride_bytes,                         \
                    activation_code_stride_bytes, activation_scale_stride_bytes, \
                    output_row_stride);                                          \
        } else {                                                                 \
        gemm_mxfp4_w4a8_wmma_kernel<(MB), false>                                 \
            <<<grid, kThreadsPerBlock, 0, stream>>>(                             \
                weight_codes, weight_scales, weight_ref, acodes, activation_scales, output,  \
                static_cast<uint32_t>(output_dtype), rows, out_features,         \
                k_padded, weight_scale_stride_bytes,                             \
                activation_code_stride_bytes, activation_scale_stride_bytes,     \
                output_row_stride);                                              \
        }                                                                        \
        return hipGetLastError();                                                \
    } while (0)

#define MXFP4_SPLITK_LAUNCH(MB, SPLIT)                                           \
    do {                                                                         \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,               \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1);   \
        if (weight_ref != nullptr) {                                             \
            gemm_mxfp4_w4a8_splitk_kernel<(MB), (SPLIT), true>                   \
                <<<grid, kThreadsPerBlock * (SPLIT), 0, stream>>>(               \
                    weight_codes, weight_scales, weight_ref, acodes,             \
                    activation_scales, output,                                   \
                    static_cast<uint32_t>(output_dtype), rows, out_features,     \
                    k_padded, weight_scale_stride_bytes,                         \
                    activation_code_stride_bytes, activation_scale_stride_bytes, \
                    output_row_stride);                                          \
        } else {                                                                 \
        gemm_mxfp4_w4a8_splitk_kernel<(MB), (SPLIT), false>                      \
            <<<grid, kThreadsPerBlock * (SPLIT), 0, stream>>>(                   \
                weight_codes, weight_scales, weight_ref, acodes, activation_scales, output,  \
                static_cast<uint32_t>(output_dtype), rows, out_features,         \
                k_padded, weight_scale_stride_bytes,                             \
                activation_code_stride_bytes, activation_scale_stride_bytes,     \
                output_row_stride);                                              \
        }                                                                        \
        return hipGetLastError();                                                \
    } while (0)

}  // namespace

bool mxfp4_splitk_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded) {
    return rows >= 1u && rows <= kMxfp4GemmSplitKMaxRows && out_features != 0u &&
           (out_features & 15u) == 0u && (k_padded & 127u) == 0u;
}

hipError_t launch_gemm_mxfp4_w4a8_wmma(
    const Mxfp4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* weight_ref,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (rows == 0u || out_features == 0u || k_padded == 0u)
        return hipSuccess;
    if ((k_padded & 127u) != 0u)
        return hipErrorNotSupported;
    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    if (config.id == Mxfp4GemmConfigId::Prefill2D) {
        // BN 128 was measured and rejected: the non-folded per-K32 rescale needs
        // 16 float8 accumulators per wave and the kernel loses 1.8x to BN 64.
        constexpr uint32_t k2dOutBlock = k2dOutBlockFixed;
        if ((k_padded % k2dKChunk) != 0u || (rows % k2dRowsPerBlock) != 0u ||
            out_features < k2dOutBlock)
            return hipErrorNotSupported;
        const dim3 grid2d((out_features + k2dOutBlock - 1u) / k2dOutBlock,
                          rows / k2dRowsPerBlock, 1);
        const bool guarded = (out_features % k2dOutBlock) != 0u;
        const bool fold = weight_ref != nullptr;
        if (guarded) {
            if (fold) {
                gemm_mxfp4_w4a8_prefill_2d_kernel<k2dOutBlock, true, true><<<grid2d, k2dThreads, 0, stream>>>(
                    weight_codes, weight_scales, weight_ref, acodes, activation_scales, output,
                    static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                    weight_scale_stride_bytes, activation_code_stride_bytes,
                    activation_scale_stride_bytes, output_row_stride);
            } else {
                gemm_mxfp4_w4a8_prefill_2d_kernel<k2dOutBlock, true, false><<<grid2d, k2dThreads, 0, stream>>>(
                    weight_codes, weight_scales, weight_ref, acodes, activation_scales, output,
                    static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                    weight_scale_stride_bytes, activation_code_stride_bytes,
                    activation_scale_stride_bytes, output_row_stride);
            }
        } else {
            if (fold) {
                gemm_mxfp4_w4a8_prefill_2d_kernel<k2dOutBlock, false, true><<<grid2d, k2dThreads, 0, stream>>>(
                    weight_codes, weight_scales, weight_ref, acodes, activation_scales, output,
                    static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                    weight_scale_stride_bytes, activation_code_stride_bytes,
                    activation_scale_stride_bytes, output_row_stride);
            } else {
                gemm_mxfp4_w4a8_prefill_2d_kernel<k2dOutBlock, false, false><<<grid2d, k2dThreads, 0, stream>>>(
                    weight_codes, weight_scales, weight_ref, acodes, activation_scales, output,
                    static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                    weight_scale_stride_bytes, activation_code_stride_bytes,
                    activation_scale_stride_bytes, output_row_stride);
            }
        }
        return hipGetLastError();
    }
    switch (config.id) {
    case Mxfp4GemmConfigId::RowBlock4:
        MXFP4_MB_LAUNCH(4);
    case Mxfp4GemmConfigId::RowBlock2:
        MXFP4_MB_LAUNCH(2);
    case Mxfp4GemmConfigId::RowBlock1:
    default:
        MXFP4_MB_LAUNCH(1);
    }
}

hipError_t launch_gemm_mxfp4_w4a8_splitk(
    uint32_t split_k,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* weight_ref,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (rows == 0u || out_features == 0u || k_padded == 0u)
        return hipSuccess;
    if ((k_padded & 127u) != 0u || (out_features & 15u) != 0u)
        return hipErrorNotSupported;
    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    const uint32_t mb = rows <= 16u ? 1u : 2u;
    if (mb == 1u) {
        if (split_k == 2u) {
            MXFP4_SPLITK_LAUNCH(1, 2);
        } else if (split_k == 8u) {
            MXFP4_SPLITK_LAUNCH(1, 8);
        }
        MXFP4_SPLITK_LAUNCH(1, 4);
    }
    if (split_k == 2u) {
        MXFP4_SPLITK_LAUNCH(2, 2);
    } else if (split_k == 4u) {
        MXFP4_SPLITK_LAUNCH(2, 4);
    }
    MXFP4_SPLITK_LAUNCH(2, 8);
}

hipError_t launch_gemm_mxfp4_w4a8_ablated_noactrescale(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* weight_ref,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (!weight_ref || (k_padded % k2dKChunk) != 0u || (rows % k2dRowsPerBlock) != 0u ||
        (out_features % k2dOutBlockFixed) != 0u)
        return hipErrorNotSupported;
    const dim3 grid2d(out_features / k2dOutBlockFixed, rows / k2dRowsPerBlock, 1);
    gemm_mxfp4_w4a8_prefill_2d_kernel<k2dOutBlockFixed, false, true, false>
        <<<grid2d, k2dThreads, 0, stream>>>(
            weight_codes, weight_scales, weight_ref,
            reinterpret_cast<const uint8_t*>(activation_codes), activation_scales, output,
            static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
            weight_scale_stride_bytes, activation_code_stride_bytes,
            activation_scale_stride_bytes, output_row_stride);
    return hipGetLastError();
}

hipError_t launch_gemm_mxfp4_w4a8_auto(
    const Mxfp4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const uint8_t* weight_ref,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Mxfp4GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (config.id == Mxfp4GemmConfigId::SplitK) {
        return launch_gemm_mxfp4_w4a8_splitk(
            config.split_k, weight_codes, weight_scales, weight_ref, activation_codes,
            activation_scales, output, output_dtype, rows, out_features, k_padded,
            weight_scale_stride_bytes, activation_code_stride_bytes,
            activation_scale_stride_bytes, output_row_stride, stream);
    }
    return launch_gemm_mxfp4_w4a8_wmma(
        config, weight_codes, weight_scales, weight_ref, activation_codes, activation_scales,
        output, output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream);
}

#undef MXFP4_MB_LAUNCH
#undef MXFP4_SPLITK_LAUNCH
#undef MXFP4_ZERO8

}  // namespace shelf::kernel
