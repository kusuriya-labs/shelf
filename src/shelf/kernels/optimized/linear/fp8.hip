#include <shelf/kernels/optimized/linear/fp8.h>
#include <shelf/core/memory/types.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include <shelf/kernels/optimized/detail/lds_barrier_device.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace ps::kernel {

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using float8_t = float __attribute__((ext_vector_type(8)));

using namespace ps::quantization::fp8;

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = kFp8FinegrainedGemmRowTile;
constexpr uint32_t kThreadsPerBlock = 32u;

#define FP8_ZERO8 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}

#define FP8_LOAD4(p, lo, hi)                                                   \
    do {                                                                        \
        lo = *reinterpret_cast<const int32x4_t*>(p);                            \
        hi = *reinterpret_cast<const int32x4_t*>((p) + 16u);                    \
    } while (0)

#define FP8_WMM4(c, al, ah, wl, wh)                                            \
    do {                                                                        \
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(               \
            int32x2_t{(al)[0], (al)[1]}, int32x2_t{(wl)[0], (wl)[1]}, c);       \
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(               \
            int32x2_t{(al)[2], (al)[3]}, int32x2_t{(wl)[2], (wl)[3]}, c);       \
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(               \
            int32x2_t{(ah)[0], (ah)[1]}, int32x2_t{(wh)[0], (wh)[1]}, c);       \
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(               \
            int32x2_t{(ah)[2], (ah)[3]}, int32x2_t{(wh)[2], (wh)[3]}, c);       \
    } while (0)

template <int MB>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_fp8_finegrained_w8a8_wmma_kernel(
    const uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
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
    const uint32_t output_row_stride,
    const uint32_t w_scale_shift) {
    const uint32_t tid = threadIdx.x;

    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4;

    const uint32_t out_start = blockIdx.x * kTileOut;
    const uint32_t row_start = blockIdx.y * (MB * kTileRow);

    extern __shared__ float smem[];

    if (out_start >= out_features || row_start >= rows)
        return;

    const bool out_valid = out_start + m_lane < out_features;
    const uint32_t o_self = out_start + m_lane;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(out_start >> 4) * (static_cast<size_t>(k_padded) * 16u) +
        tid * kFragLaneBytes;
    const float* wsrow = weight_scales +
        static_cast<size_t>(o_self >> 7) * (weight_scale_stride / 4u);

    const uint8_t* arow[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        arow[t] = activation_codes +
            (static_cast<size_t>(row_start >> 4) + static_cast<size_t>(t)) *
                (static_cast<size_t>(activation_code_stride) * 16u) +
            tid * kFragLaneBytes;
    }

    float8_t acc[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t)
        acc[t] = float8_t FP8_ZERO8;

    const bool full_tile = (out_start + kTileOut <= out_features);

    auto run_k_loop = [&]<bool kGuarded>() {
        const uint32_t groups = k_padded >> 7;
        const uint32_t as_row_floats = activation_scale_stride >> 2u;
        const uint32_t scale_col_shift = w_scale_shift - 1u;
        for (uint32_t ag = 0; ag < groups; ++ag) {
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
            const uint32_t blk0 = ag * 2u;
            #pragma unroll
            for (uint32_t bj = 0; bj < 2u; ++bj) {
                const uint32_t blk = blk0 + bj;
                const uint8_t* wp = wrow + static_cast<size_t>(blk) * kFragBlockBytes;
                int32x4_t wl = {0, 0, 0, 0};
                int32x4_t wh = {0, 0, 0, 0};
                float ws = 0.0f;
                if constexpr (!kGuarded) {
                    FP8_LOAD4(wp, wl, wh);
                    ws = wsrow[(blk >> scale_col_shift)];
                } else if (out_valid) {
                    FP8_LOAD4(wp, wl, wh);
                    ws = wsrow[(blk >> scale_col_shift)];
                }
                #pragma unroll
                for (int t = 0; t < MB; ++t) {
                    int32x4_t al = {0, 0, 0, 0};
                    int32x4_t ah = {0, 0, 0, 0};
                    const uint8_t* ap = arow[t] + static_cast<size_t>(blk) * kFragBlockBytes;
                    if constexpr (!kGuarded) {
                        FP8_LOAD4(ap, al, ah);
                    } else if ((row_start + static_cast<uint32_t>(t) * kTileRow + m_lane) < rows) {
                        FP8_LOAD4(ap, al, ah);
                    }
                    float8_t c = float8_t FP8_ZERO8;
                    FP8_WMM4(c, al, ah, wl, wh);
                    const uint32_t as_slot = (k_group * MB + static_cast<uint32_t>(t)) * 8u;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[t][j] = __builtin_fmaf(
                            c[j], ws * smem[as_slot + static_cast<uint32_t>(j)], acc[t][j]);
                }
            }
            detail::lds_barrier();
        }
    };
    if (full_tile)
        run_k_loop.template operator()<false>();
    else
        run_k_loop.template operator()<true>();

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
            const float v = acc[t][j];
            const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
            if (output_dtype == static_cast<uint32_t>(Fp8FinegrainedGemmOutputDType::BF16)) {
                static_cast<::ps::bf16_t*>(output)[out_idx] =
                    ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

constexpr uint32_t k2dThreads = 256u;
constexpr uint32_t k2dOutBlock = kFp8FinegrainedGemmPrefill2dOutBlock;

// Prefill path: one block computes a 256 (rows) x 64 (outputs) tile with 8
// waves. Each wave reads one 16-output x 64-K weight fragment block per step,
// so the weight request per wave is one contiguous 1 KiB block. The FP8
// 128x128 scale is uniform across a 16-output tile.
template <bool kGuarded>
__global__ void __launch_bounds__(k2dThreads) gemm_fp8_finegrained_w8a8_wmma_prefill_2d_kernel(
    const uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
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
    const uint32_t output_row_stride,
    const uint32_t w_scale_shift) {
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t wave_m = wave & 3u;
    const uint32_t wave_n = wave >> 2;
    const uint32_t m_lane = lane & 15u;
    const uint32_t k_group = lane >> 4;

    const uint32_t out_block_start = blockIdx.x * k2dOutBlock;
    const uint32_t row_start = blockIdx.y * 16u * kTileRow;

    extern __shared__ float smem[];

    if (out_block_start >= out_features || row_start >= rows)
        return;

    const uint32_t row_base = row_start + wave_m * 4u * kTileRow;
    const uint8_t* arow[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        arow[t] = activation_codes +
            (static_cast<size_t>(row_base >> 4) + static_cast<size_t>(t)) *
                (static_cast<size_t>(activation_code_stride) * 16u) +
            lane * kFragLaneBytes;
    }

    float8_t acc[4][2];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = float8_t FP8_ZERO8;

    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const uint32_t groups = k_padded >> 7;
    const uint32_t scale_col_shift = w_scale_shift - 1u;
    for (uint32_t ag = 0; ag < groups; ++ag) {
        {
            const uint32_t wm = tid >> 6u;
            const uint32_t rem = tid & 63u;
            const uint32_t kg = rem >> 5u;
            const uint32_t t = (rem >> 3u) & 3u;
            const uint32_t j = rem & 7u;
            const uint32_t rr =
                row_start + wm * 4u * kTileRow + t * kTileRow + 8u * kg + j;
            smem[(ag & 1u) * 256u + tid] = (rr < rows)
                            ? activation_scales[static_cast<size_t>(rr) * as_row_floats + ag]
                            : 0.0f;
        }
        detail::lds_barrier();
        const uint32_t blk0 = ag * 2u;
        #pragma unroll
        for (uint32_t bj = 0; bj < 2u; ++bj) {
            const uint32_t blk = blk0 + bj;
            const uint32_t o_nt0 = out_block_start + wave_n * 2u * kTileOut + m_lane;
            const uint32_t o_nt1 = o_nt0 + kTileOut;
            int32x4_t wl0 = {0, 0, 0, 0};
            int32x4_t wh0 = {0, 0, 0, 0};
            int32x4_t wl1 = {0, 0, 0, 0};
            int32x4_t wh1 = {0, 0, 0, 0};
            float ws0 = 0.0f;
            float ws1 = 0.0f;
            if (!kGuarded || o_nt0 < out_features) {
                const uint8_t* wb = weight_codes +
                    static_cast<size_t>(o_nt0 >> 4) * (static_cast<size_t>(k_padded) * 16u) +
                    lane * kFragLaneBytes + static_cast<size_t>(blk) * kFragBlockBytes;
                FP8_LOAD4(wb, wl0, wh0);
                ws0 = weight_scales[static_cast<size_t>(o_nt0 >> 7) *
                                        (weight_scale_stride / 4u) +
                                    (blk >> scale_col_shift)];
            }
            if (!kGuarded || o_nt1 < out_features) {
                const uint8_t* wb = weight_codes +
                    static_cast<size_t>(o_nt1 >> 4) * (static_cast<size_t>(k_padded) * 16u) +
                    lane * kFragLaneBytes + static_cast<size_t>(blk) * kFragBlockBytes;
                FP8_LOAD4(wb, wl1, wh1);
                ws1 = weight_scales[static_cast<size_t>(o_nt1 >> 7) *
                                        (weight_scale_stride / 4u) +
                                    (blk >> scale_col_shift)];
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                int32x4_t al, ah;
                FP8_LOAD4(arow[t] + static_cast<size_t>(blk) * kFragBlockBytes, al, ah);
                const uint32_t as_slot = ((wave_m * 2u + k_group) * 4u + static_cast<uint32_t>(t)) * 8u;
                float8_t c0 = float8_t FP8_ZERO8;
                float8_t c1 = float8_t FP8_ZERO8;
                FP8_WMM4(c0, al, ah, wl0, wh0);
                FP8_WMM4(c1, al, ah, wl1, wh1);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[t][0][j] = __builtin_fmaf(
                        c0[j], ws0 * smem[(ag & 1u) * 256u + as_slot + j], acc[t][0][j]);
                    acc[t][1][j] = __builtin_fmaf(
                        c1[j], ws1 * smem[(ag & 1u) * 256u + as_slot + j], acc[t][1][j]);
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint32_t o_self = out_block_start + wave_n * 2u * kTileOut +
                                    static_cast<uint32_t>(nt) * kTileOut + m_lane;
            if (kGuarded && o_self >= out_features)
                continue;
            const uint32_t r_base = row_base + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t r = r_base + static_cast<uint32_t>(j);
                if (r >= rows)
                    continue;
                const float v = acc[t][nt][j];
                const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
                if (output_dtype == static_cast<uint32_t>(Fp8FinegrainedGemmOutputDType::BF16)) {
                    static_cast<::ps::bf16_t*>(output)[out_idx] =
                        ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
                } else {
                    static_cast<float*>(output)[out_idx] = v;
                }
            }
        }
    }
}

constexpr uint32_t k2dOutBlockWide = 128u;

template <bool kDeferScale>
__global__ void __launch_bounds__(k2dThreads)
    gemm_fp8_finegrained_w8a8_wmma_prefill_2d_wide_kernel(
        const uint8_t* __restrict__ weight_codes,
        const float* __restrict__ weight_scales,
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
        const uint32_t output_row_stride,
        const uint32_t w_scale_shift,
    const uint32_t swizzle_g) {
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t ww = wave & 3u;
    const uint32_t wa = wave >> 2;
    const uint32_t m_lane = lane & 15u;
    const uint32_t k_group = lane >> 4;

    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    if (swizzle_g > 1u) {
        const uint32_t per = swizzle_g * gridDim.x;
        const uint32_t L = bx + gridDim.x * by;
        bx = (L % per) / swizzle_g;
        by = (L / per) * swizzle_g + (L % per) % swizzle_g;
    }
    const uint32_t out_block_start = bx * k2dOutBlockWide;
    const uint32_t row_start = by * 8u * kTileRow;

    __shared__ float s_as[2][128];

    if (out_block_start >= out_features || row_start >= rows)
        return;

    const uint32_t row_base = row_start + wa * 4u * kTileRow;

    float8_t acc[4][2];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = float8_t FP8_ZERO8;

    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const uint32_t groups = k_padded >> 7;
    const uint32_t scale_col_shift = w_scale_shift - 1u;

    const uint8_t* arow[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        arow[t] = activation_codes +
            (static_cast<size_t>(row_base >> 4) + static_cast<size_t>(t)) *
                (static_cast<size_t>(activation_code_stride) * 16u) +
            lane * kFragLaneBytes;
    }

    const uint32_t o_nt0 = out_block_start + ww * 2u * kTileOut + m_lane;
    const uint32_t o_nt1 = o_nt0 + kTileOut;
    const uint8_t* wb0 = weight_codes +
        static_cast<size_t>(o_nt0 >> 4u) * (static_cast<size_t>(k_padded) * 16u) +
        lane * kFragLaneBytes;
    const uint8_t* wb1 = weight_codes +
        static_cast<size_t>(o_nt1 >> 4u) * (static_cast<size_t>(k_padded) * 16u) +
        lane * kFragLaneBytes;
    const float* ws0row = weight_scales +
        static_cast<size_t>(o_nt0 >> 7u) * (weight_scale_stride / 4u);
    const float* ws1row = weight_scales +
        static_cast<size_t>(o_nt1 >> 7u) * (weight_scale_stride / 4u);

    for (uint32_t ag = 0; ag < groups; ++ag) {
        if (tid < 128u)
            s_as[ag & 1u][tid] = activation_scales[static_cast<size_t>(row_start + tid) *
                                                       as_row_floats +
                                                   ag];
        detail::lds_barrier();
        const uint32_t blk0 = ag * 2u;
        if constexpr (kDeferScale) {
            const float ws_g0 = ws0row[ag];
            const float ws_g1 = ws1row[ag];
            float8_t c0[4];
            float8_t c1[4];
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                c0[t] = float8_t FP8_ZERO8;
                c1[t] = float8_t FP8_ZERO8;
            }
            #pragma unroll
            for (uint32_t bj = 0; bj < 2u; ++bj) {
                const uint32_t blk = blk0 + bj;
                int32x4_t wl0, wh0, wl1, wh1;
                FP8_LOAD4(wb0 + static_cast<size_t>(blk) * kFragBlockBytes, wl0, wh0);
                FP8_LOAD4(wb1 + static_cast<size_t>(blk) * kFragBlockBytes, wl1, wh1);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int32x4_t al, ah;
                    FP8_LOAD4(arow[t] + static_cast<size_t>(blk) * kFragBlockBytes, al, ah);
                    FP8_WMM4(c0[t], al, ah, wl0, wh0);
                    FP8_WMM4(c1[t], al, ah, wl1, wh1);
                }
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t as_base =
                    wa * 64u + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc[t][0][j] = __builtin_fmaf(
                        c0[t][j], ws_g0 * s_as[ag & 1u][as_base + static_cast<uint32_t>(j)],
                        acc[t][0][j]);
                    acc[t][1][j] = __builtin_fmaf(
                        c1[t][j], ws_g1 * s_as[ag & 1u][as_base + static_cast<uint32_t>(j)],
                        acc[t][1][j]);
                }
            }
        } else {
            #pragma unroll
            for (uint32_t bj = 0; bj < 2u; ++bj) {
                const uint32_t blk = blk0 + bj;
                int32x4_t wl0, wh0, wl1, wh1;
                FP8_LOAD4(wb0 + static_cast<size_t>(blk) * kFragBlockBytes, wl0, wh0);
                FP8_LOAD4(wb1 + static_cast<size_t>(blk) * kFragBlockBytes, wl1, wh1);
                const float ws_b0 = ws0row[(blk >> scale_col_shift)];
                const float ws_b1 = ws1row[(blk >> scale_col_shift)];
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    int32x4_t al, ah;
                    FP8_LOAD4(arow[t] + static_cast<size_t>(blk) * kFragBlockBytes, al, ah);
                    const uint32_t as_base =
                        wa * 64u + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
                    float8_t c0 = float8_t FP8_ZERO8;
                    float8_t c1 = float8_t FP8_ZERO8;
                    FP8_WMM4(c0, al, ah, wl0, wh0);
                    FP8_WMM4(c1, al, ah, wl1, wh1);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        acc[t][0][j] = __builtin_fmaf(
                            c0[j], ws_b0 * s_as[ag & 1u][as_base + static_cast<uint32_t>(j)],
                            acc[t][0][j]);
                        acc[t][1][j] = __builtin_fmaf(
                            c1[j], ws_b1 * s_as[ag & 1u][as_base + static_cast<uint32_t>(j)],
                            acc[t][1][j]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint32_t o_self = (nt == 0) ? o_nt0 : o_nt1;
            const uint32_t r_base = row_base + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t r = r_base + static_cast<uint32_t>(j);
                const float v = acc[t][nt][j];
                const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
                if (output_dtype == static_cast<uint32_t>(Fp8FinegrainedGemmOutputDType::BF16)) {
                    static_cast<::ps::bf16_t*>(output)[out_idx] =
                        ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
                } else {
                    static_cast<float*>(output)[out_idx] = v;
                }
            }
        }
    }
}

#define FP8_MB_LAUNCH(MB)                                                      \
    do {                                                                        \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,              \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1);  \
        const std::size_t shared_needed =                                       \
            sizeof(float) * 2u * (MB) * 8u;                                     \
        gemm_fp8_finegrained_w8a8_wmma_kernel<(MB)>                             \
            <<<grid, kThreadsPerBlock, shared_needed, stream>>>(                 \
                weight_codes, weight_scales, acodes, activation_scales,          \
                output, static_cast<uint32_t>(output_dtype), rows,               \
                out_features, k_padded, weight_scale_stride_bytes,              \
                activation_code_stride_bytes, activation_scale_stride_bytes,     \
                output_row_stride, w_scale_shift);                              \
        return hipGetLastError();                                               \
    } while (0)

hipError_t launch_gemm_fp8_finegrained_w8a8_wmma_impl(
    const Fp8FinegrainedGemmConfig& config,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8FinegrainedGemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream,
    uint32_t weight_scale_group) {
    if (rows == 0 || out_features == 0 || k_padded == 0)
        return hipSuccess;
    if ((k_padded % 128u) != 0u)
        return hipErrorNotSupported;
    uint32_t w_scale_shift = 2u;
    if (weight_scale_group == 128u) w_scale_shift = 2u;
    else if (weight_scale_group == 64u) w_scale_shift = 1u;
    else return hipErrorNotSupported;
    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    if (config.id == Fp8FinegrainedGemmConfigId::Prefill2D) {
        if ((rows % (16u * kTileRow)) != 0u ||
            (out_features % k2dOutBlock) != 0u ||
            (k_padded % kFp8FinegrainedGemmPrefill2dKChunk) != 0u)
            return hipErrorNotSupported;
        if ((out_features % k2dOutBlockWide) == 0u) {
            const uint32_t gy2d = rows / (8u * kTileRow);
            static const uint32_t swz_g = []() -> uint32_t {
                const char* e = std::getenv("SHELF_GEMM2D_GROUP");
                if (e == nullptr || e[0] == '\0') return 4u;
                const int v = std::atoi(e);
                return (v >= 1 && v <= 64) ? static_cast<uint32_t>(v) : 4u;
            }();
            const uint32_t gs = (gy2d % swz_g == 0u) ? swz_g : 1u;
            const dim3 grid(out_features / k2dOutBlockWide, rows / (8u * kTileRow), 1);
            if (w_scale_shift == 2u) {
                gemm_fp8_finegrained_w8a8_wmma_prefill_2d_wide_kernel<true>
                    <<<grid, k2dThreads, 0, stream>>>(
                        weight_codes, weight_scales, acodes, activation_scales, output,
                        static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                        weight_scale_stride_bytes, activation_code_stride_bytes,
                        activation_scale_stride_bytes, output_row_stride, w_scale_shift, gs);
            } else {
                gemm_fp8_finegrained_w8a8_wmma_prefill_2d_wide_kernel<false>
                    <<<grid, k2dThreads, 0, stream>>>(
                        weight_codes, weight_scales, acodes, activation_scales, output,
                        static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                        weight_scale_stride_bytes, activation_code_stride_bytes,
                        activation_scale_stride_bytes, output_row_stride, w_scale_shift, gs);
            }
            return hipGetLastError();
        }
        const dim3 grid(out_features / k2dOutBlock, rows / (16u * kTileRow), 1);
        constexpr std::size_t shared_needed = sizeof(float) * 512u;
        gemm_fp8_finegrained_w8a8_wmma_prefill_2d_kernel<false>
            <<<grid, k2dThreads, shared_needed, stream>>>(
                weight_codes, weight_scales, acodes, activation_scales, output,
                static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                weight_scale_stride_bytes, activation_code_stride_bytes,
                activation_scale_stride_bytes, output_row_stride, w_scale_shift);
        return hipGetLastError();
    }
    switch (config.id) {
    case Fp8FinegrainedGemmConfigId::RowBlock4:
        FP8_MB_LAUNCH(4);
    case Fp8FinegrainedGemmConfigId::RowBlock2:
        FP8_MB_LAUNCH(2);
    case Fp8FinegrainedGemmConfigId::RowBlock1:
    default:
        FP8_MB_LAUNCH(1);
    }
}

#undef FP8_MB_LAUNCH
#undef FP8_WMM4
#undef FP8_LOAD4
#undef FP8_ZERO8

}  // namespace

hipError_t launch_gemm_fp8_finegrained_w8a8_wmma(
    const Fp8FinegrainedGemmConfig& config,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8FinegrainedGemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream,
    uint32_t weight_scale_group) {
    return launch_gemm_fp8_finegrained_w8a8_wmma_impl(
        config, weight_codes, weight_scales, activation_codes, activation_scales, output,
        output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream, weight_scale_group);
}

hipError_t launch_gemm_fp8_finegrained_w8a8_wmma_auto(
    const Fp8FinegrainedGemmConfig& config,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8FinegrainedGemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream,
    uint32_t weight_scale_group) {
    return launch_gemm_fp8_finegrained_w8a8_wmma(
        config, weight_codes, weight_scales, activation_codes, activation_scales, output,
        output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream, weight_scale_group);
}

}  // namespace ps::kernel
