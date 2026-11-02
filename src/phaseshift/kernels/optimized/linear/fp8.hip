#include <phaseshift/kernels/optimized/linear/fp8.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace ps::kernel {

namespace {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = kFp8GemmRowTile;
constexpr uint32_t kThreadsPerBlock = 32u;

__device__ __forceinline__ size_t fp8_act_off(uint32_t r, uint32_t code_stride) {
    return static_cast<size_t>(r >> 4) * (static_cast<size_t>(code_stride) * 16u) +
           static_cast<size_t>(r & 15u) * 16u;
}

template <int MB>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_fp8_w8a8_wmma_kernel(
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
    const uint32_t nb = k_padded >> 5;
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
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(k_padded) * 16u) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const float* wsrow = weight_scales +
        static_cast<size_t>(o_self >> 7) * (weight_scale_stride / 4u);

    const uint8_t* arow[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        const uint32_t r = row_start + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + fp8_act_off(r, activation_code_stride) + k_group * 8u;
    }

    float8_t acc[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t)
        acc[t] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const bool full_tile = (out_start + kTileOut <= out_features);

    auto run_k_loop = [&]<bool kGuarded>() {
        const uint32_t groups = nb >> 2u;
        const uint32_t as_row_floats = activation_scale_stride >> 2u;
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
            __syncthreads();
            const uint32_t ib0 = ag * 4u;
            #pragma unroll 2
            for (uint32_t ibl = 0; ibl < 4u; ++ibl) {
                const uint32_t ib = ib0 + ibl;
                int32x2_t w0 = {};
                int32x2_t w1 = {};
                float ws = 0.0f;
                if constexpr (kGuarded) {
                    if (out_valid) {
                        const uint8_t* wp = wrow + static_cast<size_t>(ib) * 512u;
                        w0 = *reinterpret_cast<const int32x2_t*>(wp);
                        w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
                        ws = wsrow[ib >> w_scale_shift];
                    }
                } else {
                    const uint8_t* wp = wrow + static_cast<size_t>(ib) * 512u;
                    w0 = *reinterpret_cast<const int32x2_t*>(wp);
                    w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
                    ws = wsrow[ib >> w_scale_shift];
                }
                #pragma unroll
                for (int t = 0; t < MB; ++t) {
                    int32x2_t a0 = {};
                    int32x2_t a1 = {};
                    if constexpr (kGuarded) {
                        const bool rv =
                            (row_start + static_cast<uint32_t>(t) * kTileRow + m_lane) < rows;
                        if (rv) {
                            const uint8_t* ap = arow[t] + static_cast<size_t>(ib) * 512u;
                            a0 = *reinterpret_cast<const int32x2_t*>(ap);
                            a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                        }
                    } else {
                        const uint8_t* ap = arow[t] + static_cast<size_t>(ib) * 512u;
                        a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                    }
                    float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                    const uint32_t as_slot =
                        (k_group * MB + static_cast<uint32_t>(t)) * 8u;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[t][j] = __builtin_fmaf(
                            c[j], ws * smem[as_slot + static_cast<uint32_t>(j)], acc[t][j]);
                }
            }
            __syncthreads();
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
            if (output_dtype == static_cast<uint32_t>(Fp8GemmOutputDType::BF16)) {
                static_cast<::ps::bf16_t*>(output)[out_idx] =
                    ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

template <Fp8GemmOutputDType kOutputType, int kUnroll>
__global__ void __launch_bounds__(kThreadsPerBlock)
gemv_fp8_w8a8_decode1_kernel(
    const uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride,
    const uint32_t w_scale_shift) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4u;

    const uint32_t o_self = blockIdx.x * kTileOut + m_lane;
    const uint32_t nb = k_padded >> 5u;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4u) * (static_cast<size_t>(k_padded) * 16u) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const float* wsrow = weight_scales +
        static_cast<size_t>(o_self >> 7u) * (weight_scale_stride / 4u);
    const uint8_t* arow = activation_codes + k_group * 8u;

    float acc0 = 0.0f;
    #pragma unroll kUnroll
    for (uint32_t ib = 0; ib < nb; ++ib) {
        const uint8_t* wp = wrow + static_cast<size_t>(ib) * 512u;
        const int32x2_t w0 = *reinterpret_cast<const int32x2_t*>(wp);
        const int32x2_t w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
        const float ws = wsrow[ib >> w_scale_shift];
        const float as = activation_scales[ib >> 2u];
        const uint8_t* ap = arow + static_cast<size_t>(ib) * 512u;
        const int32x2_t a0 = *reinterpret_cast<const int32x2_t*>(ap);
        const int32x2_t a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
        float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
        acc0 = __builtin_fmaf(c[0], ws * as, acc0);
    }

    if (k_group == 0u) {
        const float v = acc0;
        if constexpr (kOutputType == Fp8GemmOutputDType::BF16) {
            static_cast<::ps::bf16_t*>(output)[o_self] =
                ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
        } else {
            static_cast<float*>(output)[o_self] = v;
        }
    }
}

using u8v8_t = uint8_t __attribute__((ext_vector_type(8)));

constexpr uint32_t k2dThreads = 256u;
constexpr uint32_t k2dOutBlock = kFp8GemmPrefill2dOutBlock;

// Prefill path: one block computes a 256 (rows) x 64 (outputs) tile with 8
// waves. Each wave stages one 16-output x 64-K weight tile into shared memory
// so the weight is read once and reused across 16 rows. The FP8 128x128 scale
// is uniform across a 16-output tile.
template <bool kGuarded>
__global__ void __launch_bounds__(k2dThreads) gemm_fp8_w8a8_wmma_prefill_2d_kernel(
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
    const uint32_t nb2 = k_padded >> 6;
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
        const uint32_t r = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + fp8_act_off(r, activation_code_stride) + k_group * 8u;
    }

    float8_t acc[4][2];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const uint32_t g2_total = nb2 >> 1u;
    for (uint32_t g2 = 0; g2 < g2_total; ++g2) {
        {
            {
                const uint32_t wm = tid >> 6u;
                const uint32_t rem = tid & 63u;
                const uint32_t kg = rem >> 5u;
                const uint32_t t = (rem >> 3u) & 3u;
                const uint32_t j = rem & 7u;
                const uint32_t rr =
                    row_start + wm * 4u * kTileRow + t * kTileRow + 8u * kg + j;
                smem[(g2 & 1u) * 256u + tid] = (rr < rows)
                                ? activation_scales[static_cast<size_t>(rr) * as_row_floats + g2]
                                : 0.0f;
            }
        }
        __syncthreads();
        for (uint32_t inner = 0; inner < 2u; ++inner) {
            const uint32_t ib2 = g2 * 2u + inner;
            int32x2_t w0c[2][2];
            int32x2_t w1c[2][2];
            float wsc[2][2];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint32_t o_nt = out_block_start +
                    (wave_n * 2u + static_cast<uint32_t>(nt)) * kTileOut + m_lane;
                if (kGuarded && o_nt >= out_features) {
                    #pragma unroll
                    for (int il = 0; il < 2; ++il) {
                        w0c[nt][il] = int32x2_t{};
                        w1c[nt][il] = int32x2_t{};
                        wsc[nt][il] = 0.0f;
                    }
                    continue;
                }
                const uint8_t* wbase = weight_codes +
                    static_cast<size_t>(o_nt >> 4) * (static_cast<size_t>(k_padded) * 16u) +
                    static_cast<size_t>(o_nt & 15u) * 16u + k_group * 8u;
                const float* wsbase = weight_scales +
                    static_cast<size_t>(o_nt >> 7) * (weight_scale_stride / 4u);
                #pragma unroll
                for (int il = 0; il < 2; ++il) {
                    const uint32_t ib_glob = ib2 * 2u + static_cast<uint32_t>(il);
                    const uint8_t* wpn = wbase + static_cast<size_t>(ib_glob) * 512u;
                    w0c[nt][il] = *reinterpret_cast<const int32x2_t*>(wpn);
                    w1c[nt][il] = *reinterpret_cast<const int32x2_t*>(wpn + 256u);
                    wsc[nt][il] = wsbase[ib_glob >> w_scale_shift];
                }
            }
            #pragma unroll 2
            for (uint32_t ib_local = 0; ib_local < 2; ++ib_local) {
                int32x2_t w0[2];
                int32x2_t w1[2];
                float ws[2];
                const uint32_t ib_glob = ib2 * 2u + ib_local;
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    w0[nt] = w0c[nt][ib_local];
                    w1[nt] = w1c[nt][ib_local];
                    ws[nt] = wsc[nt][ib_local];
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint8_t* ap = arow[t] + static_cast<size_t>(ib_glob) * 512u;
                    const int32x2_t a0 = *reinterpret_cast<const int32x2_t*>(ap);
                    const int32x2_t a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                    const uint32_t as_slot =
                        ((wave_m * 2u + k_group) * 4u + static_cast<uint32_t>(t)) * 8u;
                    #pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0[nt], c);
                        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1[nt], c);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            acc[t][nt][j] = __builtin_fmaf(
                                c[j],
                                ws[nt] * smem[(g2 & 1u) * 256u + as_slot +
                                              static_cast<uint32_t>(j)],
                                acc[t][nt][j]);
                    }
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
                if (output_dtype == static_cast<uint32_t>(Fp8GemmOutputDType::BF16)) {
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
    gemm_fp8_w8a8_wmma_prefill_2d_wide_kernel(
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
    const uint32_t nb2 = k_padded >> 6;
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
            acc[t][nt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const uint32_t g2_total = nb2 >> 1u;

    const uint8_t* arow[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t r = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + fp8_act_off(r, activation_code_stride) + k_group * 8u;
    }

    for (uint32_t g2 = 0; g2 < g2_total; ++g2) {
        if (tid < 128u)
            s_as[g2 & 1u][tid] = activation_scales[static_cast<size_t>(row_start + tid) *
                                                       as_row_floats +
                                                   g2];
        __syncthreads();
        if constexpr (kDeferScale) {
            float ws_g2[2];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint32_t o_nt = out_block_start +
                    (ww * 2u + static_cast<uint32_t>(nt)) * kTileOut + m_lane;
                ws_g2[nt] = weight_scales[static_cast<size_t>(o_nt >> 7u) *
                                              (weight_scale_stride / 4u) +
                                          g2];
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint32_t o_nt = out_block_start +
                    (ww * 2u + static_cast<uint32_t>(nt)) * kTileOut + m_lane;
                const uint8_t* wb = weight_codes +
                    static_cast<size_t>(o_nt >> 4u) *
                        (static_cast<size_t>(k_padded) * 16u) +
                    static_cast<size_t>(o_nt & 15u) * 16u + k_group * 8u;
                float8_t c[4];
                #pragma unroll
                for (int tt = 0; tt < 4; ++tt)
                    c[tt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                #pragma unroll
                for (uint32_t inner = 0; inner < 2u; ++inner) {
                    const uint32_t ib2 = g2 * 2u + inner;
                    #pragma unroll
                    for (uint32_t il = 0; il < 2u; ++il) {
                        const uint32_t ib_glob = ib2 * 2u + il;
                        const size_t koff = static_cast<size_t>(ib_glob) * 512u;
                        const uint8_t* wpn = wb + koff;
                        const int32x2_t w0 = *reinterpret_cast<const int32x2_t*>(wpn);
                        const int32x2_t w1 = *reinterpret_cast<const int32x2_t*>(wpn + 256u);
                        const uint8_t* ap0 = arow[0] + koff;
                        const uint8_t* ap1 = arow[1] + koff;
                        const uint8_t* ap2 = arow[2] + koff;
                        const uint8_t* ap3 = arow[3] + koff;
                        const int32x2_t b00 = *reinterpret_cast<const int32x2_t*>(ap0);
                        const int32x2_t b01 = *reinterpret_cast<const int32x2_t*>(ap0 + 256u);
                        const int32x2_t b10 = *reinterpret_cast<const int32x2_t*>(ap1);
                        const int32x2_t b11 = *reinterpret_cast<const int32x2_t*>(ap1 + 256u);
                        const int32x2_t b20 = *reinterpret_cast<const int32x2_t*>(ap2);
                        const int32x2_t b21 = *reinterpret_cast<const int32x2_t*>(ap2 + 256u);
                        const int32x2_t b30 = *reinterpret_cast<const int32x2_t*>(ap3);
                        const int32x2_t b31 = *reinterpret_cast<const int32x2_t*>(ap3 + 256u);
                        c[0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b00, w0, c[0]);
                        c[1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b10, w0, c[1]);
                        c[2] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b20, w0, c[2]);
                        c[3] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b30, w0, c[3]);
                        c[0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b01, w1, c[0]);
                        c[1] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b11, w1, c[1]);
                        c[2] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b21, w1, c[2]);
                        c[3] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(b31, w1, c[3]);
                    }
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint32_t as_base =
                        wa * 64u + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[t][nt][j] = __builtin_fmaf(
                            c[t][j],
                            ws_g2[nt] * s_as[g2 & 1u][as_base +
                                              static_cast<uint32_t>(j)],
                            acc[t][nt][j]);
                }
            }
        } else {
        for (uint32_t inner = 0; inner < 2u; ++inner) {
            const uint32_t ib2 = g2 * 2u + inner;
            int32x2_t w0c[2][2];
            int32x2_t w1c[2][2];
            float wsc[2][2];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const uint32_t o_nt = out_block_start +
                    (ww * 2u + static_cast<uint32_t>(nt)) * kTileOut + m_lane;
                const uint8_t* wbase = weight_codes +
                    static_cast<size_t>(o_nt >> 4u) * (static_cast<size_t>(k_padded) * 16u) +
                    static_cast<size_t>(o_nt & 15u) * 16u + k_group * 8u;
                const float* wsbase = weight_scales +
                    static_cast<size_t>(o_nt >> 7u) * (weight_scale_stride / 4u);
                #pragma unroll
                for (int il = 0; il < 2; ++il) {
                    const uint32_t ib_glob = ib2 * 2u + static_cast<uint32_t>(il);
                    const uint8_t* wpn = wbase + static_cast<size_t>(ib_glob) * 512u;
                    w0c[nt][il] = *reinterpret_cast<const int32x2_t*>(wpn);
                    w1c[nt][il] = *reinterpret_cast<const int32x2_t*>(wpn + 256u);
                    wsc[nt][il] = wsbase[ib_glob >> w_scale_shift];
                }
            }
            #pragma unroll 2
            for (uint32_t ib_local = 0; ib_local < 2; ++ib_local) {
                int32x2_t w0[2];
                int32x2_t w1[2];
                float ws[2];
                const uint32_t ib_glob = ib2 * 2u + ib_local;
                #pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    w0[nt] = w0c[nt][ib_local];
                    w1[nt] = w1c[nt][ib_local];
                    ws[nt] = wsc[nt][ib_local];
                }
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint8_t* ap = arow[t] + static_cast<size_t>(ib_glob) * 512u;
                    const int32x2_t a0 = *reinterpret_cast<const int32x2_t*>(ap);
                    const int32x2_t a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                    const uint32_t as_base = wa * 64u + static_cast<uint32_t>(t) * kTileRow +
                                             8u * k_group;
                    #pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0[nt], c);
                        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1[nt], c);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j)
                            acc[t][nt][j] = __builtin_fmaf(
                                c[j],
                                ws[nt] * s_as[g2 & 1u][as_base + static_cast<uint32_t>(j)],
                                acc[t][nt][j]);
                    }
                }
            }
        }
    }
        }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint32_t o_self = out_block_start +
                                    (ww * 2u + static_cast<uint32_t>(nt)) * kTileOut + m_lane;
            const uint32_t r_base = row_base + static_cast<uint32_t>(t) * kTileRow + 8u * k_group;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t r = r_base + static_cast<uint32_t>(j);
                const float v = acc[t][nt][j];
                const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o_self;
                if (output_dtype == static_cast<uint32_t>(Fp8GemmOutputDType::BF16)) {
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
        gemm_fp8_w8a8_wmma_kernel<(MB)>                                         \
            <<<grid, kThreadsPerBlock, shared_needed, stream>>>(                \
                weight_codes, weight_scales, acodes, activation_scales,         \
                output, static_cast<uint32_t>(output_dtype), rows,              \
                out_features, k_padded, weight_scale_stride_bytes,              \
                activation_code_stride_bytes, activation_scale_stride_bytes,    \
                output_row_stride, w_scale_shift);                              \
        return hipGetLastError();                                               \
    } while (0)
hipError_t launch_gemm_fp8_w8a8_wmma_impl(
    const Fp8GemmConfig& config,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8GemmOutputDType output_dtype,
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
    (void)acodes;
    if (config.id == Fp8GemmConfigId::Prefill2D) {
        if ((rows % (16u * kTileRow)) != 0u ||
            (out_features % k2dOutBlock) != 0u ||
            (k_padded % kFp8GemmPrefill2dKChunk) != 0u)
            return hipErrorNotSupported;
        if ((out_features % k2dOutBlockWide) == 0u) {
            const uint32_t gy2d = rows / (8u * kTileRow);
            static const uint32_t swz_g = []() -> uint32_t {
                const char* e = std::getenv("PHASESHIFT_GEMM2D_GROUP");
                if (e == nullptr || e[0] == '\0') return 4u;
                const int v = std::atoi(e);
                return (v >= 1 && v <= 64) ? static_cast<uint32_t>(v) : 4u;
            }();
            const uint32_t gs = (gy2d % swz_g == 0u) ? swz_g : 1u;
            const dim3 grid(out_features / k2dOutBlockWide, rows / (8u * kTileRow), 1);
            if (w_scale_shift == 2u) {
                gemm_fp8_w8a8_wmma_prefill_2d_wide_kernel<true>
                    <<<grid, k2dThreads, 0, stream>>>(
                        weight_codes, weight_scales, acodes, activation_scales, output,
                        static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                        weight_scale_stride_bytes, activation_code_stride_bytes,
                        activation_scale_stride_bytes, output_row_stride, w_scale_shift, gs);
            } else {
                gemm_fp8_w8a8_wmma_prefill_2d_wide_kernel<false>
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
        gemm_fp8_w8a8_wmma_prefill_2d_kernel<false>
            <<<grid, k2dThreads, shared_needed, stream>>>(
                weight_codes, weight_scales, acodes, activation_scales, output,
                static_cast<uint32_t>(output_dtype), rows, out_features, k_padded,
                weight_scale_stride_bytes, activation_code_stride_bytes,
                activation_scale_stride_bytes, output_row_stride, w_scale_shift);
        return hipGetLastError();
    }
    switch (config.id) {
    case Fp8GemmConfigId::RowBlock8:
        FP8_MB_LAUNCH(8);
    case Fp8GemmConfigId::RowBlock4:
        FP8_MB_LAUNCH(4);
    case Fp8GemmConfigId::RowBlock2:
        FP8_MB_LAUNCH(2);
    case Fp8GemmConfigId::RowBlock1:
    default:
        FP8_MB_LAUNCH(1);
    }
}

#undef FP8_MB_LAUNCH

}  // namespace

hipError_t launch_gemm_fp8_w8a8_wmma(
    const Fp8GemmConfig& config,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream,
    uint32_t weight_scale_group) {
    return launch_gemm_fp8_w8a8_wmma_impl(
        config, weight_codes, weight_scales, activation_codes, activation_scales, output,
        output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream, weight_scale_group);
}

bool fp8_decode1_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded) {
    return rows == 1u && out_features != 0u && (out_features & 15u) == 0u &&
           (k_padded & 127u) == 0u;
}

uint32_t fp8_decode1_unroll(uint32_t out_features, uint32_t k_padded) {
    (void)out_features;
    (void)k_padded;
    return 16u;
}

namespace {

template <Fp8GemmOutputDType kOutputType, int kUnroll>
void fp8_decode1_launch(
    dim3 grid,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    hipStream_t stream,
    uint32_t w_scale_shift) {
    gemv_fp8_w8a8_decode1_kernel<kOutputType, kUnroll>
        <<<grid, kThreadsPerBlock, 0, stream>>>(
            weight_codes, weight_scales, activation_codes, activation_scales, output,
            k_padded, weight_scale_stride_bytes, w_scale_shift);
}

template <Fp8GemmOutputDType kOutputType>
hipError_t fp8_launch_decode1_dtype(
    const uint8_t* weight_codes,
    const float* weight_scales,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t unroll,
    hipStream_t stream,
    uint32_t w_scale_shift) {
    const dim3 grid(out_features >> 4u);
    switch (unroll) {
        case 2u:
            fp8_decode1_launch<kOutputType, 2>(grid, weight_codes, weight_scales,
                activation_codes, activation_scales, output, k_padded,
                weight_scale_stride_bytes, stream, w_scale_shift);
            break;
        case 8u:
            fp8_decode1_launch<kOutputType, 8>(grid, weight_codes, weight_scales,
                activation_codes, activation_scales, output, k_padded,
                weight_scale_stride_bytes, stream, w_scale_shift);
            break;
        case 16u:
            fp8_decode1_launch<kOutputType, 16>(grid, weight_codes, weight_scales,
                activation_codes, activation_scales, output, k_padded,
                weight_scale_stride_bytes, stream, w_scale_shift);
            break;
        default:
            fp8_decode1_launch<kOutputType, 4>(grid, weight_codes, weight_scales,
                activation_codes, activation_scales, output, k_padded,
                weight_scale_stride_bytes, stream, w_scale_shift);
            break;
    }
    return hipGetLastError();
}

}  // namespace

hipError_t launch_gemm_fp8_w8a8_wmma_decode1(
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    uint32_t unroll,
    hipStream_t stream,
    uint32_t weight_scale_group) {
    (void)activation_code_stride_bytes;
    (void)activation_scale_stride_bytes;
    (void)output_row_stride;
    if (!fp8_decode1_supported(rows, out_features, k_padded))
        return hipErrorNotSupported;
    if (weight_scale_group != 128u && weight_scale_group != 64u)
        return hipErrorNotSupported;
    const uint32_t w_scale_shift = weight_scale_group == 128u ? 2u : 1u;
    if (unroll == 0u)
        unroll = fp8_decode1_unroll(out_features, k_padded);
    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    if (output_dtype == Fp8GemmOutputDType::BF16)
        return fp8_launch_decode1_dtype<Fp8GemmOutputDType::BF16>(
            weight_codes, weight_scales, acodes, activation_scales, output,
            out_features, k_padded, weight_scale_stride_bytes, unroll, stream,
            w_scale_shift);
    return fp8_launch_decode1_dtype<Fp8GemmOutputDType::F32>(
        weight_codes, weight_scales, acodes, activation_scales, output,
        out_features, k_padded, weight_scale_stride_bytes, unroll, stream,
        w_scale_shift);
}

hipError_t launch_gemm_fp8_w8a8_wmma_auto(
    const Fp8GemmConfig& config,
    const uint8_t* weight_codes,
    const float* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Fp8GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream,
    uint32_t weight_scale_group) {
    if (config.id == Fp8GemmConfigId::RowBlock1 &&
        fp8_decode1_supported(rows, out_features, k_padded)) {
        return launch_gemm_fp8_w8a8_wmma_decode1(
            weight_codes, weight_scales, activation_codes, activation_scales,
            output, output_dtype, rows, out_features, k_padded,
            weight_scale_stride_bytes, activation_code_stride_bytes,
            activation_scale_stride_bytes, output_row_stride, 0u, stream,
            weight_scale_group);
    }
    return launch_gemm_fp8_w8a8_wmma(
        config, weight_codes, weight_scales, activation_codes, activation_scales, output,
        output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream, weight_scale_group);
}

}  // namespace ps::kernel
