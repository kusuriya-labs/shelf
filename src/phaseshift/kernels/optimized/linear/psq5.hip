#include <phaseshift/kernels/optimized/linear/psq5.h>
#include <phaseshift/kernels/psq5_device.h>
#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>

namespace ps::kernel {

namespace {

using float8_t = float __attribute__((ext_vector_type(8)));
using u8v8_t = uint8_t __attribute__((ext_vector_type(8)));

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = kPsq5GemmRowTile;
constexpr uint32_t kThreadsPerBlock = 32u;
constexpr uint32_t kBlockK = 32u;
constexpr uint32_t kMagRowBytes = 16u;
constexpr uint32_t kSignRowBytes = 4u;
constexpr uint32_t kScaleRowBytes = 1u;
constexpr uint32_t kMagBlockBytes = kTileRow * kMagRowBytes;
constexpr uint32_t kSignBlockBytes = kTileRow * kSignRowBytes;
constexpr uint32_t kScaleBlockBytes = kTileRow * kScaleRowBytes;
constexpr uint32_t k2dThreads = 256u;
constexpr uint32_t k2dOutBlock = kPsq5GemmPrefill2dOutBlock;
constexpr uint32_t k2dTileKBytes = 512u;
constexpr uint32_t k2dStageBytes = 1024u;
constexpr uint32_t k2dStageScales = 32u;

__device__ __forceinline__ size_t psq5_act_off(uint32_t r, uint32_t code_stride) {
    return static_cast<size_t>(r >> 4) * (static_cast<size_t>(code_stride) * 16u) +
           static_cast<size_t>(r & 15u) * 16u;
}

template <int MB>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_psq5_w8a8_wmma_kernel(
    const uint8_t* __restrict__ weight_magnitude,
    const uint8_t* __restrict__ weight_sign,
    const uint8_t* __restrict__ weight_scale,
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

    const uint32_t wtile = o_self >> 4;
    const uint32_t wrow_in_tile = o_self & 15u;
    const uint8_t* wmag = weight_magnitude +
        static_cast<size_t>(wtile) * (static_cast<size_t>(nb) * kMagBlockBytes) +
        static_cast<size_t>(wrow_in_tile) * kMagRowBytes;
    const uint8_t* wsign = weight_sign +
        static_cast<size_t>(wtile) * (static_cast<size_t>(nb) * kSignBlockBytes) +
        static_cast<size_t>(wrow_in_tile) * kSignRowBytes;
    const uint8_t* wscale = weight_scale +
        static_cast<size_t>(wtile) * (static_cast<size_t>(weight_scale_stride)) +
        static_cast<size_t>(wrow_in_tile) * kScaleRowBytes;

    const uint8_t* arow[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t) {
        const uint32_t r = row_start + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + psq5_act_off(r, activation_code_stride) + k_group * 8u;
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
                const size_t mag_off = static_cast<size_t>(ib) * kMagBlockBytes +
                                       static_cast<size_t>(wrow_in_tile) * kMagRowBytes +
                                       static_cast<size_t>(k_group) * 4u;
                const size_t sign_off = static_cast<size_t>(ib) * kSignBlockBytes +
                                        static_cast<size_t>(wrow_in_tile) * kSignRowBytes +
                                        static_cast<size_t>(k_group);
                const uint8_t* mp = weight_magnitude +
                    static_cast<size_t>(wtile) * (static_cast<size_t>(nb) * kMagBlockBytes) + mag_off;
                const uint8_t* sp = weight_sign +
                    static_cast<size_t>(wtile) * (static_cast<size_t>(nb) * kSignBlockBytes) + sign_off;
                const uint8_t* cp = weight_scale +
                    static_cast<size_t>(wtile) * static_cast<size_t>(weight_scale_stride) +
                    static_cast<size_t>(ib) * kScaleBlockBytes + wrow_in_tile;
                if constexpr (kGuarded) {
                    if (out_valid) {
                        w0 = psq5_unpack_e4m3(*reinterpret_cast<const uint32_t*>(mp), sp[0]);
                        w1 = psq5_unpack_e4m3(*reinterpret_cast<const uint32_t*>(mp + 8u), sp[2u]);
                        ws = psq5_scale_f32(*cp);
                    }
                } else {
                    w0 = psq5_unpack_e4m3(*reinterpret_cast<const uint32_t*>(mp), sp[0]);
                    w1 = psq5_unpack_e4m3(*reinterpret_cast<const uint32_t*>(mp + 8u), sp[2u]);
                    ws = psq5_scale_f32(*cp);
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
            if (output_dtype == static_cast<uint32_t>(Psq5GemmOutputDType::BF16)) {
                static_cast<::ps::bf16_t*>(output)[out_idx] =
                    ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

template <bool kGuarded>
__global__ void __launch_bounds__(k2dThreads)
gemm_psq5_w8a8_wmma_prefill_2d_kernel(
    const uint8_t* __restrict__ weight_magnitude,
    const uint8_t* __restrict__ weight_sign,
    const uint8_t* __restrict__ weight_scale,
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
    const uint32_t swizzle_g) {
    const uint32_t nb = k_padded >> 5;
    const uint32_t nb2 = k_padded >> 6;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t wave_m = wave & 3u;
    const uint32_t wave_n = wave >> 2;
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
    const uint32_t out_block_start = bx * k2dOutBlock;
    const uint32_t row_start = by * 16u * kTileRow;

    __shared__ __attribute__((aligned(16))) float s_as[2][256];
    __shared__ __attribute__((aligned(16))) uint8_t s_w[2][2][4u * k2dStageBytes];
    __shared__ float s_ws[2][2][4u * k2dStageScales];

    if (out_block_start >= out_features || row_start >= rows)
        return;

    const uint32_t row_base = row_start + wave_m * 4u * kTileRow;
    const uint8_t* arow[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t r = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + psq5_act_off(r, activation_code_stride) + k_group * 8u;
    }

    float8_t acc[4][2];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const uint32_t g2_total = nb2 >> 1u;

    const uint32_t lane_w = wave_m * 32u + lane;
    const uint32_t ibl_lane = lane_w >> 6;
    const uint32_t row_lane = (lane_w >> 2) & 15u;
    const uint32_t b_lane = (lane_w & 3u) * 4u;
    const size_t row_mag = static_cast<size_t>(row_lane) * kMagRowBytes + b_lane;
    const size_t row_sign = static_cast<size_t>(row_lane) * kSignRowBytes + (b_lane >> 2);
    const uint32_t o_tile_n[2] = {
        out_block_start + (wave_n * 2u + 0u) * kTileOut,
        out_block_start + (wave_n * 2u + 1u) * kTileOut,
    };
    const size_t mag_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * (static_cast<size_t>(nb) * kMagBlockBytes),
        static_cast<size_t>(o_tile_n[1] >> 4) * (static_cast<size_t>(nb) * kMagBlockBytes),
    };
    const size_t sign_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * (static_cast<size_t>(nb) * kSignBlockBytes),
        static_cast<size_t>(o_tile_n[1] >> 4) * (static_cast<size_t>(nb) * kSignBlockBytes),
    };
    const size_t scale_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * weight_scale_stride,
        static_cast<size_t>(o_tile_n[1] >> 4) * weight_scale_stride,
    };

    auto stage_g2 = [&](uint32_t g2i) {
        const uint32_t par = g2i & 1u;
        const uint32_t ib0 = g2i * 4u;
        for (uint32_t s = tid; s < 256u; s += k2dThreads) {
            const uint32_t wm = s >> 6u;
            const uint32_t rem = s & 63u;
            const uint32_t kg = rem >> 5u;
            const uint32_t t = (rem >> 3u) & 3u;
            const uint32_t j = rem & 7u;
            const uint32_t rr = row_start + wm * 4u * kTileRow + t * kTileRow + 8u * kg + j;
            s_as[par][s] = (rr < rows)
                                ? activation_scales[static_cast<size_t>(rr) * as_row_floats + g2i]
                                : 0.0f;
        }
        #pragma unroll
        for (uint32_t q = 0; q < 4u; ++q) {
            const uint32_t nt = q >> 1;
            const uint32_t half = q & 1u;
            const uint32_t o_tile = o_tile_n[nt];
            const uint32_t ibl_glob = ib0 + half * 2u;
            if (kGuarded && o_tile >= out_features) {
                for (uint32_t w = lane; w < k2dStageBytes; w += 32u)
                    s_w[par][wave_n][q * k2dStageBytes + w] = 0u;
                continue;
            }
            const uint32_t ib_glob = ibl_glob + ibl_lane;
            const size_t moff = mag_tile_nt[nt] +
                                static_cast<size_t>(ib_glob) * kMagBlockBytes + row_mag;
            const uint32_t m = *reinterpret_cast<const uint32_t*>(weight_magnitude + moff);
            const uint8_t sg = weight_sign[sign_tile_nt[nt] +
                                           static_cast<size_t>(ib_glob) * kSignBlockBytes +
                                           row_sign];
            const int32x2_t v = psq5_unpack_e4m3(m, sg);
            const uint32_t k16 = b_lane >> 3;
            const uint32_t kg = (b_lane >> 2) & 1u;
            *reinterpret_cast<int32x2_t*>(&s_w[par][wave_n][q * k2dStageBytes +
                                                            ibl_lane * k2dTileKBytes +
                                                            k16 * 256u + row_lane * 16u +
                                                            kg * 8u]) = v;
        }
        {
            const uint32_t q = lane_w >> 5;
            const uint32_t ibl_s = lane >> 4;
            const uint32_t row_s = lane & 15u;
            const uint32_t nt = q >> 1;
            const uint32_t half = q & 1u;
            const uint32_t o_tile = o_tile_n[nt];
            if (kGuarded && o_tile >= out_features) {
                s_ws[par][wave_n][q * k2dStageScales + ibl_s * 16u + row_s] = 0.0f;
                return;
            }
            const uint32_t ib_glob = ib0 + half * 2u + ibl_s;
            s_ws[par][wave_n][q * k2dStageScales + ibl_s * 16u + row_s] = psq5_scale_f32(
                weight_scale[scale_tile_nt[nt] +
                             static_cast<size_t>(ib_glob) * kScaleBlockBytes + row_s]);
        }
    };

    auto compute_all = [&](uint32_t ib0, uint32_t par) {
        float8_t pd[4];
        const float8_t zf = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t nt = 0; nt < 2u; ++nt) {
            #pragma unroll
            for (uint32_t rr = 0; rr < 2u; ++rr) {
                const uint32_t round = nt * 2u + rr;
                const uint32_t half = round & 1u;
                #pragma unroll
                for (uint32_t il = 0; il < 2u; ++il) {
                    const uint32_t ib_glob = ib0 + half * 2u + il;
                    const size_t woff = static_cast<size_t>(round) * k2dStageBytes +
                                        static_cast<size_t>(il) * k2dTileKBytes +
                                        static_cast<size_t>(m_lane) * 16u +
                                        static_cast<size_t>(k_group) * 8u;
                    const int32x2_t w0 = *reinterpret_cast<const int32x2_t*>(
                        &s_w[par][wave_n][woff]);
                    const int32x2_t w1 = *reinterpret_cast<const int32x2_t*>(
                        &s_w[par][wave_n][woff + 256u]);
                    const float ws = s_ws[par][wave_n][round * k2dStageScales +
                                                       il * 16u + m_lane];
                    const bool first = (rr == 0u) && (il == 0u);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const uint8_t* ap = arow[t] +
                            static_cast<size_t>(ib_glob) * k2dTileKBytes;
                        const int32x2_t a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        const int32x2_t a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                        float8_t c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, zf);
                        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            if (first) pd[t][j] = c[j] * ws;
                            else pd[t][j] = __builtin_fmaf(c[j], ws, pd[t][j]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t as_slot =
                    ((wave_m * 2u + k_group) * 4u + static_cast<uint32_t>(t)) * 8u;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[t][nt][j] = __builtin_fmaf(
                        pd[t][j], s_as[par][as_slot + static_cast<uint32_t>(j)],
                        acc[t][nt][j]);
            }
        }
    };

    stage_g2(0);
    for (uint32_t g2 = 0; g2 < g2_total; ++g2) {
        __syncthreads();
        if (g2 + 1u < g2_total)
            stage_g2(g2 + 1u);
        const uint32_t par = g2 & 1u;
        const uint32_t ib0 = g2 * 4u;
        compute_all(ib0, par);
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint32_t o_self = out_block_start + (wave_n * 2u + static_cast<uint32_t>(nt)) *
                                    kTileOut + m_lane;
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
                if (output_dtype == static_cast<uint32_t>(Psq5GemmOutputDType::BF16)) {
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

template <bool kGuarded>
__global__ void __launch_bounds__(k2dThreads)
gemm_psq5_w8a8_wmma_prefill_2d_wide_kernel(
    const uint8_t* __restrict__ weight_magnitude,
    const uint8_t* __restrict__ weight_sign,
    const uint8_t* __restrict__ weight_scale,
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
    const uint32_t swizzle_g) {
    const uint32_t nb = k_padded >> 5;
    const uint32_t nb2 = k_padded >> 6;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t wave_m = wave & 1u;
    const uint32_t wave_n = wave >> 1u;
    const uint32_t m_lane = lane & 15u;
    const uint32_t k_group = lane >> 4u;

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

    const uint32_t row_base = row_start + wave_m * 4u * kTileRow;
    const uint8_t* arow[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t r = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + psq5_act_off(r, activation_code_stride) + k_group * 8u;
    }

    float8_t acc[4][2];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const float8_t zf = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const uint32_t g2_total = nb2 >> 1u;

    const uint32_t o_tile_n[2] = {
        out_block_start + (wave_n * 2u + 0u) * kTileOut,
        out_block_start + (wave_n * 2u + 1u) * kTileOut,
    };
    const size_t mag_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * (static_cast<size_t>(nb) * kMagBlockBytes),
        static_cast<size_t>(o_tile_n[1] >> 4) * (static_cast<size_t>(nb) * kMagBlockBytes),
    };
    const size_t sign_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * (static_cast<size_t>(nb) * kSignBlockBytes),
        static_cast<size_t>(o_tile_n[1] >> 4) * (static_cast<size_t>(nb) * kSignBlockBytes),
    };
    const size_t scale_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * weight_scale_stride,
        static_cast<size_t>(o_tile_n[1] >> 4) * weight_scale_stride,
    };
    const size_t row_mag = static_cast<size_t>(m_lane) * kMagRowBytes;
    const size_t row_sign = static_cast<size_t>(m_lane) * kSignRowBytes;

    if (tid < 128u) {
        const uint32_t s = tid;
        const uint32_t wm = s >> 6u;
        const uint32_t rem = s & 63u;
        const uint32_t kg = rem >> 5u;
        const uint32_t t = (rem >> 3u) & 3u;
        const uint32_t j = rem & 7u;
        const uint32_t rr = row_start + wm * 4u * kTileRow + t * kTileRow + 8u * kg + j;
        s_as[0][s] = (rr < rows) ? activation_scales[static_cast<size_t>(rr) * as_row_floats] : 0.0f;
    }

    for (uint32_t g2 = 0; g2 < g2_total; ++g2) {
        __syncthreads();
        if (g2 + 1u < g2_total && tid < 128u) {
            const uint32_t s = tid;
            const uint32_t wm = s >> 6u;
            const uint32_t rem = s & 63u;
            const uint32_t kg = rem >> 5u;
            const uint32_t t = (rem >> 3u) & 3u;
            const uint32_t j = rem & 7u;
            const uint32_t rr = row_start + wm * 4u * kTileRow + t * kTileRow + 8u * kg + j;
            s_as[(g2 + 1u) & 1u][s] = (rr < rows)
                ? activation_scales[static_cast<size_t>(rr) * as_row_floats + (g2 + 1u)]
                : 0.0f;
        }
        const uint32_t par = g2 & 1u;
        const uint32_t ib0 = g2 * 4u;
        float8_t pd[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t)
            pd[t] = zf;
        for (uint32_t nt = 0; nt < 2u; ++nt) {
            if (kGuarded && o_tile_n[nt] >= out_features)
                continue;
            #pragma unroll
            for (uint32_t rr = 0; rr < 2u; ++rr) {
                const uint32_t ib_g = ib0 + rr * 2u;
                #pragma unroll
                for (uint32_t il = 0; il < 2u; ++il) {
                    const uint32_t ib_glob = ib_g + il;
                    const size_t mb = mag_tile_nt[nt] +
                                      static_cast<size_t>(ib_glob) * kMagBlockBytes + row_mag +
                                      static_cast<size_t>(k_group) * 4u;
                    const size_t sb = sign_tile_nt[nt] +
                                      static_cast<size_t>(ib_glob) * kSignBlockBytes + row_sign +
                                      static_cast<size_t>(k_group);
                    const int32x2_t w0 = psq5_unpack_e4m3(
                        *reinterpret_cast<const uint32_t*>(weight_magnitude + mb),
                        weight_sign[sb]);
                    const int32x2_t w1 = psq5_unpack_e4m3(
                        *reinterpret_cast<const uint32_t*>(weight_magnitude + mb + 8u),
                        weight_sign[sb + 2u]);
                    const float ws = psq5_scale_f32(
                        weight_scale[scale_tile_nt[nt] +
                                     static_cast<size_t>(ib_glob) * kScaleBlockBytes +
                                     static_cast<size_t>(m_lane)]);
                    const bool first = (rr == 0u) && (il == 0u);
                    const size_t aoff = static_cast<size_t>(ib_glob) * k2dTileKBytes;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const uint8_t* ap = arow[t] + aoff;
                        const int32x2_t a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        const int32x2_t a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                        float8_t c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, zf);
                        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            if (first) pd[t][j] = c[j] * ws;
                            else pd[t][j] = __builtin_fmaf(c[j], ws, pd[t][j]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t as_slot =
                    ((wave_m * 2u + k_group) * 4u + static_cast<uint32_t>(t)) * 8u;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[t][nt][j] = __builtin_fmaf(
                        pd[t][j], s_as[par][as_slot + static_cast<uint32_t>(j)],
                        acc[t][nt][j]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint32_t o_self = out_block_start + (wave_n * 2u + static_cast<uint32_t>(nt)) *
                                    kTileOut + m_lane;
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
                if (output_dtype == static_cast<uint32_t>(Psq5GemmOutputDType::BF16)) {
                    static_cast<::ps::bf16_t*>(output)[out_idx] =
                        ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
                } else {
                    static_cast<float*>(output)[out_idx] = v;
                }
            }
        }
    }
}

template <bool kGuarded>
__global__ void __launch_bounds__(k2dThreads)
gemm_psq5_w8a8_wmma_prefill_2d_wstage_kernel(
    const uint8_t* __restrict__ weight_magnitude,
    const uint8_t* __restrict__ weight_sign,
    const uint8_t* __restrict__ weight_scale,
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
    const uint32_t swizzle_g) {
    const uint32_t nb = k_padded >> 5;
    const uint32_t nb2 = k_padded >> 6;
    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t wave_m = wave & 1u;
    const uint32_t wave_n = wave >> 1u;
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

    __shared__ __attribute__((aligned(16))) float s_as[2][128];
    __shared__ __attribute__((aligned(16))) uint8_t s_w[2][4][4u * k2dStageBytes];
    __shared__ float s_ws[2][4][4u * k2dStageScales];

    if (out_block_start >= out_features || row_start >= rows)
        return;

    const uint32_t row_base = row_start + wave_m * 4u * kTileRow;
    const uint8_t* arow[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const uint32_t r = row_base + static_cast<uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + psq5_act_off(r, activation_code_stride) + k_group * 8u;
    }

    float8_t acc[4][2];
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const uint32_t as_row_floats = activation_scale_stride >> 2u;
    const uint32_t g2_total = nb2 >> 1u;

    const uint32_t lane_w = wave_m * 32u + lane;
    const uint32_t row_lane = (lane_w >> 2) & 15u;
    const uint32_t b_lane = (lane_w & 3u) * 4u;
    const size_t row_mag = static_cast<size_t>(row_lane) * kMagRowBytes + b_lane;
    const size_t row_sign = static_cast<size_t>(row_lane) * kSignRowBytes + (b_lane >> 2);
    const uint32_t o_tile_n[2] = {
        out_block_start + (wave_n * 2u + 0u) * kTileOut,
        out_block_start + (wave_n * 2u + 1u) * kTileOut,
    };
    const size_t mag_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * (static_cast<size_t>(nb) * kMagBlockBytes),
        static_cast<size_t>(o_tile_n[1] >> 4) * (static_cast<size_t>(nb) * kMagBlockBytes),
    };
    const size_t sign_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * (static_cast<size_t>(nb) * kSignBlockBytes),
        static_cast<size_t>(o_tile_n[1] >> 4) * (static_cast<size_t>(nb) * kSignBlockBytes),
    };
    const size_t scale_tile_nt[2] = {
        static_cast<size_t>(o_tile_n[0] >> 4) * weight_scale_stride,
        static_cast<size_t>(o_tile_n[1] >> 4) * weight_scale_stride,
    };

    auto stage_g2 = [&](uint32_t g2i) {
        const uint32_t par = g2i & 1u;
        const uint32_t ib0 = g2i * 4u;
        for (uint32_t s = tid; s < 128u; s += k2dThreads) {
            const uint32_t wm = s >> 6u;
            const uint32_t rem = s & 63u;
            const uint32_t kg = rem >> 5u;
            const uint32_t t = (rem >> 3u) & 3u;
            const uint32_t j = rem & 7u;
            const uint32_t rr = row_start + wm * 4u * kTileRow + t * kTileRow + 8u * kg + j;
            s_as[par][s] = (rr < rows)
                                ? activation_scales[static_cast<size_t>(rr) * as_row_floats + g2i]
                                : 0.0f;
        }
        const uint32_t k16 = b_lane >> 3;
        const uint32_t kg = (b_lane >> 2) & 1u;
        #pragma unroll
        for (uint32_t q = 0; q < 4u; ++q) {
            const uint32_t nt = q >> 1;
            const uint32_t half = q & 1u;
            const uint32_t o_tile = o_tile_n[nt];
            const uint32_t ibl_glob = ib0 + half * 2u;
            if (kGuarded && o_tile >= out_features) {
                for (uint32_t w = lane; w < k2dStageBytes; w += 32u)
                    s_w[par][wave_n][q * k2dStageBytes + w] = 0u;
                continue;
            }
            #pragma unroll
            for (uint32_t il = 0; il < 2u; ++il) {
                const uint32_t ib_glob = ibl_glob + il;
                const size_t moff = mag_tile_nt[nt] +
                                    static_cast<size_t>(ib_glob) * kMagBlockBytes + row_mag;
                const uint32_t m = *reinterpret_cast<const uint32_t*>(weight_magnitude + moff);
                const uint8_t sg = weight_sign[sign_tile_nt[nt] +
                                               static_cast<size_t>(ib_glob) * kSignBlockBytes +
                                               row_sign];
                const int32x2_t v = psq5_unpack_e4m3(m, sg);
                *reinterpret_cast<int32x2_t*>(&s_w[par][wave_n][q * k2dStageBytes +
                                                                il * k2dTileKBytes +
                                                                k16 * 256u + row_lane * 16u +
                                                                kg * 8u]) = v;
            }
        }
        {
            const uint32_t ibl_s = lane >> 4;
            const uint32_t row_s = lane & 15u;
            #pragma unroll
            for (uint32_t qq = 0; qq < 2u; ++qq) {
                const uint32_t q = wave_m + qq * 2u;
                const uint32_t nt = q >> 1;
                const uint32_t half = q & 1u;
                const uint32_t o_tile = o_tile_n[nt];
                if (kGuarded && o_tile >= out_features) {
                    s_ws[par][wave_n][q * k2dStageScales + ibl_s * 16u + row_s] = 0.0f;
                    continue;
                }
                const uint32_t ib_glob = ib0 + half * 2u + ibl_s;
                s_ws[par][wave_n][q * k2dStageScales + ibl_s * 16u + row_s] = psq5_scale_f32(
                    weight_scale[scale_tile_nt[nt] +
                                 static_cast<size_t>(ib_glob) * kScaleBlockBytes + row_s]);
            }
        }
    };

    auto compute_all = [&](uint32_t ib0, uint32_t par) {
        float8_t pd[4];
        const float8_t zf = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t nt = 0; nt < 2u; ++nt) {
            #pragma unroll
            for (uint32_t rr = 0; rr < 2u; ++rr) {
                const uint32_t round = nt * 2u + rr;
                const uint32_t half = round & 1u;
                #pragma unroll
                for (uint32_t il = 0; il < 2u; ++il) {
                    const uint32_t ib_glob = ib0 + half * 2u + il;
                    const size_t woff = static_cast<size_t>(round) * k2dStageBytes +
                                        static_cast<size_t>(il) * k2dTileKBytes +
                                        static_cast<size_t>(m_lane) * 16u +
                                        static_cast<size_t>(k_group) * 8u;
                    const int32x2_t w0 = *reinterpret_cast<const int32x2_t*>(
                        &s_w[par][wave_n][woff]);
                    const int32x2_t w1 = *reinterpret_cast<const int32x2_t*>(
                        &s_w[par][wave_n][woff + 256u]);
                    const float ws = s_ws[par][wave_n][round * k2dStageScales +
                                                       il * 16u + m_lane];
                    const bool first = (rr == 0u) && (il == 0u);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const uint8_t* ap = arow[t] +
                            static_cast<size_t>(ib_glob) * k2dTileKBytes;
                        const int32x2_t a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        const int32x2_t a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                        float8_t c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, zf);
                        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            if (first) pd[t][j] = c[j] * ws;
                            else pd[t][j] = __builtin_fmaf(c[j], ws, pd[t][j]);
                        }
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint32_t as_slot =
                    ((wave_m * 2u + k_group) * 4u + static_cast<uint32_t>(t)) * 8u;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[t][nt][j] = __builtin_fmaf(
                        pd[t][j], s_as[par][as_slot + static_cast<uint32_t>(j)],
                        acc[t][nt][j]);
            }
        }
    };

    stage_g2(0);
    for (uint32_t g2 = 0; g2 < g2_total; ++g2) {
        __syncthreads();
        if (g2 + 1u < g2_total)
            stage_g2(g2 + 1u);
        const uint32_t par = g2 & 1u;
        const uint32_t ib0 = g2 * 4u;
        compute_all(ib0, par);
    }

    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint32_t o_self = out_block_start + (wave_n * 2u + static_cast<uint32_t>(nt)) *
                                    kTileOut + m_lane;
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
                if (output_dtype == static_cast<uint32_t>(Psq5GemmOutputDType::BF16)) {
                    static_cast<::ps::bf16_t*>(output)[out_idx] =
                        ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
                } else {
                    static_cast<float*>(output)[out_idx] = v;
                }
            }
        }
    }
}


template <Psq5GemmOutputDType kOutputType, int kUnroll>
__global__ void __launch_bounds__(kThreadsPerBlock)
gemv_psq5_w8a8_decode1_kernel(
    const uint8_t* __restrict__ weight_magnitude,
    const uint8_t* __restrict__ weight_sign,
    const uint8_t* __restrict__ weight_scale,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4u;

    const uint32_t o_self = blockIdx.x * kTileOut + m_lane;
    const uint32_t nb = k_padded >> 5u;

    const uint32_t wtile = o_self >> 4u;
    const uint32_t wrow_in_tile = o_self & 15u;
    const uint8_t* wmag = weight_magnitude +
        static_cast<size_t>(wtile) * (static_cast<size_t>(nb) * kMagBlockBytes) +
        static_cast<size_t>(wrow_in_tile) * kMagRowBytes + k_group * 4u;
    const uint8_t* wsign = weight_sign +
        static_cast<size_t>(wtile) * (static_cast<size_t>(nb) * kSignBlockBytes) +
        static_cast<size_t>(wrow_in_tile) * kSignRowBytes + k_group;
    const uint8_t* wscale = weight_scale +
        static_cast<size_t>(wtile) * static_cast<size_t>(weight_scale_stride) + wrow_in_tile;
    const uint8_t* arow = activation_codes + k_group * 8u;

    float acc0 = 0.0f;
    #pragma unroll kUnroll
    for (uint32_t ib = 0; ib < nb; ++ib) {
        const uint8_t* mp = wmag + static_cast<size_t>(ib) * kMagBlockBytes;
        const uint8_t* sp = wsign + static_cast<size_t>(ib) * kSignBlockBytes;
        const uint32_t m0 = *reinterpret_cast<const uint32_t*>(mp);
        const uint32_t m1 = *reinterpret_cast<const uint32_t*>(mp + 8u);
        const uint8_t s0 = sp[0];
        const uint8_t s1 = sp[2u];
        const int32x2_t w0 = psq5_unpack_e4m3(m0, s0);
        const int32x2_t w1 = psq5_unpack_e4m3(m1, s1);
        const float ws = psq5_scale_f32(wscale[static_cast<size_t>(ib) * kScaleBlockBytes]);
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
        if constexpr (kOutputType == Psq5GemmOutputDType::BF16) {
            static_cast<::ps::bf16_t*>(output)[o_self] =
                ::ps::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
        } else {
            static_cast<float*>(output)[o_self] = v;
        }
    }
}

#define PSQ5_MB_LAUNCH(MB)                                                     \
    do {                                                                       \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,             \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1); \
        const std::size_t shared_needed =                                      \
            sizeof(float) * 2u * (MB) * 8u;                                    \
        gemm_psq5_w8a8_wmma_kernel<(MB)>                                       \
            <<<grid, kThreadsPerBlock, shared_needed, stream>>>(               \
                weight_magnitude, weight_sign, weight_scale, acodes,           \
                activation_scales, output, static_cast<uint32_t>(output_dtype), \
                rows, out_features, k_padded, weight_scale_stride_bytes,       \
                activation_code_stride_bytes, activation_scale_stride_bytes,   \
                output_row_stride);                                            \
        return hipGetLastError();                                              \
    } while (0)

hipError_t launch_gemm_psq5_w8a8_wmma_impl(
    const Psq5GemmConfig& config,
    const uint8_t* weight_magnitude,
    const uint8_t* weight_sign,
    const uint8_t* weight_scale,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq5GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (rows == 0 || out_features == 0 || k_padded == 0)
        return hipSuccess;
    if ((k_padded % 128u) != 0u)
        return hipErrorNotSupported;
    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    if (config.id == Psq5GemmConfigId::Prefill2DWide) {
        if ((rows % (8u * kTileRow)) != 0u ||
            (out_features % k2dOutBlockWide) != 0u ||
            (k_padded % kPsq5GemmPrefill2dKChunk) != 0u)
            return hipErrorNotSupported;
        const uint32_t gy2d = rows / (8u * kTileRow);
        static const uint32_t swz_g = []() -> uint32_t {
            const char* e = std::getenv("PHASESHIFT_GEMM2D_GROUP");
            if (e == nullptr || e[0] == '\0') return 4u;
            const int v = std::atoi(e);
            return (v >= 1 && v <= 64) ? static_cast<uint32_t>(v) : 4u;
        }();
        const uint32_t gs = (gy2d % swz_g == 0u) ? swz_g : 1u;
        const dim3 grid(out_features / k2dOutBlockWide, rows / (8u * kTileRow), 1);
        gemm_psq5_w8a8_wmma_prefill_2d_wide_kernel<false>
            <<<grid, k2dThreads, 0, stream>>>(
                weight_magnitude, weight_sign, weight_scale, acodes,
                activation_scales, output, static_cast<uint32_t>(output_dtype),
                rows, out_features, k_padded, weight_scale_stride_bytes,
                activation_code_stride_bytes, activation_scale_stride_bytes,
                output_row_stride, gs);
        return hipGetLastError();
    }
    if (config.id == Psq5GemmConfigId::Prefill2DWStage) {
        if ((rows % (8u * kTileRow)) != 0u ||
            (out_features % k2dOutBlockWide) != 0u ||
            (k_padded % kPsq5GemmPrefill2dKChunk) != 0u)
            return hipErrorNotSupported;
        const uint32_t gy2d = rows / (8u * kTileRow);
        static const uint32_t swz_g = []() -> uint32_t {
            const char* e = std::getenv("PHASESHIFT_GEMM2D_GROUP");
            if (e == nullptr || e[0] == '\0') return 4u;
            const int v = std::atoi(e);
            return (v >= 1 && v <= 64) ? static_cast<uint32_t>(v) : 4u;
        }();
        const uint32_t gs = (gy2d % swz_g == 0u) ? swz_g : 1u;
        const dim3 grid(out_features / k2dOutBlockWide, rows / (8u * kTileRow), 1);
        gemm_psq5_w8a8_wmma_prefill_2d_wstage_kernel<false>
            <<<grid, k2dThreads, 0, stream>>>(
                weight_magnitude, weight_sign, weight_scale, acodes,
                activation_scales, output, static_cast<uint32_t>(output_dtype),
                rows, out_features, k_padded, weight_scale_stride_bytes,
                activation_code_stride_bytes, activation_scale_stride_bytes,
                output_row_stride, gs);
        return hipGetLastError();
    }
    if (config.id == Psq5GemmConfigId::Prefill2D) {
        if ((rows % (16u * kTileRow)) != 0u ||
            (out_features % k2dOutBlock) != 0u ||
            (k_padded % kPsq5GemmPrefill2dKChunk) != 0u)
            return hipErrorNotSupported;
        const uint32_t gy2d = rows / (16u * kTileRow);
        static const uint32_t swz_g = []() -> uint32_t {
            const char* e = std::getenv("PHASESHIFT_GEMM2D_GROUP");
            if (e == nullptr || e[0] == '\0') return 4u;
            const int v = std::atoi(e);
            return (v >= 1 && v <= 64) ? static_cast<uint32_t>(v) : 4u;
        }();
        const uint32_t gs = (gy2d % swz_g == 0u) ? swz_g : 1u;
        const dim3 grid(out_features / k2dOutBlock, rows / (16u * kTileRow), 1);
        gemm_psq5_w8a8_wmma_prefill_2d_kernel<false>
            <<<grid, k2dThreads, 0, stream>>>(
                weight_magnitude, weight_sign, weight_scale, acodes,
                activation_scales, output, static_cast<uint32_t>(output_dtype),
                rows, out_features, k_padded, weight_scale_stride_bytes,
                activation_code_stride_bytes, activation_scale_stride_bytes,
                output_row_stride, gs);
        return hipGetLastError();
    }
    switch (config.id) {
    case Psq5GemmConfigId::RowBlock8:
        PSQ5_MB_LAUNCH(8);
    case Psq5GemmConfigId::RowBlock4:
        PSQ5_MB_LAUNCH(4);
    case Psq5GemmConfigId::RowBlock2:
        PSQ5_MB_LAUNCH(2);
    case Psq5GemmConfigId::RowBlock1:
    default:
        PSQ5_MB_LAUNCH(1);
    }
}

#undef PSQ5_MB_LAUNCH

}

hipError_t launch_gemm_psq5_w8a8_wmma(
    const Psq5GemmConfig& config,
    const uint8_t* weight_magnitude,
    const uint8_t* weight_sign,
    const uint8_t* weight_scale,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq5GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream) {
    return launch_gemm_psq5_w8a8_wmma_impl(
        config, weight_magnitude, weight_sign, weight_scale, activation_codes,
        activation_scales, output, output_dtype, rows, out_features, k_padded,
        weight_scale_stride_bytes, activation_code_stride_bytes,
        activation_scale_stride_bytes, output_row_stride, stream);
}

bool psq5_decode1_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded) {
    return rows == 1u && out_features != 0u && (out_features & 15u) == 0u &&
           (k_padded & 127u) == 0u;
}

uint32_t psq5_decode1_unroll(uint32_t out_features, uint32_t k_padded) {
    (void)out_features;
    (void)k_padded;
    return 16u;
}

namespace {

template <Psq5GemmOutputDType kOutputType, int kUnroll>
void psq5_decode1_launch(
    dim3 grid,
    const uint8_t* weight_magnitude,
    const uint8_t* weight_sign,
    const uint8_t* weight_scale,
    const uint8_t* activation_codes,
    const float* activation_scales,
    void* output,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    hipStream_t stream) {
    gemv_psq5_w8a8_decode1_kernel<kOutputType, kUnroll>
        <<<grid, kThreadsPerBlock, 0, stream>>>(
            weight_magnitude, weight_sign, weight_scale, activation_codes,
            activation_scales, output, k_padded, weight_scale_stride_bytes);
}

}

hipError_t launch_gemm_psq5_w8a8_wmma_decode1(
    const uint8_t* weight_magnitude,
    const uint8_t* weight_sign,
    const uint8_t* weight_scale,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq5GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    uint32_t unroll,
    hipStream_t stream) {
    (void)activation_code_stride_bytes;
    (void)activation_scale_stride_bytes;
    if (rows == 0 || out_features == 0 || k_padded == 0)
        return hipSuccess;
    if (!psq5_decode1_supported(rows, out_features, k_padded))
        return hipErrorNotSupported;
    const dim3 grid((out_features + kTileOut - 1u) / kTileOut, 1, 1);
    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    if (unroll == 0u) unroll = 16u;
    if (output_dtype == Psq5GemmOutputDType::BF16) {
        if (unroll >= 16u) psq5_decode1_launch<Psq5GemmOutputDType::BF16, 16>(grid, weight_magnitude, weight_sign, weight_scale, acodes, activation_scales, output, k_padded, weight_scale_stride_bytes, stream);
        else psq5_decode1_launch<Psq5GemmOutputDType::BF16, 4>(grid, weight_magnitude, weight_sign, weight_scale, acodes, activation_scales, output, k_padded, weight_scale_stride_bytes, stream);
    } else {
        if (unroll >= 16u) psq5_decode1_launch<Psq5GemmOutputDType::F32, 16>(grid, weight_magnitude, weight_sign, weight_scale, acodes, activation_scales, output, k_padded, weight_scale_stride_bytes, stream);
        else psq5_decode1_launch<Psq5GemmOutputDType::F32, 4>(grid, weight_magnitude, weight_sign, weight_scale, acodes, activation_scales, output, k_padded, weight_scale_stride_bytes, stream);
    }
    return hipGetLastError();
}

hipError_t launch_gemm_psq5_w8a8_wmma_auto(
    const Psq5GemmConfig& config,
    const uint8_t* weight_magnitude,
    const uint8_t* weight_sign,
    const uint8_t* weight_scale,
    const int8_t* activation_codes,
    const float* activation_scales,
    void* output,
    Psq5GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (rows == 1u && psq5_decode1_supported(rows, out_features, k_padded)) {
        const hipError_t err = launch_gemm_psq5_w8a8_wmma_decode1(
            weight_magnitude, weight_sign, weight_scale, activation_codes,
            activation_scales, output, output_dtype, rows, out_features, k_padded,
            weight_scale_stride_bytes, activation_code_stride_bytes,
            activation_scale_stride_bytes, output_row_stride, psq5_decode1_unroll(out_features, k_padded),
            stream);
        if (err != hipErrorNotSupported) return err;
    }
    return launch_gemm_psq5_w8a8_wmma_impl(
        config, weight_magnitude, weight_sign, weight_scale, activation_codes,
        activation_scales, output, output_dtype, rows, out_features, k_padded,
        weight_scale_stride_bytes, activation_code_stride_bytes,
        activation_scale_stride_bytes, output_row_stride, stream);
}

}
