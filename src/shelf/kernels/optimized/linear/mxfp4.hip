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
constexpr uint32_t kWeightBlockBytes = 256u;
constexpr uint32_t kFragBlockBytes = 1024u;
constexpr uint32_t kFragLaneBytes = 32u;

#define MXFP4_ZERO8 \
    { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }

// E2M1 -> E4M3 expansion for four packed bytes (eight codes). The native layout
// already pairs code k with code k + 16 in one byte, so the low nibbles and the
// high nibbles of a dword become the two operand registers of one 16 K WMMA
// fragment with no byte spread. The sign is folded into the v_perm selector
// (indices 0-3 pick the positive table, 4-7 the negative one).
__device__ __forceinline__ int32x2_t mxfp4_expand(uint32_t q4) {
    constexpr uint32_t mags0 = 0x3C383000u;
    constexpr uint32_t mags1 = 0x4C484440u;
    constexpr uint32_t negs0 = 0xBCB8B080u;
    constexpr uint32_t negs1 = 0xCCC8C4C0u;
    const uint32_t even = q4;
    const uint32_t odd = q4 >> 4u;
    const uint32_t even_mag = __builtin_amdgcn_perm(mags1, mags0, even & 0x07070707u);
    const uint32_t odd_mag = __builtin_amdgcn_perm(mags1, mags0, odd & 0x07070707u);
    const uint32_t even_neg = __builtin_amdgcn_perm(negs1, negs0, even & 0x07070707u);
    const uint32_t odd_neg = __builtin_amdgcn_perm(negs1, negs0, odd & 0x07070707u);
    const uint32_t sel_even = 0x03020100u | ((even & 0x08080808u) >> 1u);
    const uint32_t sel_odd = 0x03020100u | ((odd & 0x08080808u) >> 1u);
    return int32x2_t{
        static_cast<int32_t>(__builtin_amdgcn_perm(even_neg, even_mag, sel_even)),
        static_cast<int32_t>(__builtin_amdgcn_perm(odd_neg, odd_mag, sel_odd))};
}

// E8M0 is a biased power of two, so the scale is the f32 bit pattern with the
// exponent field set. The loader rejects byte 255 (the MX spec NaN); byte 0 is
// 2^-127, which is not representable as a normal f32 exponent and is emulated.
__device__ __forceinline__ float mxfp4_e8m0_f32(uint8_t byte) {
    const uint32_t bits = static_cast<uint32_t>(byte) << 23;
    return __uint_as_float(bits != 0u ? bits : 0x00400000u);
}

__device__ __forceinline__ void mxfp4_weight_pair(const uint8_t* wp, int32x2_t& w0,
                                                  int32x2_t& w1) {
    const uint64_t packed = *reinterpret_cast<const uint64_t*>(wp);
    const int32x2_t lo = mxfp4_expand(static_cast<uint32_t>(packed));
    const int32x2_t hi = mxfp4_expand(static_cast<uint32_t>(packed >> 32u));
    w0 = int32x2_t{lo[0], hi[0]};
    w1 = int32x2_t{lo[1], hi[1]};
}

template <int MB>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_mxfp4_w4a8_wmma_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
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
            (static_cast<size_t>(row_start >> 4) + static_cast<size_t>(t)) *
                (static_cast<size_t>(activation_code_stride) * 16u) +
            static_cast<size_t>(tid) * kFragLaneBytes;
    }

    __shared__ float smem[2u * MB * 8u];

    float8_t acc[MB];
    #pragma unroll
    for (int t = 0; t < MB; ++t)
        acc[t] = float8_t MXFP4_ZERO8;

    const bool full_tile = (out_start + kTileOut <= out_features);

    auto run_k_loop = [&]<bool kGuarded>() {
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

            #pragma unroll
            for (uint32_t u = 0; u < 4u; ++u) {
                const uint32_t ib = ag * 4u + u;
                int32x2_t w0 = {};
                int32x2_t w1 = {};
                float ws = 0.0f;
                if constexpr (!kGuarded) {
                    mxfp4_weight_pair(wrow + static_cast<size_t>(ib) * kWeightBlockBytes, w0, w1);
                    ws = mxfp4_e8m0_f32(wsrow[ib]);
                } else if (out_valid) {
                    mxfp4_weight_pair(wrow + static_cast<size_t>(ib) * kWeightBlockBytes, w0, w1);
                    ws = mxfp4_e8m0_f32(wsrow[ib]);
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
                    } else if (rv) {
                        const uint8_t* ap = arow[t] + aoff;
                        a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        a1 = *reinterpret_cast<const int32x2_t*>(ap + 8u);
                    }
                    float8_t c = float8_t MXFP4_ZERO8;
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                    const uint32_t slot = (k_group * MB + static_cast<uint32_t>(t)) * 8u;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[t][j] = __builtin_fmaf(
                            c[j], ws * smem[slot + static_cast<uint32_t>(j)], acc[t][j]);
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
            if (output_dtype == static_cast<uint32_t>(Mxfp4GemmOutputDType::BF16)) {
                static_cast<shelf::bf16_t*>(output)[out_idx] =
                    shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

// Single-token decode: one wave evaluates 16 outputs over the whole K. Only
// activation row 0 exists, so the epilogue keeps one accumulator per lane and
// the row guard, the rs shuffle and the dtype branch all disappear.
template <int kUnroll>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_mxfp4_w4a8_decode1_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
    const uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    bf16_t* __restrict__ output,
    const uint32_t k_padded,
    const uint32_t weight_scale_stride) {
    const uint32_t tid = threadIdx.x;
    const uint32_t m_lane = tid & 15u;
    const uint32_t k_group = tid >> 4u;
    const uint32_t o_self = blockIdx.x * kTileOut + m_lane;

    const uint32_t nb = k_padded >> 5;
    const uint32_t ng = k_padded >> 7;

    const uint8_t* wrow = weight_codes +
        static_cast<size_t>(o_self >> 4) * (static_cast<size_t>(nb) * kWeightBlockBytes) +
        static_cast<size_t>(o_self & 15u) * 16u + k_group * 8u;
    const uint8_t* wsrow = weight_scales + static_cast<size_t>(o_self) * weight_scale_stride;
    const uint8_t* arow = activation_codes + static_cast<size_t>(tid) * kFragLaneBytes;

    float acc = 0.0f;
    #pragma unroll kUnroll
    for (uint32_t ag = 0; ag < ng; ++ag) {
        const float as = activation_scales[ag];
        #pragma unroll
        for (uint32_t u = 0; u < 4u; ++u) {
            const uint32_t ib = ag * 4u + u;
            int32x2_t w0 = {};
            int32x2_t w1 = {};
            mxfp4_weight_pair(wrow + static_cast<size_t>(ib) * kWeightBlockBytes, w0, w1);
            const float ws = mxfp4_e8m0_f32(wsrow[ib]) * as;
            const uint8_t* ap = arow + (ib >> 1) * kFragBlockBytes + (ib & 1u) * 16u;
            const int32x2_t a0 = *reinterpret_cast<const int32x2_t*>(ap);
            const int32x2_t a1 = *reinterpret_cast<const int32x2_t*>(ap + 8u);
            float8_t c = float8_t MXFP4_ZERO8;
            c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
            c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
            acc = __builtin_fmaf(c[0], ws, acc);
        }
    }

    if (k_group != 0u)
        return;
    output[o_self] = bf16_t{__bfloat16_as_ushort(__float2bfloat16(acc))};
}

template <int MB, int kSplit>
__global__ void __launch_bounds__(kThreadsPerBlock * kSplit) gemm_mxfp4_w4a8_splitk_kernel(
    const uint8_t* __restrict__ weight_codes,
    const uint8_t* __restrict__ weight_scales,
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
            int32x2_t w0 = {};
            int32x2_t w1 = {};
            float ws = 0.0f;
            if (out_valid) {
                mxfp4_weight_pair(wrow + static_cast<size_t>(ib) * kWeightBlockBytes, w0, w1);
                ws = mxfp4_e8m0_f32(wsrow[ib]);
            }
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
                for (int j = 0; j < 8; ++j)
                    acc[t][j] = __builtin_fmaf(c[j], ws * asv[t][j], acc[t][j]);
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
            const float v = s[j];
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

#define MXFP4_MB_LAUNCH(MB)                                                      \
    do {                                                                         \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,               \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1);   \
        gemm_mxfp4_w4a8_wmma_kernel<(MB)>                                        \
            <<<grid, kThreadsPerBlock, 0, stream>>>(                             \
                weight_codes, weight_scales, acodes, activation_scales, output,  \
                static_cast<uint32_t>(output_dtype), rows, out_features,         \
                k_padded, weight_scale_stride_bytes,                             \
                activation_code_stride_bytes, activation_scale_stride_bytes,     \
                output_row_stride);                                              \
        return hipGetLastError();                                                \
    } while (0)

#define MXFP4_SPLITK_LAUNCH(MB, SPLIT)                                           \
    do {                                                                         \
        const dim3 grid((out_features + kTileOut - 1u) / kTileOut,               \
                        (rows + (MB) * kTileRow - 1u) / ((MB) * kTileRow), 1);   \
        gemm_mxfp4_w4a8_splitk_kernel<(MB), (SPLIT)>                             \
            <<<grid, kThreadsPerBlock * (SPLIT), 0, stream>>>(                   \
                weight_codes, weight_scales, acodes, activation_scales, output,  \
                static_cast<uint32_t>(output_dtype), rows, out_features,         \
                k_padded, weight_scale_stride_bytes,                             \
                activation_code_stride_bytes, activation_scale_stride_bytes,     \
                output_row_stride);                                              \
        return hipGetLastError();                                                \
    } while (0)

}  // namespace

bool mxfp4_decode1_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded) {
    return rows == 1u && out_features != 0u && (out_features & 15u) == 0u &&
           (k_padded & 127u) == 0u;
}

uint32_t mxfp4_decode1_unroll(uint32_t out_features, uint32_t k_padded) {
    (void)out_features;
    (void)k_padded;
    return 2u;
}

bool mxfp4_splitk_supported(uint32_t rows, uint32_t out_features, uint32_t k_padded) {
    return rows >= 2u && rows <= kMxfp4GemmSplitKMaxRows && out_features != 0u &&
           (out_features & 15u) == 0u && (k_padded & 127u) == 0u;
}

hipError_t launch_gemm_mxfp4_w4a8_wmma(
    const Mxfp4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
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

hipError_t launch_gemm_mxfp4_w4a8_decode1(
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
    const int8_t* activation_codes,
    const float* activation_scales,
    shelf::bf16_t* output,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k_padded,
    uint32_t weight_scale_stride_bytes,
    uint32_t activation_code_stride_bytes,
    uint32_t activation_scale_stride_bytes,
    uint32_t unroll,
    hipStream_t stream) {
    (void)activation_code_stride_bytes;
    (void)activation_scale_stride_bytes;
    if (!mxfp4_decode1_supported(rows, out_features, k_padded))
        return hipErrorNotSupported;
    if (unroll == 0u)
        unroll = mxfp4_decode1_unroll(out_features, k_padded);
    const auto* acodes = reinterpret_cast<const uint8_t*>(activation_codes);
    const dim3 grid(out_features >> 4u);
    switch (unroll) {
    case 1u:
        gemm_mxfp4_w4a8_decode1_kernel<1><<<grid, kThreadsPerBlock, 0, stream>>>(
            weight_codes, weight_scales, acodes, activation_scales, output, k_padded,
            weight_scale_stride_bytes);
        break;
    case 2u:
        gemm_mxfp4_w4a8_decode1_kernel<2><<<grid, kThreadsPerBlock, 0, stream>>>(
            weight_codes, weight_scales, acodes, activation_scales, output, k_padded,
            weight_scale_stride_bytes);
        break;
    case 8u:
        gemm_mxfp4_w4a8_decode1_kernel<8><<<grid, kThreadsPerBlock, 0, stream>>>(
            weight_codes, weight_scales, acodes, activation_scales, output, k_padded,
            weight_scale_stride_bytes);
        break;
    case 16u:
        gemm_mxfp4_w4a8_decode1_kernel<16><<<grid, kThreadsPerBlock, 0, stream>>>(
            weight_codes, weight_scales, acodes, activation_scales, output, k_padded,
            weight_scale_stride_bytes);
        break;
    default:
        gemm_mxfp4_w4a8_decode1_kernel<4><<<grid, kThreadsPerBlock, 0, stream>>>(
            weight_codes, weight_scales, acodes, activation_scales, output, k_padded,
            weight_scale_stride_bytes);
        break;
    }
    return hipGetLastError();
}

hipError_t launch_gemm_mxfp4_w4a8_splitk(
    uint32_t split_k,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
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

hipError_t launch_gemm_mxfp4_w4a8_auto(
    const Mxfp4GemmConfig& config,
    const uint8_t* weight_codes,
    const uint8_t* weight_scales,
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
    if (config.id == Mxfp4GemmConfigId::Decode1 &&
        mxfp4_decode1_supported(rows, out_features, k_padded) &&
        output_dtype == Mxfp4GemmOutputDType::BF16) {
        return launch_gemm_mxfp4_w4a8_decode1(
            weight_codes, weight_scales, activation_codes, activation_scales,
            static_cast<shelf::bf16_t*>(output), rows, out_features, k_padded,
            weight_scale_stride_bytes, activation_code_stride_bytes,
            activation_scale_stride_bytes, config.unroll, stream);
    }
    if (config.id == Mxfp4GemmConfigId::SplitK) {
        return launch_gemm_mxfp4_w4a8_splitk(
            config.split_k, weight_codes, weight_scales, activation_codes,
            activation_scales, output, output_dtype, rows, out_features, k_padded,
            weight_scale_stride_bytes, activation_code_stride_bytes,
            activation_scale_stride_bytes, output_row_stride, stream);
    }
    return launch_gemm_mxfp4_w4a8_wmma(
        config, weight_codes, weight_scales, activation_codes, activation_scales,
        output, output_dtype, rows, out_features, k_padded, weight_scale_stride_bytes,
        activation_code_stride_bytes, activation_scale_stride_bytes, output_row_stride,
        stream);
}

#undef MXFP4_MB_LAUNCH
#undef MXFP4_SPLITK_LAUNCH
#undef MXFP4_ZERO8

}  // namespace shelf::kernel
