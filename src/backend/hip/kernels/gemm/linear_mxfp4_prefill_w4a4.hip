// shelf backend/hip — MXFP4 の W4A4 prefill（A 側も E2M1 + E8M0）。
//
// 骨格は W4A8 版（linear_mxfp4_prefill.hip）と同じ 128 行 × 128 出力・8 波・256 スレッド。
// 違うのは**活性化もニブル**であること:
//
//   * LDS への展開: 64 K 半ごとに、行 r・断片 il・K 半 khalf について 8 B のニブルを読み
//     `mxfp4_low_high` で E4M3 の 8 バイトへ展開して、fp8 と同じ断片順に置く
//     （内側のループは fp8 版と同一のまま = `fp8_lane32` で読める）。
//     1 半あたり 128 行 × 4 断片 × 2 K 半 = 1024 個の 8 B ストア。256 スレッドで 4 個ずつ。
//   * スケール: W と A の両方が E8M0（K 32 ごと）。積 2^(eW+eA) はどちらも 2 の累乗なので
//     **厳密**。折りたたみはしない（方針）。したがって折りは 32 K ブロックごと。
//     A のスケールは 128 K グループ単位で 4 ブロック分（4 × 128 バイト）を LDS に置く。
#include <shelf/backend/hip/kernels/gemm/linear_mxfp4.h>

#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/gemm/detail/linear_fp8_device.h>
#include <shelf/quant/mxfp4.hpp>

namespace shelf::hipk {
namespace {

using gemm_fp8_detail::Fp8Frag16;
using gemm_fp8_detail::float8_t;
using gemm_fp8_detail::fp8_lane32;
using gemm_fp8_detail::fp8_load16;
using gemm_fp8_detail::int32x2_t;
using gemm_fp8_detail::kPrefill2dWideOut;
using gemm_fp8_detail::kPrefill2dWideRows;
using gemm_fp8_detail::kPrefill2dWideThreads;
using gemm_fp8_detail::kTileOut;
using gemm_fp8_detail::kTileRow;

constexpr std::uint32_t kBlockK = quant::mxfp4::kBlockK;   // 32
constexpr std::uint32_t kBlockBytes = 256u;                // 16 出力行 × 16 B

__device__ __forceinline__ int32x2_t mxfp4_low_high(std::uint32_t q4) {
    constexpr std::uint32_t mags0 = 0x3C383000u;
    constexpr std::uint32_t mags1 = 0x4C484440u;
    constexpr std::uint32_t negs0 = 0xBCB8B080u;
    constexpr std::uint32_t negs1 = 0xCCC8C4C0u;
    const std::uint32_t q_even = q4;
    const std::uint32_t q_odd = q4 >> 4u;
    const std::uint32_t v_even_low = __builtin_amdgcn_perm(mags1, mags0, q_even & 0x07070707u);
    const std::uint32_t v_odd_low = __builtin_amdgcn_perm(mags1, mags0, q_odd & 0x07070707u);
    const std::uint32_t v_even_high = __builtin_amdgcn_perm(negs1, negs0, q_even & 0x07070707u);
    const std::uint32_t v_odd_high = __builtin_amdgcn_perm(negs1, negs0, q_odd & 0x07070707u);
    const std::uint32_t mask_even = 0x03020100u | ((q_even & 0x08080808u) >> 1u);
    const std::uint32_t mask_odd = 0x03020100u | ((q_odd & 0x08080808u) >> 1u);
    return int32x2_t{
        static_cast<std::int32_t>(__builtin_amdgcn_perm(v_even_high, v_even_low, mask_even)),
        static_cast<std::int32_t>(__builtin_amdgcn_perm(v_odd_high, v_odd_low, mask_odd))};
}

// E8M0 バイト -> f32（2^(b-127)）。`ldexpf` は 32 K ごとの折りで 64 回呼ぶと重く、
// 分岐を入れると直列化してさらに遅い（実測 305 → 342 µs）。ビット組み立て 1 命令にする。
// b = 0（2^-127）は 0.0 になるが、そのブロックはコードも 0 なので積は 0 で一致する。
__device__ __forceinline__ float mxfp4_e8m0_to_f32(std::uint8_t b) {
    return __uint_as_float(static_cast<std::uint32_t>(b) << 23u);
}

__device__ __forceinline__ void mxfp4_w4a4_prefill_body(
    const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const std::uint8_t* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t output_dtype,
    const std::uint32_t rows,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs& sc) {
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t wave = tid >> 5u;
    const std::uint32_t lane = tid & 31u;
    const std::uint32_t m_lane = lane & 15u;
    const std::uint32_t k_group = lane >> 4u;

    const std::uint32_t wave_m = wave >> 2u;   // 0..1（64 行ずつ）
    const std::uint32_t wave_n = wave & 3u;    // 0..3（32 出力ずつ）

    const std::uint32_t out_start = blockIdx.y * kPrefill2dWideOut;
    const std::uint32_t row_start = blockIdx.x * kPrefill2dWideRows;
    if (out_start >= features || row_start >= rows) return;

    __shared__ std::uint8_t s_a2[2][kPrefill2dWideRows * 64u];   // 展開済み E4M3
    __shared__ std::uint8_t s_asc[2][4][kPrefill2dWideRows];     // A の E8M0（4 ブロック分）

    const std::uint32_t row0 = row_start + wave_m * 64u;
    const std::uint32_t o0 = out_start + wave_n * 32u;

    const std::uint32_t groups = k_padded / 128u;
    if (groups == 0u) return;
    const std::uint32_t nb = k_padded / kBlockK;

    const std::uint8_t* wrow[2];
    const std::uint8_t* wsrow[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const std::uint32_t o = o0 + static_cast<std::uint32_t>(nt) * kTileOut + m_lane;
        wrow[nt] = weight_codes + static_cast<std::size_t>(o >> 4u) *
                                      (static_cast<std::size_t>(nb) * kBlockBytes) +
                   static_cast<std::size_t>(o & 15u) * 16u +
                   static_cast<std::size_t>(k_group) * 8u;
        wsrow[nt] = weight_scales + static_cast<std::size_t>(o) * sc.weight_scale_stride;
    }

    const std::uint32_t r_local[4] = {wave_m * 64u + m_lane,
                                      wave_m * 64u + kTileRow + m_lane,
                                      wave_m * 64u + 2u * kTileRow + m_lane,
                                      wave_m * 64u + 3u * kTileRow + m_lane};

    float8_t acc[4][2];
#pragma unroll
    for (int t = 0; t < 4; ++t)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    // 64 K 半 h の活性化を、ニブルから E4M3 へ展開して LDS に置く。
    // 行 r・断片 il・K 半 khalf の 8 コードは、ブロック (2h + il/2) の
    // (r%16)*16 + 8*khalf から 8 バイトを読み、低位（il 偶数）または高位（il 奇数）
    // ニブルを取る。置き先は fp8 断片順: (r/16)*1024 + ((r%16)+16*khalf)*32 + il*8。
    auto stage_half = [&](std::uint32_t h) {
        const std::uint32_t hh = h & 1u;
        std::uint8_t* buf = s_a2[hh];
        // 128 行 × 4 断片 = 512 個を 256 スレッドで 2 個ずつ。1 個は行の 32 K ブロック
        // 16 バイト = **両方の K 半**を含むので、ロードは 16 B 1 回（以前は 8 B × 2）。
#pragma unroll
        for (std::uint32_t i = 0; i < 2u; ++i) {
            const std::uint32_t u = tid + i * kPrefill2dWideThreads;   // 0..511
            const std::uint32_t r = u >> 2u;                            // 0..127
            const std::uint32_t il = u & 3u;                            // 0..3
            const std::uint32_t ib = h * 2u + (il >> 1u);
            const std::uint8_t* src =
                activation_codes +
                static_cast<std::size_t>(row_start / 16u + r / 16u) *
                    (static_cast<std::size_t>(nb) * kBlockBytes) +
                static_cast<std::size_t>(ib) * kBlockBytes +
                static_cast<std::size_t>(r & 15u) * 16u;
            const uint4 raw16 = *reinterpret_cast<const uint4*>(src);   // 16 B = 32 K
            const std::uint32_t w[4] = {raw16.x, raw16.y, raw16.z, raw16.w};
#pragma unroll
            for (std::uint32_t khalf = 0; khalf < 2u; ++khalf) {
                const int32x2_t l0 = mxfp4_low_high(w[2u * khalf]);
                const int32x2_t l1 = mxfp4_low_high(w[2u * khalf + 1u]);
                const int32x2_t out8 = ((il & 1u) == 0u) ? int32x2_t{l0[0], l1[0]}
                                                         : int32x2_t{l0[1], l1[1]};
                std::uint8_t* dst = buf + static_cast<std::size_t>(r >> 4u) * 1024u +
                                    static_cast<std::size_t>((r & 15u) + 16u * khalf) * 32u +
                                    static_cast<std::size_t>(il) * 8u;
                *reinterpret_cast<int32x2_t*>(dst) = out8;
            }
        }
    };

    // 128 K グループ g の A スケール（4 ブロック × 128 行）を LDS へ。
    auto stage_asc = [&](std::uint32_t g, std::uint32_t slot) {
        for (std::uint32_t u = tid; u < 4u * kPrefill2dWideRows; u += kPrefill2dWideThreads) {
            const std::uint32_t blk = u >> 7u;        // 0..3
            const std::uint32_t r = u & 127u;         // 0..127
            const std::uint32_t rr = row_start + r;
            s_asc[slot][blk][r] =
                (rr < rows)
                    ? activation_scales[static_cast<std::size_t>(rr) * sc.activation_scale_stride +
                                        (g * 4u + blk)]
                    : 0u;
        }
    };

    stage_half(0u);
    stage_asc(0u, 0u);
    __syncthreads();

    const std::uint32_t halves = groups * 2u;
    float8_t raw[4][2];
    for (std::uint32_t h = 0; h < halves; ++h) {
        if (h + 1u < halves) stage_half(h + 1u);
        const std::uint32_t hh = h & 1u;
        const std::uint32_t g = h >> 1u;

#pragma unroll
        for (std::uint32_t il = 0; il < 2u; ++il) {
            const std::uint32_t ib = h * 2u + il;
            int32x2_t w0[2], w1[2];
            float we[2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const std::uint64_t packed = *reinterpret_cast<const std::uint64_t*>(
                    wrow[nt] + static_cast<std::size_t>(ib) * kBlockBytes);
                const int32x2_t l0 = mxfp4_low_high(static_cast<std::uint32_t>(packed));
                const int32x2_t l1 = mxfp4_low_high(static_cast<std::uint32_t>(packed >> 32u));
                w0[nt] = int32x2_t{l0[0], l1[0]};
                w1[nt] = int32x2_t{l0[1], l1[1]};
                we[nt] = mxfp4_e8m0_to_f32(wsrow[nt][ib]);
            }
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const std::uint8_t* ap =
                    fp8_lane32(s_a2[hh], r_local[t], 0, 64u, m_lane, k_group) +
                    static_cast<std::size_t>(il) * 16u;
                const Fp8Frag16 af = fp8_load16(ap);
                const int32x2_t a0 = af.lo, a1 = af.hi;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    // 32 K ブロックごとに折るので、毎回ゼロから始める（前のブロックの
                    // 値を積み上げると二重に足してしまう）。
                    const float8_t z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    raw[t][nt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a0, w0[nt], z);
                    raw[t][nt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a1, w1[nt], raw[t][nt]);
                }
            }
            // 32 K ブロックごとに折る（A の E8M0 が 32 K で変わるため）。A のスケールは
            // トークン行ごと（= j ごと）、W のスケールは出力行ごと（= レーンごと）。
            const std::uint32_t blk = ib - g * 4u;
            // A スケールは E8M0 のまま置き、読むたびにビット組み立てで f32 にする
            // （f32 で置くと LDS が 4 倍になり遅くなった。配列に先読みすると VGPR 増で遅い）。
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const std::uint32_t as_slot = wave_m * 64u +
                                              static_cast<std::uint32_t>(t) * kTileRow +
                                              8u * k_group;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const float ae = mxfp4_e8m0_to_f32(
                            s_asc[g & 1u][blk][as_slot + static_cast<std::uint32_t>(j)]);
                        acc[t][nt][j] = __builtin_fmaf(raw[t][nt][j], we[nt] * ae, acc[t][nt][j]);
                    }
            }
        }

        __syncthreads();
        // 次のグループの A スケールを別スロットへ（このタイミングなら他波は読み終えている）。
        if (hh == 1u) {
            const std::uint32_t g2 = g + 1u;
            if (g2 < groups) stage_asc(g2, (g2 & 1u));
        }
    }

#pragma unroll
    for (int t = 0; t < 4; ++t) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const std::uint32_t r = row0 + static_cast<std::uint32_t>(t) * kTileRow +
                                    8u * k_group;
            const std::uint32_t col = o0 + static_cast<std::uint32_t>(nt) * kTileOut + m_lane;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t rr = r + static_cast<std::uint32_t>(j);
                if (rr >= rows || col >= features) continue;
                const std::size_t idx = static_cast<std::size_t>(rr) * output_row_stride + col;
                if (output_dtype == static_cast<std::uint32_t>(Mxfp4GemmOutputDType::BF16)) {
                    static_cast<shelf::bf16_t*>(output)[idx] =
                        shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(acc[t][nt][j]))};
                } else {
                    static_cast<float*>(output)[idx] = acc[t][nt][j];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(kPrefill2dWideThreads) gemm_mxfp4_w4a4_prefill_kernel(
    const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const std::uint8_t* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t output_dtype,
    const std::uint32_t rows,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs sc) {
    mxfp4_w4a4_prefill_body(weight_codes, weight_scales, activation_codes, activation_scales,
                            output, output_dtype, rows, features, k_padded, output_row_stride,
                            sc);
}

}  // namespace

hipError_t launch_gemm_mxfp4_w4a4_prefill(
    const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
    const std::uint8_t* activation_codes, const std::uint8_t* activation_scales, void* output,
    Mxfp4GemmOutputDType output_dtype, std::uint32_t rows, std::uint32_t features,
    std::uint32_t k_padded, std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs& scales, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if ((k_padded % 128u) != 0u) return hipErrorInvalidValue;
    const dim3 grid((rows + kPrefill2dWideRows - 1u) / kPrefill2dWideRows,
                    (features + kPrefill2dWideOut - 1u) / kPrefill2dWideOut, 1);
    gemm_mxfp4_w4a4_prefill_kernel<<<grid, kPrefill2dWideThreads, 0, stream>>>(
        weight_codes, weight_scales, activation_codes, activation_scales, output,
        static_cast<std::uint32_t>(output_dtype), rows, features, k_padded, output_row_stride,
        scales);
    return hipGetLastError();
}

}  // namespace shelf::hipk
