// shelf GEMM FP8 W8A8 device layer（gfx1201 の E4M3 行列命令）。Private header、
// 使う TU は src/backend/hip/kernels/gemm/linear_fp8.hip のみ。
//
// 断片順レイアウト（`src/quant/include/shelf/quant/fp8_w8a8.hpp` と一致させる）:
//   16 行（出力特征）× 32 K を 1 タイル（512 バイト）として
//     [K 前半 16 の 16 行][K 後半 16 の 16 行]、行内は K が 16 バイト連続。
//   lane は m_lane = tid & 15（その行）、k_group = tid >> 4（16 K の中の 8 バイト組）を担う。
//   行グループ（16 行）優先、続いて 32 K タイル:
//     offset = (row/16)*(k_padded*16) + (col/32)*512 + ((col%32)/16)*256
//              + (row%16)*16 + (col%16)
// 活性化コードも同じ形（トークン行きが 16 行グループに対応）。
//
// スケール（quant/fp8 の契約）:
//   out[m,n] = Σ_k a_code[m,k]·w_code[n,k] × a_scale[m][kb] × w_scale[nb][kb]
//   * 活性化: kb は K を 128 で切ったブロック（`act_group = 128`）。
//     `act_group = 0` のときは kb = 0（= 行全体 1 スケール）を読む。
//   * 重み: PerChannel では nb は使わず `w_scale[n]`（K に依存しない）。
//     Block128 では `w_scale[(n/128)*ws_k_blocks + kb]`（移植元 PhaseShift の native 形）。
//   いずれも **128 K ブロックごとに累積へ掛ける**（コード値は未スケールで最大 448、
//   K 全体で足すと桁上がりの打ち消し誤差が出るため。移植元と同じ手順）。
//
// 元: PhaseShift（own work）の fp8.hip（断片順と命令の使い方は変更していない）。
#pragma once

#include <cstdint>

#include <hip/hip_bf16.h>

#include <shelf/backend/hip/kernels/gemm/linear_fp8.h>

namespace shelf::hipk {
namespace gemm_fp8_detail {

using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr std::uint32_t kTileOut = 16u;      // 1 ブロックの出力特征数
constexpr std::uint32_t kTileRow = 16u;      // 1 ブロックのトークン数（行タイル）
constexpr std::uint32_t kThreadsPerBlock = 32u;
constexpr std::uint32_t kScaleKChunk = 128u;  // スケールが切り替わる K の粒度

// 活性化コードの行グループ基準オフセット（quant/fp8 の tiled_offset と同一）。
__device__ __forceinline__ std::size_t fp8_code_off(std::uint32_t row, std::uint32_t col,
                                                     std::uint32_t k_padded) {
    return static_cast<std::size_t>(row >> 4) * (static_cast<std::size_t>(k_padded) << 4) +
           static_cast<std::size_t>(col >> 5) * 512u +
           static_cast<std::size_t>((col & 31u) >> 4) * 256u +
           static_cast<std::size_t>(row & 15u) * 16u +
           static_cast<std::size_t>(col & 15u);
}

// ---------------------------------------------------------------------------
// 行列命令による本体（rows <= MB*16 を 1 ブロックで扱う）
// ---------------------------------------------------------------------------
template <int MB, bool kBlockWeightScale>
__device__ __forceinline__ void fp8_wmma_body(
    const std::uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t output_dtype,
    const std::uint32_t rows,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const std::uint32_t output_row_stride,
    const Fp8GemmScaleArgs& sc) {
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t m_lane = tid & 15u;
    const std::uint32_t k_group = tid >> 4u;

    const std::uint32_t out_start = blockIdx.x * kTileOut;
    const std::uint32_t row_start = blockIdx.y * (MB * kTileRow);
    if (out_start >= features || row_start >= rows) return;

    const bool out_valid = (out_start + m_lane) < features;
    const std::uint32_t o_self = out_start + m_lane;

    const std::uint8_t* wrow =
        weight_codes + fp8_code_off(o_self, /*col=*/0, k_padded) + k_group * 8u;

    const std::uint8_t* arow[MB];
#pragma unroll
    for (int t = 0; t < MB; ++t) {
        const std::uint32_t r = row_start + static_cast<std::uint32_t>(t) * kTileRow + m_lane;
        arow[t] = activation_codes + fp8_code_off(r, 0, k_padded) + k_group * 8u;
    }

    // PerChannel のとき重みスケールは K に依らないのでループの外へ。
    const float ws_ch = (!kBlockWeightScale && out_valid) ? weight_scales[o_self] : 0.0f;
    const std::size_t ws_row =
        kBlockWeightScale ? static_cast<std::size_t>(o_self >> 7) * sc.weight_scale_k_blocks : 0u;

    __shared__ float s_as[2u * MB * 8u];     // (k_group, t, j) -> 行のスケール

    float8_t acc[MB];
#pragma unroll
    for (int t = 0; t < MB; ++t)
        acc[t] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const std::uint32_t nb = k_padded >> 5;             // 32 K タイル数
    const std::uint32_t nag = (nb + 3u) >> 2;           // 128 K ブロック数（切り上げ）
    const bool full_tile =
        (out_start + kTileOut <= features) && (row_start + MB * kTileRow <= rows);

    auto run_k_loop = [&]<bool kGuarded>() {
        for (std::uint32_t ag = 0; ag < nag; ++ag) {
            // この 128 K ブロックで行スケールを共有メモリへ（lane ごとに 8 行ずつ）。
            for (std::uint32_t s = tid; s < 2u * MB * 8u; s += kThreadsPerBlock) {
                const std::uint32_t kg = s / (MB * 8u);
                const std::uint32_t t = (s >> 3u) % static_cast<std::uint32_t>(MB);
                const std::uint32_t j = s & 7u;
                const std::uint32_t rr =
                    row_start + t * kTileRow + 8u * kg + j;
                const std::uint32_t kb = sc.act_group ? ag : 0u;
                s_as[s] = (rr < rows)
                              ? activation_scales[static_cast<std::size_t>(rr) *
                                                      sc.activation_scale_stride + kb]
                              : 0.0f;
            }
            __syncthreads();

            float ws = ws_ch;
            if constexpr (kBlockWeightScale)
                ws = out_valid ? weight_scales[ws_row + ag] : 0.0f;

#pragma unroll
            for (std::uint32_t ibl = 0; ibl < 4u; ++ibl) {
                const std::uint32_t ib = ag * 4u + ibl;
                if (ib >= nb) break;
                int32x2_t w0 = {}, w1 = {};
                if constexpr (kGuarded) {
                    if (out_valid) {
                        const std::uint8_t* wp = wrow + static_cast<std::size_t>(ib) * 512u;
                        w0 = *reinterpret_cast<const int32x2_t*>(wp);
                        w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
                    }
                } else {
                    const std::uint8_t* wp = wrow + static_cast<std::size_t>(ib) * 512u;
                    w0 = *reinterpret_cast<const int32x2_t*>(wp);
                    w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
                }
#pragma unroll
                for (int t = 0; t < MB; ++t) {
                    int32x2_t a0 = {}, a1 = {};
                    if constexpr (kGuarded) {
                        const bool rv =
                            (row_start + static_cast<std::uint32_t>(t) * kTileRow + m_lane) <
                            rows;
                        if (rv) {
                            const std::uint8_t* ap = arow[t] + static_cast<std::size_t>(ib) * 512u;
                            a0 = *reinterpret_cast<const int32x2_t*>(ap);
                            a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                        }
                    } else {
                        const std::uint8_t* ap = arow[t] + static_cast<std::size_t>(ib) * 512u;
                        a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
                    }
                    float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
                    c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
                    const std::uint32_t as_slot = (k_group * static_cast<std::uint32_t>(MB) +
                                                   static_cast<std::uint32_t>(t)) * 8u;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        acc[t][j] = __builtin_fmaf(c[j], ws * s_as[as_slot + j], acc[t][j]);
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
        if (!out_valid) continue;
        const std::uint32_t r_base =
            row_start + static_cast<std::uint32_t>(t) * kTileRow + 8u * k_group;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t r = r_base + static_cast<std::uint32_t>(j);
            if (r >= rows) continue;
            const float v = acc[t][j];   // スケールは累積の段階で掛けている
            const std::size_t out_idx = static_cast<std::size_t>(r) * output_row_stride + o_self;
            if (output_dtype == static_cast<std::uint32_t>(Fp8GemmOutputDType::BF16)) {
                static_cast<shelf::bf16_t*>(output)[out_idx] =
                    shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
            } else {
                static_cast<float*>(output)[out_idx] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// rows == 1（decode）。1 ブロック = 16 出力特征。K を 32 の組で足し積む。
// ---------------------------------------------------------------------------
template <Fp8GemmOutputDType kOutputType, int kUnroll, bool kBlockWeightScale>
__device__ __forceinline__ void fp8_decode1_body(
    const std::uint8_t* __restrict__ weight_codes,
    const float* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const Fp8GemmScaleArgs& sc) {
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t m_lane = tid & 15u;
    const std::uint32_t k_group = tid >> 4u;

    const std::uint32_t o_self = blockIdx.x * kTileOut + m_lane;
    if (o_self >= features) return;

    const std::uint8_t* wrow = weight_codes + fp8_code_off(o_self, 0, k_padded) + k_group * 8u;
    const std::uint8_t* arow = activation_codes + k_group * 8u;

    const float ws_ch = kBlockWeightScale ? 0.0f : weight_scales[o_self];
    const std::size_t ws_row =
        kBlockWeightScale ? static_cast<std::size_t>(o_self >> 7) * sc.weight_scale_k_blocks : 0u;

    float acc0 = 0.0f;
    const std::uint32_t nb = k_padded >> 5;
#pragma unroll kUnroll
    for (std::uint32_t ib = 0; ib < nb; ++ib) {
        const std::uint8_t* wp = wrow + static_cast<std::size_t>(ib) * 512u;
        const int32x2_t w0 = *reinterpret_cast<const int32x2_t*>(wp);
        const int32x2_t w1 = *reinterpret_cast<const int32x2_t*>(wp + 256u);
        const std::uint8_t* ap = arow + static_cast<std::size_t>(ib) * 512u;
        const int32x2_t a0 = *reinterpret_cast<const int32x2_t*>(ap);
        const int32x2_t a1 = *reinterpret_cast<const int32x2_t*>(ap + 256u);
        float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
        const std::uint32_t kb = sc.act_group ? (ib >> 2u) : 0u;
        float ws = ws_ch;
        if constexpr (kBlockWeightScale) ws = weight_scales[ws_row + kb];
        // 列 = トークン。有効なのは列 0 のみで、それは k_group == 0 の lane が持つ。
        if (k_group == 0u)
            acc0 = __builtin_fmaf(c[0], ws * activation_scales[kb], acc0);
    }

    if (k_group == 0u) {
        const float v = acc0;
        if constexpr (kOutputType == Fp8GemmOutputDType::BF16) {
            static_cast<shelf::bf16_t*>(output)[o_self] =
                shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
        } else {
            static_cast<float*>(output)[o_self] = v;
        }
    }
}

}  // namespace gemm_fp8_detail
}  // namespace shelf::hipk
