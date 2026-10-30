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
using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using float8_t = float __attribute__((ext_vector_type(8)));

constexpr std::uint32_t kTileOut = 16u;      // 1 ブロックの出力特征数
constexpr std::uint32_t kTileRow = 16u;      // 1 ブロックのトークン数（行タイル）
constexpr std::uint32_t kThreadsPerBlock = 32u;
constexpr std::uint32_t kScaleKChunk = 128u;  // スケールが切り替わる K の粒度

// 活性化コードの行グループ基準オフセット（quant/fp8 の tiled_offset と同一）。
// quant/fp8 の tiled_offset と同一（16 行 × 64 K タイル = 32 レーン × 32 バイト、
// レーン l の 32 バイトは 4 断片を back-to-back）。1 波の リクエストは 1 KiB。
__device__ __forceinline__ std::size_t fp8_code_off(std::uint32_t row, std::uint32_t col,
                                                     std::uint32_t k_padded) {
    const std::uint32_t lane = (row & 15u) + 16u * ((col & 15u) >> 3);
    return static_cast<std::size_t>(row >> 4) * (static_cast<std::size_t>(k_padded) << 4) +
           static_cast<std::size_t>(col >> 6) * 1024u +
           static_cast<std::size_t>(lane) * 32u +
           static_cast<std::size_t>((col & 63u) >> 4) * 8u +
           static_cast<std::size_t>(col & 7u);
}

// 16 バイト（断片 2 個）を 1 命令で読む。レーン 32 バイトは 4 断片が連続なので、
// 32-K タイル 1 個ぶんがちょうど 16 B = dwordx4 1 回になる。
struct Fp8Frag16 {
    int32x2_t lo, hi;
};

__device__ __forceinline__ Fp8Frag16 fp8_load16(const std::uint8_t* p) {
    const int32x4_t v = *reinterpret_cast<const int32x4_t*>(p);
    Fp8Frag16 f;
    f.lo = {v[0], v[1]};
    f.hi = {v[2], v[3]};
    return f;
}

// 32 K タイル `ib` の、レーン 32 バイトのうちの 16 バイト（断片 2 個）の先頭。
__device__ __forceinline__ const std::uint8_t* fp8_wp(const std::uint8_t* lane32,
                                                      std::uint32_t ib) {
    return lane32 + static_cast<std::size_t>(ib >> 1) * 1024u +
           static_cast<std::size_t>(ib & 1u) * 16u;
}

// 64 K タイル 1 個ぶんのレーンの 32 バイト先頭（4 断片 = 8 バイト × 4）。
__device__ __forceinline__ const std::uint8_t* fp8_lane32(const std::uint8_t* base,
                                                          std::uint32_t row,
                                                          std::uint32_t k64,
                                                          std::uint32_t k_padded,
                                                          std::uint32_t m_lane,
                                                          std::uint32_t k_group) {
    const std::uint32_t lane = m_lane + 16u * k_group;
    return base + static_cast<std::size_t>(row >> 4) *
                      (static_cast<std::size_t>(k_padded) << 4) +
           static_cast<std::size_t>(k64 >> 6) * 1024u +
           static_cast<std::size_t>(lane) * 32u;
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

    const std::uint8_t* wrow = fp8_lane32(weight_codes, o_self, 0, k_padded, m_lane,
                                          k_group);

    const std::uint8_t* arow[MB];
#pragma unroll
    for (int t = 0; t < MB; ++t) {
        const std::uint32_t r = row_start + static_cast<std::uint32_t>(t) * kTileRow + m_lane;
        arow[t] = fp8_lane32(activation_codes, r, 0, k_padded, m_lane, k_group);
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
            float8_t raw[MB];
#pragma unroll
            for (int t = 0; t < MB; ++t)
                raw[t] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
            for (std::uint32_t ibl = 0; ibl < 4u; ++ibl) {
                const std::uint32_t ib = ag * 4u + ibl;
                if (ib >= nb) break;
                int32x2_t w0 = {}, w1 = {};
                if constexpr (kGuarded) {
                    if (out_valid) {
                        const std::uint8_t* wp = fp8_wp(wrow, ib);
                        w0 = *reinterpret_cast<const int32x2_t*>(wp);
                        w1 = *reinterpret_cast<const int32x2_t*>(wp + 8u);
                    }
                } else {
                    const std::uint8_t* wp = fp8_wp(wrow, ib);
                    w0 = *reinterpret_cast<const int32x2_t*>(wp);
                    w1 = *reinterpret_cast<const int32x2_t*>(wp + 8u);
                }
#pragma unroll
                for (int t = 0; t < MB; ++t) {
                    int32x2_t a0 = {}, a1 = {};
                    if constexpr (kGuarded) {
                        const bool rv =
                            (row_start + static_cast<std::uint32_t>(t) * kTileRow + m_lane) <
                            rows;
                        if (rv) {
                            const std::uint8_t* ap = fp8_wp(arow[t], ib);
                            a0 = *reinterpret_cast<const int32x2_t*>(ap);
                            a1 = *reinterpret_cast<const int32x2_t*>(ap + 8u);
                        }
                    } else {
                        const std::uint8_t* ap = fp8_wp(arow[t], ib);
                        a0 = *reinterpret_cast<const int32x2_t*>(ap);
                        a1 = *reinterpret_cast<const int32x2_t*>(ap + 8u);
                    }
                    raw[t] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, raw[t]);
                    raw[t] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, raw[t]);
                }
            }
#pragma unroll
            for (int t = 0; t < MB; ++t) {
                const std::uint32_t as_slot =
                    (k_group * static_cast<std::uint32_t>(MB) +
                     static_cast<std::uint32_t>(t)) * 8u;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[t][j] =
                        __builtin_fmaf(raw[t][j], ws * s_as[as_slot + j], acc[t][j]);
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
// Prefill 2D: 1 ブロック = 64 行 × 64 出力（8 波 = 256 スレッド）。
//
//   wave_m = wave >> 2 (0..1): 32 行ずつ / wave_n = wave & 3 (0..3): 16 出力ずつ
//   各波は A 断片 2 本（16 行 × 2）と B 断片 1 本（16 出力）を持ち、
//   32 K ごとに行列命令 2 本を積む。
//
// RowBlock（16 出力 × 64 行 / 32 スレッド）との違いは**出力方向のタイル幅**:
// 活性化のコードを 1 ブロックで 64 出力ぶん使い回せるため、活性化の読みが
// 出力タイル数（N/16 → N/64）に比例して 1/4 になる。
// 数値は RowBlock と同一（128 K ブロックごとに同じ順で累積し、同じ位置で
// スケールを掛ける）ので、置き換えても結果はビット一致する。
//
// 形の前提（呼び手が `fp8_prefill2d_supported` で確認する）:
//   rows は 64 の倍数、features は 64 の倍数、k_padded は 64 の倍数。
// ---------------------------------------------------------------------------
constexpr std::uint32_t kPrefill2dRows = 64u;

// 出力タイル幅は 64 か 32（32 は小さい N でブロック数を倍にするため）。
constexpr std::uint32_t kPrefill2dThreadsFor(std::uint32_t out_w) {
    // 4 行タイル × (out_w/16) 出力タイル = 波数。1 波 32 スレッド。
    return 32u * 4u * (out_w / 16u);
}

template <std::uint32_t kOutW>
__device__ __forceinline__ void fp8_prefill2d_body(
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
    const std::uint32_t wave = tid >> 5u;
    const std::uint32_t lane = tid & 31u;
    const std::uint32_t m_lane = lane & 15u;   // 断片の行
    const std::uint32_t k_group = lane >> 4u;  // 32 K タイルの前後半

    // 1 波 = 16 行 × 16 出力の 1 タイル。ブロックは 64 行 × kOutW 出力で、
    // 波数 = 4（行タイル）× kOutW/16（出力タイル）。
    // **波数を増やすのが要点**: 小さい N ではブロック数が 64 前後しか出ず、
    // 8 波のブロックだと resident wave が 16/CU で頭打ちになる（実測 140 GB/s）。
    // 16 波にすると同じブロック数で 32/CU まで載り、帯域が伸びる。
    constexpr std::uint32_t kNT = kOutW / 16u;
    constexpr std::uint32_t kRowTiles = kPrefill2dRows / 16u;
    const std::uint32_t wave_m = wave / kNT;   // 0..3（16 行ずつ）
    const std::uint32_t wave_n = wave % kNT;   // 0..kNT-1（16 出力ずつ）

    const std::uint32_t out_start = blockIdx.x * kOutW;
    const std::uint32_t row_start = blockIdx.y * kPrefill2dRows;
    if (out_start >= features || row_start >= rows) return;

    const std::uint32_t m_row = row_start + wave_m * kTileRow;   // この波の 16 行の先頭
    const std::uint32_t o_self = out_start + wave_n * kTileOut + m_lane;

    const std::uint8_t* wrow = fp8_lane32(weight_codes, o_self, 0, k_padded, m_lane, k_group);
    const std::uint8_t* arow =
        fp8_lane32(activation_codes, m_row + m_lane, 0, k_padded, m_lane, k_group);

    // 活性化スケールはブロック内の 64 行ぶんだけ共有メモリに置き、全波で共有する。
    __shared__ float s_as[kPrefill2dRows];

    // PerChannel のとき重みスケールは K に依らないのでループの外へ。
    const float ws_ch = sc.weight_block128 ? 0.0f : weight_scales[o_self];
    const std::size_t ws_row = sc.weight_block128
                                   ? static_cast<std::size_t>(o_self >> 7) *
                                         sc.weight_scale_k_blocks
                                   : 0u;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const std::uint32_t nb2 = k_padded >> 6u;   // 64 K タイル数
    const std::uint32_t groups = nb2 >> 1u;     // 128 K ブロック数（= スケールの粒度）

    for (std::uint32_t g = 0; g < groups; ++g) {
        if (tid < kPrefill2dRows) {
            const std::uint32_t rr = row_start + tid;
            const std::uint32_t kb = sc.act_group ? g : 0u;
            s_as[tid] = activation_scales[static_cast<std::size_t>(rr) *
                                              sc.activation_scale_stride +
                                          kb];
        }
        __syncthreads();
        const float ws = sc.weight_block128 ? weight_scales[ws_row + g] : ws_ch;
        const std::uint32_t as_slot = wave_m * kTileRow + 8u * k_group;
        float8_t raw = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (std::uint32_t inner = 0; inner < 2u; ++inner) {
            const std::uint32_t ib2 = g * 2u + inner;
#pragma unroll
            for (std::uint32_t il = 0; il < 2u; ++il) {
                const std::uint32_t ib = ib2 * 2u + il;   // 32 K タイル
                const Fp8Frag16 wf = fp8_load16(fp8_wp(wrow, ib));
                const Fp8Frag16 af = fp8_load16(fp8_wp(arow, ib));
                raw = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af.lo, wf.lo, raw);
                raw = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af.hi, wf.hi, raw);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] = __builtin_fmaf(raw[j], ws * s_as[as_slot + static_cast<std::uint32_t>(j)],
                                    acc[j]);
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const std::uint32_t r = m_row + 8u * k_group + static_cast<std::uint32_t>(j);
        const float v = acc[j];
        const std::size_t out_idx = static_cast<std::size_t>(r) * output_row_stride + o_self;
        if (output_dtype == static_cast<std::uint32_t>(Fp8GemmOutputDType::BF16)) {
            static_cast<shelf::bf16_t*>(output)[out_idx] =
                shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
        } else {
            static_cast<float*>(output)[out_idx] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Prefill の広いタイル: 128 行 × 128 出力（8 波）。
//
// 64×64 版は出力タイルが 64 幅なので、活性化（1 行 K バイト）を N/64 回読み直す。
// 512 行 × N=17408 では活性化の実効トラフィックが重みの 8 倍になり、実測 57 GB/s
// （DRAM 境界の 1/10）まで落ちていた。ここでは
//   * 出力タイルを 128 に広げて活性化の読みを半分にし、
//   * 活性化を 128 K ブロックごとに LDS へ 1 回だけ置き（16 KiB）、8 波で共有し、
//   * 1 波あたり 4 行タイル × 2 出力タイル（16 WMMA/32 K）にして行列命令密度を上げる。
// 数値は 64×64 版と同一（128 K ブロックごとに同じ順で積み、同じ位置でスケールを掛ける）。
// ---------------------------------------------------------------------------
constexpr std::uint32_t kPrefill2dWideRows = 128u;
constexpr std::uint32_t kPrefill2dWideOut = 128u;
constexpr std::uint32_t kPrefill2dWideThreads = 32u * 8u;

__device__ __forceinline__ void fp8_prefill2d_wide_body(
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
    const std::uint32_t wave = tid >> 5u;
    const std::uint32_t lane = tid & 31u;
    const std::uint32_t m_lane = lane & 15u;
    const std::uint32_t k_group = lane >> 4u;

    const std::uint32_t wave_m = wave >> 2u;   // 0..1（64 行ずつ）
    const std::uint32_t wave_n = wave & 3u;    // 0..3（32 出力ずつ）

    // グリッドは **行タイルを x、出力タイルを y** に取る。ブロックは x 優先で
    // 投入されるので、同じ出力タイル（= 同じ重み 128×K）を担当する行タイルが
    // 同時に走り、重みが L2 に残る（行 512 では 4 行タイルぶんの読みが 1 回になる）。
    // 逆順（x = 出力タイル）だと行タイルごとに 89 MB を読み直して 1.4 倍遅かった。
    const std::uint32_t out_start = blockIdx.y * kPrefill2dWideOut;
    const std::uint32_t row_start = blockIdx.x * kPrefill2dWideRows;
    if (out_start >= features || row_start >= rows) return;

    constexpr std::uint32_t kKC = 128u;
    __shared__ std::uint8_t s_a[kPrefill2dWideRows * kKC];
    // 重みも 128 K ブロックごとに LDS へ 1 回だけ置く。同じ 32 出力を担当する
    // 2 つの波（wave_m = 0/1）が二重にグローバルから読むのを避ける（実測で
    // rows=512・N=5120 が 499 µs → 約 6 割）。
    __shared__ float s_as[kPrefill2dWideRows];

    // 波が持つ 4 行タイル（16 行ずつ）と 2 出力タイル。
    const std::uint32_t row0 = row_start + wave_m * 64u;
    const std::uint32_t o0 = out_start + wave_n * 32u;

    const std::uint8_t* wrow[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        wrow[nt] = fp8_lane32(weight_codes,
                              o0 + static_cast<std::uint32_t>(nt) * kTileOut + m_lane, 0,
                              k_padded, m_lane, k_group);

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

    const float ws_ch[2] = {sc.weight_block128
                                ? 0.0f
                                : weight_scales[o0 + m_lane],
                            sc.weight_block128 ? 0.0f
                                               : weight_scales[o0 + kTileOut + m_lane]};
    const std::size_t ws_row[2] = {
        sc.weight_block128
            ? static_cast<std::size_t>((o0 + m_lane) >> 7) * sc.weight_scale_k_blocks
            : 0u,
        sc.weight_block128 ? static_cast<std::size_t>((o0 + kTileOut + m_lane) >> 7) *
                                 sc.weight_scale_k_blocks
                           : 0u};

    const std::uint32_t groups = k_padded / 128u;
    if (groups == 0u) return;

    // 活性化 LDS を **64 K × 2 バッファの ping-pong** にする（总量 16,384 B は不変、
    // 追加レジスタもない）。半 h の計算中に半 h+1 を別バッファへ運ぶので、
    // グローバル読みと WMMA が重なる。`raw` は 128 K グループの頭でゼロ初期化し、
    // 折りはグループ末尾（奇数 h）で 1 回 —— サブタイルの順序（32 K × 4）も
    // 128 K 境界でスケールを掛える位置も现状と同じなので**バイト一致**。
    __shared__ std::uint8_t s_a2[2][kPrefill2dWideRows * 64u];

    auto stage_half = [&](std::uint32_t h) {          // h: 通算の 64 K 番号
        const std::uint32_t g = h >> 1u;
        const std::uint32_t hh = h & 1u;
        std::uint8_t* buf = s_a2[hh];
#pragma unroll
        for (std::uint32_t i = 0; i < 2u; ++i) {
            const std::uint32_t u = tid + i * kPrefill2dWideThreads;
            const std::uint32_t rg = u >> 6u;
            const std::uint32_t off = (u & 63u) * 16u;
            const std::uint8_t* src =
                activation_codes +
                static_cast<std::size_t>(row_start / 16u + rg) *
                    (static_cast<std::size_t>(k_padded) * 16u) +
                static_cast<std::size_t>(g) * 2048u +
                static_cast<std::size_t>(hh) * 1024u + off;
            *reinterpret_cast<uint4*>(buf + rg * 1024u + off) =
                *reinterpret_cast<const uint4*>(src);
        }
    };

    stage_half(0u);
    if (tid < kPrefill2dWideRows)
        s_as[tid] = activation_scales[static_cast<std::size_t>(row_start + tid) *
                                          sc.activation_scale_stride];
    __syncthreads();

    const std::uint32_t halves = groups * 2u;
    float8_t raw[4][2];
    for (std::uint32_t h = 0; h < halves; ++h) {
        if (h + 1u < halves) stage_half(h + 1u);
        const std::uint32_t hh = h & 1u;
        const std::uint32_t g = h >> 1u;
        if (hh == 0u) {
#pragma unroll
            for (int t = 0; t < 4; ++t)
#pragma unroll
                for (int nt = 0; nt < 2; ++nt)
                    raw[t][nt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }

#pragma unroll
        for (std::uint32_t il = 0; il < 2u; ++il) {
            const std::uint32_t ib = h * 2u + il;      // 32 K タイル（通算）
            int32x2_t w0[2], w1[2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const Fp8Frag16 wf = fp8_load16(fp8_wp(wrow[nt], ib));
                w0[nt] = wf.lo;
                w1[nt] = wf.hi;
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
                    raw[t][nt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a0, w0[nt], raw[t][nt]);
                    raw[t][nt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a1, w1[nt], raw[t][nt]);
                }
            }
        }

        if (hh == 1u) {
            const float ws[2] = {sc.weight_block128 ? weight_scales[ws_row[0] + g] : ws_ch[0],
                                 sc.weight_block128 ? weight_scales[ws_row[1] + g] : ws_ch[1]};
        // 128 K ぶんを 1 回だけ折る。
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const std::uint32_t as_slot = wave_m * 64u +
                                          static_cast<std::uint32_t>(t) * kTileRow +
                                          8u * k_group;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[t][nt][j] = __builtin_fmaf(
                        raw[t][nt][j],
                        ws[nt] * s_as[as_slot + static_cast<std::uint32_t>(j)],
                        acc[t][nt][j]);
        }
        }
        __syncthreads();
        if (hh == 1u) {
            const std::uint32_t g2 = g + 1u;
            if (g2 < groups && tid < kPrefill2dWideRows)
                s_as[tid] = activation_scales[static_cast<std::size_t>(row_start + tid) *
                                                  sc.activation_scale_stride +
                                              (sc.act_group ? g2 : 0u)];
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
                const float v = acc[t][nt][j];
                const std::size_t idx =
                    static_cast<std::size_t>(rr) * output_row_stride + col;
                if (output_dtype == static_cast<std::uint32_t>(Fp8GemmOutputDType::BF16)) {
                    static_cast<shelf::bf16_t*>(output)[idx] =
                        shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(v))};
                } else {
                    static_cast<float*>(output)[idx] = v;
                }
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

    const std::uint8_t* wrow = fp8_lane32(weight_codes, o_self, 0, k_padded, m_lane, k_group);
    // 1 行（= 0 行目）の活性化。レーンは行 0 と k の 8 要素組で決まる。
    const std::uint8_t* arow = activation_codes + static_cast<std::size_t>(m_lane + 16u * k_group) * 32u;

    const float ws_ch = kBlockWeightScale ? 0.0f : weight_scales[o_self];
    const std::size_t ws_row =
        kBlockWeightScale ? static_cast<std::size_t>(o_self >> 7) * sc.weight_scale_k_blocks : 0u;

    float acc0 = 0.0f;
    const std::uint32_t nb = k_padded >> 5;
    // 2 段先読み: 2 進分のロードを出してから演算する。参照実装の同名 kernel は
    // VGPR 120 でパイプライン化されており、素直なループ（VGPR 24）より約 5% 速い。
    // 累加順（ib = 0,1,2,… の 1 本の鎖）は変えないので**生成される値はビット同一**。
    auto scale_at = [&](std::uint32_t idx, float& ws, float& as) {
        const std::uint32_t kb = sc.act_group ? (idx >> 2u) : 0u;
        ws = ws_ch;
        if constexpr (kBlockWeightScale) ws = weight_scales[ws_row + kb];
        as = activation_scales[kb];
    };
    auto load_at = [&](std::uint32_t idx, int32x2_t& w0, int32x2_t& w1,
                       int32x2_t& a0, int32x2_t& a1) {
        // 1 レーン 32 B の並びのうち 16 B が連続（`fp8_wp` は 16 B 境界）なので、
        // 重み・活性化とも **16 B ロード 1 回**で足りる（radiance の kernel も uint4
        // 読み。8 B × 2 本のとき メモリ命令数が倍になる）。
        const std::uint8_t* wp = fp8_wp(wrow, idx);
        const std::uint8_t* ap = fp8_wp(arow, idx);
        const uint4 w = *reinterpret_cast<const uint4*>(wp);
        const uint4 a = *reinterpret_cast<const uint4*>(ap);
        w0 = int32x2_t{static_cast<int>(w.x), static_cast<int>(w.y)};
        w1 = int32x2_t{static_cast<int>(w.z), static_cast<int>(w.w)};
        a0 = int32x2_t{static_cast<int>(a.x), static_cast<int>(a.y)};
        a1 = int32x2_t{static_cast<int>(a.z), static_cast<int>(a.w)};
    };
    auto compute = [&](int32x2_t w0, int32x2_t w1, int32x2_t a0, int32x2_t a1,
                       float ws, float as) {
        float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);
        // 列 = トークン。有効なのは列 0 のみで、それは k_group == 0 の lane が持つ。
        if (k_group == 0u)
            acc0 = __builtin_fmaf(c[0], ws * as, acc0);
    };
    std::uint32_t ib = 0u;
    // 先読み 8 段（load を束ねるだけで累加順 ib = 0,1,2,… の 1 本の鎖は不変なので
    // **ビット同一**。rows=1 の decode GEMV は workgroup 数が features/16 で決まり
    // 64 CU で波数が足りない（n=5120 では 5 波/CU → 1,096 GB/s）。1 レーンの in-flight を
    // 増やすのが追加発射なしの MLP 対策（実測: 2 段 1,096 → 4 段 1,402 → 8 段 1,478 GB/s、
    // 12/16 段は 1,443〜1,487 で飽和。VGPR 76・spill 0）。
    // 2 波に割って K を半分にすることも試したが、速度は 17.73 → 17.94 µs で改善なし
    // （段階 46）。
    for (; ib + 8u <= nb; ib += 8u) {
        int32x2_t w[8][2], a[8][2];
        float wsv[8], asv[8];
#pragma unroll
        for (std::uint32_t u = 0; u < 8u; ++u) {
            load_at(ib + u, w[u][0], w[u][1], a[u][0], a[u][1]);
            scale_at(ib + u, wsv[u], asv[u]);
        }
#pragma unroll
        for (std::uint32_t u = 0; u < 8u; ++u)
            compute(w[u][0], w[u][1], a[u][0], a[u][1], wsv[u], asv[u]);
    }
    for (; ib + 2u <= nb; ib += 2u) {
        int32x2_t w0a, w1a, a0a, a1a, w0b, w1b, a0b, a1b;
        float wsa, asa, wsb, asb;
        load_at(ib, w0a, w1a, a0a, a1a);
        load_at(ib + 1u, w0b, w1b, a0b, a1b);
        scale_at(ib, wsa, asa);
        scale_at(ib + 1u, wsb, asb);
        compute(w0a, w1a, a0a, a1a, wsa, asa);
        compute(w0b, w1b, a0b, a1b, wsb, asb);
    }
    for (; ib < nb; ++ib) {
        int32x2_t w0, w1, a0, a1;
        float ws, as;
        load_at(ib, w0, w1, a0, a1);
        scale_at(ib, ws, as);
        compute(w0, w1, a0, a1, ws, as);
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
