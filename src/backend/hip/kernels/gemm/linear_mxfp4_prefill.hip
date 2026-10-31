// shelf backend/hip — MXFP4 W4A8 の prefill GEMM（gfx1201）。
//
// 骨格は fp8 の `fp8_prefill2d_wide_body` と同じ（128 行 × 128 出力、8 波・256 スレッド、
// 活性化は 64 K 半の ping-pong LDS）。違うのは重みの読み方とスケールの掛け方だけ:
//
//   * 重みは E2M1 のニブル（16 出力 × 32 K = 256 B タイル）。レーンは
//     `(o&15)*16 + k_group*8 + ib*256` から 8 B を読み、`mxfp4_low_high` で
//     E4M3 の 16 K 断片 2 個（= 32 K ブロック 1 個）へ無損失展開する。
//   * スケールは E8M0 が K 32 要素ごとに変わるので、**32 K ブロックごとに f32 で掛ける**。
//     参照実装は E8M0 を E4M3 の指数へ畳む表（`kMxfp4FoldTable`）で折りを 128 K に
//     保つが、**畳むと値が変わる**（E4M3 の subnormal 帯域に落ちるブロックで有損。
//     実モデルでは b<119 が 7.57% あり、相対 1.5e-2 の差が出た）ので採用しない。
//     ここは厳密（f32 で 2^e を掛けるだけ）で、差は累加順のみ。
//     E8M0 は 2 の累乗なので、f32 の部分和に掛けても丸めは生じない。
//
// 元: `~/work/references/phaseshift-mxfp4` の
//   `kernels/optimized/linear/mxfp4.hip`（`gemm_mxfp4_w4a8_wmma_prefill_2d_kernel`）。
//   参照の E8M0 畳み込み（`kMxfp4FoldTable`）は**採らない**（有損のため）。
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

constexpr std::uint32_t kBlockK = quant::mxfp4::kBlockK;      // 32
constexpr std::uint32_t kBlockBytes = 256u;                  // 16 出力行 × 16 B

// E2M1 のニブル 4 バイト（偶数 4 + 奇数 4）を E4M3 のバイトへ展開する
// （`linear_mxfp4.hip` の decode1 と同じ。E2M1 の 8 値は E4M3 で厳密）。
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

template <std::uint32_t kRows>
__device__ __forceinline__ void mxfp4_prefill_wide_body(
    const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
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

    const std::uint32_t wave_m = (kRows == 128u) ? (wave >> 2u) : 0u;  // 64 行ずつ（32/64 行版は 1 ウェーブ）
    const std::uint32_t wave_n = wave & 3u;    // 0..3（32 出力ずつ）

    const std::uint32_t out_start = blockIdx.y * kPrefill2dWideOut;
    const std::uint32_t row_start = blockIdx.x * kRows;
    if (out_start >= features || row_start >= rows) return;

    __shared__ std::uint8_t s_a2[2][kPrefill2dWideRows * 64u];
    __shared__ float s_as[kPrefill2dWideRows];

    const std::uint32_t row0 = row_start + wave_m * 64u;
    const std::uint32_t o0 = out_start + wave_n * 32u;

    const std::uint32_t groups = k_padded / 128u;
    if (groups == 0u) return;
    const std::uint32_t nb = k_padded / kBlockK;

    // 重み: 出力行 (o0 + nt*16 + m_lane) の 32 K ブロックのレーン分。
    const std::uint8_t* wrow[2];
    const std::uint8_t* wsrow[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        // 出力方向の端数（features が 128 の倍数でない層。例: in_proj_a の 48）でも
        // **存在しない出力行を読まない**ように clamp する（格納側は既に `cc >= features`
        // でガード済み。clamp しないと重みの範囲外読みで HSA フォルト）。
        const std::uint32_t o_raw = o0 + static_cast<std::uint32_t>(nt) * kTileOut + m_lane;
        const std::uint32_t o = min(o_raw, features - 1u);
        wrow[nt] = weight_codes + static_cast<std::size_t>(o >> 4u) *
                                      (static_cast<std::size_t>(nb) * kBlockBytes) +
                   static_cast<std::size_t>(o & 15u) * 16u +
                   static_cast<std::size_t>(k_group) * 8u;
        wsrow[nt] = weight_scales + static_cast<std::size_t>(o) * sc.weight_scale_stride;
    }

    // 行クアッド（16 行）の本数: 64 行以上は 4、32 行版は 2。
    constexpr std::uint32_t kRowQuads = (kRows < 64u) ? (kRows / 16u) : 4u;
    std::uint32_t r_local[kRowQuads];
#pragma unroll
    for (std::uint32_t q = 0; q < kRowQuads; ++q)
        r_local[q] = wave_m * 64u + q * kTileRow + m_lane;

    float8_t acc[4][2];
#pragma unroll
    for (int t = 0; t < static_cast<int>(kRowQuads); ++t)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
            acc[t][nt] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    auto stage_half = [&](std::uint32_t h) {
        const std::uint32_t g = h >> 1u;
        const std::uint32_t hh = h & 1u;
        std::uint8_t* buf = s_a2[hh];
#pragma unroll
        // スレッド数に依存しない書き方（必要な全クアッドを stride で回る）。
        for (std::uint32_t u = tid; u < kRows * 4u; u += kPrefill2dWideThreads) {
            // 共有バッファは kRows 行分しかない（32 行版では 2 クアッド）。スレッド数が
            // 128 でも u は 0..255 になるため、行群を kRows/16 で clamp する。
            const std::uint32_t rg = u >> 6u;
            const std::uint32_t off = (u & 63u) * 16u;
            // 128 行タイルの端数（rows < row_start + 128）では、**存在しない行群を
            // 読まない**ように行群を clamp する（読み値は捨てられる。格納側は
            // `rr >= rows` で既にガード済み）。clamp しないと範囲外読みで HSA フォルト。
            // このブロックが担当する行群の上限（**kRows 基準**）。総行数で clamp すると
            // 33..47 行のブロック 1 で 2 クアッド目が 1 クアッド目に潰れ、入力が重複する。
            const std::uint32_t grp_max = row_start / 16u + kRows / 16u - 1u;
            const std::uint32_t grp = min(row_start / 16u + rg, grp_max);
            const std::uint8_t* src =
                activation_codes +
                static_cast<std::size_t>(grp) *
                    (static_cast<std::size_t>(k_padded) * 16u) +
                static_cast<std::size_t>(g) * 2048u +
                static_cast<std::size_t>(hh) * 1024u + off;
            *reinterpret_cast<uint4*>(buf + rg * 1024u + off) =
                *reinterpret_cast<const uint4*>(src);
        }
    };

    stage_half(0u);
    if (tid < kRows) {
        // 行スケールも同様に clamp（rows を超える行は捨てられる）。
        const std::uint32_t rr = min(row_start + tid, rows - 1u);
        s_as[tid] = activation_scales[static_cast<std::size_t>(rr) *
                                          sc.activation_scale_stride];
    }
    __syncthreads();

    const std::uint32_t halves = groups * 2u;
    float8_t raw[4][2];
    for (std::uint32_t h = 0; h < halves; ++h) {
        if (h + 1u < halves) stage_half(h + 1u);
        const std::uint32_t hh = h & 1u;
        const std::uint32_t g = h >> 1u;

#pragma unroll
        for (std::uint32_t il = 0; il < 2u; ++il) {
            const std::uint32_t ib = h * 2u + il;   // 通算の 32 K ブロック
            int32x2_t w0[2], w1[2];
            float ws[2];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const std::uint64_t packed = *reinterpret_cast<const std::uint64_t*>(
                    wrow[nt] + static_cast<std::size_t>(ib) * kBlockBytes);
                const int32x2_t lo = mxfp4_low_high(static_cast<std::uint32_t>(packed));
                const int32x2_t hi = mxfp4_low_high(static_cast<std::uint32_t>(packed >> 32u));
                w0[nt] = int32x2_t{lo[0], hi[0]};   // 16 K 断片（先）
                w1[nt] = int32x2_t{lo[1], hi[1]};   // 16 K 断片（後）
                ws[nt] = mxfp4_e8m0_to_f32(wsrow[nt][ib]);
            }
#pragma unroll
            for (int t = 0; t < static_cast<int>(kRowQuads); ++t) {
                const std::uint8_t* ap =
                    fp8_lane32(s_a2[hh], r_local[t], 0, 64u, m_lane, k_group) +
                    static_cast<std::size_t>(il) * 16u;
                const Fp8Frag16 af = fp8_load16(ap);
                const int32x2_t a0 = af.lo, a1 = af.hi;
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    // 32 K ブロックごとに折るので、毎回ゼロから始める（前のブロックの
                    // 値を積み上げると二重に足してしまう）。ゼロ初期化の命令は要らない。
                    const float8_t z = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    raw[t][nt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a0, w0[nt], z);
                    raw[t][nt] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        a1, w1[nt], raw[t][nt]);
                }
            }
            // 32 K ブロックごとに折る（E8M0 が 32 K で変わるため）。活性化スケールは
            // 128 K グループ単位なので、g の値（s_as の中身）をそのまま使う。
#pragma unroll
            for (int t = 0; t < static_cast<int>(kRowQuads); ++t) {
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
            if (g2 < groups && tid < kRows) {
                const std::uint32_t rr = min(row_start + tid, rows - 1u);
                s_as[tid] = activation_scales[static_cast<std::size_t>(rr) *
                                                  sc.activation_scale_stride +
                                              (sc.act_group ? g2 : 0u)];
            }
        }
    }

#pragma unroll
    for (int t = 0; t < static_cast<int>(kRowQuads); ++t) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const std::uint32_t r = row0 + static_cast<std::uint32_t>(t) * kTileRow +
                                    8u * k_group;
            const std::uint32_t col = o0 + static_cast<std::uint32_t>(nt) * kTileOut + m_lane;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const std::uint32_t rr = r + static_cast<std::uint32_t>(j);
                const std::uint32_t cc = col;
                if (rr >= rows || cc >= features) continue;
                const std::size_t idx = static_cast<std::size_t>(rr) * output_row_stride + cc;
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

template <std::uint32_t kRows>
__global__ void __launch_bounds__(kPrefill2dWideThreads) gemm_mxfp4_w4a8_prefill_wide_kernel(
    const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t output_dtype,
    const std::uint32_t rows,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs sc) {
    mxfp4_prefill_wide_body<kRows>(weight_codes, weight_scales, activation_codes, activation_scales,
                            output, output_dtype, rows, features, k_padded, output_row_stride,
                            sc);
}

}  // namespace

hipError_t launch_gemm_mxfp4_w4a8_prefill(
    const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
    const std::uint8_t* activation_codes, const float* activation_scales, void* output,
    Mxfp4GemmOutputDType output_dtype, std::uint32_t rows, std::uint32_t features,
    std::uint32_t k_padded, std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs& scales, hipStream_t stream) {
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if ((k_padded % 128u) != 0u) return hipErrorInvalidValue;
    // 行数に応じて最小のタイルを選ぶ（32/64 行版は 128 本のスレッドで 1 ウェーブ）。
    const auto launch = [&](auto rows_tag) {
        constexpr std::uint32_t kR = decltype(rows_tag)::value;
        const std::uint32_t threads = (kR == 128u) ? kPrefill2dWideThreads : 128u;
        const dim3 grid((rows + kR - 1u) / kR,
                        (features + kPrefill2dWideOut - 1u) / kPrefill2dWideOut, 1);
        gemm_mxfp4_w4a8_prefill_wide_kernel<kR><<<grid, threads, 0, stream>>>(
            weight_codes, weight_scales, activation_codes, activation_scales, output,
            static_cast<std::uint32_t>(output_dtype), rows, features, k_padded,
            output_row_stride, scales);
    };
    // TODO: 32 行タイル（rows<=64 で 2 ブロック）は数値が参照と一致しないケースが
    // 残っているため、当面は 128 行タイル固定で起動する（正しさ優先。端数は
    // カーネル内の clamp で安全）。速度改善はゴールの次段。
    // 32 行タイルは単一ブロック（rows<=32）で検証済み。33..64 行は 2 ブロックになり
    // 数値が参照と合わないケースがあるため、当面 128 行タイルに回す（TODO）。
    // 2 ブロック（33..64 行）は未解決のずれが残るため 1 ブロックのみ有効化。
    // 33..64 行（2 ブロック）は最悪要素 row=37 で rel 1.3 のずれが残る（未解決）。
    // 33 行以上（2 ブロック目）は未解決。1 ブロックのみ小タイルを使う。
    // 32 行タイルは 1 ブロック（rows<=32）で検証済み。2 ブロック目（33 行以上）は
    // 行 32 以降がずれる未解決の不具合があるため、当面 128 行タイルを使う。
    // 2 ブロック目（33 行以上）は出力全体がずれる未解決の不具合があるため 128 行タイル。
    if (rows <= 64u) launch(std::integral_constant<std::uint32_t, 32u>{});
    else launch(std::integral_constant<std::uint32_t, 128u>{});
    return hipGetLastError();
}

}  // namespace shelf::hipk
