// shelf backend/hip — MXFP4 W4A8 の GEMV（rows=1、gfx1201）。
//
// 重みは E2M1 のニブル（断片順 16 出力 × 32 K = 256 B タイル）から、`v_perm_b32` で
// **E4M3 のバイトへ直接展開**してから FP8 WMMA に渡す。E2M1 の 8 値
// {0, 0.5, 1, 1.5, 2, 3, 4, 6} は E4M3（指数 4 bit・仮数 3 bit）で厳密に表せるので、
// この展開は無損失。スケールは E8M0（2^(b-127)）を K 32 要素ごとに掛ける。
//
// 活性化は既存の fp8 経路と同じ断片順（64 K スーパータイル・1 レーン 32 B）なので、
// 32 K ブロック 1 個 = `fp8_load16(fp8_wp(arow, ib*2))` の 1 回で 16 K 断片 2 個が
// 揃う。したがって fp8 の decode1 と骨格は同じで、変わるのは重みの読み方だけ。
//
// 元: `~/work/references/phaseshift-mxfp4` の
//   `kernels/optimized/linear/mxfp4.hip`（`gemv_mxfp4_w4a8_decode1_kernel` と
//   `mxfp4_low_high`）。
#include <shelf/backend/hip/kernels/gemm/linear_mxfp4.h>

#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/gemm/detail/linear_fp8_device.h>
#include <shelf/quant/mxfp4.hpp>

namespace shelf::hipk {
namespace {

using gemm_fp8_detail::Fp8Frag16;
using gemm_fp8_detail::fp8_load16;
using gemm_fp8_detail::fp8_lane32;
using gemm_fp8_detail::fp8_wp;
using gemm_fp8_detail::int32x2_t;
using gemm_fp8_detail::float8_t;

constexpr std::uint32_t kTileOut = 16u;      // 1 ブロックが担う出力行数
constexpr std::uint32_t kThreads = 32u;      // 16 出力 × 2 K 半 = 32
constexpr std::uint32_t kBlockK = quant::mxfp4::kBlockK;      // 32
constexpr std::uint32_t kBlockBytes = 256u;  // 16 出力行 × 16 B（32 K）

// E2M1 のニブル 4 バイト（= 偶数 4 + 奇数 4）を E4M3 のバイトへ展開する。
// 配置は参照実装と同じ: バイト b の下位ニブルが k=b、上位ニブルが k=16+b。
// `mags*` は E2M1 の大きさ 8 種の E4M3 ビットパターン、`negs*` はその符号付き。
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

// E8M0 バイト -> f32（2^(b-127)）。ビット組み立て 1 命令（`ldexpf` は重い）。
// b = 0（2^-127）は 0.0 になるが、そのブロックはコードも 0 なので積は 0 で一致する。
__device__ __forceinline__ float mxfp4_e8m0_to_f32(std::uint8_t b) {
    return __uint_as_float(static_cast<std::uint32_t>(b) << 23u);
}

template <Mxfp4GemmOutputDType kOutputType>
__global__ void __launch_bounds__(kThreads) gemv_mxfp4_w4a8_decode1_kernel(
    const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const float* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const Mxfp4GemmScaleArgs sc) {
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t m_lane = tid & 15u;   // 出力行（16 行タイル内）
    const std::uint32_t k_group = tid >> 4u;  // 16 K 断片内の K 半（0/1）

    const std::uint32_t o_self = blockIdx.x * kTileOut + m_lane;
    const std::uint32_t nb = k_padded >> 5u;  // 32 K ブロック数

    // 重み: 16 出力 × 32 K のタイル（256 B）。レーンは行 m_lane の 8 B を持ち、
    // k_group が K 32 の前半/後半を選ぶ。
    const std::uint8_t* wrow = weight_codes +
                               static_cast<std::size_t>(o_self >> 4u) *
                                   (static_cast<std::size_t>(nb) * kBlockBytes) +
                               static_cast<std::size_t>(o_self & 15u) * 16u +
                               static_cast<std::size_t>(k_group) * 8u;
    const std::uint8_t* wsrow = weight_scales +
                                static_cast<std::size_t>(o_self) * sc.weight_scale_stride;
    // 活性化: 行 0 の断片（fp8 と同じ断片順）。
    const std::uint8_t* arow =
        activation_codes + static_cast<std::size_t>(m_lane + 16u * k_group) * 32u;

    float acc0 = 0.0f;
    // 先読みは入れない。4 段先読みを試したが 50.7 → 56.3 µs と**悪化**した（レジスタ
    // 圧で常駐波数が落ちる）。このカーネルは遅延ではなく、32 K ごとの
    // 「8 B ロード → 展開 → WMMA → 折り」の依存と workgroup 数 5 波/CU で決まる。
    for (std::uint32_t ib = 0; ib < nb; ++ib) {
        // 8 B = 32 ニブル（k 32 ブロック分のうち k_group の 8 列 × lo/hi）。
        const std::uint64_t packed = *reinterpret_cast<const std::uint64_t*>(
            wrow + static_cast<std::size_t>(ib) * kBlockBytes);
        const int32x2_t lo = mxfp4_low_high(static_cast<std::uint32_t>(packed));
        const int32x2_t hi = mxfp4_low_high(static_cast<std::uint32_t>(packed >> 32u));
        const int32x2_t w0 = int32x2_t{lo[0], hi[0]};   // 16 K 断片（先）
        const int32x2_t w1 = int32x2_t{lo[1], hi[1]};   // 16 K 断片（後）

        // 活性化は 32 K ブロック = 16 K 断片 2 個ぶんを 1 回で読む（tile index = ib）。
        const Fp8Frag16 af = fp8_load16(fp8_wp(arow, ib));

        float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af.lo, w0, c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af.hi, w1, c);

        // 列 = トークン。有効なのは列 0 のみで、それは k_group == 0 の lane が持つ
        // （k_group == 1 の lane は WMMA にデータを供給するだけで、結果は捨てる）。
        if (k_group == 0u) {
            const float ws = mxfp4_e8m0_to_f32(wsrow[ib]);
            const std::uint32_t kb = sc.act_group ? (ib >> 2u) : 0u;   // 128 K グループ
            acc0 = __builtin_fmaf(c[0], ws * activation_scales[kb], acc0);
        }
    }

    if (k_group == 0u) {
        const std::uint32_t o = static_cast<std::uint32_t>(blockIdx.x) * kTileOut + m_lane;
        if (o >= features) return;
        if constexpr (kOutputType == Mxfp4GemmOutputDType::BF16) {
            static_cast<shelf::bf16_t*>(output)[o] =
                shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(acc0))};
        } else {
            static_cast<float*>(output)[o] = acc0;
        }
    }
}


// ---- W4A4（A 側も MXFP4）--------------------------------------------------
// モデルカードの指示どおり A 側も E2M1 + E8M0 にする。W と同じ断片順なので、
// レーン (m, kg) は行 m の 8 B（= 32 ニブル）を読み、`mxfp4_low_high` で
// E4M3 断片 2 個へ展開する。スケールは 2^(eW + eA) を 1 回掛けるだけ
// （どちらも 2 の累乗なので積は厳密）。
template <Mxfp4GemmOutputDType kOutputType>
__global__ void __launch_bounds__(kThreads) gemv_mxfp4_w4a4_decode1_kernel(
    const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales,
    const std::uint8_t* __restrict__ activation_codes,
    const std::uint8_t* __restrict__ activation_scales,
    void* __restrict__ output,
    const std::uint32_t features,
    const std::uint32_t k_padded,
    const Mxfp4GemmScaleArgs sc) {
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t m_lane = tid & 15u;
    const std::uint32_t k_group = tid >> 4u;

    const std::uint32_t o_self = blockIdx.x * kTileOut + m_lane;
    const std::uint32_t nb = k_padded >> 5u;

    const std::uint8_t* wrow = weight_codes +
                               static_cast<std::size_t>(o_self >> 4u) *
                                   (static_cast<std::size_t>(nb) * kBlockBytes) +
                               static_cast<std::size_t>(o_self & 15u) * 16u +
                               static_cast<std::size_t>(k_group) * 8u;
    const std::uint8_t* wsrow = weight_scales +
                                static_cast<std::size_t>(o_self) * sc.weight_scale_stride;
    // 活性化: 行 m_lane の 16 B タイルのうち k_group の 8 B。
    const std::uint8_t* arow = activation_codes +
                               static_cast<std::size_t>(m_lane) * 16u +
                               static_cast<std::size_t>(k_group) * 8u;
    // スケールは行 0（rows=1）の E8M0。
    const std::uint8_t* asrow = activation_scales;

    float acc0 = 0.0f;
    for (std::uint32_t ib = 0; ib < nb; ++ib) {
        const std::uint64_t wp = *reinterpret_cast<const std::uint64_t*>(
            wrow + static_cast<std::size_t>(ib) * kBlockBytes);
        const int32x2_t wlo = mxfp4_low_high(static_cast<std::uint32_t>(wp));
        const int32x2_t whi = mxfp4_low_high(static_cast<std::uint32_t>(wp >> 32u));
        const int32x2_t w0 = int32x2_t{wlo[0], whi[0]};
        const int32x2_t w1 = int32x2_t{wlo[1], whi[1]};

        const std::uint64_t ap = *reinterpret_cast<const std::uint64_t*>(
            arow + static_cast<std::size_t>(ib) * kBlockBytes);
        const int32x2_t alo = mxfp4_low_high(static_cast<std::uint32_t>(ap));
        const int32x2_t ahi = mxfp4_low_high(static_cast<std::uint32_t>(ap >> 32u));
        const int32x2_t a0 = int32x2_t{alo[0], ahi[0]};
        const int32x2_t a1 = int32x2_t{alo[1], ahi[1]};

        float8_t c = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a0, w0, c);
        c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a1, w1, c);

        if (k_group == 0u) {
            // 2^(eW) * 2^(eA)。どちらも 2 の累乗なので積は厳密（丸めなし）。
            const float s = mxfp4_e8m0_to_f32(wsrow[ib]) *
                            mxfp4_e8m0_to_f32(asrow[ib]);
            acc0 = __builtin_fmaf(c[0], s, acc0);
        }
    }

    if (k_group == 0u) {
        const std::uint32_t o = static_cast<std::uint32_t>(blockIdx.x) * kTileOut + m_lane;
        if (o >= features) return;
        if constexpr (kOutputType == Mxfp4GemmOutputDType::BF16) {
            static_cast<shelf::bf16_t*>(output)[o] =
                shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(acc0))};
        } else {
            static_cast<float*>(output)[o] = acc0;
        }
    }
}

}  // namespace

bool mxfp4_decode1_supported(std::uint32_t rows, std::uint32_t features,
                             std::uint32_t k_padded) noexcept {
    if (rows != 1u || features == 0u) return false;
    if (k_padded == 0u || (k_padded % kBlockK) != 0u) return false;
    // 32 K ブロックを 2 つ組で読むので、16 K 断片の並びが 64 K 単位である必要がある。
    if ((k_padded % 64u) != 0u) return false;
    return true;
}

std::size_t mxfp4_weight_bytes(std::uint32_t features, std::uint32_t k_padded) noexcept {
    return quant::mxfp4::tiled_bytes(features, k_padded);
}

hipError_t launch_gemm_mxfp4_w4a8_decode1(
    const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
    const std::uint8_t* activation_codes, const float* activation_scales, void* output,
    Mxfp4GemmOutputDType output_dtype, std::uint32_t rows, std::uint32_t features,
    std::uint32_t k_padded, std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs& scales, hipStream_t stream) {
    (void)output_row_stride;   // rows=1 なので行ストライドは使わない
    if (!mxfp4_decode1_supported(rows, features, k_padded)) return hipErrorInvalidValue;
    const dim3 grid((features + kTileOut - 1u) / kTileOut);
    if (output_dtype == Mxfp4GemmOutputDType::BF16) {
        gemv_mxfp4_w4a8_decode1_kernel<Mxfp4GemmOutputDType::BF16>
            <<<grid, kThreads, 0, stream>>>(weight_codes, weight_scales, activation_codes,
                                            activation_scales, output, features, k_padded,
                                            scales);
    } else {
        gemv_mxfp4_w4a8_decode1_kernel<Mxfp4GemmOutputDType::F32>
            <<<grid, kThreads, 0, stream>>>(weight_codes, weight_scales, activation_codes,
                                            activation_scales, output, features, k_padded,
                                            scales);
    }
    return hipGetLastError();
}


hipError_t launch_gemm_mxfp4_w4a4_decode1(
    const std::uint8_t* weight_codes, const std::uint8_t* weight_scales,
    const std::uint8_t* activation_codes, const std::uint8_t* activation_scales,
    void* output, Mxfp4GemmOutputDType output_dtype, std::uint32_t rows,
    std::uint32_t features, std::uint32_t k_padded, std::uint32_t output_row_stride,
    const Mxfp4GemmScaleArgs& scales, hipStream_t stream) {
    (void)output_row_stride;
    if (rows != 1u || features == 0u) return hipErrorInvalidValue;
    if (k_padded == 0u || (k_padded % 64u) != 0u) return hipErrorInvalidValue;
    const dim3 grid((features + kTileOut - 1u) / kTileOut);
    if (output_dtype == Mxfp4GemmOutputDType::BF16) {
        gemv_mxfp4_w4a4_decode1_kernel<Mxfp4GemmOutputDType::BF16>
            <<<grid, kThreads, 0, stream>>>(weight_codes, weight_scales, activation_codes,
                                            activation_scales, output, features, k_padded,
                                            scales);
    } else {
        gemv_mxfp4_w4a4_decode1_kernel<Mxfp4GemmOutputDType::F32>
            <<<grid, kThreads, 0, stream>>>(weight_codes, weight_scales, activation_codes,
                                            activation_scales, output, features, k_padded,
                                            scales);
    }
    return hipGetLastError();
}

}  // namespace shelf::hipk
