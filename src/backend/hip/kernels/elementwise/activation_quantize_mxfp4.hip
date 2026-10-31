// shelf backend/hip — 活性化の MXFP4 量子化カーネル（E2M1 + E8M0、gfx1201）。
//
// 1 ブロック = 1 行（256 スレッド = 8 波）。1 波は 32 要素ブロック 2 個を担当し、
// 16 レーンで 1 ブロックを処理する:
//   * レーン m（0..15）が k = base+m と k = base+16+m の 2 値を読み、2 値を掛け合わせた
//     amax を 16 レーンで reduce する（ブロック全体の amax になる）
//   * E8M0 スケールは `amax/6` を 2 の累乗へ切り上げた指数（`quant::mxfp4` と同じ）
//   * そのレーンが 1 バイト（下位 = k, 上位 = k+16）を書くので、同じバイトを取り合わない
//
// E2M1 と E8M0 の符号化は `quant::mxfp4` のホスト実装と**同じ式**にしてある
// （ずれるとテストの readback が落ちる）。
#include <shelf/backend/hip/kernels/elementwise/activation_quantize_mxfp4.h>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <shelf/quant/mxfp4.hpp>

namespace shelf::hipk {
namespace {

constexpr std::uint32_t kThreads = 256u;     // 1 行 1 ブロック（8 波）
constexpr std::uint32_t kLanesPerBlock = 16u;  // 1 ブロックを 16 レーンで
constexpr std::uint32_t kBlockK = quant::mxfp4::kBlockK;   // 32

__device__ __forceinline__ float bf16_to_f32(std::uint16_t w) {
    const float f = __bfloat162float(__ushort_as_bfloat16(w));
    return isfinite(f) ? f : 0.0f;
}

// E2M1 の大きさ表（ホストの kE2M1Magnitudes と同一）。
__device__ __forceinline__ float e2m1_mag(std::uint32_t i) {
    constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return kMag[i & 7u];
}

// f32 -> E2M1 ニブル（最近傍、同距離は偶数側、±6 で飽和）。ホストと同一の判定。
__device__ __forceinline__ std::uint8_t e2m1_encode(float x) {
    const std::uint32_t sign = __signbitf(x) ? 0x08u : 0x00u;
    const float a = fabsf(x);
    if (!(a > 0.0f)) return static_cast<std::uint8_t>(sign);
    if (a >= 6.0f) return static_cast<std::uint8_t>(sign | 0x07u);
    std::uint32_t lo = 0u;
#pragma unroll
    for (std::uint32_t i = 0; i < 8u; ++i) {
        if (e2m1_mag(i) <= a) lo = i;
    }
    if (lo >= 7u) return static_cast<std::uint8_t>(sign | 0x07u);
    const float vlo = e2m1_mag(lo), vhi = e2m1_mag(lo + 1u);
    const float dlo = a - vlo, dhi = vhi - a;
    std::uint32_t pick;
    if (dhi < dlo) {
        pick = lo + 1u;
    } else if (dlo < dhi) {
        pick = lo;
    } else {
        pick = (((lo + 1u) % 2u) == 0u) ? lo + 1u : lo;   // 同距離は偶数側
    }
    return static_cast<std::uint8_t>(sign | pick);
}

// amax から E8M0 バイト（`quant::mxfp4` の encode_scale_byte と同一）。
// 2 の累乗の切り上げ指数を f32 のビットから直接出す（正規数の範囲）。
__device__ __forceinline__ std::uint8_t e8m0_encode_amax(float amax) {
    if (!(amax > 0.0f) || !isfinite(amax)) return 0u;
    const float raw = amax / 6.0f;
    if (!(raw > 0.0f)) return 0u;
    const std::uint32_t bits = __float_as_uint(raw);
    std::int32_t e;
    if ((bits & 0x7F800000u) == 0u) {
        e = -127;   // 非正規: 最小指数へ
    } else {
        const std::int32_t E = static_cast<std::int32_t>((bits >> 23u) & 0xFFu) - 127;
        e = ((bits & 0x007FFFFFu) == 0u) ? E : (E + 1);   // 2 の累乗なら据え置き
    }
    if (e < -127) e = -127;
    if (e > 127) e = 127;
    return static_cast<std::uint8_t>(e + 127);
}

__device__ __forceinline__ float e8m0_to_f32(std::uint8_t b) {
    return __uint_as_float(static_cast<std::uint32_t>(b) << 23);
}

// 断片順のバイト位置（`quant::mxfp4::tiled_offset` と同一式。nibble は常に 0 で、
// 下位 = k、上位 = k+16 を自分で詰める）。
__device__ __forceinline__ std::size_t code_offset(std::uint32_t row, std::uint32_t col,
                                                   std::uint32_t k_padded) {
    const std::size_t blocks_per_tile = static_cast<std::size_t>(k_padded) / kBlockK;
    return static_cast<std::size_t>(row / 16u) * (blocks_per_tile * 256u) +
           static_cast<std::size_t>(col / kBlockK) * 256u +
           static_cast<std::size_t>(row % 16u) * 16u +
           static_cast<std::size_t>(col % 16u);
}

__device__ __forceinline__ float load_at(const shelf::bf16_t* row, std::uint32_t i,
                                         std::uint32_t k) {
    return (i < k) ? bf16_to_f32(row[i].data) : 0.0f;
}

__global__ void __launch_bounds__(kThreads) activation_quantize_mxfp4_kernel(
    const shelf::bf16_t* __restrict__ input, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, const std::uint32_t rows, const std::uint32_t k,
    const std::uint32_t k_padded, const std::uint32_t input_row_stride) {
    const std::uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const std::uint32_t wave = threadIdx.x >> 5u;    // 0..7
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t half = lane / kLanesPerBlock;  // 0 or 1（同じ波の 2 ブロック）
    const std::uint32_t m = lane % kLanesPerBlock;     // 0..15

    const shelf::bf16_t* xrow = input + static_cast<std::size_t>(r) * input_row_stride;
    const std::uint32_t nblocks = k_padded / kBlockK;
    std::uint8_t* srow = scales + static_cast<std::size_t>(r) * nblocks;

    // 1 波が 2 ブロックずつ、全体で 8 波 × 2 = 16 ブロック/周。ループは**ウェーブ内で
    // 一様**に回す（`__shfl_xor_sync` は全レーンが揃っていないと未定義。ブロック数が
    // 16 の倍数でない形で一部のレーンだけ抜けるとハングする）。範囲外は active=false で
    // 計算だけ行い、書き込みを抑止する。
    const std::uint32_t blocks_per_iter = (kThreads / 32u) * 2u;
    for (std::uint32_t it = 0; it < nblocks; it += blocks_per_iter) {
        const std::uint32_t ib = it + wave * 2u + half;
        const bool active = (ib < nblocks);
        const std::uint32_t base = active ? (ib * kBlockK) : 0u;
        const float v0 = active ? load_at(xrow, base + m, k) : 0.0f;
        const float v1 = active ? load_at(xrow, base + kLanesPerBlock + m, k) : 0.0f;
        float amax = fmaxf(fabsf(v0), fabsf(v1));
        // 16 レーン（同じ half）で reduce。
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, 1u));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, 2u));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, 4u));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, 8u));

        if (!active) continue;
        const std::uint8_t sb = e8m0_encode_amax(amax);
        const float scale = (sb == 0u) ? 0.0f : e8m0_to_f32(sb);
        const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        const std::uint8_t lo = (scale > 0.0f) ? e2m1_encode(v0 * inv) : 0u;
        const std::uint8_t hi = (scale > 0.0f) ? e2m1_encode(v1 * inv) : 0u;
        codes[code_offset(r, base + m, k_padded)] =
            static_cast<std::uint8_t>(lo | static_cast<std::uint8_t>(hi << 4u));
        if (m == 0u) srow[ib] = sb;
    }
}

// ---- SwiGLU 融合 -----------------------------------------------------------
__global__ void __launch_bounds__(kThreads) swiglu_quantize_mxfp4_kernel(
    const shelf::bf16_t* __restrict__ gate, const shelf::bf16_t* __restrict__ up,
    shelf::bf16_t* __restrict__ out, std::uint8_t* __restrict__ codes,
    std::uint8_t* __restrict__ scales, const std::uint32_t rows, const std::uint32_t k,
    const std::uint32_t k_padded, const std::uint32_t gate_row_stride,
    const std::uint32_t up_row_stride, const std::uint32_t out_row_stride) {
    const std::uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const std::uint32_t wave = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;
    const std::uint32_t half = lane / kLanesPerBlock;
    const std::uint32_t m = lane % kLanesPerBlock;

    const shelf::bf16_t* grow = gate + static_cast<std::size_t>(r) * gate_row_stride;
    const shelf::bf16_t* urow = up + static_cast<std::size_t>(r) * up_row_stride;
    shelf::bf16_t* orow = (out != nullptr) ? (out + static_cast<std::size_t>(r) * out_row_stride)
                                           : nullptr;
    const std::uint32_t nblocks = k_padded / kBlockK;
    std::uint8_t* srow = scales + static_cast<std::size_t>(r) * nblocks;
    const std::uint32_t blocks_per_iter = (kThreads / 32u) * 2u;

    // silu(gate) * up を bf16 に丸めてから量子化する（既存の融合と同じ順）。
    auto swiglu_at = [&](std::uint32_t i) -> float {
        if (i >= k) return 0.0f;
        const float g = bf16_to_f32(grow[i].data);
        const float u = bf16_to_f32(urow[i].data);
        const float y = (g / (1.0f + __expf(-g))) * u;
        const float yb = __bfloat162float(__float2bfloat16(y));
        if (orow != nullptr) orow[i] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y))};
        return isfinite(yb) ? yb : 0.0f;
    };

    for (std::uint32_t it = 0; it < nblocks; it += blocks_per_iter) {
        const std::uint32_t ib = it + wave * 2u + half;
        const bool active = (ib < nblocks);
        const std::uint32_t base = active ? (ib * kBlockK) : 0u;
        const float v0 = active ? swiglu_at(base + m) : 0.0f;
        const float v1 = active ? swiglu_at(base + kLanesPerBlock + m) : 0.0f;
        float amax = fmaxf(fabsf(v0), fabsf(v1));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, 1u));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, 2u));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, 4u));
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, amax, 8u));

        if (!active) continue;
        const std::uint8_t sb = e8m0_encode_amax(amax);
        const float scale = (sb == 0u) ? 0.0f : e8m0_to_f32(sb);
        const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        const std::uint8_t lo = (scale > 0.0f) ? e2m1_encode(v0 * inv) : 0u;
        const std::uint8_t hi = (scale > 0.0f) ? e2m1_encode(v1 * inv) : 0u;
        codes[code_offset(r, base + m, k_padded)] =
            static_cast<std::uint8_t>(lo | static_cast<std::uint8_t>(hi << 4u));
        if (m == 0u) srow[ib] = sb;
    }
}

}  // namespace

bool activation_quantize_mxfp4_supported(std::uint32_t k, std::uint32_t k_padded) noexcept {
    if (k == 0u || k_padded < k) return false;
    // 断片順のタイルは 16 行 × 32 K 単位なので 32 の倍数で足りる
    // （GEMM 側はさらに 64/128 K の倍数を要求するが、それは GEMM の契約）。
    return (k_padded % quant::mxfp4::kBlockK) == 0u;
}

hipError_t launch_activation_quantize_mxfp4(
    const shelf::bf16_t* input, std::uint8_t* codes, std::uint8_t* scales,
    std::uint32_t rows, std::uint32_t k, std::uint32_t k_padded,
    std::uint32_t input_row_stride, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (!activation_quantize_mxfp4_supported(k, k_padded)) return hipErrorNotSupported;
    if (input == nullptr || codes == nullptr || scales == nullptr) return hipErrorInvalidValue;
    activation_quantize_mxfp4_kernel<<<rows, kThreads, 0, stream>>>(
        input, codes, scales, rows, k, k_padded, input_row_stride);
    return hipGetLastError();
}

hipError_t launch_swiglu_quantize_mxfp4(
    const shelf::bf16_t* gate, const shelf::bf16_t* up, shelf::bf16_t* out,
    std::uint8_t* codes, std::uint8_t* scales, std::uint32_t rows, std::uint32_t k,
    std::uint32_t k_padded, std::uint32_t gate_row_stride, std::uint32_t up_row_stride,
    std::uint32_t out_row_stride, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (!activation_quantize_mxfp4_supported(k, k_padded)) return hipErrorNotSupported;
    if (gate == nullptr || up == nullptr || codes == nullptr || scales == nullptr)
        return hipErrorInvalidValue;
    swiglu_quantize_mxfp4_kernel<<<rows, kThreads, 0, stream>>>(
        gate, up, out, codes, scales, rows, k, k_padded, gate_row_stride, up_row_stride,
        out_row_stride);
    return hipGetLastError();
}

}  // namespace shelf::hipk
