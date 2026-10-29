// shelf backend/hip — 活性化 FP8 量子化カーネル（断片順出力）。
//
// 2 種類の粒度を持つ（同じ e4m3 手順・同じ出力レイアウト）:
//   * act_group = 128 : K を 128 要素のブロックに切り、ブロックごとに amax/scale。
//                       1 wave が 1 ブロック（32 lane × 4 要素）を扱う。
//                       宣言が blockwise（weight_block_size=[128,128] 系）のときの形。
//   * act_group = 0   : 1 行（= トークン）全体で 1 スケール（宣言 `strategy: token`）。
//                       2 パス（amax → 量子化）になる。
#include <cmath>
#include <cstdint>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <shelf/backend/hip/kernels/e4m3_codec.h>
#include <shelf/backend/hip/kernels/elementwise/activation_quantize_fp8.h>
#include <shelf/quant/fp8_w8a8.hpp>

namespace shelf::hipk {
namespace {

using shelf::quant::fp8::amax_scale;
using shelf::quant::fp8::kActGroupK;
using shelf::quant::fp8::kTileK;
using shelf::quant::fp8::kTileRows;

constexpr std::uint32_t kThreadsPerRow = 256u;              // 1 ブロック = 1 トークン
constexpr std::uint32_t kLanes = 32u;
constexpr std::uint32_t kWavesPerRow = kThreadsPerRow / kLanes;   // 8
constexpr std::uint32_t kElemsPerLane = 4u;                 // 8 バイト読み / 4 バイト書き

__device__ __forceinline__ float bf16_word_to_f32(std::uint32_t word) {
    return __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(word & 0xFFFFu)));
}

// quant/fp8 の tiled_offset と同一の式（一致は tests/unit/test_hip_gemm_fp8.cpp が
// readback で検証する — 定義の二重実装であり、片方を壊すとテストが落ちる）。
__device__ __forceinline__ std::size_t code_offset(std::uint32_t row, std::uint32_t col,
                                                   std::uint32_t k_padded) {
    return static_cast<std::size_t>(row / kTileRows) *
               (static_cast<std::size_t>(k_padded) * kTileRows) +
           static_cast<std::size_t>(col / kTileK) * (kTileRows * kTileK) +
           static_cast<std::size_t>((col % kTileK) / 16u) * (kTileRows * 16u) +
           static_cast<std::size_t>(row % kTileRows) * 16u +
           static_cast<std::size_t>(col % 16u);
}

__device__ __forceinline__ float load_f32_at(const shelf::bf16_t* xrow, std::uint32_t i,
                                             std::uint32_t k) {
    if (i >= k) return 0.0f;
    const float v = bf16_word_to_f32(xrow[i].data);
    return std::isfinite(v) ? v : 0.0f;
}

// ---- act_group = 128（ブロック単位） --------------------------------------
__global__ void __launch_bounds__(kThreadsPerRow) activation_quantize_fp8_group_kernel(
    const shelf::bf16_t* __restrict__ input,
    std::uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    const std::uint32_t rows,
    const std::uint32_t k,
    const std::uint32_t k_padded,
    const std::uint32_t input_row_stride,
    const std::uint32_t k_blocks) {
    const std::uint32_t r = blockIdx.x;
    if (r >= rows) return;
    const std::uint32_t wave = threadIdx.x >> 5u;
    const std::uint32_t lane = threadIdx.x & 31u;

    const shelf::bf16_t* xrow = input + static_cast<std::size_t>(r) * input_row_stride;
    std::uint8_t* crow = codes + code_offset(r, 0, k_padded);
    float* rscales = scales + static_cast<std::size_t>(r) * k_blocks;

    for (std::uint32_t g = wave; g < k_blocks; g += kWavesPerRow) {
        const std::uint32_t i0 = g * kActGroupK + lane * kElemsPerLane;
        float v[kElemsPerLane];
        float amax = 0.0f;
        if (i0 + kElemsPerLane <= k) {
            const std::uint64_t raw = *reinterpret_cast<const std::uint64_t*>(xrow + i0);
#pragma unroll
            for (std::uint32_t l = 0; l < kElemsPerLane; ++l) {
                const float f = bf16_word_to_f32(static_cast<std::uint32_t>(raw >> (16u * l)));
                v[l] = std::isfinite(f) ? f : 0.0f;
                amax = fmaxf(amax, fabsf(v[l]));
            }
        } else {
#pragma unroll
            for (std::uint32_t l = 0; l < kElemsPerLane; ++l) {
                v[l] = load_f32_at(xrow, i0 + l, k);
                amax = fmaxf(amax, fabsf(v[l]));
            }
        }
        for (std::uint32_t off = 16u; off > 0u; off >>= 1u)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffull, amax, off));
        const float sc = amax_scale(amax);
        const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
        if (lane == 0u) rscales[g] = sc;

        std::uint32_t packed = 0u;
#pragma unroll
        for (std::uint32_t l = 0; l < kElemsPerLane; ++l) {
            const std::uint8_t c = (sc > 0.0f) ? e4m3_encode_u8(v[l] * inv) : 0u;
            packed |= static_cast<std::uint32_t>(c) << (8u * l);
        }
        *reinterpret_cast<std::uint32_t*>(crow + code_offset(0, i0, k_padded)) = packed;
    }
}

// ---- act_group = 0（行 = トークン単位） -----------------------------------
__global__ void __launch_bounds__(kThreadsPerRow) activation_quantize_fp8_row_kernel(
    const shelf::bf16_t* __restrict__ input,
    std::uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    const std::uint32_t rows,
    const std::uint32_t k,
    const std::uint32_t k_padded,
    const std::uint32_t input_row_stride) {
    const std::uint32_t r = blockIdx.x;
    if (r >= rows) return;

    const shelf::bf16_t* xrow = input + static_cast<std::size_t>(r) * input_row_stride;
    std::uint8_t* crow = codes + code_offset(r, 0, k_padded);

    float amax = 0.0f;
    for (std::uint32_t i0 = threadIdx.x * kElemsPerLane; i0 < k;
         i0 += kThreadsPerRow * kElemsPerLane) {
#pragma unroll
        for (std::uint32_t l = 0; l < kElemsPerLane; ++l)
            amax = fmaxf(amax, fabsf(load_f32_at(xrow, i0 + l, k)));
    }
    __shared__ float s_wave[kWavesPerRow];
    for (std::uint32_t off = 16u; off > 0u; off >>= 1u)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffffffffffull, amax, off));
    if ((threadIdx.x & 31u) == 0u) s_wave[threadIdx.x >> 5u] = amax;
    __syncthreads();
    float peak = 0.0f;
#pragma unroll
    for (std::uint32_t w = 0; w < kWavesPerRow; ++w) peak = fmaxf(peak, s_wave[w]);

    const float sc = amax_scale(peak);
    const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
    if (threadIdx.x == 0u) scales[r] = sc;

    for (std::uint32_t i0 = threadIdx.x * kElemsPerLane; i0 < k_padded;
         i0 += kThreadsPerRow * kElemsPerLane) {
        std::uint32_t packed = 0u;
#pragma unroll
        for (std::uint32_t l = 0; l < kElemsPerLane; ++l) {
            const float v = load_f32_at(xrow, i0 + l, k);
            const std::uint8_t c = (sc > 0.0f) ? e4m3_encode_u8(v * inv) : 0u;
            packed |= static_cast<std::uint32_t>(c) << (8u * l);
        }
        *reinterpret_cast<std::uint32_t*>(crow + code_offset(0, i0, k_padded)) = packed;
    }
}

}  // namespace

std::size_t activation_quantize_fp8_bytes(std::uint32_t rows, std::uint32_t k_padded) noexcept {
    return shelf::quant::fp8::tiled_bytes(rows, k_padded);
}

std::size_t activation_quantize_fp8_scale_count(std::uint32_t rows, std::uint32_t k_padded,
                                                std::uint32_t act_group) noexcept {
    return shelf::quant::fp8::activation_scale_count(rows, k_padded, act_group);
}

bool activation_quantize_fp8_supported(std::uint32_t k, std::uint32_t k_padded,
                                       std::uint32_t act_group) noexcept {
    if (k == 0u || k_padded < k) return false;
    if ((k_padded % kTileK) != 0u) return false;
    if (act_group == 0u) return true;
    if (act_group != kActGroupK) return false;
    return (k_padded % kActGroupK) == 0u;      // 128 で割り切れること
}

hipError_t launch_activation_quantize_fp8(
    const shelf::bf16_t* input, std::uint8_t* codes, float* scales,
    std::uint32_t rows, std::uint32_t k, std::uint32_t k_padded,
    std::uint32_t input_row_stride, std::uint32_t act_group, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (!activation_quantize_fp8_supported(k, k_padded, act_group))
        return hipErrorNotSupported;
    if (input == nullptr || codes == nullptr || scales == nullptr)
        return hipErrorInvalidValue;
    if (act_group == 0u) {
        activation_quantize_fp8_row_kernel<<<rows, kThreadsPerRow, 0, stream>>>(
            input, codes, scales, rows, k, k_padded, input_row_stride);
    } else {
        const std::uint32_t kb = k_padded / act_group;
        activation_quantize_fp8_group_kernel<<<rows, kThreadsPerRow, 0, stream>>>(
            input, codes, scales, rows, k, k_padded, input_row_stride, kb);
    }
    return hipGetLastError();
}

}  // namespace shelf::hipk
