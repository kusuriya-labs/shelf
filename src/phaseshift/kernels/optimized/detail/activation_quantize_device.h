#pragma once
#include <phaseshift/kernels/optimized/activation_quantize.h>
#include <phaseshift/quantization/codec/e4m3_detail.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cmath>
#include <cstdint>
#include <cstddef>

namespace ps::kernel::detail {

constexpr uint32_t kQuantThreadsPerRow = 256u;
constexpr uint32_t kQuantActGroup = 128u;

__device__ __forceinline__ float quant_vec_bf16_lane_f32(uint32_t word, uint32_t e) {
    return (e & 1u) == 0u ? __uint_as_float(word << 16)
                          : __uint_as_float(word & 0xffff0000u);
}

template <uint32_t K, uint32_t KP, uint32_t Threads>
struct QuantVecRow {
    static_assert(Threads % 32u == 0u, "quant row requires whole waves");
    static_assert(K % kQuantActGroup == 0u || K == KP,
                  "group quantize requires K on group boundary or K == KP");
    static constexpr uint32_t kWaves = Threads / 32u;
    static constexpr uint32_t kGroups = (KP + kQuantActGroup - 1u) / kQuantActGroup;
    static constexpr uint32_t kElemsPerLane = kQuantActGroup / 32u;

    __device__ __forceinline__ static void run(const ::ps::bf16_t* __restrict__ xrow,
                                               uint8_t* __restrict__ codes,
                                               float* __restrict__ scales, uint32_t m,
                                               uint32_t code_row_stride,
                                               uint32_t scale_row_stride) {
        const uint32_t wave = threadIdx.x >> 5u;
        const uint32_t lane = threadIdx.x & 31u;
        uint8_t* crow = codes +
            static_cast<size_t>(m >> 4u) * (static_cast<size_t>(code_row_stride) * 16u) +
            static_cast<size_t>(m & 15u) * 16u;
        float* row_scales = scales + static_cast<size_t>(m) * (scale_row_stride / 4u);
        const __hip_bfloat16* xb = reinterpret_cast<const __hip_bfloat16*>(xrow);

        for (uint32_t g = wave; g < kGroups; g += kWaves) {
            const uint32_t base = g * kQuantActGroup;
            const uint32_t i0 = base + lane * kElemsPerLane;
            const bool full4 = (i0 + kElemsPerLane <= K);
            uint64_t raw = 0u;
            if (full4)
                raw = *reinterpret_cast<const uint64_t*>(xrow + i0);
            float peak_v = 0.0f;
#pragma unroll
            for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
                float v;
                if (full4) {
                    const uint32_t word = static_cast<uint32_t>(
                        (raw >> (16u * l)) & 0xffffu);
                    v = quant_vec_bf16_lane_f32(word, 0u);
                } else {
                    const uint32_t i = i0 + l;
                    v = (i < K) ? __bfloat162float(xb[i]) : 0.0f;
                }
                if (isfinite(v)) {
                    const float av = std::fabs(v);
                    if (av > peak_v) peak_v = av;
                }
            }
#pragma unroll
            for (uint32_t off = 16u; off > 0u; off >>= 1u)
                peak_v = fmaxf(peak_v, __shfl_xor_sync(0xffffffffULL, peak_v, off));
            const float peak = __shfl_sync(0xffffffffULL, peak_v, 0u);
            const float scale = peak > 0.0f ? peak / 448.0f : 0.0f;
            const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            if (lane == 0u) row_scales[g] = scale;
            uint32_t packed = 0u;
#pragma unroll
            for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
                float v = 0.0f;
                if (i0 + l < K) {
                    if (full4) {
                        const uint32_t word = static_cast<uint32_t>(
                            (raw >> (16u * l)) & 0xffffu);
                        const float x = quant_vec_bf16_lane_f32(word, 0u);
                        v = isfinite(x) ? x : 0.0f;
                    } else {
                        const float x = __bfloat162float(xb[i0 + l]);
                        v = isfinite(x) ? x : 0.0f;
                    }
                }
                const uint8_t c = scale > 0.0f
                    ? ::ps::quantization::codec::device::e4m3_encode_u8(v * inv)
                    : 0u;
                packed |= static_cast<uint32_t>(c) << (8u * l);
            }
            if (i0 + kElemsPerLane <= KP) {
                const uint32_t off = (i0 >> 5u) * 512u +
                                     ((i0 & 31u) >> 4u) * 256u + (i0 & 15u);
                *reinterpret_cast<uint32_t*>(crow + off) = packed;
            }
        }
    }
};

template <uint32_t K, uint32_t KP, uint32_t Threads = kQuantThreadsPerRow>
__device__ __forceinline__ void activation_quantize_e4m3_vec_row_body(
    const ::ps::bf16_t* __restrict__ input,
    uint8_t* __restrict__ codes,
    float* __restrict__ scales,
    uint32_t m,
    uint32_t input_row_stride,
    uint32_t code_row_stride,
    uint32_t scale_row_stride) {
    QuantVecRow<K, KP, Threads>::run(input + static_cast<size_t>(m) * input_row_stride,
                                     codes, scales, m, code_row_stride, scale_row_stride);
}

}  // namespace ps::kernel::detail
