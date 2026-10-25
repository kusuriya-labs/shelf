// vendored from PhaseShift (own work); namespace/type adapted only.
#pragma once

#include <shelf/backend/hip/kernels/rmsnorm.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstddef>
#include <cstdint>

namespace shelf::hipk::detail {

using bf16 = __hip_bfloat16;

struct RmsF32x4 {
    float x0;
    float x1;
    float x2;
    float x3;
};

__device__ __forceinline__ float rms_round_to_bf16_f32(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ RmsF32x4 rms_load_bf16_x4(const bf16* p) {
    uint64_t bits = *reinterpret_cast<const uint64_t*>(p);
    bf16 b[4];
    *reinterpret_cast<uint64_t*>(b) = bits;
    RmsF32x4 r;
    r.x0 = __bfloat162float(b[0]);
    r.x1 = __bfloat162float(b[1]);
    r.x2 = __bfloat162float(b[2]);
    r.x3 = __bfloat162float(b[3]);
    return r;
}

__device__ __forceinline__ RmsF32x4 rms_load_f32_x4(const float* p) {
    float4 v = *reinterpret_cast<const float4*>(p);
    RmsF32x4 r;
    r.x0 = v.x;
    r.x1 = v.y;
    r.x2 = v.z;
    r.x3 = v.w;
    return r;
}

__device__ __forceinline__ void rms_store_bf16_x4(bf16* p, RmsF32x4 v) {
    bf16 b[4];
    b[0] = __float2bfloat16(v.x0);
    b[1] = __float2bfloat16(v.x1);
    b[2] = __float2bfloat16(v.x2);
    b[3] = __float2bfloat16(v.x3);
    *reinterpret_cast<uint64_t*>(p) = *reinterpret_cast<const uint64_t*>(b);
}

__device__ __forceinline__ void rms_store_f32_x4(float* p, RmsF32x4 v) {
    *reinterpret_cast<float4*>(p) = make_float4(v.x0, v.x1, v.x2, v.x3);
}

__device__ __forceinline__ float rms_wave_reduce_sum(float v) {
    v += __shfl_down(v, 16, 32);
    v += __shfl_down(v, 8, 32);
    v += __shfl_down(v, 4, 32);
    v += __shfl_down(v, 2, 32);
    v += __shfl_down(v, 1, 32);
    return v;
}

using ::shelf::hipk::RmsNormDataType;
using ::shelf::hipk::RmsNormWeightLayout;
using ::shelf::hipk::RmsNormWeightMode;

template <RmsNormDataType InT, RmsNormDataType OutT, RmsNormDataType Wt,
          RmsNormWeightLayout WLayout, RmsNormWeightMode WMode>
__device__ __forceinline__ void rmsnorm_row_body(
    const void* input, const void* weight, void* output,
    uint32_t row, uint32_t gi, uint32_t group_size,
    uint32_t input_row_stride, uint32_t output_row_stride, float eps) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5;
    const uint32_t num_waves = blockDim.x >> 5;

    const uint32_t g0 = gi * group_size;
    const size_t in_off = static_cast<size_t>(row) * input_row_stride + g0;
    const size_t out_off = static_cast<size_t>(row) * output_row_stride + g0;
    const bool aligned = static_cast<uint32_t>(in_off & 3u) == 0u;

    __shared__ float wave_sums[8];

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint32_t base = tid * 4u; base < group_size; base += blockDim.x * 4u) {
        const bool full = (base + 4u <= group_size);
        RmsF32x4 x;
        if (full && aligned) {
            if constexpr (InT == RmsNormDataType::BF16) {
                x = rms_load_bf16_x4(reinterpret_cast<const bf16*>(input) + in_off + base);
            } else {
                x = rms_load_f32_x4(reinterpret_cast<const float*>(input) + in_off + base);
            }
        } else {
            x.x0 = (base + 0u < group_size) ? (InT == RmsNormDataType::BF16
                    ? __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + base + 0u])
                    : reinterpret_cast<const float*>(input)[in_off + base + 0u]) : 0.0f;
            x.x1 = (base + 1u < group_size) ? (InT == RmsNormDataType::BF16
                    ? __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + base + 1u])
                    : reinterpret_cast<const float*>(input)[in_off + base + 1u]) : 0.0f;
            x.x2 = (base + 2u < group_size) ? (InT == RmsNormDataType::BF16
                    ? __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + base + 2u])
                    : reinterpret_cast<const float*>(input)[in_off + base + 2u]) : 0.0f;
            x.x3 = (base + 3u < group_size) ? (InT == RmsNormDataType::BF16
                    ? __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + base + 3u])
                    : reinterpret_cast<const float*>(input)[in_off + base + 3u]) : 0.0f;
        }
        sum0 += x.x0 * x.x0;
        sum1 += x.x1 * x.x1;
        sum0 += x.x2 * x.x2;
        sum1 += x.x3 * x.x3;
    }

    float total = rms_wave_reduce_sum(sum0 + sum1);
    if (lane == 0u) wave_sums[wave] = total;
    __syncthreads();

    if (wave == 0u) {
        float block_sum = (lane < num_waves) ? wave_sums[lane] : 0.0f;
        block_sum = rms_wave_reduce_sum(block_sum);
        if (lane == 0u) {
            wave_sums[0] = rsqrtf(block_sum / static_cast<float>(group_size) + eps);
        }
    }
    __syncthreads();

    const float inv = wave_sums[0];

    for (uint32_t base = tid * 4u; base < group_size; base += blockDim.x * 4u) {
        const bool full = (base + 4u <= group_size);
        if (!full && base + 3u >= group_size) {
            for (uint32_t j = 0u; j < 4u; ++j) {
                const uint32_t h = base + j;
                if (h >= group_size) break;
                float xv;
                if constexpr (InT == RmsNormDataType::BF16) {
                    xv = __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + h]);
                } else {
                    xv = reinterpret_cast<const float*>(input)[in_off + h];
                }
                float y = xv * inv;
                if constexpr (InT == RmsNormDataType::F32) {
                    y = rms_round_to_bf16_f32(y);
                }
                if constexpr (WLayout != RmsNormWeightLayout::NONE) {
                    const uint32_t widx =
                        (WLayout == RmsNormWeightLayout::PER_FEATURE) ? (g0 + h) : h;
                    float w;
                    if constexpr (Wt == RmsNormDataType::BF16) {
                        w = __bfloat162float(reinterpret_cast<const bf16*>(weight)[widx]);
                    } else {
                        w = reinterpret_cast<const float*>(weight)[widx];
                    }
                    if constexpr (WMode == RmsNormWeightMode::ONE_PLUS) {
                        y *= 1.0f + w;
                    } else {
                        y *= w;
                    }
                }
                if constexpr (OutT == RmsNormDataType::BF16) {
                    reinterpret_cast<bf16*>(output)[out_off + h] = __float2bfloat16(y);
                } else {
                    reinterpret_cast<float*>(output)[out_off + h] = y;
                }
            }
            continue;
        }

        RmsF32x4 x;
        if (aligned) {
            if constexpr (InT == RmsNormDataType::BF16) {
                x = rms_load_bf16_x4(reinterpret_cast<const bf16*>(input) + in_off + base);
            } else {
                x = rms_load_f32_x4(reinterpret_cast<const float*>(input) + in_off + base);
            }
        } else {
            x.x0 = (InT == RmsNormDataType::BF16
                    ? __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + base + 0u])
                    : reinterpret_cast<const float*>(input)[in_off + base + 0u]);
            x.x1 = (InT == RmsNormDataType::BF16
                    ? __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + base + 1u])
                    : reinterpret_cast<const float*>(input)[in_off + base + 1u]);
            x.x2 = (InT == RmsNormDataType::BF16
                    ? __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + base + 2u])
                    : reinterpret_cast<const float*>(input)[in_off + base + 2u]);
            x.x3 = (InT == RmsNormDataType::BF16
                    ? __bfloat162float(reinterpret_cast<const bf16*>(input)[in_off + base + 3u])
                    : reinterpret_cast<const float*>(input)[in_off + base + 3u]);
        }

        RmsF32x4 y;
        y.x0 = x.x0 * inv;
        y.x1 = x.x1 * inv;
        y.x2 = x.x2 * inv;
        y.x3 = x.x3 * inv;

        if constexpr (InT == RmsNormDataType::F32) {
            y.x0 = rms_round_to_bf16_f32(y.x0);
            y.x1 = rms_round_to_bf16_f32(y.x1);
            y.x2 = rms_round_to_bf16_f32(y.x2);
            y.x3 = rms_round_to_bf16_f32(y.x3);
        }

        if constexpr (WLayout != RmsNormWeightLayout::NONE) {
            RmsF32x4 w;
            if constexpr (WLayout == RmsNormWeightLayout::PER_FEATURE) {
                const uint32_t wbase = g0 + base;
                if constexpr (Wt == RmsNormDataType::BF16) {
                    const bf16* wp = reinterpret_cast<const bf16*>(weight);
                    w.x0 = __bfloat162float(wp[wbase + 0u]);
                    w.x1 = __bfloat162float(wp[wbase + 1u]);
                    w.x2 = __bfloat162float(wp[wbase + 2u]);
                    w.x3 = __bfloat162float(wp[wbase + 3u]);
                } else {
                    const float* wp = reinterpret_cast<const float*>(weight);
                    w.x0 = wp[wbase + 0u];
                    w.x1 = wp[wbase + 1u];
                    w.x2 = wp[wbase + 2u];
                    w.x3 = wp[wbase + 3u];
                }
            } else {
                if constexpr (Wt == RmsNormDataType::BF16) {
                    const bf16* wp = reinterpret_cast<const bf16*>(weight);
                    w.x0 = __bfloat162float(wp[base + 0u]);
                    w.x1 = __bfloat162float(wp[base + 1u]);
                    w.x2 = __bfloat162float(wp[base + 2u]);
                    w.x3 = __bfloat162float(wp[base + 3u]);
                } else {
                    const float* wp = reinterpret_cast<const float*>(weight);
                    w.x0 = wp[base + 0u];
                    w.x1 = wp[base + 1u];
                    w.x2 = wp[base + 2u];
                    w.x3 = wp[base + 3u];
                }
            }
            if constexpr (WMode == RmsNormWeightMode::ONE_PLUS) {
                y.x0 *= 1.0f + w.x0;
                y.x1 *= 1.0f + w.x1;
                y.x2 *= 1.0f + w.x2;
                y.x3 *= 1.0f + w.x3;
            } else {
                y.x0 *= w.x0;
                y.x1 *= w.x1;
                y.x2 *= w.x2;
                y.x3 *= w.x3;
            }
        }

        if (aligned) {
            if constexpr (OutT == RmsNormDataType::BF16) {
                rms_store_bf16_x4(reinterpret_cast<bf16*>(output) + out_off + base, y);
            } else {
                rms_store_f32_x4(reinterpret_cast<float*>(output) + out_off + base, y);
            }
        } else {
            if constexpr (OutT == RmsNormDataType::BF16) {
                bf16* op = reinterpret_cast<bf16*>(output) + out_off + base;
                op[0] = __float2bfloat16(y.x0);
                op[1] = __float2bfloat16(y.x1);
                op[2] = __float2bfloat16(y.x2);
                op[3] = __float2bfloat16(y.x3);
            } else {
                float* op = reinterpret_cast<float*>(output) + out_off + base;
                op[0] = y.x0;
                op[1] = y.x1;
                op[2] = y.x2;
                op[3] = y.x3;
            }
        }
    }
}

}  // namespace shelf::hipk::detail
