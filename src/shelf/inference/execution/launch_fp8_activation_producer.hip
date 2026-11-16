#include <shelf/inference/execution/fp8_activation_producer.h>
#include <shelf/quantization/codec/e4m3_detail.h>
#include <shelf/quantization/fp8/fp8_frag.h>
#include "../../kernels/optimized/detail/rmsnorm_device.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::execution {
namespace {

using bf16 = __hip_bfloat16;

constexpr uint32_t kFeatures = 5120u;
constexpr uint32_t kGroup = 128u;
constexpr uint32_t kThreads = 256u;
constexpr uint32_t kWaves = kThreads / 32u;
constexpr uint32_t kElemsPerLane = kGroup / 32u;

__device__ __forceinline__ void fp8_quantize_shared(
    const bf16* x, uint8_t* codes, float* scales, uint32_t m,
    uint32_t k, uint32_t k_padded, uint32_t code_stride, uint32_t scale_stride) {
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t groups = (k_padded + kGroup - 1u) / kGroup;
    uint8_t* crow = codes +
        static_cast<size_t>(m >> 4u) * (static_cast<size_t>(code_stride) * 16u);
    const uint32_t m_tile = m & 15u;
    float* row_scales = scales + static_cast<size_t>(m) * (scale_stride / 4u);

    for (uint32_t g = wave; g < groups; g += kWaves) {
        const uint32_t i0 = g * kGroup + lane * kElemsPerLane;
        float peak = 0.0f;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            const uint32_t i = i0 + l;
            if (i < k) {
                const float v = __bfloat162float(x[i]);
                if (isfinite(v)) {
                    const float av = fabsf(v);
                    if (av > peak) peak = av;
                }
            }
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffULL, peak, off));
        const float total = __shfl_sync(0xffffffffULL, peak, 0u);
        const float scale = total > 0.0f ? total / 448.0f : 0.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        if (lane == 0u) row_scales[g] = scale;
        uint32_t packed = 0u;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            const uint32_t i = i0 + l;
            float v = 0.0f;
            if (i < k) {
                const float xv = __bfloat162float(x[i]);
                v = isfinite(xv) ? xv : 0.0f;
            }
            const uint8_t c = scale > 0.0f
                ? ::shelf::quantization::codec::device::e4m3_encode_u8(v * inv)
                : 0u;
            packed |= static_cast<uint32_t>(c) << (8u * l);
        }
        if (i0 + kElemsPerLane <= k_padded) {
            const uint32_t off =
                (i0 >> 6u) * ::shelf::quantization::fp8::kFragBlockBytes +
                ::shelf::quantization::fp8::fp8_frag_offset(m_tile, i0 & 63u);
            *reinterpret_cast<uint32_t*>(crow + off) = packed;
        }
    }
}

__device__ __forceinline__ float rms_inverse(const bf16* x, float eps, float* sums) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint32_t base = tid * 4u; base < kFeatures; base += kThreads * 4u) {
        const auto v = ::shelf::kernel::detail::rms_load_bf16_x4(x + base);
        sum0 += v.x0 * v.x0;
        sum1 += v.x1 * v.x1;
        sum0 += v.x2 * v.x2;
        sum1 += v.x3 * v.x3;
    }
    float sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum0 + sum1);
    if (lane == 0u) sums[wave] = sum;
    __syncthreads();
    if (wave == 0u) {
        sum = lane < kWaves ? sums[lane] : 0.0f;
        sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum);
        if (lane == 0u)
            sums[0] = rsqrtf(sum / static_cast<float>(kFeatures) + eps);
    }
    __syncthreads();
    return sums[0];
}

__device__ __forceinline__ void normalize_shared(
    bf16* x, const bf16* weight, float inv) {
    for (uint32_t base = threadIdx.x * 4u; base < kFeatures; base += kThreads * 4u) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t i = base + j;
            const float y = __bfloat162float(x[i]) * inv *
                            (1.0f + __bfloat162float(weight[i]));
            x[i] = __float2bfloat16(y);
        }
    }
}

__global__ void fp8_residual_norm_quantize_kernel(Fp8ResidualNormArgs args) {
    __shared__ __align__(16) bf16 x[kFeatures];
    __shared__ float sums[kWaves];
    const uint32_t row = blockIdx.x;
    const bf16* a = reinterpret_cast<const bf16*>(args.a) +
        static_cast<size_t>(row) * args.a_stride;
    const bf16* b = reinterpret_cast<const bf16*>(args.b) +
        static_cast<size_t>(row) * args.b_stride;
    bf16* residual = reinterpret_cast<bf16*>(args.residual) +
        static_cast<size_t>(row) * args.residual_stride;
    for (uint32_t base = threadIdx.x * 4u; base < kFeatures; base += kThreads * 4u) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t i = base + j;
            const bf16 v = __float2bfloat16(
                __bfloat162float(a[i]) + __bfloat162float(b[i]));
            x[i] = v;
            residual[i] = v;
        }
    }
    const float inv = rms_inverse(x, args.eps, sums);
    normalize_shared(x, reinterpret_cast<const bf16*>(args.weight), inv);
    __syncthreads();
    fp8_quantize_shared(x, args.codes, args.scales, row, args.k, args.k_padded,
                        args.code_stride, args.scale_stride);
}

__global__ void fp8_norm_quantize_kernel(Fp8NormArgs args) {
    __shared__ __align__(16) bf16 x[kFeatures];
    __shared__ float sums[kWaves];
    const uint32_t row = blockIdx.x;
    const bf16* input = reinterpret_cast<const bf16*>(args.input) +
        static_cast<size_t>(row) * args.input_stride;
    for (uint32_t base = threadIdx.x * 4u; base < kFeatures; base += kThreads * 4u) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t i = base + j;
            x[i] = input[i];
        }
    }
    const float inv = rms_inverse(x, args.eps, sums);
    normalize_shared(x, reinterpret_cast<const bf16*>(args.weight), inv);
    __syncthreads();
    fp8_quantize_shared(x, args.codes, args.scales, row, args.k, args.k_padded,
                        args.code_stride, args.scale_stride);
}

__global__ void fp8_swiglu_quantize_kernel(Fp8SwigluArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t groups = (args.k_padded + kGroup - 1u) / kGroup;
    const bf16* gate = reinterpret_cast<const bf16*>(args.gate) +
        static_cast<size_t>(row) * args.gate_stride;
    const bf16* up = reinterpret_cast<const bf16*>(args.up) +
        static_cast<size_t>(row) * args.up_stride;
    uint8_t* crow = args.codes +
        static_cast<size_t>(row >> 4u) * (static_cast<size_t>(args.code_stride) * 16u);
    const uint32_t m_tile = row & 15u;
    float* row_scales = args.scales + static_cast<size_t>(row) * (args.scale_stride / 4u);

    for (uint32_t g = wave; g < groups; g += kWaves) {
        const uint32_t i0 = g * kGroup + lane * kElemsPerLane;
        float vals[kElemsPerLane];
        float peak = 0.0f;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            const uint32_t i = i0 + l;
            float v = 0.0f;
            if (i < args.k) {
                const float gv = __bfloat162float(gate[i]);
                const float uv = __bfloat162float(up[i]);
                float y;
                if (args.bf16_intermediate) {
                    const float activated = gv / (1.0f + __expf(-gv));
                    y = uv * __bfloat162float(__float2bfloat16(activated));
                } else {
                    y = uv * (gv / (1.0f + expf(-gv)));
                }
                v = __bfloat162float(__float2bfloat16(y));
            }
            vals[l] = v;
            if (isfinite(v)) {
                const float av = fabsf(v);
                if (av > peak) peak = av;
            }
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            peak = fmaxf(peak, __shfl_xor_sync(0xffffffffULL, peak, off));
        const float total = __shfl_sync(0xffffffffULL, peak, 0u);
        const float scale = total > 0.0f ? total / 448.0f : 0.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        if (lane == 0u) row_scales[g] = scale;
        uint32_t packed = 0u;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            const float v = isfinite(vals[l]) ? vals[l] : 0.0f;
            const uint8_t c = scale > 0.0f
                ? ::shelf::quantization::codec::device::e4m3_encode_u8(v * inv)
                : 0u;
            packed |= static_cast<uint32_t>(c) << (8u * l);
        }
        if (i0 + kElemsPerLane <= args.k_padded) {
            const uint32_t off =
                (i0 >> 6u) * ::shelf::quantization::fp8::kFragBlockBytes +
                ::shelf::quantization::fp8::fp8_frag_offset(m_tile, i0 & 63u);
            *reinterpret_cast<uint32_t*>(crow + off) = packed;
        }
    }
}

__global__ void fp8_final_gather_norm_quantize_kernel(Fp8FinalProducer args) {
    __shared__ __align__(16) bf16 x[kFeatures];
    __shared__ float sums[kWaves];
    const uint32_t row = blockIdx.x;
    const bf16* input = reinterpret_cast<const bf16*>(args.input) +
        static_cast<size_t>(args.batch->output_rows[row]) * args.input_stride;
    bf16* gathered = reinterpret_cast<bf16*>(args.gathered) +
        static_cast<size_t>(row) * args.gathered_stride;
    for (uint32_t base = threadIdx.x * 4u; base < kFeatures; base += kThreads * 4u) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t i = base + j;
            const bf16 v = input[i];
            x[i] = v;
            gathered[i] = v;
        }
    }
    const float inv = rms_inverse(x, args.eps, sums);
    normalize_shared(x, reinterpret_cast<const bf16*>(args.weight), inv);
    __syncthreads();
    fp8_quantize_shared(x, args.codes, args.scales, row, args.k, args.k_padded,
                        args.code_stride, args.scale_stride);
}

}  // namespace

hipError_t launch_fp8_residual_norm_quantize(const Fp8ResidualNormArgs& args,
                                             hipStream_t stream) {
    if (args.a == nullptr || args.b == nullptr || args.residual == nullptr ||
        args.weight == nullptr || args.codes == nullptr || args.scales == nullptr ||
        args.rows == 0u || args.k != kFeatures || args.k_padded < args.k ||
        args.a_stride < args.k || args.b_stride < args.k ||
        args.residual_stride < args.k || args.code_stride == 0u ||
        args.scale_stride == 0u)
        return hipErrorInvalidValue;
    fp8_residual_norm_quantize_kernel<<<args.rows, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_fp8_norm_quantize(const Fp8NormArgs& args, hipStream_t stream) {
    if (args.input == nullptr || args.weight == nullptr || args.codes == nullptr ||
        args.scales == nullptr || args.rows == 0u || args.k != kFeatures ||
        args.k_padded < args.k || args.input_stride < args.k ||
        args.code_stride == 0u || args.scale_stride == 0u)
        return hipErrorInvalidValue;
    fp8_norm_quantize_kernel<<<args.rows, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_fp8_swiglu_quantize(const Fp8SwigluArgs& args, hipStream_t stream) {
    if (args.gate == nullptr || args.up == nullptr || args.codes == nullptr ||
        args.scales == nullptr || args.rows == 0u || args.k == 0u ||
        args.k_padded < args.k || args.gate_stride < args.k || args.up_stride < args.k ||
        args.code_stride == 0u || args.scale_stride == 0u)
        return hipErrorInvalidValue;
    fp8_swiglu_quantize_kernel<<<args.rows, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_fp8_final_gather_norm_quantize(const Fp8FinalProducer& args,
                                                 hipStream_t stream) {
    if (args.input == nullptr || args.weight == nullptr || args.gathered == nullptr ||
        args.codes == nullptr || args.scales == nullptr || args.batch == nullptr ||
        args.rows == 0u || args.k != kFeatures || args.k_padded < args.k ||
        args.input_stride < args.k || args.gathered_stride < args.k ||
        args.code_stride == 0u || args.scale_stride == 0u)
        return hipErrorInvalidValue;
    fp8_final_gather_norm_quantize_kernel<<<args.rows, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::execution
