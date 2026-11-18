#include <shelf/kernels/fused/mxfp4_activation_producer.h>
#include "../optimized/detail/rmsnorm_device.h"
#include <shelf/quantization/codec/e2m1_detail.h>
#include <shelf/quantization/codec/e4m3_detail.h>
#include <shelf/quantization/fp8/fp8_frag.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::execution {
namespace {

using bf16 = __hip_bfloat16;
using ::shelf::quantization::codec::device::e4m3_decode_f32;
using ::shelf::quantization::codec::device::e4m3_encode_u8;
using ::shelf::quantization::codec::device::e8m0_exponent_for_peak;
using ::shelf::quantization::codec::device::pow2_f32;

constexpr uint32_t kThreads = 256u;
constexpr uint32_t kDecodeThreads = 1024u;
constexpr uint32_t kMaxWaves = kDecodeThreads / 32u;
constexpr uint32_t kGroup = 128u;
constexpr uint32_t kElemsPerLane = kGroup / 32u;
constexpr float kEps = 1e-6f;

__device__ __forceinline__ float bf16_to_f32(const bf16* p) {
    return __bfloat162float(*p);
}

__device__ __forceinline__ float bf16_round(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

struct RowrefScratch {
    uint32_t* pack;
    int32_t* exp;
    int32_t* wmax;
};

__host__ __device__ __forceinline__ uint32_t align_up4(uint32_t words, uint32_t align) {
    return (words + align - 1u) & ~(align - 1u);
}

__device__ __forceinline__ void rowref_finish(
    const RowrefScratch& s, uint32_t groups, uint32_t k_padded, uint32_t row,
    uint8_t* codes, uint32_t code_stride, float* scales, uint32_t scale_stride) {
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t waves = blockDim.x >> 5u;
    int32_t wmax = -127;
    for (uint32_t g = wave; g < groups; g += waves)
        if (s.exp[g] > wmax) wmax = s.exp[g];
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        const int32_t other = __shfl_xor_sync(0xffffffffULL, wmax, off);
        if (other > wmax) wmax = other;
    }
    if (lane == 0u) s.wmax[wave] = wmax;
    __syncthreads();
    int32_t ref = -127;
    for (uint32_t w = 0u; w < waves; ++w)
        if (s.wmax[w] > ref) ref = s.wmax[w];
    if (threadIdx.x == 0u) {
        float* row_scales = scales + static_cast<size_t>(row) * (scale_stride / 4u);
        row_scales[0] = pow2_f32(ref);
        for (uint32_t g = 1u; g < groups; ++g) row_scales[g] = 0.0f;
    }
    uint8_t* crow = codes +
        static_cast<size_t>(row >> 4u) * (static_cast<size_t>(code_stride) * 16u);
    const uint32_t m_tile = row & 15u;
    for (uint32_t g = wave; g < groups; g += waves) {
        const uint32_t i0 = g * kGroup + lane * kElemsPerLane;
        const uint32_t packed = s.pack[g * 32u + lane];
        const int32_t d = ref - s.exp[g];
        uint32_t out = 0u;
        if (d <= 120) {
            const float fold = pow2_f32(-d);
            #pragma unroll
            for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
                const uint8_t c = static_cast<uint8_t>((packed >> (8u * l)) & 0xffu);
                const float folded = e4m3_decode_f32(c) * fold;
                out |= static_cast<uint32_t>(e4m3_encode_u8(folded)) << (8u * l);
            }
        }
        if (i0 + kElemsPerLane <= k_padded) {
            const uint32_t off = (i0 >> 6u) * ::shelf::quantization::fp8::kFragBlockBytes +
                                 ::shelf::quantization::fp8::fp8_frag_offset(
                                     m_tile, i0 & 63u);
            *reinterpret_cast<uint32_t*>(crow + off) = out;
        }
    }
}

__device__ __forceinline__ void rowref_stage_group(
    const RowrefScratch& s, uint32_t g, uint32_t lane, float peak, uint32_t packed) {
    s.pack[g * 32u + lane] = packed;
    if (lane == 0u) s.exp[g] = e8m0_exponent_for_peak(peak);
}

__device__ __forceinline__ float group_peak(float peak) {
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        peak = fmaxf(peak, __shfl_xor_sync(0xffffffffULL, peak, off));
    return __shfl_sync(0xffffffffULL, peak, 0u);
}

__device__ __forceinline__ uint32_t encode_group(const float* vals, float peak) {
    const int32_t e = e8m0_exponent_for_peak(peak);
    const float inv = pow2_f32(-e);
    uint32_t packed = 0u;
    #pragma unroll
    for (uint32_t l = 0u; l < kElemsPerLane; ++l)
        packed |= static_cast<uint32_t>(e4m3_encode_u8(vals[l] * inv)) << (8u * l);
    return packed;
}

__device__ __forceinline__ float rms_inverse(const bf16* x, uint32_t k, float eps,
                                             float* sums) {
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;
    const uint32_t waves = blockDim.x >> 5u;
    const uint32_t threads = blockDim.x;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint32_t base = tid * 4u; base < k; base += threads * 4u) {
        if (base + 4u <= k) {
            const auto v = ::shelf::kernel::detail::rms_load_bf16_x4(x + base);
            sum0 += v.x0 * v.x0;
            sum1 += v.x1 * v.x1;
            sum0 += v.x2 * v.x2;
            sum1 += v.x3 * v.x3;
            continue;
        }
        for (uint32_t j = 0u; j < 4u && base + j < k; ++j) {
            const float v = __bfloat162float(x[base + j]);
            if ((j & 1u) == 0u) sum0 += v * v; else sum1 += v * v;
        }
    }
    float sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum0 + sum1);
    if (lane == 0u) sums[wave] = sum;
    __syncthreads();
    if (wave == 0u) {
        sum = lane < waves ? sums[lane] : 0.0f;
        sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum);
        if (lane == 0u) sums[0] = rsqrtf(sum / static_cast<float>(k) + eps);
    }
    __syncthreads();
    return sums[0];
}

__device__ __forceinline__ void normalize_row(bf16* x, const bf16* weight,
                                              uint32_t k, float inv) {
    for (uint32_t base = threadIdx.x * 4u; base < k; base += blockDim.x * 4u) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t i = base + j;
            if (i >= k) break;
            const float y = __bfloat162float(x[i]) * inv *
                            (1.0f + __bfloat162float(weight[i]));
            x[i] = __float2bfloat16(y);
        }
    }
}

__device__ __forceinline__ void quantize_bf16_row(
    const bf16* x, const RowrefScratch& s, uint32_t groups, uint32_t k,
    uint32_t k_padded, uint32_t row, uint8_t* codes, uint32_t code_stride,
    float* scales, uint32_t scale_stride) {
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t waves = blockDim.x >> 5u;
    for (uint32_t g = wave; g < groups; g += waves) {
        const uint32_t i0 = g * kGroup + lane * kElemsPerLane;
        float vals[kElemsPerLane];
        float peak = 0.0f;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            const uint32_t i = i0 + l;
            const float v = i < k ? __bfloat162float(x[i]) : 0.0f;
            vals[l] = isfinite(v) ? v : 0.0f;
            const float av = fabsf(vals[l]);
            if (av > peak) peak = av;
        }
        peak = group_peak(peak);
        rowref_stage_group(s, g, lane, peak, encode_group(vals, peak));
    }
    __syncthreads();
    rowref_finish(s, groups, k_padded, row, codes, code_stride, scales, scale_stride);
}

template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void mxfp4_norm_quantize_kernel(
    Mxfp4NormArgs a) {
    extern __shared__ __align__(16) uint32_t smem[];
    const uint32_t groups = (a.k_padded + kGroup - 1u) / kGroup;
    bf16* x = reinterpret_cast<bf16*>(smem);
    float* sums = reinterpret_cast<float*>(x + a.k);
    const uint32_t sums_words = kMaxWaves;
    RowrefScratch s;
    s.pack = smem + align_up4(static_cast<uint32_t>(a.k / 2u) + sums_words, 4u);
    s.exp = reinterpret_cast<int32_t*>(s.pack + groups * 32u);
    s.wmax = s.exp + groups;
    const uint32_t row = blockIdx.x;
    const bf16* input = reinterpret_cast<const bf16*>(a.input) +
        static_cast<size_t>(row) * a.input_stride;
    for (uint32_t base = threadIdx.x * 4u; base < a.k; base += kBlock * 4u) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            if (base + j < a.k) x[base + j] = input[base + j];
        }
    }
    const float inv = rms_inverse(x, a.k, a.eps, sums);
    normalize_row(x, reinterpret_cast<const bf16*>(a.weight), a.k, inv);
    __syncthreads();
    quantize_bf16_row(x, s, groups, a.k, a.k_padded, row, a.codes, a.code_stride,
                      a.scales, a.scale_stride);
}

template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void mxfp4_residual_norm_quantize_kernel(
    Mxfp4ResidualNormArgs a) {
    extern __shared__ __align__(16) uint32_t smem[];
    const uint32_t groups = (a.k_padded + kGroup - 1u) / kGroup;
    bf16* x = reinterpret_cast<bf16*>(smem);
    float* sums = reinterpret_cast<float*>(x + a.k);
    const uint32_t sums_words = kMaxWaves;
    RowrefScratch s;
    s.pack = smem + align_up4(static_cast<uint32_t>(a.k / 2u) + sums_words, 4u);
    s.exp = reinterpret_cast<int32_t*>(s.pack + groups * 32u);
    s.wmax = s.exp + groups;
    const uint32_t row = blockIdx.x;
    const bf16* left = reinterpret_cast<const bf16*>(a.a) +
        static_cast<size_t>(row) * a.a_stride;
    const bf16* right = reinterpret_cast<const bf16*>(a.b) +
        static_cast<size_t>(row) * a.b_stride;
    bf16* residual = reinterpret_cast<bf16*>(a.residual) +
        static_cast<size_t>(row) * a.residual_stride;
    for (uint32_t base = threadIdx.x * 4u; base < a.k; base += kBlock * 4u) {
        #pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const uint32_t i = base + j;
            if (i >= a.k) break;
            const bf16 v = __float2bfloat16(__bfloat162float(left[i]) +
                                            __bfloat162float(right[i]));
            x[i] = v;
            residual[i] = v;
        }
    }
    const float inv = rms_inverse(x, a.k, a.eps, sums);
    normalize_row(x, reinterpret_cast<const bf16*>(a.weight), a.k, inv);
    __syncthreads();
    quantize_bf16_row(x, s, groups, a.k, a.k_padded, row, a.codes, a.code_stride,
                      a.scales, a.scale_stride);
}

template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void mxfp4_swiglu_quantize_kernel(
    Mxfp4SwigluArgs a) {
    extern __shared__ __align__(16) uint32_t smem[];
    const uint32_t groups = (a.k_padded + kGroup - 1u) / kGroup;
    RowrefScratch s;
    s.pack = smem;
    s.exp = reinterpret_cast<int32_t*>(s.pack + groups * 32u);
    s.wmax = s.exp + groups;
    const uint32_t row = blockIdx.x;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t waves = blockDim.x >> 5u;
    const bf16* gate = reinterpret_cast<const bf16*>(a.gate) +
        static_cast<size_t>(row) * a.gate_stride;
    const bf16* up = reinterpret_cast<const bf16*>(a.up) +
        static_cast<size_t>(row) * a.up_stride;
    for (uint32_t g = wave; g < groups; g += waves) {
        const uint32_t i0 = g * kGroup + lane * kElemsPerLane;
        float vals[kElemsPerLane];
        float peak = 0.0f;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            const uint32_t i = i0 + l;
            float v = 0.0f;
            if (i < a.k) {
                const float gv = __bfloat162float(gate[i]);
                const float uv = __bfloat162float(up[i]);
                float y;
                if (a.bf16_intermediate) {
                    const float activated = gv / (1.0f + __expf(-gv));
                    y = uv * __bfloat162float(__float2bfloat16(activated));
                } else {
                    y = uv * (gv / (1.0f + expf(-gv)));
                }
                v = bf16_round(y);
            }
            vals[l] = v;
            if (isfinite(v)) {
                const float av = fabsf(v);
                if (av > peak) peak = av;
            }
        }
        peak = group_peak(peak);
        rowref_stage_group(s, g, lane, peak, encode_group(vals, peak));
    }
    __syncthreads();
    rowref_finish(s, groups, a.k_padded, row, a.codes, a.code_stride, a.scales,
                  a.scale_stride);
}

template <uint32_t kBlock>
__global__ __launch_bounds__(kBlock) void mxfp4_gdn_gated_quantize_kernel(
    Mxfp4GdnGatedArgs a) {
    extern __shared__ __align__(16) uint32_t smem[];
    const uint32_t groups = a.features / a.group_size;
    RowrefScratch s;
    s.pack = smem;
    s.exp = reinterpret_cast<int32_t*>(s.pack + groups * 32u);
    s.wmax = s.exp + groups;
    const uint32_t row = blockIdx.x;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t waves = blockDim.x >> 5u;
    const float* rec = a.rec + static_cast<size_t>(row) * a.rec_stride;
    const bf16* z = reinterpret_cast<const bf16*>(a.z) +
        static_cast<size_t>(row) * a.z_stride;
    for (uint32_t g = wave; g < groups; g += waves) {
        const uint32_t i0 = g * a.group_size + lane * kElemsPerLane;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float xv[kElemsPerLane];
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            const float v = rec[i0 + l];
            xv[l] = v;
            if ((l & 1u) == 0u)
                sum0 += v * v;
            else
                sum1 += v * v;
        }
        float sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum0 + sum1);
        sum = __shfl_sync(0xffffffffULL, sum, 0u);
        const float inv = rsqrtf(sum / static_cast<float>(a.group_size) + a.eps);
        float vals[kElemsPerLane];
        float peak = 0.0f;
        #pragma unroll
        for (uint32_t l = 0u; l < kElemsPerLane; ++l) {
            const float y = bf16_round(xv[l] * inv) * __bfloat162float(reinterpret_cast<const bf16*>(a.weight)[lane * kElemsPerLane + l]);
            const float zv = __bfloat162float(z[i0 + l]);
            const float gated = bf16_round(y * (zv / (1.0f + expf(-zv))));
            vals[l] = gated;
            if (isfinite(gated)) {
                const float av = fabsf(gated);
                if (av > peak) peak = av;
            }
        }
        peak = group_peak(peak);
        rowref_stage_group(s, g, lane, peak, encode_group(vals, peak));
    }
    __syncthreads();
    rowref_finish(s, groups, a.features, row, a.codes, a.code_stride, a.scales,
                  a.scale_stride);
}

uint32_t norm_smem_bytes(uint32_t k, uint32_t groups) {
    const uint32_t words = k / 2u + kMaxWaves;
    return (align_up4(words, 4u) + groups * 32u + groups + kMaxWaves) * 4u;
}

uint32_t group_smem_bytes(uint32_t groups) {
    return (groups * 32u + groups + kMaxWaves) * 4u;
}

bool valid_group_geometry(uint32_t k, uint32_t k_padded) {
    if (k == 0u || k_padded < k || (k_padded % kGroup) != 0u) return false;
    if ((k_padded / kGroup) > 256u) return false;
    return true;
}

}  // namespace

hipError_t launch_mxfp4_norm_quantize(const Mxfp4NormArgs& args, hipStream_t stream) {
    if (args.rows == 0u || args.input == nullptr || args.weight == nullptr ||
        args.codes == nullptr || args.scales == nullptr)
        return hipErrorInvalidValue;
    if (!valid_group_geometry(args.k, args.k_padded)) return hipErrorNotSupported;
    if ((args.code_stride % 16u) != 0u || (args.scale_stride % 16u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t groups = args.k_padded / kGroup;
    if (args.rows <= 32u)
        mxfp4_norm_quantize_kernel<kDecodeThreads>
            <<<args.rows, kDecodeThreads, norm_smem_bytes(args.k, groups), stream>>>(args);
    else
        mxfp4_norm_quantize_kernel<kThreads>
            <<<args.rows, kThreads, norm_smem_bytes(args.k, groups), stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_mxfp4_residual_norm_quantize(const Mxfp4ResidualNormArgs& args,
                                               hipStream_t stream) {
    if (args.rows == 0u || args.a == nullptr || args.b == nullptr ||
        args.residual == nullptr || args.weight == nullptr || args.codes == nullptr ||
        args.scales == nullptr)
        return hipErrorInvalidValue;
    if (!valid_group_geometry(args.k, args.k_padded)) return hipErrorNotSupported;
    if ((args.code_stride % 16u) != 0u || (args.scale_stride % 16u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t groups = args.k_padded / kGroup;
    if (args.rows <= 32u)
        mxfp4_residual_norm_quantize_kernel<kDecodeThreads>
            <<<args.rows, kDecodeThreads, norm_smem_bytes(args.k, groups), stream>>>(args);
    else
        mxfp4_residual_norm_quantize_kernel<kThreads>
            <<<args.rows, kThreads, norm_smem_bytes(args.k, groups), stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_mxfp4_swiglu_quantize(const Mxfp4SwigluArgs& args, hipStream_t stream) {
    if (args.rows == 0u || args.gate == nullptr || args.up == nullptr ||
        args.codes == nullptr || args.scales == nullptr)
        return hipErrorInvalidValue;
    if (!valid_group_geometry(args.k, args.k_padded)) return hipErrorNotSupported;
    if ((args.code_stride % 16u) != 0u || (args.scale_stride % 16u) != 0u)
        return hipErrorInvalidValue;
    const uint32_t groups = args.k_padded / kGroup;
    if (args.rows <= 32u)
        mxfp4_swiglu_quantize_kernel<kDecodeThreads>
            <<<args.rows, kDecodeThreads, group_smem_bytes(groups), stream>>>(args);
    else
        mxfp4_swiglu_quantize_kernel<kThreads>
            <<<args.rows, kThreads, group_smem_bytes(groups), stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_mxfp4_gdn_gated_quantize(const Mxfp4GdnGatedArgs& args,
                                           hipStream_t stream) {
    if (args.rows == 0u || args.rec == nullptr || args.z == nullptr ||
        args.weight == nullptr || args.codes == nullptr || args.scales == nullptr)
        return hipErrorInvalidValue;
    if (args.features == 0u || args.group_size != kGroup ||
        (args.features % args.group_size) != 0u)
        return hipErrorNotSupported;
    const uint32_t groups = args.features / args.group_size;
    if (groups > 256u) return hipErrorNotSupported;
    if ((args.code_stride % 16u) != 0u || (args.scale_stride % 16u) != 0u)
        return hipErrorInvalidValue;
    if (args.rows <= 32u)
        mxfp4_gdn_gated_quantize_kernel<kDecodeThreads>
            <<<args.rows, kDecodeThreads, group_smem_bytes(groups), stream>>>(args);
    else
        mxfp4_gdn_gated_quantize_kernel<kThreads>
            <<<args.rows, kThreads, group_smem_bytes(groups), stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::execution
