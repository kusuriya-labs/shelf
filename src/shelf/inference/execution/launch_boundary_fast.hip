#include <shelf/inference/execution/boundary_deferred.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include "../../kernels/optimized/detail/rmsnorm_device.h"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace shelf::execution {
namespace {
using bf16 = __hip_bfloat16;
using ::shelf::kernel::PlamoExl3ProjectionGroup;
using namespace ::shelf::kernel::exl3_detail;
using namespace ::shelf::kernel::exl3_detail::mul24;

__device__ __forceinline__ bf16 boundary_normalize(bf16 x, bf16 weight, float inv) {
    float y = __bfloat162float(x) * inv;
    y *= 1.f + __bfloat162float(weight);
    return __float2bfloat16(y);
}

template<bool FromOutput>
__global__ void boundary_fast(PlamoExl3ProjectionGroup group, BoundaryDeferred source,
    const bf16* a, const bf16* b, bf16* residual, const bf16* weight, bf16* output,
    uint32_t as, uint32_t bs, uint32_t rs, uint32_t os, float eps, bool source_first) {
    __shared__ bf16 x[5120];
    __shared__ float sums[8];
    __shared__ float smem[256];
    __shared__ float peaks[8];
    const uint32_t row = blockIdx.x, g = blockIdx.y;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5u;
    const uint32_t projection = blockIdx.z * 2u + tid / 128u;
    const uint32_t safe_projection = projection < group.count ? projection : group.count - 1u;
    float sum0 = 0.f, sum1 = 0.f;
    if constexpr (FromOutput) {
        for (uint32_t block = wave; block < 40u; block += 8u) {
            const uint32_t col = block * 128u + lane * 4u;
            float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
            for (uint32_t z = 0; z < source.splits; ++z) {
                const float4 v = *reinterpret_cast<const float4*>(source.yprime +
                    (static_cast<size_t>(z) * source.rows + row) * 5120u + col);
                v0 += v.x; v1 += v.y; v2 += v.z; v3 += v.w;
            }
            fwht128(v0, v1, v2, v3, lane);
            const float v[4] = {v0, v1, v2, v3};
            float r[4];
            for (uint32_t j = 0; j < 4u; ++j) {
                const float transformed = (v[j] * kInvSqrt128) * f16_bits_to_f32(source.svh[col + j]);
                const float rounded = __bfloat162float(__float2bfloat16(transformed));
                const float skip = __bfloat162float(b[static_cast<size_t>(row) * bs + col + j]);
                const bf16 value = __float2bfloat16(source_first ? rounded + skip : skip + rounded);
                x[col + j] = value;
                r[j] = __bfloat162float(value);
            }
            sum0 += r[0] * r[0];
            sum1 += r[1] * r[1];
            sum0 += r[2] * r[2];
            sum1 += r[3] * r[3];
        }
    } else {
        for (uint32_t base = tid * 4u; base < 5120u; base += 1024u) {
            float r[4];
            for (uint32_t j = 0; j < 4u; ++j) {
                const uint32_t i = base + j;
                const bf16 value = __float2bfloat16(__bfloat162float(a[static_cast<size_t>(row) * as + i]) +
                                                    __bfloat162float(b[static_cast<size_t>(row) * bs + i]));
                x[i] = value;
                r[j] = __bfloat162float(value);
            }
            sum0 += r[0] * r[0];
            sum1 += r[1] * r[1];
            sum0 += r[2] * r[2];
            sum1 += r[3] * r[3];
        }
    }
    float sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum0 + sum1);
    if (lane == 0u) sums[wave] = sum;
    __syncthreads();
    if (wave == 0u) {
        sum = lane < 8u ? sums[lane] : 0.f;
        sum = ::shelf::kernel::detail::rms_wave_reduce_sum(sum);
        if (lane == 0u) sums[0] = rsqrtf(sum / 5120.f + eps);
    }
    __syncthreads();
    const float inv = sums[0];
    if (g == 0u && blockIdx.z == 0u) {
        for (uint32_t i = tid; i < 5120u; i += 256u) {
            residual[static_cast<size_t>(row) * rs + i] = x[i];
            output[static_cast<size_t>(row) * os + i] = boundary_normalize(x[i], weight[i], inv);
        }
    }
    const uint32_t t = tid & 127u, col = g * 128u + t;
    const bf16 norm = boundary_normalize(x[col], weight[col], inv);
    float v = __bfloat162float(norm) * f16_bits_to_f32(group.p[safe_projection].suh[col]);
    v = fwht128_smem(v, smem + (tid / 128u) * 128u, t) * kInvSqrt128;
    float peak = fabsf(v);
    for (uint32_t off = 16u; off; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off));
    if (lane == 0u) peaks[wave] = peak;
    __syncthreads();
    const uint32_t base = (tid / 128u) * 4u;
    peak = fmaxf(fmaxf(peaks[base], peaks[base + 1u]), fmaxf(peaks[base + 2u], peaks[base + 3u]));
    const float scale = peak / 127.f;
    const float q = scale > 0.f ? fminf(fmaxf(rintf(v / scale), -128.f), 127.f) : 0.f;
    if (projection < group.count) {
        group.p[projection].codes[static_cast<size_t>(row) * group.k + col] = static_cast<int8_t>(static_cast<int>(q));
        if (t == 0u) group.p[projection].scales[static_cast<size_t>(row) * (group.scale_stride_bytes / 4u) + g] = scale;
    }
}
}

hipError_t launch_boundary_rotate_fast(const PlamoExl3ProjectionGroup& group,
    const bf16* a, const bf16* b, bf16* residual, const bf16* weight,
    bf16* output, uint32_t as, uint32_t bs, uint32_t rs, uint32_t os, float eps, hipStream_t stream) {
    boundary_fast<false><<<dim3(group.rows, 40, (group.count + 1u) / 2u), 256, 0, stream>>>(
        group, {}, a, b, residual, weight, output, as, bs, rs, os, eps, false);
    return hipGetLastError();
}

hipError_t launch_boundary_out_rotate_fast(const PlamoExl3ProjectionGroup& group, const BoundaryDeferred& source,
    const bf16* other, bf16* residual, const bf16* weight, bf16* output,
    uint32_t other_stride, uint32_t residual_stride, uint32_t output_stride, float eps, bool source_first, hipStream_t stream) {
    boundary_fast<true><<<dim3(group.rows, 40, (group.count + 1u) / 2u), 256, 0, stream>>>(
        group, source, nullptr, other, residual, weight, output, 0, other_stride, residual_stride, output_stride, eps, source_first);
    return hipGetLastError();
}
}
