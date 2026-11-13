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

template<bool Serial>
__global__ void boundary_out_rotate(PlamoExl3ProjectionGroup group, BoundaryDeferred source,
    const bf16* other, bf16* residual, const bf16* weight, bf16* output,
    uint32_t other_stride, uint32_t residual_stride, uint32_t output_stride, float eps, bool source_first) {
    __shared__ bf16 x[5120];
    __shared__ bf16 y[5120];
    __shared__ float smem[256];
    __shared__ float peaks[8];
    const uint32_t row = blockIdx.x, tid = threadIdx.x, lane = tid & 31u, wave = tid >> 5u;
    for (uint32_t block = wave; block < 40; block += 8) {
        const uint32_t col = block * 128u + lane * 4u;
        float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
        for (uint32_t z = 0; z < source.splits; ++z) {
            const float4 v = *reinterpret_cast<const float4*>(source.yprime + (static_cast<size_t>(z) * source.rows + row) * 5120u + col);
            v0 += v.x; v1 += v.y; v2 += v.z; v3 += v.w;
        }
        fwht128(v0, v1, v2, v3, lane);
        const float v[4] = {v0, v1, v2, v3};
        for (uint32_t j = 0; j < 4; ++j) {
            const float transformed = (v[j] * kInvSqrt128) * f16_bits_to_f32(source.svh[col + j]);
            const float rounded = __bfloat162float(__float2bfloat16(transformed));
            const float skip = __bfloat162float(other[static_cast<size_t>(row) * other_stride + col + j]);
            x[col + j] = __float2bfloat16(source_first ? rounded + skip : skip + rounded);
        }
    }
    __syncthreads();
    ::shelf::kernel::detail::rmsnorm_row_body<
        ::shelf::kernel::RmsNormDataType::BF16, ::shelf::kernel::RmsNormDataType::BF16,
        ::shelf::kernel::RmsNormDataType::BF16, ::shelf::kernel::RmsNormWeightLayout::PER_FEATURE,
        ::shelf::kernel::RmsNormWeightMode::ONE_PLUS>(x, weight, y, 0, 0, 5120, 5120, 5120, eps);
    __syncthreads();
    if (Serial || (blockIdx.y == 0 && blockIdx.z == 0)) {
        for (uint32_t i = tid; i < 5120; i += 256) {
            residual[static_cast<size_t>(row) * residual_stride + i] = x[i];
            output[static_cast<size_t>(row) * output_stride + i] = y[i];
        }
    }
    const uint32_t count = Serial ? 40u * group.count : 1u;
    for (uint32_t index = 0; index < count; ++index) {
        const uint32_t g = Serial ? index % 40u : blockIdx.y;
        const uint32_t projection = Serial ? index / 40u : blockIdx.z;
        const uint32_t t = tid & 127u, col = g * 128u + t;
        float v = __bfloat162float(y[col]) * f16_bits_to_f32(group.p[projection].suh[col]);
        v = fwht128_smem(v, smem + (tid / 128u) * 128u, t) * kInvSqrt128;
        float peak = fabsf(v);
        for (uint32_t off = 16u; off; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off));
        if ((tid & 31u) == 0u) peaks[tid >> 5u] = peak;
        __syncthreads();
        const uint32_t base = (tid / 128u) * 4u;
        peak = fmaxf(fmaxf(peaks[base], peaks[base + 1]), fmaxf(peaks[base + 2], peaks[base + 3]));
        const float scale = peak / 127.f;
        const float q = scale > 0.f ? fminf(fmaxf(rintf(v / scale), -128.f), 127.f) : 0.f;
        if (tid < 128) {
            group.p[projection].codes[static_cast<size_t>(row) * 5120u + col] = static_cast<int8_t>(static_cast<int>(q));
            if (tid == 0) group.p[projection].scales[static_cast<size_t>(row) * (group.scale_stride_bytes / 4u) + g] = scale;
        }
        __syncthreads();
    }
}
}

hipError_t launch_boundary_out_rotate(const PlamoExl3ProjectionGroup& group, const BoundaryDeferred& source,
    const bf16* other, bf16* residual, const bf16* weight, bf16* output,
    uint32_t other_stride, uint32_t residual_stride, uint32_t output_stride, float eps, bool source_first, bool serial, hipStream_t stream) {
    if (serial) boundary_out_rotate<true><<<group.rows, 256, 0, stream>>>(group, source, other, residual, weight, output, other_stride, residual_stride, output_stride, eps, source_first);
    else boundary_out_rotate<false><<<dim3(group.rows, 40, group.count), 256, 0, stream>>>(group, source, other, residual, weight, output, other_stride, residual_stride, output_stride, eps, source_first);
    return hipGetLastError();
}
}
