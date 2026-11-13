#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include "../../kernels/optimized/detail/rmsnorm_device.h"
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

namespace shelf::execution {
namespace {
using bf16 = __hip_bfloat16;
using namespace ::shelf::kernel::exl3_detail;
using namespace ::shelf::kernel::exl3_detail::mul24;
using ::shelf::kernel::PlamoExl3ProjectionGroup;

__global__ void residual_norm_rotate(PlamoExl3ProjectionGroup group,
    const bf16* a, const bf16* b, bf16* residual, const bf16* weight,
    bf16* output, uint32_t as, uint32_t bs, uint32_t rs, uint32_t os, float eps) {
    __shared__ bf16 x[5120];
    __shared__ bf16 y[5120];
    __shared__ float smem[256];
    __shared__ float peaks[8];
    const uint32_t row = blockIdx.x, g = blockIdx.y, projection = blockIdx.z;
    const uint32_t tid = threadIdx.x, t = tid & 127u;
    for (uint32_t i = tid; i < 5120; i += 256)
        x[i] = __float2bfloat16(__bfloat162float(a[row * as + i]) +
                                __bfloat162float(b[row * bs + i]));
    __syncthreads();
    ::shelf::kernel::detail::rmsnorm_row_body<
        ::shelf::kernel::RmsNormDataType::BF16, ::shelf::kernel::RmsNormDataType::BF16,
        ::shelf::kernel::RmsNormDataType::BF16, ::shelf::kernel::RmsNormWeightLayout::PER_FEATURE,
        ::shelf::kernel::RmsNormWeightMode::ONE_PLUS>(x, weight, y, 0, 0, 5120, 5120, 5120, eps);
    __syncthreads();
    if (g == 0 && projection == 0) {
        for (uint32_t i = tid; i < 5120; i += 256) {
            residual[row * rs + i] = x[i];
            output[row * os + i] = y[i];
        }
    }
    const uint32_t col = g * 128u + t;
    float v = __bfloat162float(y[col]) * f16_bits_to_f32(group.p[projection].suh[col]);
    v = fwht128_smem(v, smem + (tid / 128u) * 128u, t) * kInvSqrt128;
    float peak = fabsf(v);
    for (uint32_t off = 16u; off; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off));
    if ((tid & 31u) == 0u) peaks[tid >> 5u] = peak;
    __syncthreads();
    const uint32_t base = (tid / 128u) * 4u;
    peak = fmaxf(fmaxf(peaks[base], peaks[base + 1]), fmaxf(peaks[base + 2], peaks[base + 3]));
    const float scale = peak / 127.0f;
    const float q = scale > 0.0f ? fminf(fmaxf(rintf(v / scale), -128.0f), 127.0f) : 0.0f;
    if (tid < 128) {
        group.p[projection].codes[static_cast<size_t>(row) * group.k + col] = static_cast<int8_t>(static_cast<int>(q));
        if (tid == 0) group.p[projection].scales[static_cast<size_t>(row) * (group.scale_stride_bytes / 4u) + g] = scale;
    }
}
}

hipError_t launch_boundary_rotate(const PlamoExl3ProjectionGroup& group,
    const bf16* a, const bf16* b, bf16* residual, const bf16* weight,
    bf16* output, uint32_t as, uint32_t bs, uint32_t rs, uint32_t os, float eps, hipStream_t stream) {
    residual_norm_rotate<<<dim3(group.rows, 40, group.count), 256, 0, stream>>>(
        group, a, b, residual, weight, output, as, bs, rs, os, eps);
    return hipGetLastError();
}
}
