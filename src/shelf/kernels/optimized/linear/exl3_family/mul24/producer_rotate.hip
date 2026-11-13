#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_bf16.h>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;

__global__ void __launch_bounds__(128) swiglu_rotate(PlamoExl3ProjectionGroup args,
                                                    const bf16_t* gate, const bf16_t* up,
                                                    bf16_t* product, bool intermediate) {
    const uint32_t t = threadIdx.x;
    const uint32_t g = blockIdx.x;
    const uint32_t col = g * 128u + t;
    const float a = bf16_to_f32(gate[col]);
    const float b = bf16_to_f32(up[col]);
    float activated = a / (1.0f + __expf(-a));
    if (intermediate)
        activated = bf16_bits_to_f32(f32_to_bf16_rne_device(activated));
    const auto vbf = bf16_t{f32_to_bf16_rne_device(activated * b)};
    product[col] = vbf;
    float v = bf16_to_f32(vbf) * f16_bits_to_f32(args.p[0].suh[col]);
    __shared__ float smem[128];
    __shared__ float peaks[4];
    v = fwht128_smem(v, smem, t) * kInvSqrt128;
    float peak = fabsf(v);
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        peak = fmaxf(peak, __shfl_xor(peak, off));
    if ((t & 31u) == 0u) peaks[t >> 5u] = peak;
    __syncthreads();
    peak = fmaxf(fmaxf(peaks[0], peaks[1]), fmaxf(peaks[2], peaks[3]));
    const float scale = peak / 127.0f;
    const float q = scale > 0.0f ? fminf(fmaxf(rintf(v / scale), -128.0f), 127.0f) : 0.0f;
    args.p[0].codes[col] = static_cast<int8_t>(static_cast<int>(q));
    if (t == 0u) args.p[0].scales[g] = scale;
}
}

hipError_t launch_plamo_exl3_swiglu_rotate(const PlamoExl3ProjectionGroup& args,
                                            const bf16_t* gate, const bf16_t* up,
                                            bf16_t* product, bool intermediate, hipStream_t stream) {
    if (args.rows != 1u || args.count != 1u || !gate || !up || !product ||
        !args.p[0].suh || !args.p[0].codes || !args.p[0].scales || args.k % 128u)
        return hipErrorInvalidValue;
    swiglu_rotate<<<args.k / 128u, 128, 0, stream>>>(args, gate, up, product, intermediate);
    return hipGetLastError();
}
}
