#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul1/decode.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul1;

constexpr float kInvSqrt128 = 0.08838834764831844055f;

__global__ __launch_bounds__(256) void exl3_dense_epilogue_kernel(Exl3GemmArgs args,
                                                                  const float* accum,
                                                                  const float* rowsum) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + warp;
    if (row >= args.rows) return;
    const float bias = exl3_detail::mul1::kMul1BiasOverInv * rowsum[row];
    const float scale = kInvSqrt128 * exl3_detail::mul1::kMul1Inv;
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const uint32_t n_blocks = args.out_features / 128u;
    for (uint32_t cb = 0u; cb < n_blocks; ++cb) {
        const float* src =
            accum + static_cast<uint64_t>(row) * args.out_features + cb * 128u + lane * 4u;
        float v0 = src[0] + bias;
        float v1 = src[1] + bias;
        float v2 = src[2] + bias;
        float v3 = src[3] + bias;
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        const uint32_t col = cb * 128u + lane * 4u;
        const float s0 = __half2float(__ushort_as_half(svh[col + 0]));
        const float s1 = __half2float(__ushort_as_half(svh[col + 1]));
        const float s2 = __half2float(__ushort_as_half(svh[col + 2]));
        const float s3 = __half2float(__ushort_as_half(svh[col + 3]));
        const uint64_t out_base =
            static_cast<uint64_t>(row) * args.output_row_stride + col;
        const float y0 = v0 * scale * s0;
        const float y1 = v1 * scale * s1;
        const float y2 = v2 * scale * s2;
        const float y3 = v3 * scale * s3;
        if (args.output_dtype == 0u) {
            shelf::bf16_t* dst = static_cast<shelf::bf16_t*>(args.output) + out_base;
            dst[0] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y0))};
            dst[1] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y1))};
            dst[2] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y2))};
            dst[3] = shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(y3))};
        } else {
            float* dst = static_cast<float*>(args.output) + out_base;
            dst[0] = y0;
            dst[1] = y1;
            dst[2] = y2;
            dst[3] = y3;
        }
    }
}

}  // namespace

hipError_t launch_epilogue_exl3_dense(const Exl3GemmArgs& args, const void* rowsum,
                                      const void* accum, hipStream_t stream) {
    if (rowsum == nullptr || accum == nullptr || args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if ((args.out_features % 128u) != 0u) return hipErrorInvalidValue;
    const uint32_t blocks = (args.rows + 7u) / 8u;
    exl3_dense_epilogue_kernel<<<blocks, 256, 0, stream>>>(
        args, static_cast<const float*>(accum), static_cast<const float*>(rowsum));
    return hipGetLastError();
}

}  // namespace shelf::kernel
