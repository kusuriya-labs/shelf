#include <shelf/kernels/fused/exl3/row_prepare.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {
namespace {
using namespace exl3_detail::mul24;

// One block per row: the row peak / squared sum decide whether a single row scale is good
// enough, flagged rows get a compact slot with an atomic append, the rest are re-rounded in
// place to the peak. This replaces the former rowstat + rowslot + rowfinish launch chain.
__global__ void __launch_bounds__(256) exl3_mul24_rowprepare_kernel(
    int8_t* __restrict__ codes,
    const float* __restrict__ scales,
    float* __restrict__ rowscale,
    int32_t* __restrict__ slots,
    uint32_t* __restrict__ count,
    int8_t* __restrict__ codes_f,
    float* __restrict__ scales_f,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride_floats,
    const uint32_t groups,
    const uint32_t k,
    const uint32_t cap,
    const float share) {
    __shared__ float red[2][8];
    __shared__ float s_peak;
    __shared__ int32_t s_slot;
    const uint32_t m = blockIdx.x;
    const uint32_t t = threadIdx.x;
    float peak = 0.0f, sq = 0.0f;
    for (uint32_t g = t; g < groups; g += 256u) {
        const float sg = scales[static_cast<size_t>(m) * scale_row_stride_floats + g];
        peak = fmaxf(peak, sg);
        sq += sg * sg;
    }
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        peak = fmaxf(peak, __shfl_xor(peak, off));
        sq += __shfl_xor(sq, off);
    }
    if ((t & 31u) == 0u) { red[0][t >> 5u] = peak; red[1][t >> 5u] = sq; }
    __syncthreads();
    if (t == 0u) {
        float pk = red[0][0], sm = red[1][0];
        for (uint32_t i = 1u; i < 8u; ++i) { pk = fmaxf(pk, red[0][i]); sm += red[1][i]; }
        rowscale[m] = pk;
        s_peak = pk;
        int32_t slot = -1;
        if (pk > 0.0f && pk * pk > share * sm) {
            const uint32_t at = atomicAdd(count, 1u);
            if (at < cap) {
                slot = static_cast<int32_t>(at);
            } else {
                atomicSub(count, 1u);
            }
        }
        slots[m] = slot;
        s_slot = slot;
    }
    __syncthreads();
    const int32_t slot = s_slot;
    const float pk = s_peak;
    int8_t* row = codes + static_cast<size_t>(m) * code_row_stride;
    const float* srow = scales + static_cast<size_t>(m) * scale_row_stride_floats;
    const uint32_t vecs = k / 16u;
    if (slot >= 0) {
        int8_t* dst = codes_f + static_cast<size_t>(slot) * code_row_stride;
        const uint4* src4 = reinterpret_cast<const uint4*>(row);
        uint4* dst4 = reinterpret_cast<uint4*>(dst);
        for (uint32_t i = t; i < vecs; i += 256u) dst4[i] = src4[i];
        float* sdst = scales_f + static_cast<size_t>(slot) * scale_row_stride_floats;
        for (uint32_t g = t; g < groups; g += 256u) sdst[g] = srow[g];
        return;
    }
    if (pk <= 0.0f) return;
    uint4* row4 = reinterpret_cast<uint4*>(row);
    for (uint32_t i = t; i < vecs; i += 256u) {
        const float ratio = srow[i / 8u] / pk;
        if (ratio == 1.0f) continue;
        uint4 v = row4[i];
        uint32_t* w = reinterpret_cast<uint32_t*>(&v);
        for (uint32_t j = 0u; j < 4u; ++j) {
            uint32_t x = w[j];
            for (uint32_t b = 0u; b < 4u; ++b) {
                const uint32_t shift = b * 8u;
                const int8_t c = static_cast<int8_t>((x >> shift) & 0xFFu);
                const int8_t q = static_cast<int8_t>(static_cast<int>(rintf(static_cast<float>(c) * ratio)));
                x = (x & ~(0xFFu << shift)) | (static_cast<uint32_t>(static_cast<uint8_t>(q)) << shift);
            }
            w[j] = x;
        }
        row4[i] = v;
    }
}

}  // namespace

hipError_t launch_exl3_mul24_rowscale_fused(
    int8_t* codes, const float* scales, float* rowscale, int32_t* slots, uint32_t* count,
    int8_t* codes_f, float* scales_f, uint32_t rows, uint32_t cap, uint32_t k,
    uint32_t code_row_stride_bytes, uint32_t scale_row_stride_bytes, float share,
    hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (count == nullptr || codes == nullptr || scales == nullptr || rowscale == nullptr ||
        slots == nullptr || codes_f == nullptr || scales_f == nullptr || cap == 0u)
        return hipErrorInvalidValue;
    if (k == 0u || (k % kBlock) != 0u || (scale_row_stride_bytes % 4u) != 0u ||
        (code_row_stride_bytes % 16u) != 0u)
        return hipErrorInvalidValue;
    exl3_mul24_rowprepare_kernel<<<rows, 256, 0, stream>>>(
        codes, scales, rowscale, slots, count, codes_f, scales_f, code_row_stride_bytes,
        scale_row_stride_bytes / 4u, k / kBlock, k, cap, share);
    return hipGetLastError();
}

}  // namespace shelf::kernel
