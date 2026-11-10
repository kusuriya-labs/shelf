#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul1/decode.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {
namespace {
using namespace exl3_detail::mul1;

constexpr float kInvSqrt128 = 0.08838834764831844055f;

struct ReconWork {
    uint16_t tile[8][256];
    uint32_t ring[8][64];
};

__global__ __launch_bounds__(256) void recon_exl3_dense_kernel(Exl3GemmArgs args,
                                                                uint16_t* dense) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    __shared__ __align__(16) ReconWork work;
    const uint32_t n_tiles = args.out_features / 16u;
    const uint32_t k_tiles = args.k / 16u;
    const uint64_t total = static_cast<uint64_t>(k_tiles) * n_tiles;
    uint16_t* tile = work.tile[warp];
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * 8u + warp; idx < total;
         idx += static_cast<uint64_t>(gridDim.x) * 8u) {
        const uint32_t kt = static_cast<uint32_t>(idx / n_tiles);
        const uint32_t nt = static_cast<uint32_t>(idx % n_tiles);
        decode_tile<0u>(args, kt, nt, lane, work.ring[warp], tile);
        const uint32_t row = lane >> 1u;
        const uint32_t half = lane & 1u;
        uint16_t vals[8];
#pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) {
            vals[j] = tile[(half * 8u + j) * 16u + row];
        }
        uint16_t* dst = dense + (static_cast<uint64_t>(kt * 16u + row)) * args.out_features +
                        nt * 16u + half * 8u;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(vals);
    }
}


template <bool kRowSum>
__global__ __launch_bounds__(256) void rotate_exl3_kernel(const shelf::bf16_t* input,
                                                         const uint16_t* suh, __half* xrot,
                                                         uint32_t rows, uint32_t k,
                                                         uint32_t stride, float* rowsum) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t kb_count = k / 128u;
    float row_acc = 0.f;
    for (uint32_t kb = warp; kb < kb_count; kb += 8u) {
        const uint32_t k_base = kb * 128u;
        float suh_v[4];
#pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            suh_v[i] = __half2float(__ushort_as_half(suh[k_base + lane * 4u + i]));
        }
        const shelf::bf16_t* src =
            input + static_cast<uint64_t>(row) * stride + k_base + lane * 4u;
        float v0 = bf16_value(src + 0) * suh_v[0];
        float v1 = bf16_value(src + 1) * suh_v[1];
        float v2 = bf16_value(src + 2) * suh_v[2];
        float v3 = bf16_value(src + 3) * suh_v[3];
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        const __half hv0 = __float2half(v0 * kInvSqrt128);
        const __half hv1 = __float2half(v1 * kInvSqrt128);
        const __half hv2 = __float2half(v2 * kInvSqrt128);
        const __half hv3 = __float2half(v3 * kInvSqrt128);
        __half* dst = xrot + static_cast<uint64_t>(row) * k + k_base + lane * 4u;
        dst[0] = hv0;
        dst[1] = hv1;
        dst[2] = hv2;
        dst[3] = hv3;
        if constexpr (kRowSum) {
            row_acc += __half2float(hv0) + __half2float(hv1) + __half2float(hv2) +
                       __half2float(hv3);
        }
    }
    if constexpr (kRowSum) {
        row_acc += __shfl_down(row_acc, 16, 32);
        row_acc += __shfl_down(row_acc, 8, 32);
        row_acc += __shfl_down(row_acc, 4, 32);
        row_acc += __shfl_down(row_acc, 2, 32);
        row_acc += __shfl_down(row_acc, 1, 32);
        __shared__ float warp_sums[8];
        if (lane == 0u) warp_sums[warp] = row_acc;
        __syncthreads();
        if (tid == 0u) {
            float total = 0.f;
#pragma unroll
            for (uint32_t i = 0u; i < 8u; ++i) total += warp_sums[i];
            rowsum[row] = total;
        }
    }
}

}  // namespace

hipError_t launch_rotate_exl3_rowsum(const Exl3GemmArgs& args, void* xrot, float* rowsum,
                                     hipStream_t stream) {
    rotate_exl3_kernel<true><<<dim3(args.rows), 256, 0, stream>>>(
        static_cast<const shelf::bf16_t*>(args.input),
        static_cast<const uint16_t*>(args.suh), static_cast<__half*>(xrot), args.rows,
        args.k, args.input_row_stride, rowsum);
    return hipGetLastError();
}
hipError_t launch_recon_exl3_dense(const Exl3GemmArgs& args, void* dense, hipStream_t stream) {
    if (args.trellis == nullptr || dense == nullptr) return hipErrorInvalidValue;
    if ((args.k % 16u) != 0u || (args.out_features % 16u) != 0u) {
        return hipErrorInvalidValue;
    }
    const uint32_t n_tiles = args.out_features / 16u;
    const uint32_t k_tiles = args.k / 16u;
    const uint64_t total = static_cast<uint64_t>(k_tiles) * n_tiles;
    const uint32_t blocks = static_cast<uint32_t>((total + 7u) / 8u);
    recon_exl3_dense_kernel<<<blocks, 256, 0, stream>>>(
        args, static_cast<uint16_t*>(dense));
    return hipGetLastError();
}
hipError_t launch_rotate_exl3(const Exl3GemmArgs& args, void* xrot, hipStream_t stream) {
    rotate_exl3_kernel<false><<<dim3(args.rows), 256, 0, stream>>>(
        static_cast<const shelf::bf16_t*>(args.input),
        static_cast<const uint16_t*>(args.suh), static_cast<__half*>(xrot), args.rows,
        args.k, args.input_row_stride, nullptr);
    return hipGetLastError();
}

}  // namespace shelf::kernel
