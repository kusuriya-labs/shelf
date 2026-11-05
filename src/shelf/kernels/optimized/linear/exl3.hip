#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/kernels/optimized/linear/exl3_decode_device.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {
namespace {

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kOutBlock = kExl3W16A16OutBlock;
constexpr uint32_t kRowTile = kExl3W16A16RowTile;
constexpr uint32_t kWarps = kOutBlock / 16u;
constexpr uint32_t kThreads = kExl3W16A16Threads;
constexpr float kInvSqrt128 = 0.08838834764831844055f;

struct WorkArea {
    uint16_t tile[kWarps][2][16u * 16u];
    uint32_t ring[kWarps][64];
};

union SharedArea {
    WorkArea work;
    float accum[kRowTile * kOutBlock];
};

__device__ __forceinline__ float bf16_value(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<uint32_t>(p->data) << 16u);
}

__device__ __forceinline__ f16x8_t load_f16x8(const void* p) {
    return *reinterpret_cast<const f16x8_t*>(p);
}

__device__ __forceinline__ void decode_tile(const Exl3GemmArgs& args, uint32_t kt, uint32_t nt,
                                            uint32_t lane, uint32_t* ring, uint16_t* tile) {
    const uint32_t words = 16u * args.ka + (args.half != 0u ? 8u : 0u);
    const uint32_t word_count = words / 2u;
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * words;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);

    for (uint32_t i = lane; i < word_count; i += 32u) ring[i] = source[i];
    __syncwarp();

    uint32_t* tile32 = reinterpret_cast<uint32_t*>(tile);
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p) {
        const uint32_t position = lane * 8u + p * 2u;
        const uint16_t state0 = exl3_detail::state_from_ring(
            ring, word_count, position, args.ka, args.half);
        const uint16_t state1 = exl3_detail::state_from_ring(
            ring, word_count, position + 1u, args.ka, args.half);
        const uint32_t packed = exl3_detail::codebook_pair_h_bits(state0, state1);
        uint32_t row = 0;
        uint32_t col = 0;
        exl3_detail::tile_coords(position, &row, &col);
        tile32[col * 8u + (row >> 1)] = packed;
    }
    __syncwarp();
}

template <bool kPartial>
__global__ __launch_bounds__(kThreads) void gemm_exl3_w16a16_kernel(Exl3GemmArgs args,
                                                                    uint32_t splits) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_high = lane >> 4;

    const uint32_t n_tile_base = kPartial ? blockIdx.x / splits : blockIdx.x;
    const uint32_t split = kPartial ? blockIdx.x % splits : 0u;
    const uint32_t n0 = n_tile_base * kOutBlock;
    const uint32_t r0 = blockIdx.y * kRowTile;
    const uint32_t active_rows =
        args.rows - r0 < kRowTile ? args.rows - r0 : kRowTile;

    __shared__ __align__(16) __half xrot[kRowTile * kOutBlock];
    __shared__ __align__(16) SharedArea area;
    __shared__ __align__(16) float rowsum[kRowTile];

    f32x8_t acc = f32x8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float row_acc[2] = {0.f, 0.f};

    const uint16_t* suh = static_cast<const uint16_t*>(args.suh);
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input);

    const uint32_t k_chunks = args.k / 128u;
    for (uint32_t kc = split; kc < k_chunks; kc += (kPartial ? splits : 1u)) {
        const uint32_t k_base = kc * 128u;
        __syncthreads();
        float suh_v[4];
#pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i) {
            suh_v[i] = __half2float(__ushort_as_half(suh[k_base + lane * 4u + i]));
        }

#pragma unroll
        for (uint32_t ri = 0u; ri < 2u; ++ri) {
            const uint32_t r = warp * 2u + ri;
            if (r >= active_rows) break;
            const uint32_t row = r0 + r;
            const shelf::bf16_t* src =
                input + static_cast<uint64_t>(row) * args.input_row_stride + k_base +
                lane * 4u;
            float v0 = bf16_value(src + 0) * suh_v[0];
            float v1 = bf16_value(src + 1) * suh_v[1];
            float v2 = bf16_value(src + 2) * suh_v[2];
            float v3 = bf16_value(src + 3) * suh_v[3];
            exl3_detail::fwht128(v0, v1, v2, v3, lane);
            const __half hv0 = __float2half(v0 * kInvSqrt128);
            const __half hv1 = __float2half(v1 * kInvSqrt128);
            const __half hv2 = __float2half(v2 * kInvSqrt128);
            const __half hv3 = __float2half(v3 * kInvSqrt128);
            row_acc[ri] += __half2float(hv0) + __half2float(hv1) +
                           __half2float(hv2) + __half2float(hv3);
            __half* dst = xrot + r * kOutBlock + lane * 4u;
            dst[0] = hv0;
            dst[1] = hv1;
            dst[2] = hv2;
            dst[3] = hv3;
        }
        __syncthreads();

        const uint32_t n_tile = n_tile_base * kWarps + warp;
#pragma unroll
        for (uint32_t ks = 0u; ks < 8u; ++ks) {
            const uint32_t buf = ks & 1u;
            uint16_t* tile = area.work.tile[warp][buf];
            decode_tile(args, kc * 8u + ks, n_tile, lane, area.work.ring[warp], tile);
            const f16x8_t a = load_f16x8(tile + lane16 * 16u + lane_high * 8u);
            const f16x8_t b =
                load_f16x8(xrot + lane16 * kOutBlock + ks * 16u + lane_high * 8u);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
        }
    }

    if constexpr (kPartial) {
        const uint64_t partial_plane =
            static_cast<uint64_t>(args.rows) * args.out_features;
        float* rowsum_base =
            args.splitk_scratch + static_cast<uint64_t>(splits) * partial_plane;
#pragma unroll
        for (uint32_t ri = 0u; ri < 2u; ++ri) {
            float sum = row_acc[ri];
            sum += __shfl_down(sum, 16, 32);
            sum += __shfl_down(sum, 8, 32);
            sum += __shfl_down(sum, 4, 32);
            sum += __shfl_down(sum, 2, 32);
            sum += __shfl_down(sum, 1, 32);
            const uint32_t r = warp * 2u + ri;
            if (lane == 0u && r < active_rows) {
                rowsum_base[static_cast<uint64_t>(split) * args.rows + (r0 + r)] = sum;
            }
        }
        if (lane16 < active_rows) {
            const uint64_t row_index =
                static_cast<uint64_t>(split) * args.rows + (r0 + lane16);
            const uint32_t out_col = n0 + warp * 16u + lane_high * 8u;
            float* dst = args.splitk_scratch + row_index * args.out_features + out_col;
#pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j) dst[j] = acc[j];
        }
    } else {
#pragma unroll
        for (uint32_t ri = 0u; ri < 2u; ++ri) {
            float sum = row_acc[ri];
            sum += __shfl_down(sum, 16, 32);
            sum += __shfl_down(sum, 8, 32);
            sum += __shfl_down(sum, 4, 32);
            sum += __shfl_down(sum, 2, 32);
            sum += __shfl_down(sum, 1, 32);
            if (lane == 0u) rowsum[warp * 2u + ri] = sum;
        }
        __syncthreads();

#pragma unroll
        for (uint32_t j = 0u; j < 8u; ++j) {
            const uint32_t out_col = warp * 16u + lane_high * 8u + j;
            area.accum[lane16 * kOutBlock + out_col] = acc[j];
        }
        __syncthreads();

#pragma unroll
        for (uint32_t ri = 0u; ri < 2u; ++ri) {
            const uint32_t r = warp * 2u + ri;
            if (r >= active_rows) break;
            const uint32_t row = r0 + r;
            const float* src = area.accum + r * kOutBlock + lane * 4u;
            const float bias = exl3_detail::kMul1BiasOverInv * rowsum[r];
            float v0 = src[0] + bias;
            float v1 = src[1] + bias;
            float v2 = src[2] + bias;
            float v3 = src[3] + bias;
            exl3_detail::fwht128(v0, v1, v2, v3, lane);
            const uint32_t col = n0 + lane * 4u;
            const float s0 = __half2float(__ushort_as_half(svh[col + 0]));
            const float s1 = __half2float(__ushort_as_half(svh[col + 1]));
            const float s2 = __half2float(__ushort_as_half(svh[col + 2]));
            const float s3 = __half2float(__ushort_as_half(svh[col + 3]));
            const uint64_t out_base =
                static_cast<uint64_t>(row) * args.output_row_stride + col;
            const float scale = kInvSqrt128 * exl3_detail::kMul1Inv;
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
}

__global__ __launch_bounds__(kThreads) void exl3_splitk_reduce_kernel(Exl3GemmArgs args,
                                                                     uint32_t splits) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t n0 = blockIdx.x * kOutBlock;
    const uint32_t r0 = blockIdx.y * kRowTile;
    const uint32_t active_rows =
        args.rows - r0 < kRowTile ? args.rows - r0 : kRowTile;
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const uint64_t partial_plane = static_cast<uint64_t>(args.rows) * args.out_features;
    const float* partials = args.splitk_scratch;
    const float* rowsums = partials + static_cast<uint64_t>(splits) * partial_plane;
    const float scale = kInvSqrt128 * exl3_detail::kMul1Inv;

#pragma unroll
    for (uint32_t ri = 0u; ri < 2u; ++ri) {
        const uint32_t r = warp * 2u + ri;
        if (r >= active_rows) break;
        const uint32_t row = r0 + r;
        float row_sum = 0.f;
        for (uint32_t s = 0u; s < splits; ++s) {
            row_sum += rowsums[static_cast<uint64_t>(s) * args.rows + row];
        }
        const float bias = exl3_detail::kMul1BiasOverInv * row_sum;
        const uint32_t col = n0 + lane * 4u;
        float v0 = bias;
        float v1 = bias;
        float v2 = bias;
        float v3 = bias;
        for (uint32_t s = 0u; s < splits; ++s) {
            const float* src = partials +
                               (static_cast<uint64_t>(s) * args.rows + row) *
                                   args.out_features +
                               col;
            v0 += src[0];
            v1 += src[1];
            v2 += src[2];
            v3 += src[3];
        }
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
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

}

hipError_t launch_gemm_exl3_w16a16(const Exl3GemmArgs& args, hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.input == nullptr || args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.out_features == 0u || args.k == 0u) return hipErrorInvalidValue;
    if ((args.out_features % kOutBlock) != 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    const uint32_t grid_x = args.out_features / kOutBlock;
    const uint32_t grid_y = (args.rows + kRowTile - 1u) / kRowTile;
    uint32_t splits = 1u;
    if (args.splitk_scratch != nullptr) {
        uint32_t desired = args.splitk_splits;
        if (desired == 0u) {
            desired = grid_x <= 64u ? 8u : (grid_x <= 160u ? 4u : 1u);
        }
        const uint32_t k_chunks = args.k / 128u;
        if (desired > k_chunks) desired = k_chunks;
        const uint64_t per_split =
            static_cast<uint64_t>(args.rows) * args.out_features + args.rows;
        while (desired > 1u &&
               static_cast<uint64_t>(desired) * per_split > args.splitk_floats) {
            --desired;
        }
        splits = desired;
    }
    if (splits > 1u) {
        gemm_exl3_w16a16_kernel<true>
            <<<dim3(grid_x * splits, grid_y), kThreads, 0, stream>>>(args, splits);
        const hipError_t launch_error = hipGetLastError();
        if (launch_error != hipSuccess) return launch_error;
        exl3_splitk_reduce_kernel<<<dim3(grid_x, grid_y), kThreads, 0, stream>>>(args,
                                                                                  splits);
    } else {
        gemm_exl3_w16a16_kernel<false>
            <<<dim3(grid_x, grid_y), kThreads, 0, stream>>>(args, 1u);
    }
    return hipGetLastError();
}

}
