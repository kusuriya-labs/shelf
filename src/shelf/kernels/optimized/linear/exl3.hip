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

#pragma unroll
    for (uint32_t j = 0u; j < 8u; ++j) {
        const uint32_t position = lane * 8u + j;
        const uint16_t state = exl3_detail::state_from_ring(
            ring, word_count, position, args.ka, args.half);
        const uint16_t bits = exl3_detail::codebook_f16_bits(state);
        uint32_t row = 0;
        uint32_t col = 0;
        exl3_detail::tile_coords(position, &row, &col);
        tile[col * 16u + row] = bits;
    }
    __syncwarp();
}

__global__ __launch_bounds__(kThreads) void gemm_exl3_w16a16_kernel(Exl3GemmArgs args) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_high = lane >> 4;

    const uint32_t n0 = blockIdx.x * kOutBlock;
    const uint32_t r0 = blockIdx.y * kRowTile;
    const uint32_t active_rows =
        args.rows - r0 < kRowTile ? args.rows - r0 : kRowTile;

    __shared__ __align__(16) __half xrot[kRowTile * kOutBlock];
    __shared__ __align__(16) SharedArea area;

    f32x8_t acc = f32x8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const uint16_t* suh = static_cast<const uint16_t*>(args.suh);
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input);

    const uint32_t k_chunks = args.k / 128u;
    for (uint32_t kc = 0u; kc < k_chunks; ++kc) {
        const uint32_t k_base = kc * 128u;
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
            __half* dst = xrot + r * kOutBlock + lane * 4u;
            dst[0] = __float2half(v0 * kInvSqrt128);
            dst[1] = __float2half(v1 * kInvSqrt128);
            dst[2] = __float2half(v2 * kInvSqrt128);
            dst[3] = __float2half(v3 * kInvSqrt128);
        }
        __syncthreads();

        const uint32_t n_tile = blockIdx.x * kWarps + warp;
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
        float v0 = src[0];
        float v1 = src[1];
        float v2 = src[2];
        float v3 = src[3];
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        const uint32_t col = n0 + lane * 4u;
        const float s0 = __half2float(__ushort_as_half(svh[col + 0]));
        const float s1 = __half2float(__ushort_as_half(svh[col + 1]));
        const float s2 = __half2float(__ushort_as_half(svh[col + 2]));
        const float s3 = __half2float(__ushort_as_half(svh[col + 3]));
        const uint64_t out_base =
            static_cast<uint64_t>(row) * args.output_row_stride + col;
        const float y0 = v0 * kInvSqrt128 * s0;
        const float y1 = v1 * kInvSqrt128 * s1;
        const float y2 = v2 * kInvSqrt128 * s2;
        const float y3 = v3 * kInvSqrt128 * s3;
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
    gemm_exl3_w16a16_kernel<<<dim3(grid_x, grid_y), kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

}
