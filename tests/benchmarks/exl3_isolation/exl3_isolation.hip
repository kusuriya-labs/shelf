#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/kernels/optimized/linear/exl3/detail/decode_mul1.h>
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

__device__ __forceinline__ uint32_t pick4(uint32_t r, uint32_t a, uint32_t b, uint32_t c,
                                          uint32_t d) {
    uint32_t v = a;
    v = r == 1u ? b : v;
    v = r == 2u ? c : v;
    v = r == 3u ? d : v;
    return v;
}

__device__ __forceinline__ void decode_tile_ka4(const Exl3GemmArgs& args, uint32_t kt,
                                               uint32_t nt, uint32_t lane, uint32_t* ring,
                                               uint16_t* tile) {
    (void) ring;
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * 64u;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);
    const uint32_t b = source[lane];
    const uint32_t a = __shfl_sync(0xFFFFFFFFull, b, (lane + 31u) & 31u, 32);

    uint32_t st[8];
    st[0] = __builtin_amdgcn_alignbit(a, b, 28) & 0xFFFFu;
    st[1] = __builtin_amdgcn_alignbit(a, b, 24) & 0xFFFFu;
    st[2] = __builtin_amdgcn_alignbit(a, b, 20) & 0xFFFFu;
    st[3] = __builtin_amdgcn_alignbit(b, b, 16) & 0xFFFFu;
    st[4] = __builtin_amdgcn_alignbit(b, b, 12) & 0xFFFFu;
    st[5] = __builtin_amdgcn_alignbit(b, b, 8) & 0xFFFFu;
    st[6] = __builtin_amdgcn_alignbit(b, b, 4) & 0xFFFFu;
    st[7] = b & 0xFFFFu;

    uint32_t* tile32 = reinterpret_cast<uint32_t*>(tile);
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p) {
        const uint32_t position = lane * 8u + p * 2u;
        const uint32_t packed =
            exl3_detail::codebook_pair_h_bits(st[2u * p], st[2u * p + 1u]);
        uint32_t row = 0;
        uint32_t col = 0;
        exl3_detail::tile_coords(position, &row, &col);
        tile32[col * 8u + (row >> 1)] = packed;
    }
    __syncwarp();
}

__device__ __forceinline__ void decode_tile(const Exl3GemmArgs& args, uint32_t kt, uint32_t nt,
                                            uint32_t lane, uint32_t* ring, uint16_t* tile) {
    if (args.ka == 4u && args.half == 0u) {
        decode_tile_ka4(args, kt, nt, lane, ring, tile);
        return;
    }
    const uint32_t words = 16u * args.ka + (args.half != 0u ? 8u : 0u);
    const uint32_t word_count = words / 2u;
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * words;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);

    for (uint32_t i = lane; i < word_count; i += 32u) ring[i] = source[i];
    __syncwarp();

    const uint32_t bit_length = 256u * args.ka + (args.half != 0u ? 128u : 0u);
    const uint32_t p0 = lane * 8u;
    const uint32_t step0 = p0 + 1u;
    const uint32_t end0 =
        step0 * args.ka + (args.half != 0u ? step0 / 2u : 0u) + bit_length;
    const uint32_t w0 = (end0 - 16u) >> 5;
    uint32_t i0 = w0;
    uint32_t i1 = w0 + 1u;
    uint32_t i2 = w0 + 2u;
    uint32_t i3 = w0 + 3u;
    if (i0 >= word_count) i0 -= word_count;
    if (i1 >= word_count) i1 -= word_count;
    if (i2 >= word_count) i2 -= word_count;
    if (i3 >= word_count) i3 -= word_count;
    if (i0 >= word_count) i0 -= word_count;
    if (i1 >= word_count) i1 -= word_count;
    if (i2 >= word_count) i2 -= word_count;
    if (i3 >= word_count) i3 -= word_count;
    const uint32_t v0 = ring[i0];
    const uint32_t v1 = ring[i1];
    const uint32_t v2 = ring[i2];
    const uint32_t v3 = ring[i3];

    uint32_t* tile32 = reinterpret_cast<uint32_t*>(tile);
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p) {
        const uint32_t position = p0 + p * 2u;
        const uint32_t step = position + 1u;
        const uint32_t end =
            step * args.ka + (args.half != 0u ? step / 2u : 0u) + bit_length;
        const uint32_t shift = (((end - 1u) >> 5) + 1u) * 32u - end;
        const uint32_t lo = ((end - 16u) >> 5) - w0;
        const uint32_t hi = ((end - 1u) >> 5) - w0;
        const uint32_t state0 = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo, v0, v1, v2, v3)) << 32) |
             pick4(hi, v0, v1, v2, v3)) >>
            shift);
        const uint32_t step1 = position + 2u;
        const uint32_t end1 =
            step1 * args.ka + (args.half != 0u ? step1 / 2u : 0u) + bit_length;
        const uint32_t shift1 = (((end1 - 1u) >> 5) + 1u) * 32u - end1;
        const uint32_t lo1 = ((end1 - 16u) >> 5) - w0;
        const uint32_t hi1 = ((end1 - 1u) >> 5) - w0;
        const uint32_t state1 = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo1, v0, v1, v2, v3)) << 32) |
             pick4(hi1, v0, v1, v2, v3)) >>
            shift1);
        const uint32_t packed = exl3_detail::codebook_pair_h_bits(state0, state1);
        uint32_t row = 0;
        uint32_t col = 0;
        exl3_detail::tile_coords(position, &row, &col);
        tile32[col * 8u + (row >> 1)] = packed;
    }
    __syncwarp();
}

__global__ __launch_bounds__(256) void rotate_exl3_kernel(const shelf::bf16_t* input,
                                                         const uint16_t* suh, __half* xrot,
                                                         uint32_t rows, uint32_t k,
                                                         uint32_t stride) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    const uint32_t lane = tid & 31u;
    const uint32_t kb_count = k / 128u;
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
        __half* dst = xrot + static_cast<uint64_t>(row) * k + k_base + lane * 4u;
        dst[0] = __float2half(v0 * kInvSqrt128);
        dst[1] = __float2half(v1 * kInvSqrt128);
        dst[2] = __float2half(v2 * kInvSqrt128);
        dst[3] = __float2half(v3 * kInvSqrt128);
    }
}

__global__ __launch_bounds__(kThreads) void recon_tiles_kernel(Exl3GemmArgs args,
                                                            uint16_t* dense) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t n_tiles = args.out_features / 16u;
    const uint32_t n_tile = blockIdx.x * kWarps + warp;
    __shared__ __align__(16) uint32_t ring[kWarps][64];
    __shared__ __align__(16) uint16_t tile[kWarps][16u * 16u];
    const uint32_t kt_count = args.k / 16u;
    for (uint32_t kt = 0u; kt < kt_count; ++kt) {
        decode_tile(args, kt, n_tile, lane, ring[warp], tile[warp]);
        const uint16_t* src = tile[warp];
        uint16_t* dst = dense + (static_cast<uint64_t>(kt) * n_tiles + n_tile) * 256u;
#pragma unroll
        for (uint32_t i = 0u; i < 8u; ++i) dst[lane * 8u + i] = src[lane * 8u + i];
    }
}

template <uint32_t kGroups, bool kPartial, bool kPre, bool kDense>
__global__ __launch_bounds__(kThreads) void gemm_exl3_w16a16_kernel(Exl3GemmArgs args,
                                                                    uint32_t splits,
                                                                    const uint16_t* dense) {
    constexpr uint32_t kRows = kRowTile * kGroups;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_high = lane >> 4;

    const uint32_t n_tile_base = kPartial ? blockIdx.x / splits : blockIdx.x;
    const uint32_t split = kPartial ? blockIdx.x % splits : 0u;
    const uint32_t n0 = n_tile_base * kOutBlock;
    const uint32_t r0 = blockIdx.y * kRows;
    const uint32_t active_rows = args.rows - r0 < kRows ? args.rows - r0 : kRows;
    const __half* xrot_global = static_cast<const __half*>(args.xrot);

    __shared__ __align__(16) __half xrot[kRows * kOutBlock];
    __shared__ __align__(16) SharedArea area;
    __shared__ __align__(16) float rowsum[kRows];

    f32x8_t acc[kGroups];
#pragma unroll
    for (uint32_t g = 0u; g < kGroups; ++g) {
        acc[g] = f32x8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    }
    float row_acc[kGroups][2];
#pragma unroll
    for (uint32_t g = 0u; g < kGroups; ++g) {
        row_acc[g][0] = 0.f;
        row_acc[g][1] = 0.f;
    }

    const uint16_t* suh = static_cast<const uint16_t*>(args.suh);
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input);

    const uint32_t k_chunks = args.k / 128u;
    for (uint32_t kc = split; kc < k_chunks; kc += (kPartial ? splits : 1u)) {
        const uint32_t k_base = kc * 128u;
        __syncthreads();
        if constexpr (kPre) {
            const uint32_t copy_rows = active_rows < kRows ? active_rows : kRows;
            for (uint32_t i = threadIdx.x; i < copy_rows * kOutBlock; i += kThreads) {
                const uint32_t r = i / kOutBlock;
                const uint32_t c = i % kOutBlock;
                xrot[i] = xrot_global[static_cast<uint64_t>(r0 + r) * args.k + k_base + c];
            }
        } else {
            float suh_v[4];
#pragma unroll
            for (uint32_t i = 0u; i < 4u; ++i) {
                suh_v[i] = __half2float(__ushort_as_half(suh[k_base + lane * 4u + i]));
            }

#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
                for (uint32_t ri = 0u; ri < 2u; ++ri) {
                    const uint32_t r = g * kRowTile + warp * 2u + ri;
                    if (r >= active_rows) continue;
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
                    row_acc[g][ri] += __half2float(hv0) + __half2float(hv1) +
                                      __half2float(hv2) + __half2float(hv3);
                    __half* dst = xrot + r * kOutBlock + lane * 4u;
                    dst[0] = hv0;
                    dst[1] = hv1;
                    dst[2] = hv2;
                    dst[3] = hv3;
                }
            }
        }
        __syncthreads();
        if constexpr (kPre) {
#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
                for (uint32_t ri = 0u; ri < 2u; ++ri) {
                    const uint32_t r = g * kRowTile + warp * 2u + ri;
                    if (r >= active_rows) continue;
                    const __half* p = xrot + r * kOutBlock + lane * 4u;
                    row_acc[g][ri] += __half2float(p[0]) + __half2float(p[1]) +
                                      __half2float(p[2]) + __half2float(p[3]);
                }
            }
        }

        const uint32_t n_tile = n_tile_base * kWarps + warp;
#pragma unroll
        for (uint32_t ks = 0u; ks < 8u; ++ks) {
            const uint32_t buf = ks & 1u;
            f16x8_t a;
            if constexpr (kDense) {
                const uint32_t n_tiles = args.out_features / 16u;
                const uint16_t* dsrc = dense +
                    (static_cast<uint64_t>(kc * 8u + ks) * n_tiles + n_tile) * 256u;
                a = load_f16x8(dsrc + lane16 * 16u + lane_high * 8u);
            } else {
                uint16_t* tile = area.work.tile[warp][buf];
                decode_tile(args, kc * 8u + ks, n_tile, lane, area.work.ring[warp], tile);
                a = load_f16x8(tile + lane16 * 16u + lane_high * 8u);
            }
#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
                if constexpr (kGroups > 1u) {
                    if (g * kRowTile >= active_rows) continue;
                }
                const f16x8_t b = load_f16x8(xrot +
                                             (g * kRowTile + lane16) * kOutBlock +
                                             ks * 16u + lane_high * 8u);
                acc[g] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[g]);
            }
        }
    }

    if constexpr (kPartial) {
        const uint64_t partial_plane =
            static_cast<uint64_t>(args.rows) * args.out_features;
        float* rowsum_base =
            args.splitk_scratch + static_cast<uint64_t>(splits) * partial_plane;
#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                float sum = row_acc[g][ri];
                sum += __shfl_down(sum, 16, 32);
                sum += __shfl_down(sum, 8, 32);
                sum += __shfl_down(sum, 4, 32);
                sum += __shfl_down(sum, 2, 32);
                sum += __shfl_down(sum, 1, 32);
                const uint32_t r = g * kRowTile + warp * 2u + ri;
                if (lane == 0u && r < active_rows) {
                    rowsum_base[static_cast<uint64_t>(split) * args.rows + (r0 + r)] = sum;
                }
            }
        }
#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
            const uint32_t r = g * kRowTile + lane16;
            if (r < active_rows) {
                const uint64_t row_index =
                    static_cast<uint64_t>(split) * args.rows + (r0 + r);
                const uint32_t out_col = n0 + warp * 16u + lane_high * 8u;
                float* dst = args.splitk_scratch + row_index * args.out_features + out_col;
#pragma unroll
                for (uint32_t j = 0u; j < 8u; ++j) dst[j] = acc[g][j];
            }
        }
    } else {
#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                float sum = row_acc[g][ri];
                sum += __shfl_down(sum, 16, 32);
                sum += __shfl_down(sum, 8, 32);
                sum += __shfl_down(sum, 4, 32);
                sum += __shfl_down(sum, 2, 32);
                sum += __shfl_down(sum, 1, 32);
                if (lane == 0u && g * kRowTile + warp * 2u + ri < active_rows) {
                    rowsum[g * kRowTile + warp * 2u + ri] = sum;
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
            if constexpr (kGroups > 1u) {
                if (g * kRowTile >= active_rows) continue;
            }
#pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j) {
                const uint32_t out_col = warp * 16u + lane_high * 8u + j;
                area.accum[lane16 * kOutBlock + out_col] = acc[g][j];
            }
            __syncthreads();

#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                const uint32_t r = g * kRowTile + warp * 2u + ri;
                if (r >= active_rows) continue;
                const uint32_t row = r0 + r;
                const float* src =
                    area.accum + (r - g * kRowTile) * kOutBlock + lane * 4u;
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
            if constexpr (kGroups > 1u) {
                __syncthreads();
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

struct LaunchPlan {
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t splits = 1;
    uint32_t groups = 1;
};

LaunchPlan make_launch_plan(const Exl3GemmArgs& args) {
    LaunchPlan plan;
    plan.grid_x = args.out_features / kOutBlock;
    if (args.splitk_scratch != nullptr) {
        uint32_t desired = args.splitk_splits;
        if (desired == 0u) {
            desired = plan.grid_x <= 16u ? 8u : (plan.grid_x <= 160u ? 4u : 1u);
        }
        const uint32_t k_chunks = args.k / 128u;
        if (desired > k_chunks) desired = k_chunks;
        const uint64_t per_split =
            static_cast<uint64_t>(args.rows) * args.out_features + args.rows;
        while (desired > 1u &&
               static_cast<uint64_t>(desired) * per_split > args.splitk_floats) {
            --desired;
        }
        plan.splits = desired;
    }
    const uint32_t row_blocks2 = (args.rows + 2u * kRowTile - 1u) / (2u * kRowTile);
    if (args.groups == 1u) {
        plan.groups = 1u;
    } else if (args.groups == 2u) {
        plan.groups = 2u;
    } else if (args.rows >= 2u * kRowTile &&
               plan.grid_x * row_blocks2 * plan.splits >= 128u) {
        plan.groups = 2u;
    }
    const uint32_t rows_per_block = kRowTile * plan.groups;
    plan.grid_y = (args.rows + rows_per_block - 1u) / rows_per_block;
    return plan;
}

template <uint32_t kGroups, bool kPre, bool kDense>
hipError_t launch_kernel(const Exl3GemmArgs& args, const LaunchPlan& plan,
                         hipStream_t stream, const uint16_t* dense) {
    if (plan.splits > 1u) {
        gemm_exl3_w16a16_kernel<kGroups, true, kPre, kDense>
            <<<dim3(plan.grid_x * plan.splits, plan.grid_y), kThreads, 0, stream>>>(
                args, plan.splits, dense);
    } else {
        gemm_exl3_w16a16_kernel<kGroups, false, kPre, kDense>
            <<<dim3(plan.grid_x, plan.grid_y), kThreads, 0, stream>>>(args, 1u, dense);
    }
    return hipGetLastError();
}

template <bool kPre, bool kDense = false>
hipError_t launch_plan(const Exl3GemmArgs& args, const LaunchPlan& plan,
                       hipStream_t stream, const uint16_t* dense = nullptr) {
    const hipError_t err =
        plan.groups == 2u ? launch_kernel<2u, kPre, kDense>(args, plan, stream, dense)
                          : launch_kernel<1u, kPre, kDense>(args, plan, stream, dense);
    if (err != hipSuccess) return err;
    if (plan.splits > 1u) {
        const uint32_t reduce_grid_y = (args.rows + kRowTile - 1u) / kRowTile;
        exl3_splitk_reduce_kernel<<<dim3(plan.grid_x, reduce_grid_y), kThreads, 0,
                                    stream>>>(args, plan.splits);
    }
    return hipGetLastError();
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
    return launch_plan<false, false>(args, make_launch_plan(args), stream);
}

hipError_t launch_rotate_exl3(const Exl3GemmArgs& args, void* xrot, hipStream_t stream) {
    rotate_exl3_kernel<<<dim3(args.rows), 256, 0, stream>>>(
        static_cast<const shelf::bf16_t*>(args.input),
        static_cast<const uint16_t*>(args.suh), static_cast<__half*>(xrot), args.rows,
        args.k, args.input_row_stride);
    return hipGetLastError();
}

hipError_t launch_gemm_exl3_w16a16_pre(const Exl3GemmArgs& args, const void* xrot,
                                       hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.out_features == 0u || args.k == 0u) return hipErrorInvalidValue;
    if ((args.out_features % kOutBlock) != 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    Exl3GemmArgs pre_args = args;
    pre_args.xrot = const_cast<void*>(xrot);
    return launch_plan<true, false>(pre_args, make_launch_plan(pre_args), stream);
}

hipError_t launch_gemm_exl3_w16a16_dense_pre(const Exl3GemmArgs& args, const void* xrot,
                                             const void* dense, hipStream_t stream) {
    Exl3GemmArgs pre_args = args;
    pre_args.xrot = const_cast<void*>(xrot);
    return launch_plan<true, true>(pre_args, make_launch_plan(pre_args), stream,
                                   static_cast<const uint16_t*>(dense));
}

hipError_t launch_recon_tiles(const Exl3GemmArgs& args, void* dense, hipStream_t stream) {
    const uint32_t grid_x = args.out_features / kOutBlock;
    recon_tiles_kernel<<<dim3(grid_x), kThreads, 0, stream>>>(
        args, static_cast<uint16_t*>(dense));
    return hipGetLastError();
}

}
