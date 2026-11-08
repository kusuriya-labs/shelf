#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/kernels/optimized/linear/exl3_decode_device.h>
#include <shelf/kernels/optimized/linear/exl3_group_plan_device.h>
#include <shelf/kernels/optimized/linear/exl3_grouped.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace shelf::kernel {
namespace {

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kOutBlockBase = kExl3W16A16OutBlock;
constexpr uint32_t kRowTile = kExl3W16A16RowTile;
constexpr uint32_t kThreads = kExl3W16A16Threads;
constexpr uint32_t kWarps = kThreads / 32u;
constexpr uint32_t kXrotPitch = 136u;
constexpr float kInvSqrt128 = 0.08838834764831844055f;

template <uint32_t kNTiles>
struct WorkAreaT {
    uint16_t tile[kWarps][2][kNTiles * 16u * 16u];
    uint32_t ring[kWarps][64];
};

template <uint32_t kNTiles>
union SharedAreaT {
    WorkAreaT<kNTiles> work;
    float accum[kRowTile * kOutBlockBase * kNTiles];
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

__device__ __forceinline__ void decode_tile_ka5(const Exl3GemmArgs& args, uint32_t kt, uint32_t nt,
                                                uint32_t lane, uint32_t* ring, uint16_t* tile) {
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * 80u;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);
    for (uint32_t i = lane; i < 40u; i += 32u) ring[i] = source[i];
    __syncwarp();

    const uint32_t off0 = 40u * lane + 1269u;
    uint32_t w0 = off0 >> 5u;
    if (w0 >= 40u) w0 -= 40u;
    uint32_t i1 = w0 + 1u;
    uint32_t i2 = w0 + 2u;
    uint32_t i3 = w0 + 3u;
    if (i1 >= 40u) i1 -= 40u;
    if (i2 >= 40u) i2 -= 40u;
    if (i3 >= 40u) i3 -= 40u;
    const uint32_t v0 = ring[w0];
    const uint32_t v1 = ring[i1];
    const uint32_t v2 = ring[i2];
    const uint32_t v3 = ring[i3];
    const uint32_t r = off0 & 31u;

    uint32_t st[8];
#pragma unroll
    for (uint32_t j = 0u; j < 8u; ++j) {
        const uint32_t off = r + 5u * j;
        const uint32_t lo = off >> 5u;
        const uint32_t hi = (off + 15u) >> 5u;
        const uint32_t shift = ((hi + 1u) << 5u) - (off + 16u);
        st[j] = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo, v0, v1, v2, v3)) << 32) |
             pick4(hi, v0, v1, v2, v3)) >>
            shift);
    }

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

template <uint32_t KA, bool HALF>
__device__ __forceinline__ void decode_tile_fixed(const Exl3GemmArgs& args, uint32_t kt,
                                                  uint32_t nt, uint32_t lane, uint32_t* ring,
                                                  uint16_t* tile) {
    constexpr uint32_t kWords = 16u * KA + (HALF ? 8u : 0u);
    constexpr uint32_t kWordCount = kWords / 2u;
    constexpr uint32_t kBitLength = 256u * KA + (HALF ? 128u : 0u);
    const uint32_t n_tiles = args.out_features / 16u;
    const uint16_t* tile_base =
        static_cast<const uint16_t*>(args.trellis) +
        (static_cast<uint64_t>(kt) * n_tiles + nt) * kWords;
    const uint32_t* source = reinterpret_cast<const uint32_t*>(tile_base);
    for (uint32_t i = lane; i < kWordCount; i += 32u) ring[i] = source[i];
    __syncwarp();

    const uint32_t p0 = lane * 8u;
    const uint32_t step0 = p0 + 1u;
    const uint32_t end0 = step0 * KA + (HALF ? step0 / 2u : 0u) + kBitLength;
    const uint32_t w0 = (end0 - 16u) >> 5u;
    const uint32_t v0 = ring[(w0 + 0u) % kWordCount];
    const uint32_t v1 = ring[(w0 + 1u) % kWordCount];
    const uint32_t v2 = ring[(w0 + 2u) % kWordCount];
    const uint32_t v3 = ring[(w0 + 3u) % kWordCount];

    uint32_t* tile32 = reinterpret_cast<uint32_t*>(tile);
#pragma unroll
    for (uint32_t p = 0u; p < 4u; ++p) {
        const uint32_t position = p0 + p * 2u;
        const uint32_t step = position + 1u;
        const uint32_t end = step * KA + (HALF ? step / 2u : 0u) + kBitLength;
        const uint32_t shift = (((end - 1u) >> 5) + 1u) * 32u - end;
        const uint32_t lo = ((end - 16u) >> 5) - w0;
        const uint32_t hi = ((end - 1u) >> 5) - w0;
        const uint32_t state0 = static_cast<uint16_t>(
            ((static_cast<uint64_t>(pick4(lo, v0, v1, v2, v3)) << 32) |
             pick4(hi, v0, v1, v2, v3)) >>
            shift);
        const uint32_t step1 = position + 2u;
        const uint32_t end1 = step1 * KA + (HALF ? step1 / 2u : 0u) + kBitLength;
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

template <uint32_t kDecode>
__device__ __forceinline__ void decode_tile(const Exl3GemmArgs& args, uint32_t kt, uint32_t nt,
                                            uint32_t lane, uint32_t* ring, uint16_t* tile) {
    if constexpr (kDecode == 1u) {
        decode_tile_ka5(args, kt, nt, lane, ring, tile);
        return;
    }
    if constexpr (kDecode == 2u) {
        decode_tile_fixed<3u, false>(args, kt, nt, lane, ring, tile);
        return;
    }
    if constexpr (kDecode == 3u) {
        decode_tile_fixed<3u, true>(args, kt, nt, lane, ring, tile);
        return;
    }
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

__global__ __launch_bounds__(256) void exl3_dense_epilogue_kernel(Exl3GemmArgs args,
                                                                  const float* accum,
                                                                  const float* rowsum) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + warp;
    if (row >= args.rows) return;
    const float bias = exl3_detail::kMul1BiasOverInv * rowsum[row];
    const float scale = kInvSqrt128 * exl3_detail::kMul1Inv;
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

__global__ __launch_bounds__(256) void quantize_dense_w8_kernel(uint16_t* dense, uint32_t k,
                                                               uint32_t n) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    float amax = 0.f;
    for (uint32_t r = 0u; r < k; ++r) {
        const float v =
            fabsf(__half2float(__ushort_as_half(dense[static_cast<uint64_t>(r) * n + col])));
        amax = fmaxf(amax, v);
    }
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    const float inv = 1.f / scale;
    for (uint32_t r = 0u; r < k; ++r) {
        const uint64_t idx = static_cast<uint64_t>(r) * n + col;
        const float v = __half2float(__ushort_as_half(dense[idx]));
        const float q = static_cast<float>(__float2int_rn(v * inv));
        dense[idx] = __half_as_ushort(__float2half(q * scale));
    }
}

template <bool kResidual>
__global__ __launch_bounds__(256) void quantize_xrot_x8_kernel(__half* xrot, float* rowsum,
                                                              uint32_t rows, uint32_t k) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + warp;
    if (row >= rows) return;
    const uint64_t base = static_cast<uint64_t>(row) * k;
    float amax = 0.f;
    for (uint32_t i = lane * 4u; i < k; i += 128u) {
#pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            amax = fmaxf(amax, fabsf(__half2float(xrot[base + i + j])));
        }
    }
    amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 8, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 4, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    const float inv = 1.f / scale;
    float rmax = 0.f;
    float rscale = 1.f;
    float rinv = 1.f;
    if constexpr (kResidual) {
        for (uint32_t i = lane * 4u; i < k; i += 128u) {
#pragma unroll
            for (uint32_t j = 0u; j < 4u; ++j) {
                const float v = __half2float(xrot[base + i + j]);
                const float q = static_cast<float>(__float2int_rn(v * inv)) * scale;
                rmax = fmaxf(rmax, fabsf(v - q));
            }
        }
        rmax = fmaxf(rmax, __shfl_xor(rmax, 16, 32));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 8, 32));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 4, 32));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 2, 32));
        rmax = fmaxf(rmax, __shfl_xor(rmax, 1, 32));
        rscale = rmax > 0.f ? rmax / 127.f : 1.f;
        rinv = 1.f / rscale;
    }
    float sum = 0.f;
    for (uint32_t i = lane * 4u; i < k; i += 128u) {
#pragma unroll
        for (uint32_t j = 0u; j < 4u; ++j) {
            const float v = __half2float(xrot[base + i + j]);
            float q = static_cast<float>(__float2int_rn(v * inv)) * scale;
            if constexpr (kResidual) {
                const float r = v - q;
                q += static_cast<float>(__float2int_rn(r * rinv)) * rscale;
            }
            xrot[base + i + j] = __float2half(q);
            sum += q;
        }
    }
    sum += __shfl_down(sum, 16, 32);
    sum += __shfl_down(sum, 8, 32);
    sum += __shfl_down(sum, 4, 32);
    sum += __shfl_down(sum, 2, 32);
    sum += __shfl_down(sum, 1, 32);
    if (lane == 0u) rowsum[row] = sum;
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

template <uint32_t kGroups, bool kPartial, bool kPre, uint32_t kNTiles, uint32_t kDecode>
__global__ __launch_bounds__(kThreads) void gemm_exl3_w16a16_kernel(Exl3GemmArgs args,
                                                                    uint32_t splits) {
    constexpr uint32_t kOutBlock = kOutBlockBase * kNTiles;
    constexpr uint32_t kNTilesPerBlock = kOutBlock / 16u;
    constexpr uint32_t kRows = kRowTile * kGroups;
    using SharedArea = SharedAreaT<kNTiles>;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_high = lane >> 4;

    const uint32_t n_tile_base = kPartial ? blockIdx.x / splits : blockIdx.x;
    const uint32_t split = kPartial ? blockIdx.x % splits : 0u;
    const uint32_t n0 = n_tile_base * kOutBlock;
    uint32_t r0 = blockIdx.y * kRows;
    uint32_t active_rows = args.rows - r0 < kRows ? args.rows - r0 : kRows;
    const uint32_t group_index = blockIdx.z;
    if (args.group_expert != nullptr &&
        !exl3_detail::resolve_group_plan<kGroups>(args, group_index, r0, active_rows))
        return;
    const __half* xrot_global = static_cast<const __half*>(args.xrot);

    __shared__ __align__(16) __half xrot[kPre ? 1u : kRows * kXrotPitch];
    __shared__ __align__(16) SharedArea area;
    __shared__ __align__(16) float rowsum[kRows];

    f32x8_t acc[kGroups][kNTiles];
#pragma unroll
    for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
        for (uint32_t t = 0u; t < kNTiles; ++t) {
            acc[g][t] = f32x8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        }
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
        if constexpr (!kPre) {
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
                    uint32_t input_row = r0 + r;
                    if (args.row_token != nullptr) {
                        const int32_t token =
                            args.row_token[group_index * kRowTile + r];
                        if (token < 0) continue;
                        input_row = static_cast<uint32_t>(token);
                    }
                    const shelf::bf16_t* src =
                        input + static_cast<uint64_t>(input_row) * args.input_row_stride +
                        k_base + lane * 4u;
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
                    __half* dst = xrot + r * kXrotPitch + lane * 4u;
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
                    const __half* p = static_cast<const __half*>(args.xrot) +
                                      static_cast<uint64_t>(r0 + r) * args.k + k_base +
                                      lane * 4u;
                    row_acc[g][ri] += __half2float(p[0]) + __half2float(p[1]) +
                                      __half2float(p[2]) + __half2float(p[3]);
                }
            }
        }

        const uint32_t n_tile = n_tile_base * kNTilesPerBlock + warp * kNTiles;
#pragma unroll
        for (uint32_t ks = 0u; ks < 8u; ++ks) {
            const uint32_t buf = ks & 1u;
            uint16_t* tile = area.work.tile[warp][buf];
            f16x8_t a[kNTiles];
#pragma unroll
            for (uint32_t t = 0u; t < kNTiles; ++t) {
                decode_tile<kDecode>(args, kc * 8u + ks, n_tile + t, lane,
                                  area.work.ring[warp], tile + t * 256u);
                a[t] = load_f16x8(tile + t * 256u + lane16 * 16u + lane_high * 8u);
            }
#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
                if constexpr (kGroups > 1u) {
                    if (g * kRowTile >= active_rows) continue;
                }
                f16x8_t b;
                if constexpr (kPre) {
                    b = load_f16x8(static_cast<const __half*>(args.xrot) +
                                   static_cast<uint64_t>(r0 + g * kRowTile + lane16) *
                                       args.k +
                                   k_base + ks * 16u + lane_high * 8u);
                } else {
                    b = load_f16x8(xrot + (g * kRowTile + lane16) * kXrotPitch +
                                   ks * 16u + lane_high * 8u);
                }
#pragma unroll
                for (uint32_t t = 0u; t < kNTiles; ++t) {
                    acc[g][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                        a[t], b, acc[g][t]);
                }
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
#pragma unroll
                for (uint32_t t = 0u; t < kNTiles; ++t) {
                    const uint32_t out_col =
                        n0 + (warp * kNTiles + t) * 16u + lane_high * 8u;
                    float* dst =
                        args.splitk_scratch + row_index * args.out_features + out_col;
#pragma unroll
                    for (uint32_t j = 0u; j < 8u; ++j) dst[j] = acc[g][t][j];
                }
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
            for (uint32_t t = 0u; t < kNTiles; ++t) {
#pragma unroll
                for (uint32_t j = 0u; j < 8u; ++j) {
                    const uint32_t out_col =
                        (warp * kNTiles + t) * 16u + lane_high * 8u + j;
                    area.accum[lane16 * kOutBlock + out_col] = acc[g][t][j];
                }
            }
            __syncthreads();

#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                const uint32_t r = g * kRowTile + warp * 2u + ri;
                if (r >= active_rows) continue;
                const uint32_t row = r0 + r;
                const float bias = exl3_detail::kMul1BiasOverInv * rowsum[r];
#pragma unroll
                for (uint32_t cb = 0u; cb < kNTiles; ++cb) {
                    const float* src = area.accum + (r - g * kRowTile) * kOutBlock +
                                       cb * 128u + lane * 4u;
                    float v0 = src[0] + bias;
                    float v1 = src[1] + bias;
                    float v2 = src[2] + bias;
                    float v3 = src[3] + bias;
                    exl3_detail::fwht128(v0, v1, v2, v3, lane);
                    const uint32_t col = n0 + cb * 128u + lane * 4u;
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
                        shelf::bf16_t* dst =
                            static_cast<shelf::bf16_t*>(args.output) + out_base;
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
            if constexpr (kGroups > 1u) {
                __syncthreads();
            }
        }
    }
}

template <uint32_t kNTiles>
__global__ __launch_bounds__(kThreads) void exl3_splitk_reduce_kernel(Exl3GemmArgs args,
                                                                     uint32_t splits) {
    constexpr uint32_t kOutBlock = kOutBlockBase * kNTiles;
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
#pragma unroll
        for (uint32_t cb = 0u; cb < kNTiles; ++cb) {
            const uint32_t col = n0 + cb * 128u + lane * 4u;
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

struct LaunchPlan {
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t grid_z = 1;
    uint32_t splits = 1;
    uint32_t groups = 1;
};

uint32_t pick_n_tiles(const Exl3GemmArgs& args) {
    if (args.n_tiles != 0u) return args.n_tiles;
    if (args.rows < 4u) return 1u;
    if (args.out_features % (kOutBlockBase * 2u) != 0u) return 1u;
    if (args.out_features < kOutBlockBase * 8u) return 1u;
    return 2u;
}

LaunchPlan make_launch_plan(const Exl3GemmArgs& args) {
    LaunchPlan plan;
    plan.grid_x = args.out_features / (kOutBlockBase * pick_n_tiles(args));
    if (args.splitk_scratch != nullptr) {
        const uint32_t tuning_rows =
            args.splitk_tuning_rows != 0u ? args.splitk_tuning_rows : args.rows;
        uint32_t desired = args.splitk_splits;
        if (desired == 0u) {
            if (tuning_rows == 1u) {
                desired = plan.grid_x <= 160u ? 8u : 1u;
            } else if (tuning_rows <= 8u) {
                desired = 8u;
            } else {
                desired = plan.grid_x <= 16u ? 8u : (plan.grid_x <= 160u ? 4u : 1u);
            }
        }
        const uint32_t k_chunks = args.k / 128u;
        if (desired > k_chunks) desired = k_chunks;
        const uint64_t per_split =
            static_cast<uint64_t>(tuning_rows) * args.out_features + tuning_rows;
        while (desired > 1u &&
               static_cast<uint64_t>(desired) * per_split > args.splitk_floats) {
            --desired;
        }
        plan.splits = desired;
    }
    const uint32_t row_blocks2 = (args.rows + 2u * kRowTile - 1u) / (2u * kRowTile);
    if (args.groups == 1u) {
        plan.groups = 1u;
    } else if (args.groups == 2u || args.groups == 4u) {
        plan.groups = args.groups;
    } else if (pick_n_tiles(args) == 2u && args.rows >= 16u * kRowTile &&
               plan.grid_x >= 8u) {
        plan.groups = 4u;
    } else if (args.rows >= 2u * kRowTile &&
               plan.grid_x * row_blocks2 * plan.splits >= 128u) {
        plan.groups = 2u;
    }
    const uint32_t rows_per_block = kRowTile * plan.groups;
    plan.grid_y = (args.rows + rows_per_block - 1u) / rows_per_block;
    return plan;
}

template <uint32_t kGroups, bool kPre, uint32_t kNTiles, uint32_t kDecode>
hipError_t launch_kernel(const Exl3GemmArgs& args, const LaunchPlan& plan,
                         hipStream_t stream) {
    if (plan.splits > 1u) {
        gemm_exl3_w16a16_kernel<kGroups, true, kPre, kNTiles, kDecode>
            <<<dim3(plan.grid_x * plan.splits, plan.grid_y, plan.grid_z), kThreads, 0,
               stream>>>(args, plan.splits);
    } else {
        gemm_exl3_w16a16_kernel<kGroups, false, kPre, kNTiles, kDecode>
            <<<dim3(plan.grid_x, plan.grid_y, plan.grid_z), kThreads, 0, stream>>>(args,
                                                                                  1u);
    }
    return hipGetLastError();
}

template <bool kPre, uint32_t kDecode>
hipError_t launch_plan_typed(const Exl3GemmArgs& args, const LaunchPlan& plan,
                             hipStream_t stream) {
    const uint32_t n_tiles = pick_n_tiles(args);
    hipError_t err = hipSuccess;
    if (n_tiles == 2u) {
        err = plan.groups == 4u   ? launch_kernel<4u, kPre, 2u, kDecode>(args, plan, stream)
              : plan.groups == 2u ? launch_kernel<2u, kPre, 2u, kDecode>(args, plan, stream)
                                  : launch_kernel<1u, kPre, 2u, kDecode>(args, plan, stream);
    } else {
        err = plan.groups == 4u   ? launch_kernel<4u, kPre, 1u, kDecode>(args, plan, stream)
              : plan.groups == 2u ? launch_kernel<2u, kPre, 1u, kDecode>(args, plan, stream)
                                  : launch_kernel<1u, kPre, 1u, kDecode>(args, plan, stream);
    }
    if (err != hipSuccess) return err;
    if (plan.splits > 1u) {
        const uint32_t reduce_grid_y = (args.rows + kRowTile - 1u) / kRowTile;
        if (n_tiles == 2u) {
            exl3_splitk_reduce_kernel<2u><<<dim3(plan.grid_x, reduce_grid_y), kThreads,
                                            0, stream>>>(args, plan.splits);
        } else {
            exl3_splitk_reduce_kernel<1u><<<dim3(plan.grid_x, reduce_grid_y), kThreads,
                                            0, stream>>>(args, plan.splits);
        }
    }
    return hipGetLastError();
}

template <bool kPre>
hipError_t launch_plan(const Exl3GemmArgs& args, const LaunchPlan& plan,
                       hipStream_t stream) {
    if (args.fast_decode != 0u) {
        if (args.ka == 5u && args.half == 0u) {
            return launch_plan_typed<kPre, 1u>(args, plan, stream);
        }
        if (args.ka == 3u && args.half == 0u) {
            return launch_plan_typed<kPre, 2u>(args, plan, stream);
        }
        if (args.ka == 3u && args.half == 1u) {
            return launch_plan_typed<kPre, 3u>(args, plan, stream);
        }
    }
    return launch_plan_typed<kPre, 0u>(args, plan, stream);
}

}

hipError_t launch_gemm_exl3_w16a16(const Exl3GemmArgs& args, hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.input == nullptr || args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.out_features == 0u || args.k == 0u) return hipErrorInvalidValue;
    if ((args.out_features % kOutBlockBase) != 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    return launch_plan<false>(args, make_launch_plan(args), stream);
}

hipError_t launch_gemm_exl3_w16a16_grouped(const Exl3GroupedArgs& args,
                                           hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.input == nullptr || args.output == nullptr || args.group_expert == nullptr ||
        args.group_rows == nullptr || args.group_count == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.out_features == 0u || (args.out_features % kOutBlockBase) != 0u ||
        args.k == 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    if (args.input_row_stride < args.k || args.output_row_stride < args.out_features ||
        args.group_capacity == 0u || args.group_capacity > 65535u) {
        return hipErrorInvalidValue;
    }
    if (args.mul24 != 0u && args.half != 0u) return hipErrorInvalidValue;  // mul24 has integer bit rates only
    if (args.int8_act != 0u && args.mul24 == 0u) return hipErrorInvalidValue;  // INT8 activations exist for mul24 only

    Exl3GemmArgs dense;
    dense.trellis = args.trellis;
    dense.suh = args.suh;
    dense.svh = args.svh;
    dense.input = args.input;
    dense.output = args.output;
    dense.rows = kRowTile;
    dense.out_features = args.out_features;
    dense.k = args.k;
    dense.input_row_stride = args.input_row_stride;
    dense.output_row_stride = args.output_row_stride;
    dense.output_dtype = args.output_dtype;
    dense.ka = args.ka;
    dense.half = args.half;
    static const uint32_t fast_decode = [] {
        const char* e = std::getenv("SHELF_EXL3_FAST_DECODE");
        return e != nullptr ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 1u;
    }();
    dense.fast_decode = fast_decode;
    static const uint32_t fuse_override = [] {
        const char* e = std::getenv("SHELF_EXL3_GROUPED_FUSE");
        if (e == nullptr) return kExl3GroupedFuse;
        return static_cast<uint32_t>(std::strtoul(e, nullptr, 10));
    }();
    static const uint32_t ntiles_override = [] {
        const char* e = std::getenv("SHELF_EXL3_GROUPED_NTILES");
        return e != nullptr ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    dense.n_tiles = ntiles_override;
    dense.groups = (args.expert_group_begin != nullptr && fuse_override > 1u) ? fuse_override
                                                                            : 1u;
    dense.group_expert = args.group_expert;
    dense.group_rows = args.group_rows;
    dense.group_count = args.group_count;
    dense.expert_group_begin = args.expert_group_begin;
    dense.row_token = args.row_token;
    dense.trellis_batch_stride = args.trellis_batch_stride;
    dense.suh_batch_stride = args.suh_batch_stride;
    dense.svh_batch_stride = args.svh_batch_stride;

    // An n-tile override (SHELF_EXL3_GROUPED_NTILES) that does not divide the output would silently drop columns.
    if ((args.out_features % (kOutBlockBase * pick_n_tiles(dense))) != 0u) return hipErrorInvalidValue;
    LaunchPlan plan;
    plan.grid_x = args.out_features / (kOutBlockBase * pick_n_tiles(dense));
    plan.grid_y = 1u;
    plan.grid_z = args.group_capacity;
    plan.splits = 1u;
    plan.groups = dense.groups;
    // mul24 has its own kernel (exl3_mul24_grouped.hip): same group plan, mul24 decode, FP16 or INT8 activations.
    if (args.mul24 != 0u)
        return launch_exl3_mul24_grouped(dense, args.int8_act != 0u, pick_n_tiles(dense), plan.grid_x,
                                         plan.grid_z, stream);
    return launch_plan<false>(dense, plan, stream);
}

hipError_t launch_rotate_exl3_rowsum(const Exl3GemmArgs& args, void* xrot, float* rowsum,
                                     hipStream_t stream) {
    rotate_exl3_kernel<true><<<dim3(args.rows), 256, 0, stream>>>(
        static_cast<const shelf::bf16_t*>(args.input),
        static_cast<const uint16_t*>(args.suh), static_cast<__half*>(xrot), args.rows,
        args.k, args.input_row_stride, rowsum);
    return hipGetLastError();
}

hipError_t launch_quantize_dense_w8(const Exl3GemmArgs& args, void* dense,
                                   hipStream_t stream) {
    if (dense == nullptr) return hipErrorInvalidValue;
    const uint32_t blocks = (args.out_features + 255u) / 256u;
    quantize_dense_w8_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<uint16_t*>(dense), args.k, args.out_features);
    return hipGetLastError();
}

hipError_t launch_quantize_xrot_x8(const Exl3GemmArgs& args, void* xrot, float* rowsum,
                                   bool residual, hipStream_t stream) {
    if (xrot == nullptr || rowsum == nullptr) return hipErrorInvalidValue;
    const uint32_t blocks = (args.rows + 7u) / 8u;
    if (residual) {
        quantize_xrot_x8_kernel<true><<<blocks, 256, 0, stream>>>(
            static_cast<__half*>(xrot), rowsum, args.rows, args.k);
    } else {
        quantize_xrot_x8_kernel<false><<<blocks, 256, 0, stream>>>(
            static_cast<__half*>(xrot), rowsum, args.rows, args.k);
    }
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

hipError_t launch_rotate_exl3(const Exl3GemmArgs& args, void* xrot, hipStream_t stream) {
    rotate_exl3_kernel<false><<<dim3(args.rows), 256, 0, stream>>>(
        static_cast<const shelf::bf16_t*>(args.input),
        static_cast<const uint16_t*>(args.suh), static_cast<__half*>(xrot), args.rows,
        args.k, args.input_row_stride, nullptr);
    return hipGetLastError();
}

hipError_t launch_gemm_exl3_w16a16_pre(const Exl3GemmArgs& args, const void* xrot,
                                       hipStream_t stream) {
    if (args.trellis == nullptr || args.suh == nullptr || args.svh == nullptr ||
        args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.out_features == 0u || args.k == 0u) return hipErrorInvalidValue;
    if ((args.out_features % kOutBlockBase) != 0u || (args.k % 128u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.ka < 1u || args.ka > 8u || (args.half != 0u && args.ka > 3u)) {
        return hipErrorInvalidValue;
    }
    Exl3GemmArgs pre_args = args;
    pre_args.xrot = const_cast<void*>(xrot);
    return launch_plan<true>(pre_args, make_launch_plan(pre_args), stream);
}

}
