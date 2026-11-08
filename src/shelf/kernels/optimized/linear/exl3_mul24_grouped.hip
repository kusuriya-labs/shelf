#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/kernels/optimized/linear/exl3_decode_device.h>
#include <shelf/kernels/optimized/linear/exl3_group_plan_device.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

// Grouped (routed-expert) GEMM for mul24 EXL3 (EXL3-INT8 weights): q = (B - 509) >> 2 in [-128, 127], weight =
// q * kExl3Int8GridStep. Two activation forms share the structure:
//   FP16: xrot = H128(suh * x) / sqrt(128) in FP16, WMMA f16 on the exact integer q, step applied in the epilogue.
//   INT8: xrot quantized per row and 128-K block to int8 (scale amax / 127), iu8 WMMA with an exact int32 block sum,
//         the fp32 row scale applied per block (the dense EXL3-INT8 decode path's group scale).
// The rotation needs the expert's own suh, so it cannot be shared between experts and is fused here (no workspace).
// The expert trellis stays in the canonical [kt][nt] tile order.
namespace shelf::kernel {
namespace {

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f32x8_t = float __attribute__((ext_vector_type(8)));
using int32x2_t = int32_t __attribute__((ext_vector_type(2)));
using int32x8_t = int32_t __attribute__((ext_vector_type(8)));

constexpr uint32_t kOutBlockBase = kExl3W16A16OutBlock;
constexpr uint32_t kRowTile = kExl3W16A16RowTile;
constexpr uint32_t kThreads = kExl3W16A16Threads;
constexpr float kInvSqrt128 = 0.08838834764831844055f;
constexpr uint32_t kXrotPitch = 136u;  // FP16 elements per row in LDS
constexpr uint32_t kXqPitch = 144u;    // int8 elements per row in LDS (16-byte aligned rows)

__device__ __forceinline__ float bf16_value(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<uint32_t>(p->data) << 16u);
}

// Lane l of a 16x16 tile holds stream positions 8l..8l+7: output column l % 16 and the consecutive k rows
// 8 * (l / 16) .. +7, which is the lane's WMMA operand fragment as decoded. All offsets are compile time for a given
// BITS (the same decode as the dense kernel). biased[j] = q + 128 in [0, 255].
template <uint32_t BITS>
__device__ __forceinline__ void decode_biased(const uint32_t* tile, uint32_t lane, uint32_t (&biased)[8]) {
    constexpr uint32_t kWords = 8u * BITS;
    constexpr uint32_t kBits = 256u * BITS;
    constexpr bool kAlignedWindows = (BITS % 4) == 0;
    // The third word is needed when a lane's windows reach past 64 bits of its stream.
    constexpr bool kNeedW2 = kAlignedWindows ? ((((BITS + 16u) % 32u) + 7u * BITS + 16u) > 64u) : true;
    const uint32_t base_bit = (lane * 8u * BITS + BITS + kBits - 16u) % kBits;
    const uint32_t off = base_bit & 31u;
    const uint32_t i0 = base_bit >> 5u;
    const uint32_t i1 = (i0 + 1u) % kWords;
    const uint32_t i2 = (i0 + 2u) % kWords;
    const uint32_t w0 = tile[i0];
    const uint32_t w1 = tile[i1];
    const uint32_t w2 = kNeedW2 ? tile[i2] : 0u;
#pragma unroll
    for (uint32_t i = 0u; i < 8u; ++i) {
        uint32_t pos;
        if constexpr (kAlignedWindows) {
            constexpr uint32_t off_c = (BITS + 32u - 16u) % 32u;
            pos = off_c + i * BITS;
        } else {
            pos = off + i * BITS;
        }
        const uint32_t e = pos + 16u;
        const uint32_t wl = (e - 1u) >> 5u;
        const uint32_t r = 32u - (e - 32u * wl);
        const uint32_t lo = wl == 0u ? w0 : (wl == 1u ? w1 : w2);
        const uint32_t hi = wl == 0u ? 0u : (wl == 1u ? w0 : w1);
        const uint32_t state = __builtin_amdgcn_alignbit(hi, lo, r) & 0xFFFFu;
        biased[i] = __builtin_amdgcn_udot4(__umul24(state, 0xDCD12Du), 0x01010101u, 3u, false) >> 2u;
    }
}

__global__ __launch_bounds__(kThreads) void rotate_quant_grouped(Exl3GemmArgs args) {
    const uint32_t group = blockIdx.x;
    const uint32_t kc = blockIdx.y;
    if (group >= args.group_count[0]) return;
    const uint32_t expert = args.group_expert[group];
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t pitch = args.k + (args.k / 128u) * sizeof(float);
    const uint16_t* suh = reinterpret_cast<const uint16_t*>(
        static_cast<const uint8_t*>(args.suh) + static_cast<uint64_t>(expert) * args.suh_batch_stride);
    for (uint32_t ri = 0; ri < 2u; ++ri) {
        const uint32_t r = warp * 2u + ri;
        if (r >= args.group_rows[group]) continue;
        const uint32_t row = group * kRowTile + r;
        const int32_t input_row = args.row_token ? args.row_token[row] : static_cast<int32_t>(row);
        if (input_row < 0) continue;
        const uint32_t k = kc * 128u + lane * 4u;
        const auto* src = static_cast<const shelf::bf16_t*>(args.input) +
                          static_cast<uint64_t>(input_row) * args.input_row_stride + k;
        float v0 = bf16_value(src) * __half2float(__ushort_as_half(suh[k]));
        float v1 = bf16_value(src + 1) * __half2float(__ushort_as_half(suh[k + 1]));
        float v2 = bf16_value(src + 2) * __half2float(__ushort_as_half(suh[k + 2]));
        float v3 = bf16_value(src + 3) * __half2float(__ushort_as_half(suh[k + 3]));
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        v0 *= kInvSqrt128; v1 *= kInvSqrt128; v2 *= kInvSqrt128; v3 *= kInvSqrt128;
        float peak = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        peak = fmaxf(peak, __shfl_xor(peak, 16, 32));
        peak = fmaxf(peak, __shfl_xor(peak, 8, 32));
        peak = fmaxf(peak, __shfl_xor(peak, 4, 32));
        peak = fmaxf(peak, __shfl_xor(peak, 2, 32));
        peak = fmaxf(peak, __shfl_xor(peak, 1, 32));
        const float scale = peak / 127.0f;
        const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        auto quant = [&](float v) -> uint32_t {
            return static_cast<uint32_t>(static_cast<int32_t>(fminf(fmaxf(rintf(v * inv), -128.0f), 127.0f))) & 255u;
        };
        auto* dst = static_cast<uint8_t*>(args.mul24_activation_workspace) + static_cast<uint64_t>(row) * pitch;
        *reinterpret_cast<uint32_t*>(dst + k) = quant(v0) | (quant(v1) << 8) | (quant(v2) << 16) | (quant(v3) << 24);
        if (lane == 0u) reinterpret_cast<float*>(dst + args.k)[kc] = scale;
    }
}

template <bool kInt8, bool kPre, uint32_t kGroups, uint32_t kNTiles, uint32_t BITS>
__global__ __launch_bounds__(kThreads) void gemm_mul24_grouped_kernel(Exl3GemmArgs args) {
    constexpr uint32_t kOutBlock = kOutBlockBase * kNTiles;
    constexpr uint32_t kNTilesPerBlock = kOutBlock / 16u;
    constexpr uint32_t kRows = kRowTile * kGroups;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_high = lane >> 4;
    const uint32_t n0 = blockIdx.x * kOutBlock;
    const uint32_t group_index = blockIdx.z;
    uint32_t r0 = 0u;
    uint32_t active_rows = 0u;
    if (!exl3_detail::resolve_group_plan<kGroups>(args, group_index, r0, active_rows)) return;

    // INT8: int8 codes (+ a scale per row). FP16: xrot as half.
    __shared__ __align__(16) uint8_t xbuf[kInt8 ? kRows * kXqPitch : kRows * kXrotPitch * 2u];
    __shared__ __align__(16) float xs[kInt8 ? kRows : 1u];
    __shared__ __align__(16) float accum[kRowTile * kOutBlock];

    f32x8_t acc[kGroups][kNTiles];
#pragma unroll
    for (uint32_t g = 0u; g < kGroups; ++g)
#pragma unroll
        for (uint32_t t = 0u; t < kNTiles; ++t) acc[g][t] = f32x8_t{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    const uint16_t* suh = static_cast<const uint16_t*>(args.suh);
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input);
    const uint32_t n_tiles_total = args.out_features / 16u;
    const uint32_t* trellis = static_cast<const uint32_t*>(args.trellis);

    const uint32_t k_chunks = args.k / 128u;
    for (uint32_t kc = 0u; kc < k_chunks; ++kc) {
        const uint32_t k_base = kc * 128u;
        __syncthreads();
        if constexpr (kPre) {
            const uint32_t pitch = args.k + (args.k / 128u) * sizeof(float);
            for (uint32_t r = warp; r < kRows; r += 8u) {
                if (r >= active_rows) continue;
                const uint32_t row = group_index * kRowTile + r;
                if (args.row_token && args.row_token[row] < 0) {
                    if (lane == 0u) xs[r] = 0.f;
                    continue;
                }
                const auto* src = static_cast<const uint8_t*>(args.mul24_activation_workspace) +
                                  static_cast<uint64_t>(row) * pitch;
                *reinterpret_cast<uint32_t*>(xbuf + r * kXqPitch + lane * 4u) =
                    *reinterpret_cast<const uint32_t*>(src + k_base + lane * 4u);
                if (lane == 0u) xs[r] = reinterpret_cast<const float*>(src + args.k)[kc];
            }
        } else {
        float suh_v[4];
#pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i)
            suh_v[i] = __half2float(__ushort_as_half(suh[k_base + lane * 4u + i]));
#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                const uint32_t r = g * kRowTile + warp * 2u + ri;
                int32_t token = static_cast<int32_t>(r);
                if (args.row_token != nullptr && r < active_rows)
                    token = args.row_token[group_index * kRowTile + r];
                if (r >= active_rows || token < 0) {
                    // Nothing to compute: scale 0 (the stale codes are finite and contribute nothing).
                    if constexpr (kInt8) {
                        if (lane == 0u) xs[r] = 0.f;
                    }
                    continue;
                }
                const uint32_t input_row = args.row_token != nullptr ? static_cast<uint32_t>(token) : r;
                const shelf::bf16_t* src =
                    input + static_cast<uint64_t>(input_row) * args.input_row_stride + k_base + lane * 4u;
                float v0 = bf16_value(src + 0) * suh_v[0];
                float v1 = bf16_value(src + 1) * suh_v[1];
                float v2 = bf16_value(src + 2) * suh_v[2];
                float v3 = bf16_value(src + 3) * suh_v[3];
                exl3_detail::fwht128(v0, v1, v2, v3, lane);
                v0 *= kInvSqrt128;
                v1 *= kInvSqrt128;
                v2 *= kInvSqrt128;
                v3 *= kInvSqrt128;
                if constexpr (kInt8) {
                    float peak = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
                    peak = fmaxf(peak, __shfl_xor(peak, 16, 32));
                    peak = fmaxf(peak, __shfl_xor(peak, 8, 32));
                    peak = fmaxf(peak, __shfl_xor(peak, 4, 32));
                    peak = fmaxf(peak, __shfl_xor(peak, 2, 32));
                    peak = fmaxf(peak, __shfl_xor(peak, 1, 32));
                    const float scale = peak / 127.0f;
                    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;  // one division per row, not per element
                    auto quant = [&](float v) -> uint32_t {
                        const float q = fminf(fmaxf(rintf(v * inv), -128.0f), 127.0f);
                        return static_cast<uint32_t>(static_cast<int32_t>(q)) & 0xFFu;
                    };
                    const uint32_t packed = quant(v0) | (quant(v1) << 8) | (quant(v2) << 16) | (quant(v3) << 24);
                    *reinterpret_cast<uint32_t*>(xbuf + r * kXqPitch + lane * 4u) = packed;
                    if (lane == 0u) xs[r] = scale;
                } else {
                    __half* dst = reinterpret_cast<__half*>(xbuf) + r * kXrotPitch + lane * 4u;
                    dst[0] = __float2half(v0);
                    dst[1] = __float2half(v1);
                    dst[2] = __float2half(v2);
                    dst[3] = __float2half(v3);
                }
            }
        }
        }
        __syncthreads();

        int32x8_t acci[kGroups][kNTiles];
        if constexpr (kInt8) {
#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g)
#pragma unroll
                for (uint32_t t = 0u; t < kNTiles; ++t)
#pragma unroll
                    for (uint32_t j = 0u; j < 8u; ++j) acci[g][t][j] = 0;
        }
        const uint32_t n_tile = blockIdx.x * kNTilesPerBlock + warp * kNTiles;
#pragma unroll
        for (uint32_t ks = 0u; ks < 8u; ++ks) {
            uint32_t biased[kNTiles][8];
#pragma unroll
            for (uint32_t t = 0u; t < kNTiles; ++t) {
                const uint32_t* tile =
                    trellis + (static_cast<uint64_t>(kc * 8u + ks) * n_tiles_total + n_tile + t) * (8u * BITS);
                decode_biased<BITS>(tile, lane, biased[t]);
            }
#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
                if constexpr (kGroups > 1u) {
                    if (g * kRowTile >= active_rows) continue;
                }
                if constexpr (kInt8) {
                    const int32x2_t b = *reinterpret_cast<const int32x2_t*>(
                        xbuf + (g * kRowTile + lane16) * kXqPitch + ks * 16u + lane_high * 8u);
#pragma unroll
                    for (uint32_t t = 0u; t < kNTiles; ++t) {
                        int32x2_t a;  // four biased bytes per dword, flipped to signed with one xor
                        a[0] = static_cast<int32_t>((biased[t][0] | (biased[t][1] << 8) | (biased[t][2] << 16) |
                                                     (biased[t][3] << 24)) ^ 0x80808080u);
                        a[1] = static_cast<int32_t>((biased[t][4] | (biased[t][5] << 8) | (biased[t][6] << 16) |
                                                     (biased[t][7] << 24)) ^ 0x80808080u);
                        acci[g][t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b,
                                                                                      acci[g][t], false);
                    }
                } else {
                    const f16x8_t b = *reinterpret_cast<const f16x8_t*>(
                        reinterpret_cast<const __half*>(xbuf) + (g * kRowTile + lane16) * kXrotPitch + ks * 16u +
                        lane_high * 8u);
#pragma unroll
                    for (uint32_t t = 0u; t < kNTiles; ++t) {
                        f16x8_t a;
#pragma unroll
                        for (uint32_t j = 0u; j < 8u; ++j)
                            // 1024 + (q + 128) is exact in FP16; subtracting 1152 leaves q.
                            a[j] = __builtin_bit_cast(_Float16, static_cast<uint16_t>(0x6400u + biased[t][j])) -
                                   static_cast<_Float16>(1152.0f);
                        acc[g][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[g][t]);
                    }
                }
            }
        }
        if constexpr (kInt8) {
#pragma unroll
            for (uint32_t g = 0u; g < kGroups; ++g) {
                const float row_scale = xs[g * kRowTile + lane16];
#pragma unroll
                for (uint32_t t = 0u; t < kNTiles; ++t)
#pragma unroll
                    for (uint32_t j = 0u; j < 8u; ++j)
                        acc[g][t][j] += static_cast<float>(acci[g][t][j]) * row_scale;
            }
        }
    }

    // Output: H128 over each 128-wide output block, then * svh (mul24 has no codebook bias).
    const float scale = kInvSqrt128 * ::shelf::quantization::exl3::kExl3Int8GridStep;
#pragma unroll
    for (uint32_t g = 0u; g < kGroups; ++g) {
        if constexpr (kGroups > 1u) {
            if (g * kRowTile >= active_rows) continue;
        }
#pragma unroll
        for (uint32_t t = 0u; t < kNTiles; ++t)
#pragma unroll
            for (uint32_t j = 0u; j < 8u; ++j)
                accum[lane16 * kOutBlock + (warp * kNTiles + t) * 16u + lane_high * 8u + j] = acc[g][t][j];
        __syncthreads();
#pragma unroll
        for (uint32_t ri = 0u; ri < 2u; ++ri) {
            const uint32_t r = g * kRowTile + warp * 2u + ri;
            if (r >= active_rows || (args.row_token && args.row_token[group_index * kRowTile + r] < 0)) continue;
#pragma unroll
            for (uint32_t cb = 0u; cb < kNTiles; ++cb) {
                const float* src = accum + (r - g * kRowTile) * kOutBlock + cb * 128u + lane * 4u;
                float v0 = src[0];
                float v1 = src[1];
                float v2 = src[2];
                float v3 = src[3];
                exl3_detail::fwht128(v0, v1, v2, v3, lane);
                const uint32_t col = n0 + cb * 128u + lane * 4u;
                const float y0 = v0 * scale * __half2float(__ushort_as_half(svh[col + 0]));
                const float y1 = v1 * scale * __half2float(__ushort_as_half(svh[col + 1]));
                const float y2 = v2 * scale * __half2float(__ushort_as_half(svh[col + 2]));
                const float y3 = v3 * scale * __half2float(__ushort_as_half(svh[col + 3]));
                const uint64_t out_base = static_cast<uint64_t>(r) * args.output_row_stride + col;
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
        if constexpr (kGroups > 1u) {
            __syncthreads();
        }
    }
}

template <bool kInt8, bool kPre, uint32_t BITS>
__global__ __launch_bounds__(kThreads) void gemm_mul24_lds_kernel(Exl3GemmArgs args) {
    constexpr uint32_t kGroups = 3u;
    constexpr uint32_t kNTiles = 1u;
    constexpr uint32_t kNTilesPerBlock = 8u;
    constexpr uint32_t kOutBlock = 128u;
    constexpr uint32_t kRows = kRowTile * kGroups;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t lane_high = lane >> 4;
    const uint32_t n0 = blockIdx.x * kOutBlock;
    const uint32_t group_index = blockIdx.z;
    uint32_t r0 = 0u, active_rows = 0u;
    if (!exl3_detail::resolve_group_plan<kGroups>(args, group_index, r0, active_rows)) return;
    __shared__ __align__(16) uint8_t xbuf[kInt8 ? kRows * kXqPitch : kRows * kXrotPitch * 2u];
    __shared__ __align__(16) float xs[kInt8 ? kRows : 1u];
    __shared__ __align__(16) float accum[kRows * kOutBlock];
    __shared__ __align__(16) int32x2_t weights[8u * 8u * 32u];
    const uint16_t* suh = static_cast<const uint16_t*>(args.suh);
    const uint16_t* svh = static_cast<const uint16_t*>(args.svh);
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input);
    const uint32_t n_tiles_total = args.out_features / 16u;
    const uint32_t* trellis = static_cast<const uint32_t*>(args.trellis);

    for (uint32_t i = threadIdx.x; i < kRows * kOutBlock; i += kThreads) accum[i] = 0.0f;
    const uint32_t k_chunks = args.k / 128u;
    for (uint32_t kc = 0u; kc < k_chunks; ++kc) {
        const uint32_t k_base = kc * 128u;
        __syncthreads();
        if constexpr (kPre) {
            const uint32_t pitch = args.k + (args.k / 128u) * sizeof(float);
            for (uint32_t r = warp; r < kRows; r += 8u) {
                if (r >= active_rows) continue;
                const uint32_t row = group_index * kRowTile + r;
                if (args.row_token && args.row_token[row] < 0) {
                    if (lane == 0u) xs[r] = 0.f;
                    continue;
                }
                const auto* src = static_cast<const uint8_t*>(args.mul24_activation_workspace) +
                                  static_cast<uint64_t>(row) * pitch;
                *reinterpret_cast<uint32_t*>(xbuf + r * kXqPitch + lane * 4u) =
                    *reinterpret_cast<const uint32_t*>(src + k_base + lane * 4u);
                if (lane == 0u) xs[r] = reinterpret_cast<const float*>(src + args.k)[kc];
            }
        } else {
        float suh_v[4];
#pragma unroll
        for (uint32_t i = 0u; i < 4u; ++i)
            suh_v[i] = __half2float(__ushort_as_half(suh[k_base + lane * 4u + i]));
#pragma unroll
        for (uint32_t g = 0u; g < kGroups; ++g) {
#pragma unroll
            for (uint32_t ri = 0u; ri < 2u; ++ri) {
                const uint32_t r = g * kRowTile + warp * 2u + ri;
                int32_t token = static_cast<int32_t>(r);
                if (args.row_token != nullptr && r < active_rows)
                    token = args.row_token[group_index * kRowTile + r];
                if (r >= active_rows || token < 0) {

                    if constexpr (kInt8) {
                        if (lane == 0u) xs[r] = 0.f;
                    }
                    continue;
                }
                const uint32_t input_row = args.row_token != nullptr ? static_cast<uint32_t>(token) : r;
                const shelf::bf16_t* src =
                    input + static_cast<uint64_t>(input_row) * args.input_row_stride + k_base + lane * 4u;
                float v0 = bf16_value(src + 0) * suh_v[0];
                float v1 = bf16_value(src + 1) * suh_v[1];
                float v2 = bf16_value(src + 2) * suh_v[2];
                float v3 = bf16_value(src + 3) * suh_v[3];
                exl3_detail::fwht128(v0, v1, v2, v3, lane);
                v0 *= kInvSqrt128;
                v1 *= kInvSqrt128;
                v2 *= kInvSqrt128;
                v3 *= kInvSqrt128;
                if constexpr (kInt8) {
                    float peak = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
                    peak = fmaxf(peak, __shfl_xor(peak, 16, 32));
                    peak = fmaxf(peak, __shfl_xor(peak, 8, 32));
                    peak = fmaxf(peak, __shfl_xor(peak, 4, 32));
                    peak = fmaxf(peak, __shfl_xor(peak, 2, 32));
                    peak = fmaxf(peak, __shfl_xor(peak, 1, 32));
                    const float scale = peak / 127.0f;
                    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
                    auto quant = [&](float v) -> uint32_t {
                        const float q = fminf(fmaxf(rintf(v * inv), -128.0f), 127.0f);
                        return static_cast<uint32_t>(static_cast<int32_t>(q)) & 0xFFu;
                    };
                    const uint32_t packed = quant(v0) | (quant(v1) << 8) | (quant(v2) << 16) | (quant(v3) << 24);
                    *reinterpret_cast<uint32_t*>(xbuf + r * kXqPitch + lane * 4u) = packed;
                    if (lane == 0u) xs[r] = scale;
                } else {
                    __half* dst = reinterpret_cast<__half*>(xbuf) + r * kXrotPitch + lane * 4u;
                    dst[0] = __float2half(v0);
                    dst[1] = __float2half(v1);
                    dst[2] = __float2half(v2);
                    dst[3] = __float2half(v3);
                }
            }
        }
        }
        __syncthreads();


        for (uint32_t ks = 0u; ks < 8u; ++ks) {
            const uint32_t nt = blockIdx.x * kNTilesPerBlock + warp;
            const uint32_t* tile = trellis +
                (static_cast<uint64_t>(kc * 8u + ks) * n_tiles_total + nt) * (8u * BITS);
            uint32_t q[8];
            decode_biased<BITS>(tile, lane, q);
            int32x2_t a;
            a[0] = static_cast<int32_t>((q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24)) ^ 0x80808080u);
            a[1] = static_cast<int32_t>((q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24)) ^ 0x80808080u);
            weights[(ks * 8u + warp) * 32u + lane] = a;
        }
        __syncthreads();
        for (uint32_t g = 0; g < kGroups && g * kRowTile < active_rows; ++g) {
            const uint32_t offset = g * kRowTile * kOutBlock + lane16 * kOutBlock +
                                    warp * 16u + lane_high * 8u;
            f32x8_t acc;
            for (uint32_t j = 0; j < 8u; ++j) acc[j] = accum[offset + j];
            int32x8_t acci = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t ks = 0; ks < 8u; ++ks) {
                const int32x2_t a8 = weights[(ks * 8u + warp) * 32u + lane];
                if constexpr (kInt8) {
                    const int32x2_t b = *reinterpret_cast<const int32x2_t*>(
                        xbuf + (g * kRowTile + lane16) * kXqPitch + ks * 16u + lane_high * 8u);
                    acci = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a8, true, b, acci, false);
                } else {
                    const f16x8_t b = *reinterpret_cast<const f16x8_t*>(
                        reinterpret_cast<const __half*>(xbuf) + (g * kRowTile + lane16) * kXrotPitch +
                        ks * 16u + lane_high * 8u);
                    f16x8_t a;
                    for (uint32_t j = 0; j < 8u; ++j) {
                        const uint32_t bits = static_cast<uint32_t>(a8[j / 4u]);
                        const int32_t q = static_cast<int8_t>(bits >> (8u * (j % 4u)));
                        a[j] = static_cast<_Float16>(q);
                    }
                    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
                }
            }
            if constexpr (kInt8) {
                const float row_scale = xs[g * kRowTile + lane16];
                for (uint32_t j = 0; j < 8u; ++j) acc[j] += static_cast<float>(acci[j]) * row_scale;
            }
            for (uint32_t j = 0; j < 8u; ++j) accum[offset + j] = acc[j];
        }
    }
    __syncthreads();

    const float scale = kInvSqrt128 * ::shelf::quantization::exl3::kExl3Int8GridStep;
#pragma unroll
    for (uint32_t g = 0u; g < kGroups; ++g) {
        if constexpr (kGroups > 1u) {
            if (g * kRowTile >= active_rows) continue;
        }
#pragma unroll
        for (uint32_t ri = 0u; ri < 2u; ++ri) {
            const uint32_t r = g * kRowTile + warp * 2u + ri;
            if (r >= active_rows || (args.row_token && args.row_token[group_index * kRowTile + r] < 0)) continue;
#pragma unroll
            for (uint32_t cb = 0u; cb < kNTiles; ++cb) {
                const float* src = accum + g * kRowTile * kOutBlock + (r - g * kRowTile) * kOutBlock + cb * 128u + lane * 4u;
                float v0 = src[0];
                float v1 = src[1];
                float v2 = src[2];
                float v3 = src[3];
                exl3_detail::fwht128(v0, v1, v2, v3, lane);
                const uint32_t col = n0 + cb * 128u + lane * 4u;
                const float y0 = v0 * scale * __half2float(__ushort_as_half(svh[col + 0]));
                const float y1 = v1 * scale * __half2float(__ushort_as_half(svh[col + 1]));
                const float y2 = v2 * scale * __half2float(__ushort_as_half(svh[col + 2]));
                const float y3 = v3 * scale * __half2float(__ushort_as_half(svh[col + 3]));
                const uint64_t out_base = static_cast<uint64_t>(r) * args.output_row_stride + col;
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
        if constexpr (kGroups > 1u) {
            __syncthreads();
        }
    }
}

template <bool kInt8, bool kPre>
hipError_t launch_lds_bits(const Exl3GemmArgs& args, dim3 grid, hipStream_t stream) {
#define SHELF_MUL24_LDS_BITS(B) case B: gemm_mul24_lds_kernel<kInt8, kPre, B><<<grid, kThreads, 0, stream>>>(args); break
    switch (args.ka) {
        SHELF_MUL24_LDS_BITS(1u); SHELF_MUL24_LDS_BITS(2u); SHELF_MUL24_LDS_BITS(3u); SHELF_MUL24_LDS_BITS(4u);
        SHELF_MUL24_LDS_BITS(5u); SHELF_MUL24_LDS_BITS(6u); SHELF_MUL24_LDS_BITS(7u); SHELF_MUL24_LDS_BITS(8u);
        default: return hipErrorInvalidValue;
    }
#undef SHELF_MUL24_LDS_BITS
    return hipGetLastError();
}

template <bool kInt8, bool kPre, uint32_t kGroups, uint32_t kNTiles>
hipError_t launch_bits(const Exl3GemmArgs& args, dim3 grid, hipStream_t stream) {
#define SHELF_MUL24_BITS(B) \
    case B: gemm_mul24_grouped_kernel<kInt8, kPre, kGroups, kNTiles, B><<<grid, kThreads, 0, stream>>>(args); break
    switch (args.ka) {
        SHELF_MUL24_BITS(1u);
        SHELF_MUL24_BITS(2u);
        SHELF_MUL24_BITS(3u);
        SHELF_MUL24_BITS(4u);
        SHELF_MUL24_BITS(5u);
        SHELF_MUL24_BITS(6u);
        SHELF_MUL24_BITS(7u);
        SHELF_MUL24_BITS(8u);
        default: return hipErrorInvalidValue;
    }
#undef SHELF_MUL24_BITS
    return hipGetLastError();
}

template <bool kInt8, bool kPre, uint32_t kNTiles>
hipError_t launch_groups(const Exl3GemmArgs& args, dim3 grid, hipStream_t stream) {
    // Fusing more than two groups spills registers and measured slower, so it runs as two.
    return args.groups >= 2u ? launch_bits<kInt8, kPre, 2u, kNTiles>(args, grid, stream)
                             : launch_bits<kInt8, kPre, 1u, kNTiles>(args, grid, stream);
}

}  // namespace

hipError_t launch_exl3_mul24_grouped(const Exl3GemmArgs& args, bool int8_act, uint32_t n_tiles, uint32_t grid_x,
                                     uint32_t grid_z, hipStream_t stream) {
    if (args.half != 0u || args.group_expert == nullptr || (n_tiles != 1u && n_tiles != 2u)) return hipErrorInvalidValue;
    const dim3 grid(args.mul24_lds ? args.out_features / 128u : grid_x, 1u, grid_z);
    if (args.mul24_lds && args.expert_group_begin == nullptr) return hipErrorInvalidValue;
    if (args.mul24_activation_workspace != nullptr) {
        if (!int8_act) return hipErrorInvalidValue;
        rotate_quant_grouped<<<dim3(grid_z, args.k / 128u), kThreads, 0, stream>>>(args);
        const hipError_t err = hipGetLastError();
        if (err != hipSuccess) return err;
        if (args.mul24_lds) return launch_lds_bits<true, true>(args, grid, stream);
        return n_tiles == 2u ? launch_groups<true, true, 2u>(args, grid, stream)
                             : launch_groups<true, true, 1u>(args, grid, stream);
    }
    if (args.mul24_lds) return int8_act ? launch_lds_bits<true, false>(args, grid, stream)
                                       : launch_lds_bits<false, false>(args, grid, stream);
    if (int8_act)
        return n_tiles == 2u ? launch_groups<true, false, 2u>(args, grid, stream) : launch_groups<true, false, 1u>(args, grid, stream);
    return n_tiles == 2u ? launch_groups<false, false, 2u>(args, grid, stream) : launch_groups<false, false, 1u>(args, grid, stream);
}

}  // namespace shelf::kernel
