#include <shelf/kernels/optimized/linear/lm_head_fp8.h>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::kernel {
namespace {

constexpr uint32_t kBlock = 32u;
constexpr uint32_t kVec = 4u;
constexpr uint32_t kChunk = 32u * kVec;
constexpr uint32_t kWarps = 8u;
constexpr uint32_t kThreads = kWarps * 32u;
constexpr uint32_t kMaxRows = 16u;
constexpr float kE4M3Max = 448.0f;

using f16x8_t = _Float16 __attribute__((ext_vector_type(8)));
using f16x2_t = _Float16 __attribute__((ext_vector_type(2)));
using f32x8_t = float __attribute__((ext_vector_type(8)));

constexpr uint32_t kWmmaVocabTile = 16u;
constexpr uint32_t kWmmaWarps = 8u;
constexpr uint32_t kWmmaThreads = kWmmaWarps * 32u;
constexpr uint32_t kWmmaKc = 256u;
constexpr uint32_t kWmmaPitch = kWmmaKc + 8u;
constexpr uint32_t kWmmaPrefetch = 8u;
constexpr uint32_t kWmmaBlocksPerChunk = kWmmaKc / kBlock;
constexpr uint32_t kWmmaBlockVocab = kWmmaVocabTile * kWmmaWarps;
constexpr uint32_t kWmmaMinRows = 5u;

// Smallest positive FP16 subnormal. A scale below it rounds to zero and would make
// the whole block unrepresentable, so it is the floor for a stored scale.
constexpr float kMinHalfSubnormal = 0x1p-24f;

__device__ __forceinline__ float bf16_value(const bf16_t* p) {
    return __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(p));
}

__device__ __forceinline__ float fp16_value(const fp16_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

__device__ __forceinline__ fp16_t fp16_from_float(float v) {
    fp16_t out;
    *reinterpret_cast<__half*>(&out) = __float2half(v);
    return out;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u)
        v += __shfl_xor(v, offset, 32);
    return v;
}

template <uint32_t kRows, bool kExact>
__device__ __forceinline__ void lm_head_row_body(
    const bf16_t* __restrict__ hidden, uint32_t hidden_row_stride,
    const uint8_t* __restrict__ code_row, const fp16_t* __restrict__ scale_row,
    float* __restrict__ logits, uint32_t logits_row_stride,
    uint32_t rows, uint32_t row, uint32_t k, uint32_t lane) {
    float acc[kRows];
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) acc[r] = 0.0f;

    for (uint32_t base = 0u; base < k; base += kChunk) {
        const uint32_t offset = base + lane * kVec;
        const uint32_t packed = *reinterpret_cast<const uint32_t*>(code_row + offset);
        const float scale = fp16_value(scale_row + offset / kBlock);
        const auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(packed), false);
        const auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(packed), true);
        const float w0 = lo[0] * scale;
        const float w1 = lo[1] * scale;
        const float w2 = hi[0] * scale;
        const float w3 = hi[1] * scale;
#pragma unroll
        for (uint32_t r = 0; r < kRows; ++r) {
            if (kExact || r < rows) {
                const bf16_t* h =
                    hidden + static_cast<std::size_t>(r) * hidden_row_stride + offset;
                acc[r] += w0 * bf16_value(h + 0) + w1 * bf16_value(h + 1) +
                          w2 * bf16_value(h + 2) + w3 * bf16_value(h + 3);
            }
        }
    }

#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
        if (kExact || r < rows) {
            const float sum = warp_sum(acc[r]);
            if (lane == 0u)
                logits[static_cast<std::size_t>(r) * logits_row_stride + row] = sum;
        }
    }
}

template <uint32_t kRows, bool kExact>
__global__ __launch_bounds__(kThreads) void lm_head_fp8_w32_kernel(
    const bf16_t* __restrict__ hidden, uint32_t hidden_row_stride,
    const uint8_t* __restrict__ codes, const fp16_t* __restrict__ scales,
    float* __restrict__ logits, uint32_t logits_row_stride,
    uint32_t rows, uint32_t vocab, uint32_t k) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row = blockIdx.x * kWarps + warp;
    if (row >= vocab) return;
    const std::size_t code_offset = static_cast<std::size_t>(row) * k;
    const std::size_t scale_offset = static_cast<std::size_t>(row) * (k / kBlock);
    lm_head_row_body<kRows, kExact>(hidden, hidden_row_stride, codes + code_offset,
                                    scales + scale_offset, logits, logits_row_stride,
                                    rows, row, k, lane);
}

__device__ __forceinline__ f16x8_t w32_codes_to_f16x8(const uint2 raw) {
    const auto lo0 = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(raw.x), false);
    const auto lo1 = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(raw.x), true);
    const auto hi0 = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(raw.y), false);
    const auto hi1 = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(raw.y), true);
    f16x8_t out;
    out[0] = static_cast<_Float16>(lo0[0]);
    out[1] = static_cast<_Float16>(lo0[1]);
    out[2] = static_cast<_Float16>(lo1[0]);
    out[3] = static_cast<_Float16>(lo1[1]);
    out[4] = static_cast<_Float16>(hi0[0]);
    out[5] = static_cast<_Float16>(hi0[1]);
    out[6] = static_cast<_Float16>(hi1[0]);
    out[7] = static_cast<_Float16>(hi1[1]);
    return out;
}

__device__ __forceinline__ void w32_scales_to_floats(const uint4 raw, float* out) {
#pragma unroll
    for (uint32_t q = 0u; q < 4u; ++q) {
        const uint32_t bits =
            (q == 0u) ? raw.x : (q == 1u) ? raw.y : (q == 2u) ? raw.z : raw.w;
        const f16x2_t pair = __builtin_bit_cast(f16x2_t, bits);
        out[2u * q] = static_cast<float>(pair[0]);
        out[2u * q + 1u] = static_cast<float>(pair[1]);
    }
}

__global__ __launch_bounds__(kWmmaThreads) void lm_head_fp8_w32_wmma_kernel(
    const bf16_t* __restrict__ hidden, uint32_t hidden_row_stride,
    const uint8_t* __restrict__ codes, const fp16_t* __restrict__ scales,
    float* __restrict__ logits, uint32_t logits_row_stride,
    uint32_t rows, uint32_t vocab, uint32_t k) {
    if (blockIdx.x * kWmmaBlockVocab >= vocab) return;

    __shared__ _Float16 stage[kWmmaVocabTile * kWmmaPitch];

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane16 = lane & 15u;
    const uint32_t a_shift = (lane >> 4u) * 8u;
    const uint32_t code_shift = (lane >> 4u) * 16u;
    const uint32_t vocab_base = blockIdx.x * kWmmaBlockVocab + warp * kWmmaVocabTile;
    const uint32_t vrow = vocab_base + lane16;
    const uint32_t vrow_safe = vrow < vocab ? vrow : vocab - 1u;
    const uint8_t* __restrict__ code_row = codes + static_cast<std::size_t>(vrow_safe) * k;
    const fp16_t* __restrict__ scale_row =
        scales + static_cast<std::size_t>(vrow_safe) * (k / kBlock);

    f32x8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint32_t k0 = 0u; k0 < k; k0 += kWmmaKc) {
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < kWmmaVocabTile * kWmmaBlocksPerChunk * 4u;
             i += kWmmaThreads) {
            const uint32_t slot = i & 3u;
            const uint32_t block = (i >> 2u) % kWmmaBlocksPerChunk;
            const uint32_t r = (i >> 2u) / kWmmaBlocksPerChunk;
            _Float16* dst = stage + r * kWmmaPitch + (2u * block + (slot & 1u)) * 16u +
                            (slot >> 1u) * 8u;
            if (r < rows) {
                const bf16_t* src = hidden + static_cast<std::size_t>(r) * hidden_row_stride +
                                    k0 + block * kBlock + slot * 8u;
                dst[0] = static_cast<_Float16>(bf16_value(src + 0));
                dst[1] = static_cast<_Float16>(bf16_value(src + 1));
                dst[2] = static_cast<_Float16>(bf16_value(src + 2));
                dst[3] = static_cast<_Float16>(bf16_value(src + 3));
                dst[4] = static_cast<_Float16>(bf16_value(src + 4));
                dst[5] = static_cast<_Float16>(bf16_value(src + 5));
                dst[6] = static_cast<_Float16>(bf16_value(src + 6));
                dst[7] = static_cast<_Float16>(bf16_value(src + 7));
            } else {
                *reinterpret_cast<uint4*>(dst) = uint4{0u, 0u, 0u, 0u};
            }
        }
        __syncthreads();

        float scale[kWmmaBlocksPerChunk];
        w32_scales_to_floats(
            *reinterpret_cast<const uint4*>(scale_row + k0 / kBlock), scale);

        for (uint32_t group = 0u; group < kWmmaBlocksPerChunk; group += kWmmaPrefetch) {
            uint4 raw[kWmmaPrefetch];
#pragma unroll
            for (uint32_t u = 0u; u < kWmmaPrefetch; ++u)
                raw[u] = *reinterpret_cast<const uint4*>(
                    code_row + k0 + (group + u) * kBlock + code_shift);
#pragma unroll
            for (uint32_t u = 0u; u < kWmmaPrefetch; ++u) {
                const uint32_t base = (group + u) * kBlock + a_shift;
                const f16x8_t a0 =
                    *reinterpret_cast<const f16x8_t*>(stage + lane16 * kWmmaPitch + base);
                const f16x8_t a1 =
                    *reinterpret_cast<const f16x8_t*>(stage + lane16 * kWmmaPitch + base + 16u);
                const f32x8_t zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                f32x8_t step = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a0, w32_codes_to_f16x8(uint2{raw[u].x, raw[u].y}), zero);
                step = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                    a1, w32_codes_to_f16x8(uint2{raw[u].z, raw[u].w}), step);
                const float s = scale[group + u];
#pragma unroll
                for (uint32_t j = 0u; j < 8u; ++j) acc[j] = __builtin_fmaf(step[j], s, acc[j]);
            }
        }
    }

    const uint32_t row_base = (lane >> 4u) * 8u;
#pragma unroll
    for (uint32_t j = 0u; j < 8u; ++j) {
        const uint32_t row = row_base + j;
        if (row < rows && vrow < vocab)
            logits[static_cast<std::size_t>(row) * logits_row_stride + vrow] = acc[j];
    }
}

__global__ __launch_bounds__(256) void quantize_bf16_to_fp8_w32_kernel(
    const bf16_t* __restrict__ src, uint32_t src_row_stride,
    uint8_t* __restrict__ codes, fp16_t* __restrict__ scales,
    uint32_t rows, uint32_t k) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t group = lane >> 3u;
    const uint32_t lane_in_group = lane & 7u;
    const uint32_t blocks_per_row = k / kBlock;
    const uint32_t groups_per_wave = 4u;
    const uint32_t waves_per_row =
        (blocks_per_row + groups_per_wave - 1u) / groups_per_wave;
    const uint32_t warps_per_block = blockDim.x >> 5u;
    const std::size_t total_waves = static_cast<std::size_t>(rows) * waves_per_row;
    const std::size_t wave_stride = static_cast<std::size_t>(gridDim.x) * warps_per_block;
    for (std::size_t wave =
             static_cast<std::size_t>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5u);
         wave < total_waves; wave += wave_stride) {
        const uint32_t row = static_cast<uint32_t>(wave / waves_per_row);
        const uint32_t block =
            static_cast<uint32_t>(wave % waves_per_row) * groups_per_wave + group;
        if (block >= blocks_per_row) continue;
        const uint32_t element = block * kBlock + lane_in_group * kVec;
        const bf16_t* in = src + static_cast<std::size_t>(row) * src_row_stride + element;
        const float v0 = bf16_value(in + 0);
        const float v1 = bf16_value(in + 1);
        const float v2 = bf16_value(in + 2);
        const float v3 = bf16_value(in + 3);
        float amax = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        amax = fmaxf(amax, __shfl_xor(amax, 1u, 8u));
        amax = fmaxf(amax, __shfl_xor(amax, 2u, 8u));
        amax = fmaxf(amax, __shfl_xor(amax, 4u, 8u));
        const fp16_t stored = fp16_from_float(
            amax > 0.0f ? fmaxf(amax / kE4M3Max, kMinHalfSubnormal) : 1.0f);
        const float inv = 1.0f / fp16_value(&stored);
        int packed = __builtin_amdgcn_cvt_pk_fp8_f32(v0 * inv, v1 * inv, 0, false);
        packed = __builtin_amdgcn_cvt_pk_fp8_f32(v2 * inv, v3 * inv, packed, true);
        *reinterpret_cast<uint32_t*>(codes + static_cast<std::size_t>(row) * k + element) =
            static_cast<uint32_t>(packed);
        if (lane_in_group == 0u)
            scales[static_cast<std::size_t>(row) * blocks_per_row + block] = stored;
    }
}

template <uint32_t kRows, bool kExact>
inline hipError_t launch_lm_head_rows(
    const bf16_t* hidden, uint32_t hidden_row_stride,
    const uint8_t* weight_codes, const fp16_t* weight_scales,
    float* logits, uint32_t logits_row_stride,
    uint32_t rows, uint32_t vocab, uint32_t k, hipStream_t stream) {
    const uint32_t grid = (vocab + kWarps - 1u) / kWarps;
    lm_head_fp8_w32_kernel<kRows, kExact><<<grid, kThreads, 0, stream>>>(
        hidden, hidden_row_stride, weight_codes, weight_scales, logits,
        logits_row_stride, rows, vocab, k);
    return hipGetLastError();
}

inline hipError_t launch_lm_head_wmma(
    const bf16_t* hidden, uint32_t hidden_row_stride,
    const uint8_t* weight_codes, const fp16_t* weight_scales,
    float* logits, uint32_t logits_row_stride,
    uint32_t rows, uint32_t vocab, uint32_t k, hipStream_t stream) {
    const uint32_t grid = (vocab + kWmmaBlockVocab - 1u) / kWmmaBlockVocab;
    lm_head_fp8_w32_wmma_kernel<<<grid, kWmmaThreads, 0, stream>>>(
        hidden, hidden_row_stride, weight_codes, weight_scales, logits,
        logits_row_stride, rows, vocab, k);
    return hipGetLastError();
}

}  // namespace

hipError_t launch_lm_head_fp8_w32(
    const bf16_t* hidden, uint32_t hidden_row_stride,
    const uint8_t* weight_codes, const fp16_t* weight_scales,
    float* logits, uint32_t logits_row_stride,
    uint32_t rows, uint32_t vocab, uint32_t k, hipStream_t stream) {
    if (hidden == nullptr || weight_codes == nullptr || weight_scales == nullptr ||
        logits == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || rows > kMaxRows || vocab == 0u || k == 0u) return hipErrorInvalidValue;
    if ((k % kChunk) != 0u || hidden_row_stride < k || logits_row_stride < vocab)
        return hipErrorInvalidValue;
    if (rows >= kWmmaMinRows && (k % kWmmaKc) == 0u)
        return launch_lm_head_wmma(hidden, hidden_row_stride, weight_codes, weight_scales,
                                   logits, logits_row_stride, rows, vocab, k, stream);
    switch (rows) {
        case 1u: return launch_lm_head_rows<1u, true>(hidden, hidden_row_stride,
                      weight_codes, weight_scales, logits, logits_row_stride,
                      rows, vocab, k, stream);
        case 2u: return launch_lm_head_rows<2u, true>(hidden, hidden_row_stride,
                      weight_codes, weight_scales, logits, logits_row_stride,
                      rows, vocab, k, stream);
        case 3u: return launch_lm_head_rows<4u, false>(hidden, hidden_row_stride,
                      weight_codes, weight_scales, logits, logits_row_stride,
                      rows, vocab, k, stream);
        case 4u: return launch_lm_head_rows<4u, true>(hidden, hidden_row_stride,
                      weight_codes, weight_scales, logits, logits_row_stride,
                      rows, vocab, k, stream);
        case 5u:
        case 6u:
        case 7u: return launch_lm_head_rows<8u, false>(hidden, hidden_row_stride,
                      weight_codes, weight_scales, logits, logits_row_stride,
                      rows, vocab, k, stream);
        case 8u: return launch_lm_head_rows<8u, true>(hidden, hidden_row_stride,
                      weight_codes, weight_scales, logits, logits_row_stride,
                      rows, vocab, k, stream);
        case 16u: return launch_lm_head_rows<kMaxRows, true>(hidden, hidden_row_stride,
                       weight_codes, weight_scales, logits, logits_row_stride,
                       rows, vocab, k, stream);
        default: return launch_lm_head_rows<kMaxRows, false>(hidden, hidden_row_stride,
                       weight_codes, weight_scales, logits, logits_row_stride,
                       rows, vocab, k, stream);
    }
}

hipError_t launch_quantize_bf16_to_fp8_w32(
    const bf16_t* src, uint32_t src_row_stride,
    uint8_t* codes, fp16_t* scales,
    uint32_t rows, uint32_t k, hipStream_t stream) {
    if (src == nullptr || codes == nullptr || scales == nullptr) return hipErrorInvalidValue;
    if (rows == 0u || k == 0u || (k % kBlock) != 0u || src_row_stride < k)
        return hipErrorInvalidValue;
    const std::size_t work = static_cast<std::size_t>(rows) * (k / kBlock);
    if (work == 0u) return hipErrorInvalidValue;
    constexpr uint32_t kThreadsPerBlock = 256u;
    uint32_t grid = static_cast<uint32_t>((work + kThreadsPerBlock - 1u) / kThreadsPerBlock);
    if (grid > 65535u) grid = 65535u;
    quantize_bf16_to_fp8_w32_kernel<<<grid, kThreadsPerBlock, 0, stream>>>(
        src, src_row_stride, codes, scales, rows, k);
    return hipGetLastError();
}

}  // namespace shelf::kernel
