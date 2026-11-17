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
