#include <shelf/kernels/dflash2/coarse_head.h>
#include <shelf/kernels/fused/exl3/group_body.h>

namespace shelf::kernel {
namespace {
using namespace exl3_detail::mul24;

constexpr float kLevelScale = 10.85f;

__device__ uint32_t coarse_code(const uint32_t* tile, uint32_t lane, uint32_t i) {
    constexpr uint32_t bits = 6;
    constexpr uint32_t words = 8 * bits;
    const uint32_t start = ((lane * 8 + i + 1) * bits + words * 32 - 16) % (words * 32);
    const uint32_t word = start / 32;
    const uint32_t shifted = __builtin_amdgcn_alignbit(tile[word], tile[(word + 1) % words],
                                                       32 - start % 32);
    const uint32_t state = (start % 32 ? shifted : tile[word]) >> 16;
    const uint32_t sum = __builtin_amdgcn_udot4(__umul24(state, 0xDCD12Du), 0x01010101u, 3u, false);
    const int q = int(sum >> 2) - 128;
    return min(7, max(0, 4 + int(floorf(float(q) / (2 * kLevelScale)))));
}

__global__ void pack_coarse(const uint32_t* trellis, uint32_t* packed) {
    const uint32_t pair = blockIdx.x * 8 + threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t* tiles = trellis + size_t(pair) * 96;
    uint32_t low = 0;
    uint32_t high = 0;
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) {
        const auto code = coarse_code(tiles + (i / 8) * 48, lane, i % 8);
        high |= (code >> 2) << i;
        low |= (code & 3) << ((i % 4) * 8 + (i / 4) * 2);
    }
    packed[size_t(pair) * 48 + lane] = low;
    reinterpret_cast<uint16_t*>(packed + size_t(pair) * 48 + 32)[lane] = high;
}

__device__ int32_t unpack(uint32_t word, uint32_t high, uint32_t shift) {
    const uint32_t upper = ((((high >> (shift * 2)) & 15u) * 0x00204081u) & 0x01010101u) << 2;
    const uint32_t code = ((word >> shift) & 0x03030303u) | upper;
    return int32_t(((code * 2u) + 0x79797979u) ^ 0x80808080u);
}

__global__ void __launch_bounds__(128) coarse_gemm(DFlash2CoarseHeadArgs a) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t nt = blockIdx.x * 4 + threadIdx.x / 32;
    const uint32_t col = lane % 16, half = lane / 16;
    const uint32_t row = col < a.rows ? col : 0;
    const auto* x = a.activations + row * a.code_stride + half * 8;
    const auto* weights = a.packed + size_t(nt) * a.hidden * 3 / 2;
    float sums[8]{};
    for (uint32_t g = 0; g < a.hidden / 128; ++g) {
        int32x8_t acc{};
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
            const uint32_t* pair = weights + g * 192 + j * 48;
            const uint32_t w = pair[lane];
            const uint32_t high = reinterpret_cast<const uint16_t*>(pair + 32)[lane];
#pragma unroll
            for (uint32_t h = 0; h < 2; ++h) {
                const auto input = *reinterpret_cast<const int32x2_t*>(x + g * 128 + j * 32 + h * 16);
                const int32x2_t b{unpack(w, high, h * 4), unpack(w, high, h * 4 + 2)};
                acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, input, true, b, acc, false);
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t m = half * 8 + i;
            if (m < a.rows) sums[i] += float(acc[i]) * a.scales[m * a.scale_stride + g];
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t m = half * 8 + i;
        if (m < a.rows) a.accum[m * a.vocab + nt * 16 + col] =
            sums[i] * (kLevelScale * quantization::exl3::kExl3Int8GridStep);
    }
}

__global__ void coarse_rotate(DFlash2CoarseHeadArgs a) {
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t group = blockIdx.x * 8 + threadIdx.x / 32;
    if (group >= a.vocab / 128) return;
    const uint32_t col = group * 128 + lane * 4;
    const uint32_t offset = blockIdx.y * a.vocab + col;
    auto v = *reinterpret_cast<const float4*>(a.accum + offset);
    exl3_detail::fwht128(v.x, v.y, v.z, v.w, lane);
    v.x *= kInvSqrt128 * f16_bits_to_f32(a.svh[col]);
    v.y *= kInvSqrt128 * f16_bits_to_f32(a.svh[col + 1]);
    v.z *= kInvSqrt128 * f16_bits_to_f32(a.svh[col + 2]);
    v.w *= kInvSqrt128 * f16_bits_to_f32(a.svh[col + 3]);
    *reinterpret_cast<float4*>(a.logits + offset) = v;
    float peak = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
    for (int d = 16; d; d >>= 1) peak = fmaxf(peak, __shfl_xor(peak, d));
    if (!lane) a.maxima[blockIdx.y * (a.vocab / 128) + group] = peak;
}

__global__ void rank_blocks(DFlash2CoarseHeadArgs a) {
    constexpr uint32_t capacity = kDFlash2CoarseMaxBlocks;
    __shared__ float sorted[capacity];
    const uint32_t groups = a.vocab / 128;
    float* maxima = a.maxima + blockIdx.x * groups;
    for (uint32_t i = threadIdx.x; i < capacity; i += blockDim.x)
        sorted[i] = i < groups ? maxima[i] : -INFINITY;
    __syncthreads();
    for (uint32_t width = 2; width <= capacity; width *= 2) {
        for (uint32_t stride = width / 2; stride; stride /= 2) {
            for (uint32_t pair = threadIdx.x; pair < capacity / 2; pair += blockDim.x) {
                const uint32_t left = pair * 2 - (pair & (stride - 1));
                const uint32_t right = left + stride;
                const float x = sorted[left], y = sorted[right];
                const bool descending = (left & width) == 0;
                sorted[left] = descending ? fmaxf(x, y) : fminf(x, y);
                sorted[right] = descending ? fminf(x, y) : fmaxf(x, y);
            }
            __syncthreads();
        }
    }
    const float threshold = sorted[min(a.keep, groups) - 1];
    for (uint32_t i = threadIdx.x; i < groups; i += blockDim.x)
        maxima[i] = maxima[i] >= threshold;
}

__global__ void shortlist(DFlash2CoarseHeadArgs a) {
    const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t groups = a.vocab / 128;
    if (group >= groups) return;
    bool selected = false;
    for (uint32_t m = 0; m < a.rows; ++m)
        selected |= a.maxima[m * groups + group] != 0;
    a.selected[group] = selected;
}

__global__ void __launch_bounds__(128) refine_gemm(DFlash2CoarseHeadArgs a) {
    if (!a.selected[blockIdx.x / 2]) return;
    group_body<6, 1>(a.activations, a.scales, a.code_stride, a.scale_stride,
                     reinterpret_cast<const uint32_t*>(a.trellis), a.accum, a.rows,
                     nullptr, a.vocab, a.hidden, a.hidden / 128, 1, blockIdx.x, 0);
}

__global__ void refine_rotate(DFlash2CoarseHeadArgs a) {
    __shared__ float scratch[128];
    const uint32_t col = blockIdx.x * 128 + threadIdx.x;
    const uint32_t offset = blockIdx.y * a.vocab + col;
    if (a.selected[blockIdx.x]) {
        const float v = exl3_detail::fwht128_smem(a.accum[offset], scratch, threadIdx.x);
        const float logit = v * kInvSqrt128 * f16_bits_to_f32(a.svh[col]);
        a.logits[offset] = logit;
        a.accum[offset] = logit;
    } else {
        a.accum[offset] = -INFINITY;
    }
}
}

hipError_t build_dflash2_coarse_head(const uint16_t* trellis, uint32_t* packed,
                                     uint32_t vocab, uint32_t hidden, hipStream_t stream) {
    if (!trellis || !packed || !vocab || vocab % 128 || !hidden || hidden % 128)
        return hipErrorInvalidValue;
    pack_coarse<<<size_t(vocab) * hidden / 4096, 256, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(trellis), packed);
    return hipGetLastError();
}

hipError_t launch_dflash2_coarse_head(const DFlash2CoarseHeadArgs& a, hipStream_t stream) {
    if (!a.packed || !a.trellis || !a.svh || !a.activations || !a.scales ||
        !a.accum || !a.logits || !a.maxima || !a.selected || !a.rows ||
        a.rows > kDFlash2CoarseMaxRows || !a.vocab || a.vocab % 128 ||
        !a.hidden || a.hidden % 128 || a.code_stride < a.hidden || a.code_stride % 8 ||
        a.scale_stride < a.hidden / 128 || a.vocab / 128 > kDFlash2CoarseMaxBlocks) return hipErrorInvalidValue;
    coarse_gemm<<<a.vocab / 64, 128, 0, stream>>>(a);
    coarse_rotate<<<dim3((a.vocab / 128 + 7) / 8, a.rows), 256, 0, stream>>>(a);
    if (a.keep) {
        rank_blocks<<<a.rows, 256, 0, stream>>>(a);
        shortlist<<<(a.vocab / 128 + 127) / 128, 128, 0, stream>>>(a);
        refine_gemm<<<a.vocab / 64, 128, 0, stream>>>(a);
        refine_rotate<<<dim3(a.vocab / 128, a.rows), 128, 0, stream>>>(a);
    }
    return hipGetLastError();
}
}
