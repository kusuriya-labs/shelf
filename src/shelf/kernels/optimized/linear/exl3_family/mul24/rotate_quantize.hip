#include <shelf/kernels/optimized/linear/exl3_mul1.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul1/decode.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>


namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;

__global__ void __launch_bounds__(kBlock) exl3_mul24_rotate_quantize_kernel(
    const ::shelf::bf16_t* __restrict__ input,
    const uint32_t input_row_stride,
    const uint16_t* __restrict__ suh,
    int8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride_floats) {
    __shared__ float smem[kBlock];
    __shared__ float wave_peak[kBlock / 32u];
    const uint32_t t = threadIdx.x;
    const uint32_t m = blockIdx.x;
    const uint32_t g = blockIdx.y;
    const uint32_t col = g * kBlock + t;
    float v = bf16_to_f32(input[static_cast<size_t>(m) * input_row_stride + col]) * f16_bits_to_f32(suh[col]);
    v = fwht128_smem(v, smem, t) * kInvSqrt128;

    float peak = fabsf(v);
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off));
    if ((t & 31u) == 0u) wave_peak[t >> 5u] = peak;
    __syncthreads();
    peak = fmaxf(fmaxf(wave_peak[0], wave_peak[1]), fmaxf(wave_peak[2], wave_peak[3]));

    const float scale = peak / 127.0f;
    float q = 0.0f;
    if (scale > 0.0f) q = fminf(fmaxf(rintf(v / scale), -128.0f), 127.0f);
    codes[static_cast<size_t>(m) * code_row_stride + col] = static_cast<int8_t>(static_cast<int>(q));
    if (t == 0u) scales[static_cast<size_t>(m) * scale_row_stride_floats + g] = scale;
}

// Same math, float4 form: one CTA per row, 8 waves, one wave per 128-K group (the rotate_exl3_kernel
// structure). H128 stays inside one wave, so there is no shared memory and no barrier; the amax is a
// warp shuffle and each quartet is one 64-bit input load / one 32-bit code store.
__global__ void __launch_bounds__(256) exl3_mul24_rotate_quantize_f4_kernel(
    const ::shelf::bf16_t* __restrict__ input,
    const uint32_t input_row_stride,
    const uint16_t* __restrict__ suh,
    int8_t* __restrict__ codes,
    float* __restrict__ scales,
    const uint32_t code_row_stride,
    const uint32_t scale_row_stride_floats,
    const uint32_t groups) {
    const uint32_t m = blockIdx.x;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t g = wave; g < groups; g += 8u) {
        const uint32_t col = g * kBlock + lane * 4u;
        const uint2 raw = *reinterpret_cast<const uint2*>(input + static_cast<size_t>(m) * input_row_stride + col);
        float v0 = bf16_bits_to_f32(static_cast<uint16_t>(raw.x & 0xFFFFu)) * f16_bits_to_f32(suh[col + 0u]);
        float v1 = bf16_bits_to_f32(static_cast<uint16_t>(raw.x >> 16u)) * f16_bits_to_f32(suh[col + 1u]);
        float v2 = bf16_bits_to_f32(static_cast<uint16_t>(raw.y & 0xFFFFu)) * f16_bits_to_f32(suh[col + 2u]);
        float v3 = bf16_bits_to_f32(static_cast<uint16_t>(raw.y >> 16u)) * f16_bits_to_f32(suh[col + 3u]);
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        v0 *= kInvSqrt128;
        v1 *= kInvSqrt128;
        v2 *= kInvSqrt128;
        v3 *= kInvSqrt128;
        float peak = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u) peak = fmaxf(peak, __shfl_xor(peak, off, 32));
        const float scale = peak / 127.0f;
        float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
        if (scale > 0.0f) {
            q0 = fminf(fmaxf(rintf(v0 / scale), -128.0f), 127.0f);
            q1 = fminf(fmaxf(rintf(v1 / scale), -128.0f), 127.0f);
            q2 = fminf(fmaxf(rintf(v2 / scale), -128.0f), 127.0f);
            q3 = fminf(fmaxf(rintf(v3 / scale), -128.0f), 127.0f);
        }
        const uint32_t packed =
            (static_cast<uint32_t>(static_cast<int32_t>(q0)) & 0xFFu) |
            ((static_cast<uint32_t>(static_cast<int32_t>(q1)) & 0xFFu) << 8u) |
            ((static_cast<uint32_t>(static_cast<int32_t>(q2)) & 0xFFu) << 16u) |
            ((static_cast<uint32_t>(static_cast<int32_t>(q3)) & 0xFFu) << 24u);
        *reinterpret_cast<uint32_t*>(codes + static_cast<size_t>(m) * code_row_stride + col) = packed;
        if (lane == 0u) scales[static_cast<size_t>(m) * scale_row_stride_floats + g] = scale;
    }
}

// Weights stay in the EXL3 trellis (mul24 format, rocm_exl3 research/codebook) and are decoded to int8 on
// every read:
//   state = 16-bit window of the tile's circular MSB-first bit stream,
//   x = state * 0xDCD12D (uint32, one v_mul_u32_u24), B = byte sum of x, q = (B - 509) >> 2.
// q + 128 = (B + 3) >> 2 is taken unsigned (the +3 rides on the dot4 accumulator) and four of them
// are flipped to signed with one xor.
//
// A 16x16 trellis tile holds 256 weights in the WMMA B-fragment order: lane l of 32 decodes
// t = 8l..8l+7, which are column n = l % 16 at the input rows 8 * (l / 16) .. +7. Those eight bytes are
// the lane's iu8 B fragment as decoded, so there is no LDS and no k reordering; the activation codes
// are in natural k order.
//
// One wave owns a 16-wide output tile and MT 16-row tiles; a block has 4 waves along N.
}  // namespace

template <bool kF4>
hipError_t launch_rotate_quantize_impl(const ::shelf::bf16_t* input, uint32_t input_row_stride, const uint16_t* suh,
                                       int8_t* codes, float* scales, uint32_t rows, uint32_t k,
                                       uint32_t code_row_stride_bytes, uint32_t scale_row_stride_bytes,
                                       hipStream_t stream) {
    if constexpr (kF4) {
        exl3_mul24_rotate_quantize_f4_kernel<<<rows, 256, 0, stream>>>(
            input, input_row_stride, suh, codes, scales, code_row_stride_bytes,
            scale_row_stride_bytes / 4u, k / kBlock);
    } else {
        exl3_mul24_rotate_quantize_kernel<<<dim3(rows, k / kBlock), kBlock, 0, stream>>>(
            input, input_row_stride, suh, codes, scales, code_row_stride_bytes, scale_row_stride_bytes / 4u);
    }
    return hipGetLastError();
}

hipError_t launch_exl3_mul24_rotate_quantize(
    const ::shelf::bf16_t* input, uint32_t input_row_stride, const uint16_t* suh, int8_t* codes,
    float* scales, uint32_t rows, uint32_t k, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, hipStream_t stream) {
    if (rows == 0u) return hipSuccess;
    if (k == 0u || (k % kBlock) != 0u || (scale_row_stride_bytes % 4u) != 0u) return hipErrorInvalidValue;
    const bool f4 = rotate_f4_enabled() && rows >= kRotateQuantizeF4MinRows &&
                    (input_row_stride % 4u) == 0u && (code_row_stride_bytes % 4u) == 0u;
    return f4 ? launch_rotate_quantize_impl<true>(input, input_row_stride, suh, codes, scales, rows, k,
                                                  code_row_stride_bytes, scale_row_stride_bytes, stream)
              : launch_rotate_quantize_impl<false>(input, input_row_stride, suh, codes, scales, rows, k,
                                                   code_row_stride_bytes, scale_row_stride_bytes, stream);
}


}  // namespace shelf::kernel