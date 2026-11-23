#pragma once

#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstddef>

namespace shelf::kernel::detail {

using short8_t = short __attribute__((ext_vector_type(8)));
using short2_t = short __attribute__((ext_vector_type(2)));

inline constexpr uint32_t kExactRowsThreads = kBf16GemmExactRowsThreads;

__device__ __forceinline__ float bf16_f32(short s) {
    const uint32_t ui = static_cast<uint32_t>(static_cast<unsigned short>(s)) << 16u;
    union { uint32_t u; float f; } b;
    b.u = ui;
    return b.f;
}

template <uint32_t kRows, bool kDot2 = false>
__device__ __forceinline__ void gemm_bf16_exact_rows_body(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t n,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride,
    const uint32_t block = blockIdx.x) {
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r = block * 8u + wave;
    if (r >= n)
        return;
    const shelf::bf16_t* wrow = weight + static_cast<size_t>(r) * k;
    float acc[kRows];
    #pragma unroll
    for (uint32_t m = 0; m < kRows; ++m)
        acc[m] = 0.0f;
    const uint32_t iters = k >> 8u;
    uint32_t i = 0;
    if constexpr (!kDot2 && kRows == 1) {
        for (; i + 4u <= iters; i += 4u) {
            const uint32_t off = (i << 8) + (lane << 3);
            float wf[4][8];
            #pragma unroll
            for (uint32_t u = 0; u < 4u; ++u) {
                const short8_t w = *reinterpret_cast<const short8_t*>(
                    wrow + off + (u << 8));
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) wf[u][j] = bf16_f32(w[j]);
            }
            short8_t x[4];
            #pragma unroll
            for (uint32_t u = 0; u < 4u; ++u)
                x[u] = *reinterpret_cast<const short8_t*>(input + off + (u << 8));
            #pragma unroll
            for (uint32_t u = 0; u < 4u; ++u) {
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j)
                    acc[0] += wf[u][j] * bf16_f32(x[u][j]);
            }
        }
    }
    for (; i < iters; ++i) {
        const uint32_t off = (i << 8) + (lane << 3);
        const short8_t w = *reinterpret_cast<const short8_t*>(wrow + off);
        if constexpr (kDot2) {
            #pragma unroll
            for (uint32_t m = 0; m < kRows; ++m) {
                const short8_t x = *reinterpret_cast<const short8_t*>(
                    input + static_cast<size_t>(m) * input_row_stride + off);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; j += 2u) {
                    const short2_t w2 = short2_t{w[j], w[j + 1u]};
                    const short2_t x2 = short2_t{x[j], x[j + 1u]};
                    acc[m] = __builtin_amdgcn_fdot2_f32_bf16(w2, x2, acc[m], false);
                }
            }
        } else {
            float wf[8];
            #pragma unroll
            for (uint32_t j = 0; j < 8u; ++j)
                wf[j] = bf16_f32(w[j]);
            #pragma unroll
            for (uint32_t m = 0; m < kRows; ++m) {
                const short8_t x = *reinterpret_cast<const short8_t*>(
                    input + static_cast<size_t>(m) * input_row_stride + off);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j)
                    acc[m] += wf[j] * bf16_f32(x[j]);
            }
        }
    }
    const uint32_t tail = k & 255u;
    if (tail != 0u) {
        const uint32_t base = iters << 8;
        #pragma unroll
        for (uint32_t m = 0; m < kRows; ++m) {
            const shelf::bf16_t* xrow = input + static_cast<size_t>(m) * input_row_stride;
            for (uint32_t j = lane; j < tail; j += 32u)
                acc[m] += bf16_f32(static_cast<short>(wrow[base + j].data))
                        * bf16_f32(static_cast<short>(xrow[base + j].data));
        }
    }
    #pragma unroll
    for (uint32_t m = 0; m < kRows; ++m) {
        float a = acc[m];
        #pragma unroll
        for (uint32_t s = 16u; s > 0u; s >>= 1)
            a += __shfl_down(a, s, 32);
        if (lane == 0u) {
            const size_t oidx = static_cast<size_t>(m) * output_row_stride + r;
            if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16))
                static_cast<shelf::bf16_t*>(output)[oidx] =
                    shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(a))};
            else
                static_cast<float*>(output)[oidx] = a;
        }
    }
}

}  // namespace shelf::kernel::detail
