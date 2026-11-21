#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel::detail {
__device__ __forceinline__ uint32_t float_radix_key(float value) {
    const uint32_t bits = value == 0.f ? 0u : __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <unsigned IndexBytes>
__device__ __forceinline__ void radix_select(const float *scores, uint32_t usable, uint32_t k, uint32_t *hist,
                                             uint32_t *state, std::int32_t *out, float *out_scores,
                                             uint32_t index_base) {
    const uint32_t t = threadIdx.x;
    const uint32_t stride = blockDim.x;
    uint32_t rank = k;
    uint32_t prefix = 0u;
    uint32_t above = 0u;
    for (uint32_t pass = 0u; pass < 4u; ++pass) {
        const uint32_t shift = 24u - 8u * pass;
        for (uint32_t i = t; i < 256u; i += stride)
            hist[i] = 0u;
        __syncthreads();
        for (uint32_t i = t; i < usable; i += stride) {
            const uint32_t key = float_radix_key(scores[i]);
            if (pass == 0u || (key >> (shift + 8u)) == (prefix >> (shift + 8u)))
                atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (t == 0u) {
            uint32_t acc = 0u;
            uint32_t bucket = 0u;
            for (int b = 255; b >= 0; --b) {
                const uint32_t count = hist[static_cast<uint32_t>(b)];
                if (acc + count >= rank) {
                    bucket = static_cast<uint32_t>(b);
                    break;
                }
                acc += count;
            }
            state[0] = bucket;
            state[1] = acc;
            state[2] = rank - acc;
        }
        __syncthreads();
        above += state[1];
        rank = state[2];
        prefix |= state[0] << shift;
        __syncthreads();
    }

    uint32_t need = k - above;
    uint32_t index_prefix = 0u;
    for (uint32_t pass = 0u; pass < IndexBytes && need != 0u; ++pass) {
        const uint32_t shift = 8u * (IndexBytes - 1u - pass);
        for (uint32_t i = t; i < 256u; i += stride)
            hist[i] = 0u;
        __syncthreads();
        for (uint32_t i = t; i < usable; i += stride) {
            if (float_radix_key(scores[i]) != prefix)
                continue;
            if (pass != 0u && (i >> (shift + 8u)) != (index_prefix >> (shift + 8u)))
                continue;
            atomicAdd(&hist[(i >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (t == 0u) {
            uint32_t acc = 0u;
            uint32_t bucket = 0u;
            for (uint32_t b = 0u; b < 256u; ++b) {
                const uint32_t count = hist[b];
                if (acc + count >= need) {
                    bucket = b;
                    break;
                }
                acc += count;
            }
            state[0] = bucket;
            state[1] = need - acc;
        }
        __syncthreads();
        index_prefix |= state[0] << shift;
        need = state[1];
        __syncthreads();
    }
    if (t == 0u)
        state[3] = 0u;
    __syncthreads();
    for (uint32_t i = t; i < usable; i += stride) {
        const float value = scores[i];
        const auto key = float_radix_key(value);
        if (key > prefix || (key == prefix && i <= index_prefix)) {
            const uint32_t pos = atomicAdd(&state[3], 1u);
            out[pos] = static_cast<std::int32_t>(index_base + i);
            if (out_scores != nullptr)
                out_scores[pos] = value;
        }
    }
    __syncthreads();
}
}
