#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel::detail {

__device__ __forceinline__ bool topn_better(float va, int32_t ia, float vb, int32_t ib) {
    if (va > vb)
        return true;
    if (va < vb)
        return false;
    return ia < ib;
}

template <uint32_t PerThread>
__device__ __forceinline__ void topn_insert(
    float* values, int32_t* ids, float value, int32_t id) {
    if (!topn_better(value, id, values[PerThread - 1u], ids[PerThread - 1u]))
        return;
    uint32_t pos = PerThread - 1u;
    while (pos > 0u && topn_better(value, id, values[pos - 1u], ids[pos - 1u])) {
        values[pos] = values[pos - 1u];
        ids[pos] = ids[pos - 1u];
        --pos;
    }
    values[pos] = value;
    ids[pos] = id;
}

struct TopnBest {
    float value;
    int32_t id;
    int32_t index;
};

template <uint32_t Lanes>
__device__ __forceinline__ TopnBest topn_warp_best(TopnBest b) {
    #pragma unroll
    for (uint32_t off = Lanes / 2u; off > 0u; off >>= 1u) {
        const float ov = __shfl_down(b.value, off);
        const int32_t oi = __shfl_down(b.id, off);
        const int32_t ox = __shfl_down(b.index, off);
        if (topn_better(ov, oi, b.value, b.id)) {
            b.value = ov;
            b.id = oi;
            b.index = ox;
        }
    }
    return b;
}

struct TopnKeyBest {
    uint64_t key;
    float value;
    int32_t id;
    int32_t index;
};

__device__ __forceinline__ uint64_t topn_key(float v, int32_t id) {
    const uint32_t bits = __float_as_uint(v);
    const uint32_t ordered =
        (v != v) ? 0u : ((v == 0.0f) ? 0x80000000u
                                      : ((bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u)));
    return (static_cast<uint64_t>(ordered) << 32) |
           static_cast<uint32_t>(~id);
}

template <uint32_t Lanes>
__device__ __forceinline__ TopnKeyBest topn_warp_best_key(TopnKeyBest b) {
    #pragma unroll
    for (uint32_t off = Lanes / 2u; off > 0u; off >>= 1u) {
        const uint32_t khi = __shfl_down(static_cast<uint32_t>(b.key >> 32), off);
        const uint32_t klo = __shfl_down(static_cast<uint32_t>(b.key), off);
        const uint64_t ok = (static_cast<uint64_t>(khi) << 32) | klo;
        const float ov = __shfl_down(b.value, off);
        const int32_t oi = __shfl_down(b.id, off);
        const int32_t ox = __shfl_down(b.index, off);
        if (ok > b.key) {
            b.key = ok;
            b.value = ov;
            b.id = oi;
            b.index = ox;
        }
    }
    return b;
}

__device__ __forceinline__ TopnKeyBest topn_broadcast(TopnKeyBest b) {
    TopnKeyBest r;
    r.key = (static_cast<uint64_t>(__shfl(static_cast<uint32_t>(b.key >> 32), 0)) << 32) |
            static_cast<uint64_t>(__shfl(static_cast<uint32_t>(b.key), 0));
    r.value = __shfl(b.value, 0);
    r.id = __shfl(b.id, 0);
    r.index = __shfl(b.index, 0);
    return r;
}

template <uint32_t Warps>
__device__ __forceinline__ TopnKeyBest topn_block_best_key(
    TopnKeyBest b,
    uint64_t* __restrict__ shared_key,
    float* __restrict__ shared_value,
    int32_t* __restrict__ shared_id,
    int32_t* __restrict__ shared_index) {
    b = topn_warp_best_key<32u>(b);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (lane == 0u) {
        shared_key[warp] = b.key;
        shared_value[warp] = b.value;
        shared_id[warp] = b.id;
        shared_index[warp] = b.index;
    }
    __syncthreads();
    if (warp == 0u) {
        TopnKeyBest w{0ull, -INFINITY, 0x7FFFFFFF, -1};
        if (lane < Warps) {
            w.key = shared_key[lane];
            w.value = shared_value[lane];
            w.id = shared_id[lane];
            w.index = shared_index[lane];
        }
        w = topn_warp_best_key<32u>(w);
        if (lane == 0u) {
            shared_key[0] = w.key;
            shared_value[0] = w.value;
            shared_id[0] = w.id;
            shared_index[0] = w.index;
        }
    }
    __syncthreads();
    TopnKeyBest r;
    r.key = shared_key[0];
    r.value = shared_value[0];
    r.id = shared_id[0];
    r.index = shared_index[0];
    return r;
}

template <uint32_t PerThread>
__device__ __forceinline__ void topn_insert_key(
    uint64_t* keys, float* values, int32_t* ids, float value, int32_t id) {
    const uint64_t k = topn_key(value, id);
    if (k <= keys[PerThread - 1u])
        return;
    uint32_t pos = PerThread - 1u;
    while (pos > 0u && k > keys[pos - 1u]) {
        keys[pos] = keys[pos - 1u];
        values[pos] = values[pos - 1u];
        ids[pos] = ids[pos - 1u];
        --pos;
    }
    keys[pos] = k;
    values[pos] = value;
    ids[pos] = id;
}

}  // namespace ps::kernel::detail
