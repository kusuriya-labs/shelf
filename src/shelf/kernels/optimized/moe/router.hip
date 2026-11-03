#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/kernels/optimized/moe/router.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::kernel {
namespace {

constexpr uint32_t kRouterThreads = 64u;
constexpr uint32_t kRouterWaves = kRouterThreads >> 5u;
constexpr uint32_t kRouterMaxPerThread = kMoeRouterMaxExperts / kRouterThreads;
constexpr uint32_t kScanThreads = 256u;
constexpr uint32_t kGroupRows = kMxfp4GemmRowTile;

__device__ __forceinline__ bool better_pair(float a_val, int32_t a_idx, float b_val, int32_t b_idx) {
    return a_val > b_val || (a_val == b_val && a_idx < b_idx);
}

__global__ void __launch_bounds__(kRouterThreads) moe_router_kernel(
    const float* __restrict__ logits,
    uint32_t logits_row_stride,
    int32_t* __restrict__ sel_expert,
    float* __restrict__ sel_weight,
    uint32_t* __restrict__ tile_counts,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    uint32_t tiles) {
    const uint32_t t = blockIdx.x;
    if (t >= rows)
        return;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t wave = tid >> 5u;

    const float* row = logits + static_cast<size_t>(t) * logits_row_stride;
    float v[kRouterMaxPerThread];
    uint32_t idx[kRouterMaxPerThread];
    uint32_t n = 0u;
    float best = -INFINITY;
    for (uint32_t e = tid; e < experts; e += kRouterThreads) {
        const float x = row[e];
        idx[n] = e;
        v[n] = x;
        ++n;
        if (x > best)
            best = x;
    }

    __shared__ float s_val[kRouterWaves];
    __shared__ int32_t s_idx[kRouterWaves];
    __shared__ int32_t s_win_idx[kMoeRouterMaxTopK];
    __shared__ float s_win_val[kMoeRouterMaxTopK];

    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        best = fmaxf(best, __shfl_xor_sync(0xffffffffULL, best, off));
    if (lane == 0u)
        s_val[wave] = best;
    __syncthreads();
    float row_max = s_val[0];
    #pragma unroll
    for (uint32_t w = 1u; w < kRouterWaves; ++w)
        if (s_val[w] > row_max)
            row_max = s_val[w];
    if (!isfinite(row_max))
        row_max = 0.0f;

    float sum = 0.0f;
    for (uint32_t i = 0u; i < n; ++i) {
        v[i] = expf(v[i] - row_max);
        sum += v[i];
    }
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        sum += __shfl_xor_sync(0xffffffffULL, sum, off);
    if (lane == 0u)
        s_val[wave] = sum;
    __syncthreads();
    float row_sum = 0.0f;
    #pragma unroll
    for (uint32_t w = 0u; w < kRouterWaves; ++w)
        row_sum += s_val[w];
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    for (uint32_t i = 0u; i < n; ++i)
        v[i] *= inv_sum;

    const uint32_t tile = t / kMoeSlotTileTokens;
    for (uint32_t j = 0u; j < top_k; ++j) {
        float lv = -1.0f;
        int32_t li = static_cast<int32_t>(kMoeRouterMaxExperts);
        for (uint32_t i = 0u; i < n; ++i) {
            if (v[i] > lv || (v[i] == lv && idx[i] < static_cast<uint32_t>(li))) {
                lv = v[i];
                li = static_cast<int32_t>(idx[i]);
            }
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1u) {
            const float ov = __shfl_xor_sync(0xffffffffULL, lv, off);
            const int32_t oi = __shfl_xor_sync(0xffffffffULL, li, off);
            if (better_pair(ov, oi, lv, li)) {
                lv = ov;
                li = oi;
            }
        }
        if (lane == 0u) {
            s_val[wave] = lv;
            s_idx[wave] = li;
        }
        __syncthreads();
        float bv = s_val[0];
        int32_t bi = s_idx[0];
        #pragma unroll
        for (uint32_t w = 1u; w < kRouterWaves; ++w) {
            if (better_pair(s_val[w], s_idx[w], bv, bi)) {
                bv = s_val[w];
                bi = s_idx[w];
            }
        }
        if (tid == 0u) {
            s_win_idx[j] = bi;
            s_win_val[j] = bv;
        }
        for (uint32_t i = 0u; i < n; ++i) {
            if (static_cast<int32_t>(idx[i]) == bi)
                v[i] = -1.0f;
        }
        __syncthreads();
    }

    float sel_sum = 0.0f;
    for (uint32_t j = 0u; j < top_k; ++j)
        sel_sum += s_win_val[j];
    if (tid < top_k) {
        const int32_t e = s_win_idx[tid];
        const float w = sel_sum > 0.0f ? s_win_val[tid] / sel_sum : 0.0f;
        sel_expert[static_cast<size_t>(t) * top_k + tid] = e;
        sel_weight[static_cast<size_t>(t) * top_k + tid] = w;
        atomicAdd(&tile_counts[static_cast<size_t>(e) * tiles + tile], 1u);
    }
}

__global__ void __launch_bounds__(kScanThreads) moe_slot_scan_kernel(
    const uint32_t* __restrict__ tile_counts,
    uint32_t* __restrict__ expert_row_begin,
    uint32_t* __restrict__ tile_base,
    uint32_t* __restrict__ expert_group_begin,
    uint32_t* __restrict__ group_expert,
    uint32_t* __restrict__ group_rows,
    uint32_t* __restrict__ group_count,
    int32_t* __restrict__ row_token,
    uint32_t experts,
    uint32_t tiles,
    uint32_t group_capacity) {
    const uint32_t tid = threadIdx.x;
    if (tid == 0u) {
        uint32_t slots = 0u;
        uint32_t groups = 0u;
        for (uint32_t e = 0u; e < experts; ++e) {
            uint32_t expert_total = 0u;
            for (uint32_t tl = 0u; tl < tiles; ++tl)
                expert_total += tile_counts[static_cast<size_t>(e) * tiles + tl];
            expert_row_begin[e] = slots;
            expert_group_begin[e] = groups;
            uint32_t at = slots;
            for (uint32_t tl = 0u; tl < tiles; ++tl) {
                tile_base[static_cast<size_t>(e) * tiles + tl] = at;
                at += tile_counts[static_cast<size_t>(e) * tiles + tl];
            }
            for (uint32_t done = 0u; done < expert_total; done += kGroupRows) {
                if (groups >= group_capacity)
                    break;
                group_expert[groups] = e;
                group_rows[groups] =
                    expert_total - done < kGroupRows ? expert_total - done : kGroupRows;
                ++groups;
            }
            slots += expert_total;
        }
        expert_row_begin[experts] = slots;
        expert_group_begin[experts] = groups;
        group_count[0] = groups;
    }
    __syncthreads();

    const uint32_t groups = group_count[0];
    const uint32_t rows_here = groups * kGroupRows;
    for (uint32_t r = tid; r < rows_here; r += kScanThreads)
        row_token[r] = -1;
}

__global__ void __launch_bounds__(kRouterThreads) moe_slot_scatter_kernel(
    const int32_t* __restrict__ sel_expert,
    const uint32_t* __restrict__ tile_base,
    const uint32_t* __restrict__ expert_row_begin,
    const uint32_t* __restrict__ expert_group_begin,
    int32_t* __restrict__ row_token,
    int32_t* __restrict__ slot_row_of,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    uint32_t tiles) {
    const uint32_t tile = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t s_base[kMoeRouterMaxExperts];
    __shared__ uint32_t s_first[kMoeRouterMaxExperts];
    __shared__ uint32_t s_group[kMoeRouterMaxExperts];
    for (uint32_t e = tid; e < experts; e += kRouterThreads) {
        s_base[e] = tile_base[static_cast<size_t>(e) * tiles + tile];
        s_first[e] = expert_row_begin[e];
        s_group[e] = expert_group_begin[e];
    }
    __syncthreads();

    const uint32_t t_begin = tile * kMoeSlotTileTokens;
    const uint32_t t_end = t_begin + kMoeSlotTileTokens < rows ? t_begin + kMoeSlotTileTokens : rows;
    for (uint32_t t = t_begin; t < t_end; ++t) {
        if (tid < top_k) {
            const uint32_t j = static_cast<size_t>(t) * top_k + tid;
            const uint32_t e = static_cast<uint32_t>(sel_expert[j]);
            const uint32_t slot = atomicAdd(&s_base[e], 1u);
            const uint32_t rank = slot - s_first[e];
            const uint32_t row = (s_group[e] + rank / kGroupRows) * kGroupRows + rank % kGroupRows;
            row_token[row] = static_cast<int32_t>(t);
            slot_row_of[j] = static_cast<int32_t>(row);
        }
    }
}

}  // namespace

hipError_t launch_moe_router(
    const float* logits,
    uint32_t logits_row_stride,
    int32_t* sel_expert,
    float* sel_weight,
    uint32_t* tile_counts,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    hipStream_t stream) {
    if (logits == nullptr || sel_expert == nullptr || sel_weight == nullptr || tile_counts == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || experts == 0u || experts > kMoeRouterMaxExperts)
        return hipErrorInvalidValue;
    if (top_k == 0u || top_k > kMoeRouterMaxTopK || top_k > experts)
        return hipErrorInvalidValue;
    if (logits_row_stride < experts)
        return hipErrorInvalidValue;
    const uint32_t tiles = moe_slot_tile_count(rows);
    moe_router_kernel<<<rows, kRouterThreads, 0, stream>>>(
        logits, logits_row_stride, sel_expert, sel_weight, tile_counts, rows, experts, top_k, tiles);
    return hipGetLastError();
}

hipError_t launch_moe_slot_scan(
    const uint32_t* tile_counts,
    uint32_t* expert_row_begin,
    uint32_t* tile_base,
    uint32_t* expert_group_begin,
    uint32_t* group_expert,
    uint32_t* group_rows,
    uint32_t* group_count,
    int32_t* row_token,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    hipStream_t stream) {
    if (tile_counts == nullptr || expert_row_begin == nullptr || tile_base == nullptr ||
        expert_group_begin == nullptr || group_expert == nullptr || group_rows == nullptr ||
        group_count == nullptr || row_token == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || experts == 0u || experts > kMoeRouterMaxExperts)
        return hipErrorInvalidValue;
    if (top_k == 0u || top_k > kMoeRouterMaxTopK || top_k > experts)
        return hipErrorInvalidValue;
    const uint32_t tiles = moe_slot_tile_count(rows);
    const uint32_t group_capacity = moe_group_count_upper_bound(rows, experts, top_k);
    moe_slot_scan_kernel<<<1, kScanThreads, 0, stream>>>(
        tile_counts, expert_row_begin, tile_base, expert_group_begin, group_expert, group_rows,
        group_count, row_token, experts, tiles, group_capacity);
    return hipGetLastError();
}

hipError_t launch_moe_slot_scatter(
    const int32_t* sel_expert,
    const uint32_t* tile_base,
    const uint32_t* expert_row_begin,
    const uint32_t* expert_group_begin,
    int32_t* row_token,
    int32_t* slot_row_of,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    hipStream_t stream) {
    if (sel_expert == nullptr || tile_base == nullptr || expert_row_begin == nullptr ||
        expert_group_begin == nullptr || row_token == nullptr || slot_row_of == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || experts == 0u || experts > kMoeRouterMaxExperts)
        return hipErrorInvalidValue;
    if (top_k == 0u || top_k > kMoeRouterMaxTopK || top_k > experts)
        return hipErrorInvalidValue;
    const uint32_t tiles = moe_slot_tile_count(rows);
    moe_slot_scatter_kernel<<<tiles, kRouterThreads, 0, stream>>>(
        sel_expert, tile_base, expert_row_begin, expert_group_begin, row_token, slot_row_of, rows,
        experts, top_k, tiles);
    return hipGetLastError();
}

}  // namespace shelf::kernel
