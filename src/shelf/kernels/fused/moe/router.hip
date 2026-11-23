#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/kernels/fused/moe/router.h>
#include <shelf/runtime/program/device_program.h>

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
    uint32_t tiles,
    uint32_t expert_begin,
    uint32_t owned_experts,
    uint32_t* error_word,
    uint32_t* route_freq) {
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
    uint32_t non_finite = 0u;
    float best = -INFINITY;
    for (uint32_t e = tid; e < experts; e += kRouterThreads) {
        // A non-finite logit is a numeric anomaly, and this row is reported for it. It may not
        // win the selection either: a winner taken from NaN probabilities falls back to the
        // search sentinel, and an expert index outside [0, experts) walks the plan arrays and
        // the scatter past their capacity.
        const float raw = row[e];
        const bool finite = isfinite(raw);
        if (!finite)
            ++non_finite;
        const float x = finite ? raw : -INFINITY;
        idx[n] = e;
        v[n] = x;
        ++n;
        if (x > best)
            best = x;
    }

    __shared__ float s_val[kRouterWaves];
    __shared__ int32_t s_idx[kRouterWaves];
    __shared__ uint32_t s_non_finite[kRouterWaves];
    __shared__ int32_t s_win_idx[kMoeRouterMaxTopK];
    __shared__ float s_win_val[kMoeRouterMaxTopK];

    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        non_finite += __shfl_xor_sync(0xffffffffULL, non_finite, off);
    if (lane == 0u)
        s_non_finite[wave] = non_finite;
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        best = fmaxf(best, __shfl_xor_sync(0xffffffffULL, best, off));
    if (lane == 0u)
        s_val[wave] = best;
    __syncthreads();
    uint32_t row_non_finite = s_non_finite[0];
    #pragma unroll
    for (uint32_t w = 1u; w < kRouterWaves; ++w)
        row_non_finite += s_non_finite[w];
    if (row_non_finite != 0u && tid == 0u && error_word != nullptr) {
        atomicCAS(error_word,
                  static_cast<uint32_t>(::shelf::runtime::ProgramStatus::COMPLETE),
                  static_cast<uint32_t>(::shelf::runtime::ProgramStatus::NUMERIC_ERROR));
    }
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
        // The weight is renormalized over the whole selection on every rank. A rank that owns
        // only part of the selection therefore contributes part of the same sum, and the
        // combine outputs of the ranks add to what one device computes.
        const bool owned = static_cast<uint32_t>(e) >= expert_begin &&
                           static_cast<uint32_t>(e) < expert_begin + owned_experts;
        const float w = sel_sum > 0.0f ? s_win_val[tid] / sel_sum : 0.0f;
        sel_expert[static_cast<size_t>(t) * top_k + tid] =
            owned ? e - static_cast<int32_t>(expert_begin) : kMoeSlotNotOwned;
        sel_weight[static_cast<size_t>(t) * top_k + tid] = w;
        if (owned) {
            const uint32_t local = static_cast<uint32_t>(e - expert_begin);
            atomicAdd(&tile_counts[static_cast<size_t>(local) * tiles + tile], 1u);
            if (route_freq != nullptr) atomicAdd(&route_freq[local], 1u);
        }
    }
}

__device__ __forceinline__ void moe_slot_scan_body(
    uint32_t* __restrict__ tile_counts,
    uint32_t* __restrict__ expert_row_begin,
    uint32_t* __restrict__ tile_base,
    uint32_t* __restrict__ expert_group_begin,
    uint32_t* __restrict__ group_expert,
    uint32_t* __restrict__ group_rows,
    uint32_t* __restrict__ group_count,
    int32_t* __restrict__ row_token,
    const int32_t* __restrict__ expert_slot,
    uint32_t* __restrict__ hit_slot,
    uint32_t* __restrict__ hit_rows,
    uint32_t* __restrict__ hit_base,
    uint32_t* __restrict__ hit_count,
    uint32_t* __restrict__ miss_slot,
    uint32_t* __restrict__ miss_rows,
    uint32_t* __restrict__ miss_base,
    uint32_t* __restrict__ miss_count,
    uint32_t* __restrict__ hit_expert_group_begin,
    uint32_t* __restrict__ miss_expert_group_begin,
    uint32_t experts,
    uint32_t tiles,
    uint32_t group_capacity) {
    const uint32_t tid = threadIdx.x;
    // thread 0 alone would serialize experts * tiles dependent loads before it can lay out
    // any row, which costs more than the scan itself at decode. Sum the per-expert totals
    // with all threads into LDS first; the layout below then only reads LDS and stores.
    __shared__ uint32_t s_total[kMoeRouterMaxExperts];
    for (uint32_t e = tid; e < experts; e += kScanThreads) {
        uint32_t expert_total = 0u;
        for (uint32_t tl = 0u; tl < tiles; ++tl)
            expert_total += tile_counts[static_cast<size_t>(e) * tiles + tl];
        s_total[e] = expert_total;
    }
    __syncthreads();
    __shared__ uint32_t s_rows[kScanThreads];
    __shared__ uint32_t s_all[kScanThreads];
    __shared__ uint32_t s_hit[kScanThreads];
    __shared__ uint32_t s_miss[kScanThreads];
    __shared__ uint32_t s_slot_hit[kMoeRouterMaxExperts];
    __shared__ uint32_t s_slot_miss[kMoeRouterMaxExperts];
    const bool split = hit_count != nullptr;
    constexpr uint32_t kPerThread = (kMoeRouterMaxExperts + kScanThreads - 1u) / kScanThreads;
    for (uint32_t e = tid; e < kMoeRouterMaxExperts; e += kScanThreads) {
        s_slot_hit[e] = ~0u;
        s_slot_miss[e] = ~0u;
    }
    uint32_t local_rows = 0u;
    uint32_t local_all = 0u;
    uint32_t local_hit = 0u;
    uint32_t local_miss = 0u;
    for (uint32_t q = 0u; q < kPerThread; ++q) {
        const uint32_t e = tid * kPerThread + q;
        if (e >= experts) break;
        const uint32_t expert_total = s_total[e];
        const uint32_t groups_here = expert_total / kGroupRows +
            (expert_total % kGroupRows != 0u ? 1u : 0u);
        const int32_t location = expert_slot != nullptr ? expert_slot[e]
                                                        : static_cast<int32_t>(e);
        local_rows += expert_total;
        local_all += groups_here;
        if (split) {
            if (location >= 0) local_hit += groups_here;
            else local_miss += groups_here;
        }
    }
    if (tid < kScanThreads) {
        s_rows[tid] = local_rows;
        s_all[tid] = local_all;
        s_hit[tid] = local_hit;
        s_miss[tid] = local_miss;
    }
    __syncthreads();
    for (uint32_t off = 1u; off < kScanThreads; off <<= 1u) {
        const uint32_t r = tid >= off ? s_rows[tid - off] : 0u;
        const uint32_t a = tid >= off ? s_all[tid - off] : 0u;
        const uint32_t hh = tid >= off ? s_hit[tid - off] : 0u;
        const uint32_t mm = tid >= off ? s_miss[tid - off] : 0u;
        __syncthreads();
        s_rows[tid] += r;
        s_all[tid] += a;
        s_hit[tid] += hh;
        s_miss[tid] += mm;
        __syncthreads();
    }
    uint32_t rows_off = tid == 0u ? 0u : s_rows[tid - 1u];
    uint32_t group_off = tid == 0u ? 0u : s_all[tid - 1u];
    uint32_t hit_off = tid == 0u ? 0u : s_hit[tid - 1u];
    uint32_t miss_off = tid == 0u ? 0u : s_miss[tid - 1u];
    for (uint32_t q = 0u; q < kPerThread; ++q) {
        const uint32_t e = tid * kPerThread + q;
        if (e >= experts) break;
        const uint32_t expert_total = s_total[e];
        const int32_t location = expert_slot != nullptr ? expert_slot[e]
                                                        : static_cast<int32_t>(e);
        expert_row_begin[e] = rows_off;
        expert_group_begin[e] = group_off < group_capacity ? group_off : group_capacity;
        uint32_t at = rows_off;
        for (uint32_t tl = 0u; tl < tiles; ++tl) {
            tile_base[static_cast<size_t>(e) * tiles + tl] = at;
            at += tile_counts[static_cast<size_t>(e) * tiles + tl];
        }
        uint32_t gidx = group_off;
        for (uint32_t done = 0u; done < expert_total; done += kGroupRows) {
            if (gidx >= group_capacity) break;
            const uint32_t rows_here =
                expert_total - done < kGroupRows ? expert_total - done : kGroupRows;
            group_expert[gidx] = e;
            group_rows[gidx] = rows_here;
            if (split) {
                if (location >= 0) {
                    const uint32_t slot = static_cast<uint32_t>(location);
                    hit_slot[hit_off] = slot;
                    hit_rows[hit_off] = rows_here;
                    hit_base[hit_off] = gidx;
                    atomicMin(&s_slot_hit[slot], hit_off);
                    ++hit_off;
                } else {
                    const uint32_t slot = static_cast<uint32_t>(-1 - location);
                    miss_slot[miss_off] = slot;
                    miss_rows[miss_off] = rows_here;
                    miss_base[miss_off] = gidx;
                    atomicMin(&s_slot_miss[slot], miss_off);
                    ++miss_off;
                }
            }
            ++gidx;
        }
        rows_off += expert_total;
        group_off += expert_total / kGroupRows +
            (expert_total % kGroupRows != 0u ? 1u : 0u);
    }
    __syncthreads();
    for (uint32_t e = tid; e < experts; e += kScanThreads) {
        if (split) {
            if (s_slot_hit[e] != ~0u) hit_expert_group_begin[e] = s_slot_hit[e];
            if (s_slot_miss[e] != ~0u) miss_expert_group_begin[e] = s_slot_miss[e];
        }
    }
    if (tid == 0u) {
        expert_row_begin[experts] = s_rows[kScanThreads - 1u];
        const uint32_t groups_total = s_all[kScanThreads - 1u];
        expert_group_begin[experts] = groups_total < group_capacity ? groups_total
                                                                   : group_capacity;
        group_count[0] = expert_group_begin[experts];
        if (split) {
            hit_count[0] = s_hit[kScanThreads - 1u];
            miss_count[0] = s_miss[kScanThreads - 1u];
        }
    }
    __syncthreads();

    // The router accumulates these counters, so a round that consumed them has to leave
    // them at zero: a model with several MoE layers shares one plan scratch, and the host
    // cannot zero them per layer without one more launch per layer.
    const uint32_t counts_here = experts * tiles;
    for (uint32_t c = tid; c < counts_here; c += kScanThreads) tile_counts[c] = 0u;

    // The activation producer launches over the host capacity, so every row that no group
    // claims has to say so, not just the tail past the last active group.
    const uint32_t rows_here = group_capacity * kGroupRows;
    for (uint32_t r = tid; r < rows_here; r += kScanThreads)
        row_token[r] = -1;
}

__global__ void __launch_bounds__(kScanThreads) moe_slot_scan_kernel(
    uint32_t* __restrict__ tile_counts,
    uint32_t* __restrict__ expert_row_begin,
    uint32_t* __restrict__ tile_base,
    uint32_t* __restrict__ expert_group_begin,
    uint32_t* __restrict__ group_expert,
    uint32_t* __restrict__ group_rows,
    uint32_t* __restrict__ group_count,
    int32_t* __restrict__ row_token,
    const int32_t* __restrict__ expert_slot,
    uint32_t* __restrict__ hit_slot,
    uint32_t* __restrict__ hit_rows,
    uint32_t* __restrict__ hit_base,
    uint32_t* __restrict__ hit_count,
    uint32_t* __restrict__ miss_slot,
    uint32_t* __restrict__ miss_rows,
    uint32_t* __restrict__ miss_base,
    uint32_t* __restrict__ miss_count,
    uint32_t* __restrict__ hit_expert_group_begin,
    uint32_t* __restrict__ miss_expert_group_begin,
    uint32_t experts,
    uint32_t tiles,
    uint32_t group_capacity) {
    moe_slot_scan_body(
        tile_counts, expert_row_begin, tile_base, expert_group_begin, group_expert,
        group_rows, group_count, row_token, expert_slot, hit_slot, hit_rows, hit_base,
        hit_count, miss_slot, miss_rows, miss_base, miss_count, hit_expert_group_begin,
        miss_expert_group_begin, experts, tiles, group_capacity);
}

__device__ __forceinline__ void moe_slot_scatter_body(
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
    for (uint32_t e = tid; e < experts; e += blockDim.x) {
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
            const int32_t sel = sel_expert[j];
            if (sel < 0) {
                slot_row_of[j] = sel;
                continue;
            }
            const uint32_t e = static_cast<uint32_t>(sel);
            const uint32_t slot = atomicAdd(&s_base[e], 1u);
            const uint32_t rank = slot - s_first[e];
            const uint32_t row = (s_group[e] + rank / kGroupRows) * kGroupRows + rank % kGroupRows;
            row_token[row] = static_cast<int32_t>(t);
            slot_row_of[j] = static_cast<int32_t>(row);
        }
    }
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
    moe_slot_scatter_body(
        sel_expert, tile_base, expert_row_begin, expert_group_begin, row_token, slot_row_of,
        rows, experts, top_k, tiles);
}

__global__ void __launch_bounds__(kScanThreads) moe_slot_scan_scatter_kernel(
    uint32_t* __restrict__ tile_counts,
    uint32_t* __restrict__ expert_row_begin,
    uint32_t* __restrict__ tile_base,
    uint32_t* __restrict__ expert_group_begin,
    uint32_t* __restrict__ group_expert,
    uint32_t* __restrict__ group_rows,
    uint32_t* __restrict__ group_count,
    int32_t* __restrict__ row_token,
    const int32_t* __restrict__ expert_slot,
    uint32_t* __restrict__ hit_slot,
    uint32_t* __restrict__ hit_rows,
    uint32_t* __restrict__ hit_base,
    uint32_t* __restrict__ hit_count,
    uint32_t* __restrict__ miss_slot,
    uint32_t* __restrict__ miss_rows,
    uint32_t* __restrict__ miss_base,
    uint32_t* __restrict__ miss_count,
    uint32_t* __restrict__ hit_expert_group_begin,
    uint32_t* __restrict__ miss_expert_group_begin,
    uint32_t experts,
    uint32_t tiles,
    uint32_t group_capacity,
    const int32_t* __restrict__ sel_expert,
    int32_t* __restrict__ slot_row_of,
    uint32_t rows,
    uint32_t top_k) {
    moe_slot_scan_body(
        tile_counts, expert_row_begin, tile_base, expert_group_begin, group_expert,
        group_rows, group_count, row_token, expert_slot, hit_slot, hit_rows, hit_base,
        hit_count, miss_slot, miss_rows, miss_base, miss_count, hit_expert_group_begin,
        miss_expert_group_begin, experts, tiles, group_capacity);
    __syncthreads();
    moe_slot_scatter_body(
        sel_expert, tile_base, expert_row_begin, expert_group_begin, row_token, slot_row_of,
        rows, experts, top_k, 1u);
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
    uint32_t expert_begin,
    uint32_t owned_experts,
    uint32_t* error_word,
    hipStream_t stream,
    uint32_t* route_freq) {
    if (logits == nullptr || sel_expert == nullptr || sel_weight == nullptr || tile_counts == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || experts == 0u || experts > kMoeRouterMaxExperts)
        return hipErrorInvalidValue;
    if (top_k == 0u || top_k > kMoeRouterMaxTopK || top_k > experts)
        return hipErrorInvalidValue;
    if (owned_experts == 0u || owned_experts > experts)
        return hipErrorInvalidValue;
    if (expert_begin > experts - owned_experts)
        return hipErrorInvalidValue;
    if (logits_row_stride < experts)
        return hipErrorInvalidValue;
    const uint32_t tiles = moe_slot_tile_count(rows);
    moe_router_kernel<<<rows, kRouterThreads, 0, stream>>>(
        logits, logits_row_stride, sel_expert, sel_weight, tile_counts, rows, experts, top_k,
        tiles, expert_begin, owned_experts, error_word, route_freq);
    return hipGetLastError();
}

hipError_t launch_moe_slot_scan_split(
    uint32_t* tile_counts,
    uint32_t* expert_row_begin,
    uint32_t* tile_base,
    uint32_t* expert_group_begin,
    uint32_t* group_expert,
    uint32_t* group_rows,
    uint32_t* group_count,
    int32_t* row_token,
    const int32_t* expert_slot,
    uint32_t* hit_slot,
    uint32_t* hit_rows,
    uint32_t* hit_base,
    uint32_t* hit_count,
    uint32_t* miss_slot,
    uint32_t* miss_rows,
    uint32_t* miss_base,
    uint32_t* miss_count,
    uint32_t* hit_expert_group_begin,
    uint32_t* miss_expert_group_begin,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    hipStream_t stream) {
    if (tile_counts == nullptr || expert_row_begin == nullptr || tile_base == nullptr ||
        expert_group_begin == nullptr || group_expert == nullptr || group_rows == nullptr ||
        group_count == nullptr || row_token == nullptr || hit_slot == nullptr ||
        hit_rows == nullptr || hit_base == nullptr || hit_count == nullptr ||
        miss_slot == nullptr || miss_rows == nullptr || miss_base == nullptr ||
        miss_count == nullptr)
        return hipErrorInvalidValue;
    if (hit_count != nullptr &&
        (hit_expert_group_begin == nullptr || miss_expert_group_begin == nullptr))
        return hipErrorInvalidValue;
    if (rows == 0u || experts == 0u || experts > kMoeRouterMaxExperts)
        return hipErrorInvalidValue;
    if (top_k == 0u || top_k > kMoeRouterMaxTopK || top_k > experts)
        return hipErrorInvalidValue;
    const uint32_t tiles = moe_slot_tile_count(rows);
    const uint32_t group_capacity = moe_group_count_upper_bound(rows, experts, top_k);
    moe_slot_scan_kernel<<<1, kScanThreads, 0, stream>>>(
        tile_counts, expert_row_begin, tile_base, expert_group_begin, group_expert, group_rows,
        group_count, row_token, expert_slot, hit_slot, hit_rows, hit_base, hit_count,
        miss_slot, miss_rows, miss_base, miss_count, hit_expert_group_begin,
        miss_expert_group_begin, experts, tiles, group_capacity);
    return hipGetLastError();
}

hipError_t launch_moe_slot_scan(
    uint32_t* tile_counts,
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
        group_count, row_token, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, experts, tiles, group_capacity);
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

hipError_t launch_moe_slot_scan_scatter_split(
    uint32_t* tile_counts,
    uint32_t* expert_row_begin,
    uint32_t* tile_base,
    uint32_t* expert_group_begin,
    uint32_t* group_expert,
    uint32_t* group_rows,
    uint32_t* group_count,
    int32_t* row_token,
    const int32_t* expert_slot,
    uint32_t* hit_slot,
    uint32_t* hit_rows,
    uint32_t* hit_base,
    uint32_t* hit_count,
    uint32_t* miss_slot,
    uint32_t* miss_rows,
    uint32_t* miss_base,
    uint32_t* miss_count,
    uint32_t* hit_expert_group_begin,
    uint32_t* miss_expert_group_begin,
    const int32_t* sel_expert,
    int32_t* slot_row_of,
    uint32_t rows,
    uint32_t experts,
    uint32_t top_k,
    hipStream_t stream) {
    if (tile_counts == nullptr || expert_row_begin == nullptr || tile_base == nullptr ||
        expert_group_begin == nullptr || group_expert == nullptr || group_rows == nullptr ||
        group_count == nullptr || row_token == nullptr || sel_expert == nullptr ||
        slot_row_of == nullptr)
        return hipErrorInvalidValue;
    if (rows == 0u || experts == 0u || experts > kMoeRouterMaxExperts)
        return hipErrorInvalidValue;
    if (top_k == 0u || top_k > kMoeRouterMaxTopK || top_k > experts)
        return hipErrorInvalidValue;
    if (moe_slot_tile_count(rows) != 1u) return hipErrorNotSupported;
    const uint32_t group_capacity = moe_group_count_upper_bound(rows, experts, top_k);
    moe_slot_scan_scatter_kernel<<<1, kScanThreads, 0, stream>>>(
        tile_counts, expert_row_begin, tile_base, expert_group_begin, group_expert, group_rows,
        group_count, row_token, expert_slot, hit_slot, hit_rows, hit_base, hit_count,
        miss_slot, miss_rows, miss_base, miss_count, hit_expert_group_begin,
        miss_expert_group_begin, experts, 1u, group_capacity, sel_expert, slot_row_of, rows,
        top_k);
    return hipGetLastError();
}

}  // namespace shelf::kernel
