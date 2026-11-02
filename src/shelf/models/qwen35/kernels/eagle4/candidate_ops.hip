#include <shelf/models/qwen35/kernels/eagle4/candidate_ops.h>
#include <shelf/kernels/optimized/detail/vector_io.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

namespace {

using bf16 = ::ps::kernel::detail::bf16;

constexpr uint32_t kRoughThreads = 256u;
constexpr uint32_t kRoughWarps = kRoughThreads / 32u;
constexpr uint32_t kRoughRowsMax = 16u;
constexpr uint32_t kRoughRankMax = 256u;
constexpr uint32_t kRankWarps = 4u;

constexpr uint32_t kLogitThreads = 128u;
constexpr uint32_t kLogitWarps = kLogitThreads / 32u;

__device__ __forceinline__ float eagle4_warp_reduce_sum(float value) {
    #pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFFFFFFFFFFull, value, offset);
    }
    return value;
}

__global__ void eagle4_rank_project_kernel(
    const bf16* hidden,
    uint32_t hidden_row_stride,
    const bf16* down_proj,
    bf16* projected,
    uint32_t rows,
    uint32_t rank,
    uint32_t hidden_size) {
    const uint32_t row = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t feature = blockIdx.x * kRankWarps + warp;
    if (row >= rows || feature >= rank) return;
    const bf16* weight_row = down_proj + static_cast<std::size_t>(feature) * hidden_size;
    const bf16* hidden_row = hidden + static_cast<std::size_t>(row) * hidden_row_stride;
    float acc = 0.0f;
    for (uint32_t d = lane * 8u; d < hidden_size; d += 32u * 8u) {
        const uint32_t remaining = hidden_size - d;
        if (remaining >= 8u) {
            const uint4 wv = *reinterpret_cast<const uint4*>(weight_row + d);
            const uint4 hv = *reinterpret_cast<const uint4*>(hidden_row + d);
            const bf16* wp = reinterpret_cast<const bf16*>(&wv);
            const bf16* hp = reinterpret_cast<const bf16*>(&hv);
            #pragma unroll
            for (uint32_t e = 0; e < 8u; ++e) {
                acc += __bfloat162float(wp[e]) * __bfloat162float(hp[e]);
            }
        } else {
            for (uint32_t e = 0; e < remaining; ++e) {
                acc += __bfloat162float(weight_row[d + e]) *
                       __bfloat162float(hidden_row[d + e]);
            }
        }
    }
    acc = eagle4_warp_reduce_sum(acc);
    if (lane == 0u) {
        projected[static_cast<std::size_t>(row) * rank + feature] =
            __float2bfloat16(acc);
    }
}

__global__ void eagle4_rough_scores_kernel(
    const bf16* projected,
    uint32_t projected_row_stride,
    const bf16* score_head,
    float* scores,
    uint32_t rows,
    uint32_t rank,
    uint32_t vocab) {
    extern __shared__ bf16 s_projected[];
    for (uint32_t index = threadIdx.x; index < rows * projected_row_stride;
         index += blockDim.x) {
        const uint32_t r = index / projected_row_stride;
        const uint32_t c = index % projected_row_stride;
        s_projected[index] = c < rank ? projected[static_cast<std::size_t>(r) *
                                                 projected_row_stride + c]
                                      : __float2bfloat16(0.0f);
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t v = blockIdx.x * kRoughWarps + warp;
    if (v >= vocab) return;
    const bf16* head_row = score_head + static_cast<std::size_t>(v) * rank;
    for (uint32_t r = 0; r < rows; ++r) {
        const bf16* u_row = s_projected + r * projected_row_stride;
        float acc = 0.0f;
        for (uint32_t j = lane; j < rank; j += 32u) {
            acc += __bfloat162float(u_row[j]) * __bfloat162float(head_row[j]);
        }
        acc = eagle4_warp_reduce_sum(acc);
        if (lane == 0u) {
            scores[static_cast<std::size_t>(r) * vocab + v] =
                __bfloat162float(__float2bfloat16(acc));
        }
    }
}

__global__ void eagle4_candidate_logits_kernel(
    const bf16* lm_head,
    uint32_t lm_head_row_stride,
    const bf16* hidden,
    uint32_t hidden_row_stride,
    const int32_t* candidate_ids,
    uint32_t rows,
    uint32_t candidates,
    uint32_t hidden_size,
    float* logits) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y * kLogitWarps + (threadIdx.x >> 5u);
    if (row >= rows || slot >= candidates) return;
    const uint32_t lane = threadIdx.x & 31u;
    const int32_t id = candidate_ids[static_cast<std::size_t>(row) * candidates + slot];
    if (id < 0) {
        if (lane == 0u) {
            logits[static_cast<std::size_t>(row) * candidates + slot] = -INFINITY;
        }
        return;
    }
    const bf16* weight_row =
        lm_head + static_cast<std::size_t>(id) * lm_head_row_stride;
    const bf16* hidden_row = hidden + static_cast<std::size_t>(row) * hidden_row_stride;
    float acc = 0.0f;
    for (uint32_t d = lane; d < hidden_size; d += 32u) {
        acc += __bfloat162float(weight_row[d]) * __bfloat162float(hidden_row[d]);
    }
    acc = eagle4_warp_reduce_sum(acc);
    if (lane == 0u) {
        logits[static_cast<std::size_t>(row) * candidates + slot] = acc;
    }
}

__device__ __forceinline__ void eagle4_candidate_select(
    float value,
    int32_t id,
    float* s_scratch,
    float* out_value,
    int32_t* out_id) {
    s_scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step > 0u; step >>= 1u) {
        if (threadIdx.x < step) {
            s_scratch[threadIdx.x] = fmaxf(s_scratch[threadIdx.x], s_scratch[threadIdx.x + step]);
        }
        __syncthreads();
    }
    const float max_value = s_scratch[0];
    __syncthreads();
    s_scratch[threadIdx.x] =
        value == max_value ? static_cast<float>(id) : static_cast<float>(0x7FFFFFFF);
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step > 0u; step >>= 1u) {
        if (threadIdx.x < step) {
            s_scratch[threadIdx.x] = fminf(s_scratch[threadIdx.x], s_scratch[threadIdx.x + step]);
        }
        __syncthreads();
    }
    *out_value = max_value;
    *out_id = static_cast<int32_t>(s_scratch[0]);
    __syncthreads();
}

__global__ void eagle4_candidate_topk_kernel(
    const float* logits,
    const int32_t* candidate_ids,
    uint32_t rows,
    uint32_t candidates,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    float* out_probs) {
    __shared__ float s_scratch[kEagle4CandidateThreads];

    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t tid = threadIdx.x;
    const float* row_logits = logits + static_cast<std::size_t>(row) * candidates;
    const int32_t* row_ids = candidate_ids + static_cast<std::size_t>(row) * candidates;

    float value = tid < candidates ? row_logits[tid] : -INFINITY;
    int32_t id = tid < candidates ? row_ids[tid] : 0x7FFFFFFF;

    s_scratch[tid] = value;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step > 0u; step >>= 1u) {
        if (tid < step) {
            s_scratch[tid] = fmaxf(s_scratch[tid], s_scratch[tid + step]);
        }
        __syncthreads();
    }
    const float row_max = s_scratch[0];
    float local_sum = 0.0f;
    if (tid < candidates) local_sum = __expf(value - row_max);
    s_scratch[tid] = local_sum;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step > 0u; step >>= 1u) {
        if (tid < step) {
            s_scratch[tid] += s_scratch[tid + step];
        }
        __syncthreads();
    }
    const float row_sum = s_scratch[0];

    float best = value;
    int32_t best_id = id;
    for (uint32_t rank_index = 0; rank_index < k; ++rank_index) {
        float chosen_value = -INFINITY;
        int32_t chosen_id = 0x7FFFFFFF;
        eagle4_candidate_select(best, best_id, s_scratch, &chosen_value, &chosen_id);
        if (tid == 0u) {
            out_ids[static_cast<std::size_t>(row) * k + rank_index] = chosen_id;
            out_logits[static_cast<std::size_t>(row) * k + rank_index] = chosen_value;
            out_probs[static_cast<std::size_t>(row) * k + rank_index] =
                chosen_value > -INFINITY ? __expf(chosen_value - row_max) / row_sum : 0.0f;
        }
        if (best == chosen_value && best_id == chosen_id) {
            best = -INFINITY;
            best_id = 0x7FFFFFFF;
        }
        __syncthreads();
    }
}

}  // namespace

hipError_t launch_eagle4_rank_project_bf16(
    const bf16_t* hidden,
    uint32_t hidden_row_stride,
    const bf16_t* down_proj,
    bf16_t* projected,
    uint32_t rows,
    uint32_t rank,
    uint32_t hidden_size,
    hipStream_t stream) {
    if (hidden == nullptr || down_proj == nullptr || projected == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || rank == 0u || hidden_size == 0u) return hipErrorInvalidValue;
    if (hidden_row_stride < hidden_size) return hipErrorInvalidValue;
    const dim3 grid((rank + kRankWarps - 1u) / kRankWarps, rows, 1u);
    eagle4_rank_project_kernel<<<grid, kEagle4CandidateThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(hidden), hidden_row_stride,
        reinterpret_cast<const bf16*>(down_proj), reinterpret_cast<bf16*>(projected), rows,
        rank, hidden_size);
    return hipGetLastError();
}

hipError_t launch_eagle4_rough_scores_bf16(
    const bf16_t* projected,
    uint32_t projected_row_stride,
    const bf16_t* score_head,
    float* scores,
    uint32_t rows,
    uint32_t rank,
    uint32_t vocab,
    hipStream_t stream) {
    if (projected == nullptr || score_head == nullptr || scores == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || vocab == 0u || rank == 0u) return hipErrorInvalidValue;
    if (rows > kRoughRowsMax || rank > kRoughRankMax) return hipErrorNotSupported;
    if (projected_row_stride < rank) return hipErrorInvalidValue;
    const std::size_t shared_bytes =
        static_cast<std::size_t>(rows) * projected_row_stride * sizeof(bf16);
    if (shared_bytes > 24u * 1024u) return hipErrorInvalidValue;
    const uint32_t grid = (vocab + kRoughWarps - 1u) / kRoughWarps;
    eagle4_rough_scores_kernel<<<grid, kRoughThreads, shared_bytes, stream>>>(
        reinterpret_cast<const bf16*>(projected), projected_row_stride,
        reinterpret_cast<const bf16*>(score_head), scores, rows, rank, vocab);
    return hipGetLastError();
}

hipError_t launch_eagle4_candidate_logits(
    const bf16_t* lm_head,
    uint32_t lm_head_row_stride,
    const bf16_t* hidden,
    uint32_t hidden_row_stride,
    const int32_t* candidate_ids,
    uint32_t rows,
    uint32_t candidates,
    uint32_t hidden_size,
    float* logits,
    hipStream_t stream) {
    if (lm_head == nullptr || hidden == nullptr || candidate_ids == nullptr ||
        logits == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || candidates == 0u || hidden_size == 0u) return hipErrorInvalidValue;
    if (candidates % kLogitWarps != 0u) return hipErrorInvalidValue;
    const dim3 grid(rows, candidates / kLogitWarps, 1u);
    eagle4_candidate_logits_kernel<<<grid, kLogitThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(lm_head), lm_head_row_stride,
        reinterpret_cast<const bf16*>(hidden), hidden_row_stride, candidate_ids, rows,
        candidates, hidden_size, logits);
    return hipGetLastError();
}

hipError_t launch_eagle4_candidate_topk(
    const float* logits,
    const int32_t* candidate_ids,
    uint32_t rows,
    uint32_t candidates,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    float* out_probs,
    hipStream_t stream) {
    if (logits == nullptr || candidate_ids == nullptr || out_ids == nullptr ||
        out_logits == nullptr || out_probs == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || candidates == 0u || k == 0u || k > candidates) {
        return hipErrorInvalidValue;
    }
    if (candidates > kEagle4CandidateThreads) return hipErrorNotSupported;
    eagle4_candidate_topk_kernel<<<rows, kEagle4CandidateThreads, 0, stream>>>(
        logits, candidate_ids, rows, candidates, k, out_ids, out_logits, out_probs);
    return hipGetLastError();
}

}  // namespace ps::kernel
