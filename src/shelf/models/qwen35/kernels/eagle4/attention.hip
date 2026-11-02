#include <shelf/models/qwen35/kernels/eagle4/attention.h>
#include <shelf/kernels/optimized/detail/vector_io.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

namespace {

using bf16 = ::ps::kernel::detail::bf16;

constexpr uint32_t kEagle4AttentionSharedBytesMax = 48u * 1024u;

__device__ __forceinline__ float eagle4_attention_reduce_max(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step > 0u; step >>= 1u) {
        if (threadIdx.x < step) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + step]);
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float eagle4_attention_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t step = blockDim.x / 2u; step > 0u; step >>= 1u) {
        if (threadIdx.x < step) {
            scratch[threadIdx.x] += scratch[threadIdx.x + step];
        }
        __syncthreads();
    }
    return scratch[0];
}

__device__ __forceinline__ float eagle4_dot_bf16(
    const float* q, const bf16* k, uint32_t head_dim) {
    const uint4* k_vec = reinterpret_cast<const uint4*>(k);
    constexpr uint32_t kElementsPerVec = sizeof(uint4) / sizeof(bf16);
    const uint32_t vecs = head_dim / kElementsPerVec;
    float acc = 0.0f;
    for (uint32_t v = 0; v < vecs; ++v) {
        const uint4 raw = k_vec[v];
        const bf16* packed = reinterpret_cast<const bf16*>(&raw);
        const uint32_t base = v * kElementsPerVec;
        #pragma unroll
        for (uint32_t e = 0; e < kElementsPerVec; ++e) {
            acc += q[base + e] * __bfloat162float(packed[e]);
        }
    }
    return acc;
}

__global__ void eagle4_ring_attention_kernel(
    const bf16* q,
    uint32_t q_row_stride,
    const bf16* k_ring,
    const bf16* v_ring,
    uint32_t ring_capacity,
    uint32_t kv_features,
    uint32_t block_position_start,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    bf16* output,
    uint32_t output_row_stride) {
    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_q = s_scores + sliding_window;
    float* s_scratch = s_q + head_dim;

    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (row >= rows) return;

    const uint32_t tid = threadIdx.x;
    const int64_t qpos = static_cast<int64_t>(block_position_start) + row;
    const int64_t lo = qpos - static_cast<int64_t>(sliding_window) + 1 > 0
                           ? qpos - static_cast<int64_t>(sliding_window) + 1
                           : 0;
    const uint32_t nkeys = static_cast<uint32_t>(qpos - lo + 1u);

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __bfloat162float(
            q[static_cast<std::size_t>(row) * q_row_stride + head * head_dim + d]);
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (uint32_t k = tid; k < nkeys; k += blockDim.x) {
        const uint32_t slot =
            static_cast<uint32_t>((lo + static_cast<int64_t>(k)) % ring_capacity);
        const bf16* krow = k_ring + static_cast<std::size_t>(slot) * kv_features +
                           static_cast<std::size_t>(head) * head_dim;
        const float score = eagle4_dot_bf16(s_q, krow, head_dim) * scale;
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    const float max_score = eagle4_attention_reduce_max(local_max, s_scratch);

    float local_sum = 0.0f;
    for (uint32_t k = tid; k < nkeys; k += blockDim.x) {
        const float e = __expf(s_scores[k] - max_score);
        s_scores[k] = e;
        local_sum += e;
    }
    const float sum = eagle4_attention_reduce_sum(local_sum, s_scratch);
    const float inv_sum = 1.0f / sum;

    float acc = 0.0f;
    if (tid < head_dim) {
        const uint32_t d = tid;
        for (uint32_t k = 0; k < nkeys; ++k) {
            const uint32_t slot =
                static_cast<uint32_t>((lo + static_cast<int64_t>(k)) % ring_capacity);
            const bf16* vrow = v_ring + static_cast<std::size_t>(slot) * kv_features +
                               static_cast<std::size_t>(head) * head_dim;
            acc += s_scores[k] * __bfloat162float(vrow[d]);
        }
        acc *= inv_sum;
        output[static_cast<std::size_t>(row) * output_row_stride + head * head_dim + d] =
            __float2bfloat16(acc);
    }
}

__global__ void eagle4_tree_attention_kernel(
    const bf16* q,
    uint32_t q_row_stride,
    const bf16* k_ring,
    const bf16* v_ring,
    uint32_t ring_capacity,
    uint32_t kv_features,
    const bf16* k_round,
    const bf16* v_round,
    uint32_t round_capacity,
    uint32_t committed_end,
    uint32_t row_position,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    const int32_t* ancestor_offsets,
    const int32_t* ancestor_nodes,
    bf16* output,
    uint32_t output_row_stride) {
    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_q = s_scores + sliding_window + (row_position - committed_end + 1u);
    float* s_scratch = s_q + head_dim;

    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    if (row >= rows) return;

    const uint32_t tid = threadIdx.x;
    const int64_t qpos = static_cast<int64_t>(row_position);
    const int64_t lo = qpos - static_cast<int64_t>(sliding_window) + 1 > 0
                           ? qpos - static_cast<int64_t>(sliding_window) + 1
                           : 0;
    const uint32_t n_committed =
        lo < static_cast<int64_t>(committed_end)
            ? static_cast<uint32_t>(static_cast<int64_t>(committed_end) - lo)
            : 0u;
    const int32_t anc_base = ancestor_offsets[row];
    const int32_t anc_count = ancestor_offsets[row + 1u] - anc_base;
    const int32_t anc_first =
        lo > static_cast<int64_t>(committed_end)
            ? static_cast<int32_t>(lo - static_cast<int64_t>(committed_end))
            : 0;
    const uint32_t n_anc =
        anc_count > anc_first ? static_cast<uint32_t>(anc_count - anc_first) : 0u;
    const uint32_t total = n_committed + n_anc;

    for (uint32_t d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = __bfloat162float(
            q[static_cast<std::size_t>(row) * q_row_stride + head * head_dim + d]);
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (uint32_t k = tid; k < total; k += blockDim.x) {
        const bf16* krow = nullptr;
        if (k < n_committed) {
            const uint32_t slot =
                static_cast<uint32_t>((lo + static_cast<int64_t>(k)) % ring_capacity);
            krow = k_ring + static_cast<std::size_t>(slot) * kv_features +
                   static_cast<std::size_t>(head) * head_dim;
        } else {
            const int32_t node = ancestor_nodes[anc_base + anc_first +
                                                static_cast<int32_t>(k - n_committed)];
            if (node < 0 || static_cast<uint32_t>(node) >= round_capacity) continue;
            krow = k_round + static_cast<std::size_t>(node) * kv_features +
                   static_cast<std::size_t>(head) * head_dim;
        }
        const float score = eagle4_dot_bf16(s_q, krow, head_dim) * scale;
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    const float max_score = eagle4_attention_reduce_max(local_max, s_scratch);

    float local_sum = 0.0f;
    for (uint32_t k = tid; k < total; k += blockDim.x) {
        const float e = __expf(s_scores[k] - max_score);
        s_scores[k] = e;
        local_sum += e;
    }
    const float sum = eagle4_attention_reduce_sum(local_sum, s_scratch);
    const float inv_sum = 1.0f / sum;

    float acc = 0.0f;
    if (tid < head_dim) {
        const uint32_t d = tid;
        for (uint32_t k = 0; k < total; ++k) {
            const bf16* vrow = nullptr;
            if (k < n_committed) {
                const uint32_t slot =
                    static_cast<uint32_t>((lo + static_cast<int64_t>(k)) % ring_capacity);
                vrow = v_ring + static_cast<std::size_t>(slot) * kv_features +
                       static_cast<std::size_t>(head) * head_dim;
            } else {
                const int32_t node = ancestor_nodes[anc_base + anc_first +
                                                    static_cast<int32_t>(k - n_committed)];
                vrow = v_round + static_cast<std::size_t>(node) * kv_features +
                       static_cast<std::size_t>(head) * head_dim;
            }
            acc += s_scores[k] * __bfloat162float(vrow[d]);
        }
        acc *= inv_sum;
        output[static_cast<std::size_t>(row) * output_row_stride + head * head_dim + d] =
            __float2bfloat16(acc);
    }
}

}  // namespace

hipError_t launch_eagle4_ring_attention_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ring,
    const bf16_t* v_ring,
    uint32_t ring_capacity,
    uint32_t kv_features,
    uint32_t block_position_start,
    uint32_t rows,
    uint32_t q_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    bf16_t* output,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (q == nullptr || k_ring == nullptr || v_ring == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || q_heads == 0u) {
        return hipErrorInvalidValue;
    }
    if (head_dim != kEagle4AttentionHeadDim) {
        return hipErrorNotSupported;
    }
    if (sliding_window == 0u || sliding_window > ring_capacity) {
        return hipErrorInvalidValue;
    }
    const std::size_t shared_bytes =
        static_cast<std::size_t>(sliding_window) * sizeof(float) +
        static_cast<std::size_t>(head_dim) * sizeof(float) +
        static_cast<std::size_t>(kEagle4AttentionThreads) * sizeof(float);
    if (shared_bytes > kEagle4AttentionSharedBytesMax) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(rows, q_heads, 1u);
    eagle4_ring_attention_kernel<<<grid, kEagle4AttentionThreads, shared_bytes, stream>>>(
        reinterpret_cast<const bf16*>(q), q_row_stride,
        reinterpret_cast<const bf16*>(k_ring), reinterpret_cast<const bf16*>(v_ring),
        ring_capacity, kv_features, block_position_start, rows, head_dim, sliding_window,
        scale, reinterpret_cast<bf16*>(output), output_row_stride);
    return hipGetLastError();
}

hipError_t launch_eagle4_tree_attention_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ring,
    const bf16_t* v_ring,
    uint32_t ring_capacity,
    uint32_t kv_features,
    const bf16_t* k_round,
    const bf16_t* v_round,
    uint32_t round_capacity,
    uint32_t committed_end,
    uint32_t row_position,
    uint32_t rows,
    uint32_t q_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    const int32_t* ancestor_offsets,
    const int32_t* ancestor_nodes,
    bf16_t* output,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (q == nullptr || k_ring == nullptr || v_ring == nullptr || k_round == nullptr ||
        v_round == nullptr || ancestor_offsets == nullptr || ancestor_nodes == nullptr ||
        output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || q_heads == 0u) {
        return hipErrorInvalidValue;
    }
    if (head_dim != kEagle4AttentionHeadDim) {
        return hipErrorNotSupported;
    }
    if (sliding_window == 0u || sliding_window > ring_capacity) {
        return hipErrorInvalidValue;
    }
    if (row_position < committed_end) {
        return hipErrorInvalidValue;
    }
    const std::size_t score_slots =
        static_cast<std::size_t>(sliding_window) +
        static_cast<std::size_t>(row_position - committed_end) + 1u;
    const std::size_t shared_bytes =
        score_slots * sizeof(float) + static_cast<std::size_t>(head_dim) * sizeof(float) +
        static_cast<std::size_t>(kEagle4AttentionThreads) * sizeof(float);
    if (shared_bytes > kEagle4AttentionSharedBytesMax) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(rows, q_heads, 1u);
    eagle4_tree_attention_kernel<<<grid, kEagle4AttentionThreads, shared_bytes, stream>>>(
        reinterpret_cast<const bf16*>(q), q_row_stride,
        reinterpret_cast<const bf16*>(k_ring), reinterpret_cast<const bf16*>(v_ring),
        ring_capacity, kv_features, reinterpret_cast<const bf16*>(k_round),
        reinterpret_cast<const bf16*>(v_round), round_capacity, committed_end, row_position,
        rows, head_dim, sliding_window, scale, ancestor_offsets, ancestor_nodes,
        reinterpret_cast<bf16*>(output), output_row_stride);
    return hipGetLastError();
}

}  // namespace ps::kernel
