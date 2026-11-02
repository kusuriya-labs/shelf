#include <shelf/models/qwen35/kernels/dflash2/attention.h>
#include <shelf/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace ps::kernel {

using namespace ::ps::kernel::detail;

constexpr uint32_t kAttentionThreads = 128u;
constexpr uint32_t kAttentionSharedBytesLimit = 32u * 1024u;

constexpr uint32_t kAttentionGqaGroup = 4u;
constexpr uint32_t kAttentionGqaTile = 32u;
constexpr uint32_t kAttentionGqaSharedBytesLimit = 60u * 1024u;

namespace {

struct RingAttentionView {
    const bf16* k_ring;
    const bf16* v_ring;
    uint32_t ring_capacity;
    uint32_t ring_row_stride;
    uint32_t context_rows;
    uint32_t context_start_position;
    uint32_t block_position_start;
    uint32_t block_rows;
};

__device__ float block_reduce_max(float value, float* scratch) {
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

__device__ float block_reduce_sum(float value, float* scratch) {
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

__global__ void dflash2_attention_kernel(
    const bf16* q,
    uint32_t q_row_stride,
    const bf16* k_ctx,
    const bf16* v_ctx,
    uint32_t context_rows,
    uint32_t context_row_stride,
    const bf16* k_noise,
    const bf16* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t context_position_start,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale) {
    extern __shared__ float smem[];
    float* sq = smem;
    float* scores = sq + head_dim;
    float* scratch = scores + context_rows + block_rows;

    const uint32_t row = blockIdx.y;
    const uint32_t head = blockIdx.x;
    if (row >= block_rows || head >= q_heads) return;

    const uint32_t keys = context_rows + block_rows;
    const uint32_t group = q_heads / kv_heads;
    const uint32_t kv_head = head / group;
    const int64_t query_position = static_cast<int64_t>(block_position_start) + row;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        sq[d] = bf16_to_f32(q[static_cast<std::size_t>(row) * q_row_stride +
                              static_cast<std::size_t>(head) * head_dim + d]);
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) {
        const bool is_context = k < context_rows;
        const uint32_t index = is_context ? k : (k - context_rows);
        const int64_t key_position = is_context
                                         ? static_cast<int64_t>(context_position_start) + index
                                         : static_cast<int64_t>(block_position_start) + index;
        const int64_t distance = key_position - query_position;
        const int64_t abs_distance = distance < 0 ? -distance : distance;
        float score = -INFINITY;
        if (abs_distance < static_cast<int64_t>(sliding_window)) {
            const bf16* k_row = is_context
                                    ? k_ctx + static_cast<std::size_t>(index) * context_row_stride
                                    : k_noise + static_cast<std::size_t>(index) * noise_row_stride;
            const bf16* k_base = k_row + static_cast<std::size_t>(kv_head) * head_dim;
            float acc = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                acc += sq[d] * bf16_to_f32(k_base[d]);
            }
            score = acc * scale;
        }
        scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    const float max_score = block_reduce_max(local_max, scratch);

    float local_sum = 0.0f;
    for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) {
        const float e = expf(scores[k] - max_score);
        scores[k] = e;
        local_sum += e;
    }
    const float sum_score = block_reduce_sum(local_sum, scratch);
    const float inv_sum = sum_score > 0.0f ? 1.0f / sum_score : 0.0f;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < keys; ++k) {
            const float weight = scores[k];
            if (weight == 0.0f) continue;
            const bool is_context = k < context_rows;
            const uint32_t index = is_context ? k : (k - context_rows);
            const bf16* v_row = is_context
                                    ? v_ctx + static_cast<std::size_t>(index) * context_row_stride
                                    : v_noise + static_cast<std::size_t>(index) * noise_row_stride;
            acc += weight * bf16_to_f32(v_row[static_cast<std::size_t>(kv_head) * head_dim + d]);
        }
        output[static_cast<std::size_t>(row) * output_row_stride +
               static_cast<std::size_t>(head) * head_dim + d] = f32_to_bf16(acc * inv_sum);
    }
}

__device__ __forceinline__ void attention_ring_body(
    float* smem,
    const RingAttentionView& view,
    uint32_t keys_alloc,
    uint32_t row,
    uint32_t head,
    uint32_t noise_row_stride,
    uint32_t sliding_window,
    const bf16* q_row,
    const bf16* k_noise,
    const bf16* v_noise,
    bf16* out_row,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    float scale) {
    float* sq = smem;
    float* scores = sq + head_dim;
    float* scratch = scores + keys_alloc;

    const uint32_t keys = view.context_rows + view.block_rows;
    const uint32_t group = q_heads / kv_heads;
    const uint32_t kv_head = head / group;
    const int64_t query_position = static_cast<int64_t>(view.block_position_start) + row;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        sq[d] = bf16_to_f32(q_row[static_cast<std::size_t>(head) * head_dim + d]);
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) {
        const bool is_context = k < view.context_rows;
        const uint32_t index = is_context ? k : (k - view.context_rows);
        const int64_t key_position =
            is_context ? static_cast<int64_t>(view.context_start_position) + index
                       : static_cast<int64_t>(view.block_position_start) + index;
        const int64_t distance = key_position - query_position;
        const int64_t abs_distance = distance < 0 ? -distance : distance;
        float score = -INFINITY;
        if (abs_distance < static_cast<int64_t>(sliding_window)) {
            const bf16* k_row;
            if (is_context) {
                const uint32_t slot =
                    static_cast<uint32_t>(key_position % static_cast<int64_t>(view.ring_capacity));
                k_row = view.k_ring + static_cast<std::size_t>(slot) * view.ring_row_stride;
            } else {
                k_row = k_noise + static_cast<std::size_t>(index) * noise_row_stride;
            }
            const bf16* k_base = k_row + static_cast<std::size_t>(kv_head) * head_dim;
            float acc = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                acc += sq[d] * bf16_to_f32(k_base[d]);
            }
            score = acc * scale;
        }
        scores[k] = score;
        local_max = fmaxf(local_max, score);
    }
    const float max_score = block_reduce_max(local_max, scratch);

    float local_sum = 0.0f;
    for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) {
        const float e = expf(scores[k] - max_score);
        scores[k] = e;
        local_sum += e;
    }
    const float sum_score = block_reduce_sum(local_sum, scratch);
    const float inv_sum = sum_score > 0.0f ? 1.0f / sum_score : 0.0f;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t k = 0; k < keys; ++k) {
            const float weight = scores[k];
            if (weight == 0.0f) continue;
            const bool is_context = k < view.context_rows;
            const uint32_t index = is_context ? k : (k - view.context_rows);
            const bf16* v_row;
            if (is_context) {
                const int64_t key_position =
                    static_cast<int64_t>(view.context_start_position) + index;
                const uint32_t slot =
                    static_cast<uint32_t>(key_position % static_cast<int64_t>(view.ring_capacity));
                v_row = view.v_ring + static_cast<std::size_t>(slot) * view.ring_row_stride;
            } else {
                v_row = v_noise + static_cast<std::size_t>(index) * noise_row_stride;
            }
            acc += weight * bf16_to_f32(v_row[static_cast<std::size_t>(kv_head) * head_dim + d]);
        }
        out_row[static_cast<std::size_t>(head) * head_dim + d] = f32_to_bf16(acc * inv_sum);
    }
}

__global__ void dflash2_attention_ring_kernel(
    const bf16* q,
    uint32_t q_row_stride,
    const bf16* k_ring,
    const bf16* v_ring,
    uint32_t ring_capacity,
    uint32_t context_rows,
    uint32_t context_start_position,
    uint32_t context_row_stride,
    const bf16* k_noise,
    const bf16* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale) {
    extern __shared__ float smem[];
    const uint32_t row = blockIdx.y;
    const uint32_t head = blockIdx.x;
    if (row >= block_rows || head >= q_heads) return;
    RingAttentionView view;
    view.k_ring = k_ring;
    view.v_ring = v_ring;
    view.ring_capacity = ring_capacity;
    view.ring_row_stride = context_row_stride;
    view.context_rows = context_rows;
    view.context_start_position = context_start_position;
    view.block_position_start = block_position_start;
    view.block_rows = block_rows;
    attention_ring_body(
        smem, view, context_rows + block_rows, row, head, noise_row_stride, sliding_window,
        q + static_cast<std::size_t>(row) * q_row_stride, k_noise, v_noise,
        output + static_cast<std::size_t>(row) * output_row_stride, q_heads, kv_heads, head_dim,
        scale);
}

__global__ void dflash2_attention_ring_requests_kernel(
    const bf16* q,
    uint32_t q_row_stride,
    const DFlash2RequestDescriptor* requests,
    uint64_t ring_layer_offset,
    uint32_t ring_capacity,
    const bf16* k_noise,
    const bf16* v_noise,
    uint32_t noise_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    uint32_t keys_alloc) {
    extern __shared__ float smem[];
    const DFlash2RequestDescriptor& request = requests[blockIdx.z];
    const uint32_t row = blockIdx.y;
    if (row >= request.block_rows) return;
    RingAttentionView view;
    view.k_ring = reinterpret_cast<const bf16*>(request.ring_k) + ring_layer_offset;
    view.v_ring = reinterpret_cast<const bf16*>(request.ring_v) + ring_layer_offset;
    view.ring_capacity = ring_capacity;
    view.ring_row_stride = noise_row_stride;
    view.context_rows = request.context_rows;
    view.context_start_position = request.context_start_position;
    view.block_position_start = request.block_position_start;
    view.block_rows = request.block_rows;
    attention_ring_body(
        smem, view, keys_alloc, row, blockIdx.x, noise_row_stride, sliding_window,
        q + static_cast<std::size_t>(request.row_begin + row) * q_row_stride,
        k_noise + static_cast<std::size_t>(request.row_begin) * noise_row_stride,
        v_noise + static_cast<std::size_t>(request.row_begin) * noise_row_stride,
        output + static_cast<std::size_t>(request.row_begin + row) * output_row_stride, q_heads,
        kv_heads, head_dim, scale);
}

__device__ __forceinline__ const bf16* gqa_row_base(
    const bf16* ring, const bf16* noise, uint32_t ring_capacity, uint32_t context_rows,
    uint32_t context_start_position, uint32_t key,
    uint32_t row_stride, uint32_t head_dim, uint32_t kv_head) {
    if (key < context_rows) {
        const int64_t key_position = static_cast<int64_t>(context_start_position) + key;
        const uint32_t slot =
            static_cast<uint32_t>(key_position % static_cast<int64_t>(ring_capacity));
        return ring + static_cast<std::size_t>(slot) * row_stride +
               static_cast<std::size_t>(kv_head) * head_dim;
    }
    const uint32_t index = key - context_rows;
    return noise + static_cast<std::size_t>(index) * row_stride +
           static_cast<std::size_t>(kv_head) * head_dim;
}

__device__ __forceinline__ void gqa_stage_tile(
    const bf16* ring, const bf16* noise, uint32_t ring_capacity, uint32_t context_rows,
    uint32_t context_start_position, uint32_t tile_base,
    uint32_t keys, uint32_t context_row_stride, uint32_t noise_row_stride, uint32_t head_dim,
    uint32_t kv_head, bf16* tile) {
    const uint32_t slot = threadIdx.x >> 2;
    const uint32_t part = threadIdx.x & 3u;
    const uint32_t key = tile_base + slot;
    if (key >= keys)
        return;
    const bf16* row_src = gqa_row_base(ring, noise, ring_capacity, context_rows,
                                        context_start_position, key,
                                        key < context_rows ? context_row_stride
                                                           : noise_row_stride,
                                        head_dim, kv_head);
    const uint32_t quarter = head_dim / 4u;
    uint4* dst = reinterpret_cast<uint4*>(tile + static_cast<std::size_t>(slot) * head_dim +
                                            part * quarter);
    const uint4* g = reinterpret_cast<const uint4*>(row_src + part * quarter);
    dst[0] = g[0];
    dst[1] = g[1];
    dst[2] = g[2];
    dst[3] = g[3];
}

__device__ __forceinline__ void gqa_attention_body(
    char* smem_raw,
    const RingAttentionView& view,
    uint32_t keys_alloc,
    uint32_t row,
    uint32_t kv_head,
    uint32_t noise_row_stride,
    uint32_t sliding_window,
    const bf16* q_row,
    const bf16* k_noise,
    const bf16* v_noise,
    bf16* out_row,
    uint32_t head_dim,
    float scale) {
    constexpr uint32_t kGroup = kAttentionGqaGroup;
    float* sq = reinterpret_cast<float*>(smem_raw);
    float* scores = sq + kGroup * head_dim;
    const uint32_t keys = view.context_rows + view.block_rows;
    float* scratch = scores + static_cast<std::size_t>(keys_alloc) * kGroup;
    bf16* tile = reinterpret_cast<bf16*>(scratch + blockDim.x);

    const uint32_t head0 = kv_head * kGroup;
    const int64_t query_position = static_cast<int64_t>(view.block_position_start) + row;

    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        #pragma unroll
        for (uint32_t h = 0; h < kGroup; ++h) {
            sq[h * head_dim + d] = bf16_to_f32(
                q_row[static_cast<std::size_t>(head0 + h) * head_dim + d]);
        }
    }
    __syncthreads();

    for (uint32_t tile_base = 0; tile_base < keys; tile_base += kAttentionGqaTile) {
        gqa_stage_tile(view.k_ring, k_noise, view.ring_capacity, view.context_rows,
                       view.context_start_position, tile_base, keys, view.ring_row_stride,
                       noise_row_stride, head_dim, kv_head, tile);
        __syncthreads();
        const uint32_t slot = threadIdx.x >> 2;
        const uint32_t h = threadIdx.x & 3u;
        const uint32_t key = tile_base + slot;
        if (key < keys) {
            int64_t key_position;
            if (key < view.context_rows)
                key_position = static_cast<int64_t>(view.context_start_position) + key;
            else
                key_position = static_cast<int64_t>(view.block_position_start) +
                               static_cast<int64_t>(key - view.context_rows);
            const int64_t distance = key_position - query_position;
            const int64_t abs_distance = distance < 0 ? -distance : distance;
            float score = -INFINITY;
            if (abs_distance < static_cast<int64_t>(sliding_window)) {
                const bf16* k_row = tile + static_cast<std::size_t>(slot) * head_dim;
                const float* q_head = sq + h * head_dim;
                float acc = 0.0f;
                for (uint32_t d = 0; d < head_dim; ++d)
                    acc += q_head[d] * bf16_to_f32(k_row[d]);
                score = acc * scale;
            }
            scores[static_cast<std::size_t>(key) * kGroup + h] = score;
        }
        __syncthreads();
    }

    float max_scores[kGroup];
    #pragma unroll
    for (uint32_t h = 0; h < kGroup; ++h) {
        float local_max = -INFINITY;
        for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x)
            local_max = fmaxf(local_max, scores[static_cast<std::size_t>(k) * kGroup + h]);
        max_scores[h] = block_reduce_max(local_max, scratch);
    }

    float inv_sums[kGroup];
    #pragma unroll
    for (uint32_t h = 0; h < kGroup; ++h) {
        float local_sum = 0.0f;
        for (uint32_t k = threadIdx.x; k < keys; k += blockDim.x) {
            const std::size_t idx = static_cast<std::size_t>(k) * kGroup + h;
            const float e = expf(scores[idx] - max_scores[h]);
            scores[idx] = e;
            local_sum += e;
        }
        const float sum_score = block_reduce_sum(local_sum, scratch);
        inv_sums[h] = sum_score > 0.0f ? 1.0f / sum_score : 0.0f;
    }

    float acc[kGroup] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint32_t d = threadIdx.x;
    for (uint32_t tile_base = 0; tile_base < keys; tile_base += kAttentionGqaTile) {
        gqa_stage_tile(view.v_ring, v_noise, view.ring_capacity, view.context_rows,
                       view.context_start_position, tile_base, keys, view.ring_row_stride,
                       noise_row_stride, head_dim, kv_head, tile);
        __syncthreads();
        const uint32_t tail = keys - tile_base;
        const uint32_t count = tail < kAttentionGqaTile ? tail : kAttentionGqaTile;
        for (uint32_t s = 0; s < count; ++s) {
            const uint32_t key = tile_base + s;
            const float v = bf16_to_f32(tile[static_cast<std::size_t>(s) * head_dim + d]);
            #pragma unroll
            for (uint32_t h = 0; h < kGroup; ++h) {
                const float weight = scores[static_cast<std::size_t>(key) * kGroup + h];
                if (weight == 0.0f)
                    continue;
                acc[h] += weight * v;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t h = 0; h < kGroup; ++h) {
        out_row[static_cast<std::size_t>(head0 + h) * head_dim + d] =
            f32_to_bf16(acc[h] * inv_sums[h]);
    }
}

__global__ void dflash2_attention_ring_gqa_kernel(
    const bf16* q,
    uint32_t q_row_stride,
    const bf16* k_ring,
    const bf16* v_ring,
    uint32_t ring_capacity,
    uint32_t context_rows,
    uint32_t context_start_position,
    uint32_t context_row_stride,
    const bf16* k_noise,
    const bf16* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t head_dim,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale) {
    extern __shared__ char smem_raw[];
    const uint32_t row = blockIdx.y;
    const uint32_t kv_head = blockIdx.x;
    RingAttentionView view;
    view.k_ring = k_ring;
    view.v_ring = v_ring;
    view.ring_capacity = ring_capacity;
    view.ring_row_stride = context_row_stride;
    view.context_rows = context_rows;
    view.context_start_position = context_start_position;
    view.block_position_start = block_position_start;
    view.block_rows = block_rows;
    gqa_attention_body(
        smem_raw, view, context_rows + block_rows, row, kv_head, noise_row_stride,
        sliding_window, q + static_cast<std::size_t>(row) * q_row_stride, k_noise, v_noise,
        output + static_cast<std::size_t>(row) * output_row_stride, head_dim, scale);
}

__global__ void dflash2_attention_ring_gqa_requests_kernel(
    const bf16* q,
    uint32_t q_row_stride,
    const DFlash2RequestDescriptor* requests,
    uint64_t ring_layer_offset,
    uint32_t ring_capacity,
    const bf16* k_noise,
    const bf16* v_noise,
    uint32_t noise_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    uint32_t keys_alloc) {
    extern __shared__ char smem_raw[];
    const DFlash2RequestDescriptor& request = requests[blockIdx.z];
    const uint32_t row = blockIdx.y;
    if (row >= request.block_rows) return;
    RingAttentionView view;
    view.k_ring = reinterpret_cast<const bf16*>(request.ring_k) + ring_layer_offset;
    view.v_ring = reinterpret_cast<const bf16*>(request.ring_v) + ring_layer_offset;
    view.ring_capacity = ring_capacity;
    view.ring_row_stride = noise_row_stride;
    view.context_rows = request.context_rows;
    view.context_start_position = request.context_start_position;
    view.block_position_start = request.block_position_start;
    view.block_rows = request.block_rows;
    gqa_attention_body(
        smem_raw, view, keys_alloc, row, blockIdx.x, noise_row_stride, sliding_window,
        q + static_cast<std::size_t>(request.row_begin + row) * q_row_stride,
        k_noise + static_cast<std::size_t>(request.row_begin) * noise_row_stride,
        v_noise + static_cast<std::size_t>(request.row_begin) * noise_row_stride,
        output + static_cast<std::size_t>(request.row_begin + row) * output_row_stride, head_dim,
        scale);
}

}  // namespace

hipError_t launch_dflash2_attention_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ctx,
    const bf16_t* v_ctx,
    uint32_t context_rows,
    uint32_t context_row_stride,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t context_position_start,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream) {
    if (q == nullptr || k_ctx == nullptr || v_ctx == nullptr || k_noise == nullptr ||
        v_noise == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (q_heads == 0u || kv_heads == 0u || head_dim == 0u || q_heads % kv_heads != 0u) {
        return hipErrorInvalidValue;
    }
    if (block_rows == 0u || context_rows == 0u || sliding_window == 0u) {
        return hipErrorInvalidValue;
    }
    if (q_row_stride < q_heads * head_dim || output_row_stride < q_heads * head_dim ||
        context_row_stride < kv_heads * head_dim || noise_row_stride < kv_heads * head_dim) {
        return hipErrorInvalidValue;
    }
    if (head_dim > kAttentionThreads) {
        return hipErrorInvalidValue;
    }
    const uint32_t keys = context_rows + block_rows;
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(head_dim) + keys + kAttentionThreads) * sizeof(float);
    if (shared_bytes > kAttentionSharedBytesLimit) {
        return hipErrorInvalidValue;
    }
    dim3 grid(q_heads, block_rows);
    dflash2_attention_kernel<<<grid, kAttentionThreads,
                               static_cast<std::size_t>(shared_bytes), stream>>>(
        reinterpret_cast<const bf16*>(q), q_row_stride,
        reinterpret_cast<const bf16*>(k_ctx), reinterpret_cast<const bf16*>(v_ctx),
        context_rows, context_row_stride,
        reinterpret_cast<const bf16*>(k_noise), reinterpret_cast<const bf16*>(v_noise),
        block_rows, noise_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride,
        q_heads, kv_heads, head_dim, context_position_start, block_position_start,
        sliding_window, scale);
    return hipGetLastError();
}

hipError_t launch_dflash2_attention_ring_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ring,
    const bf16_t* v_ring,
    uint32_t ring_capacity,
    uint32_t context_rows,
    uint32_t context_start_position,
    uint32_t context_row_stride,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream) {
    if (q == nullptr || k_ring == nullptr || v_ring == nullptr || k_noise == nullptr ||
        v_noise == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (q_heads == 0u || kv_heads == 0u || head_dim == 0u || q_heads % kv_heads != 0u) {
        return hipErrorInvalidValue;
    }
    if (block_rows == 0u || context_rows == 0u || sliding_window == 0u ||
        ring_capacity == 0u) {
        return hipErrorInvalidValue;
    }
    if (context_rows > ring_capacity) {
        return hipErrorInvalidValue;
    }
    if (q_row_stride < q_heads * head_dim || output_row_stride < q_heads * head_dim ||
        context_row_stride < kv_heads * head_dim || noise_row_stride < kv_heads * head_dim) {
        return hipErrorInvalidValue;
    }
    if (head_dim > kAttentionThreads) {
        return hipErrorInvalidValue;
    }
    const uint32_t keys = context_rows + block_rows;
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(head_dim) + keys + kAttentionThreads) * sizeof(float);
    if (shared_bytes > kAttentionSharedBytesLimit) {
        return hipErrorInvalidValue;
    }
    dim3 grid(q_heads, block_rows);
    dflash2_attention_ring_kernel<<<grid, kAttentionThreads,
                                    static_cast<std::size_t>(shared_bytes), stream>>>(
        reinterpret_cast<const bf16*>(q), q_row_stride,
        reinterpret_cast<const bf16*>(k_ring), reinterpret_cast<const bf16*>(v_ring),
        ring_capacity, context_rows, context_start_position, context_row_stride,
        reinterpret_cast<const bf16*>(k_noise), reinterpret_cast<const bf16*>(v_noise),
        block_rows, noise_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride,
        q_heads, kv_heads, head_dim, block_position_start, sliding_window, scale);
    return hipGetLastError();
}

hipError_t launch_dflash2_attention_ring_gqa_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const bf16_t* k_ring,
    const bf16_t* v_ring,
    uint32_t ring_capacity,
    uint32_t context_rows,
    uint32_t context_start_position,
    uint32_t context_row_stride,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t block_rows,
    uint32_t noise_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t block_position_start,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream) {
    if (q == nullptr || k_ring == nullptr || v_ring == nullptr || k_noise == nullptr ||
        v_noise == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (q_heads == 0u || kv_heads == 0u || head_dim == 0u || q_heads % kv_heads != 0u) {
        return hipErrorInvalidValue;
    }
    if (block_rows == 0u || context_rows == 0u || sliding_window == 0u || ring_capacity == 0u) {
        return hipErrorInvalidValue;
    }
    if (context_rows > ring_capacity) {
        return hipErrorInvalidValue;
    }
    if (q_heads != kv_heads * kAttentionGqaGroup || head_dim != kAttentionThreads) {
        return hipErrorNotSupported;
    }
    if ((context_row_stride & 7u) != 0u || (noise_row_stride & 7u) != 0u ||
        (q_row_stride & 7u) != 0u || (output_row_stride & 7u) != 0u) {
        return hipErrorNotSupported;
    }
    const uint32_t keys = context_rows + block_rows;
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(kAttentionGqaGroup) * head_dim +
         static_cast<std::size_t>(keys) * kAttentionGqaGroup + kAttentionThreads) *
            sizeof(float) +
        static_cast<std::size_t>(kAttentionGqaTile) * head_dim * sizeof(bf16_t);
    if (shared_bytes > kAttentionGqaSharedBytesLimit) {
        return hipErrorNotSupported;
    }
    dim3 grid(kv_heads, block_rows);
    dflash2_attention_ring_gqa_kernel<<<grid, kAttentionThreads, shared_bytes, stream>>>(
        reinterpret_cast<const bf16*>(q), q_row_stride,
        reinterpret_cast<const bf16*>(k_ring), reinterpret_cast<const bf16*>(v_ring),
        ring_capacity, context_rows, context_start_position, context_row_stride,
        reinterpret_cast<const bf16*>(k_noise), reinterpret_cast<const bf16*>(v_noise),
        block_rows, noise_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride,
        head_dim, block_position_start, sliding_window, scale);
    return hipGetLastError();
}

hipError_t launch_dflash2_attention_ring_requests_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const DFlash2RequestDescriptor* requests,
    uint32_t request_count,
    uint32_t max_block_rows,
    uint32_t max_keys,
    uint32_t ring_capacity,
    uint64_t ring_layer_offset,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t kv_features,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream) {
    if (q == nullptr || requests == nullptr || k_noise == nullptr || v_noise == nullptr ||
        output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (q_heads == 0u || kv_heads == 0u || head_dim == 0u || q_heads % kv_heads != 0u) {
        return hipErrorInvalidValue;
    }
    if (request_count == 0u || max_block_rows == 0u || max_keys == 0u ||
        ring_capacity == 0u || sliding_window == 0u) {
        return hipErrorInvalidValue;
    }
    if (q_row_stride < q_heads * head_dim || output_row_stride < q_heads * head_dim ||
        kv_features < kv_heads * head_dim) {
        return hipErrorInvalidValue;
    }
    if (head_dim > kAttentionThreads) {
        return hipErrorInvalidValue;
    }
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(head_dim) + max_keys + kAttentionThreads) * sizeof(float);
    if (shared_bytes > kAttentionSharedBytesLimit) {
        return hipErrorInvalidValue;
    }
    dim3 grid(q_heads, max_block_rows, request_count);
    dflash2_attention_ring_requests_kernel<<<grid, kAttentionThreads, shared_bytes, stream>>>(
        reinterpret_cast<const bf16*>(q), q_row_stride, requests, ring_layer_offset,
        ring_capacity, reinterpret_cast<const bf16*>(k_noise),
        reinterpret_cast<const bf16*>(v_noise), kv_features,
        reinterpret_cast<bf16*>(output), output_row_stride, q_heads, kv_heads, head_dim,
        sliding_window, scale, max_keys);
    return hipGetLastError();
}

hipError_t launch_dflash2_attention_ring_gqa_requests_bf16(
    const bf16_t* q,
    uint32_t q_row_stride,
    const DFlash2RequestDescriptor* requests,
    uint32_t request_count,
    uint32_t max_block_rows,
    uint32_t max_keys,
    uint32_t ring_capacity,
    uint64_t ring_layer_offset,
    const bf16_t* k_noise,
    const bf16_t* v_noise,
    uint32_t kv_features,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t q_heads,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t sliding_window,
    float scale,
    hipStream_t stream) {
    if (q == nullptr || requests == nullptr || k_noise == nullptr || v_noise == nullptr ||
        output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (q_heads == 0u || kv_heads == 0u || head_dim == 0u || q_heads % kv_heads != 0u) {
        return hipErrorInvalidValue;
    }
    if (request_count == 0u || max_block_rows == 0u || max_keys == 0u || ring_capacity == 0u ||
        sliding_window == 0u) {
        return hipErrorInvalidValue;
    }
    if (q_heads != kv_heads * kAttentionGqaGroup || head_dim != kAttentionThreads) {
        return hipErrorNotSupported;
    }
    if ((kv_features & 7u) != 0u || (q_row_stride & 7u) != 0u ||
        (output_row_stride & 7u) != 0u) {
        return hipErrorNotSupported;
    }
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(kAttentionGqaGroup) * head_dim +
         static_cast<std::size_t>(max_keys) * kAttentionGqaGroup + kAttentionThreads) *
            sizeof(float) +
        static_cast<std::size_t>(kAttentionGqaTile) * head_dim * sizeof(bf16_t);
    if (shared_bytes > kAttentionGqaSharedBytesLimit) {
        return hipErrorNotSupported;
    }
    dim3 grid(kv_heads, max_block_rows, request_count);
    dflash2_attention_ring_gqa_requests_kernel<<<grid, kAttentionThreads, shared_bytes,
                                                 stream>>>(
        reinterpret_cast<const bf16*>(q), q_row_stride, requests, ring_layer_offset,
        ring_capacity, reinterpret_cast<const bf16*>(k_noise),
        reinterpret_cast<const bf16*>(v_noise), kv_features,
        reinterpret_cast<bf16*>(output), output_row_stride, head_dim, sliding_window, scale,
        max_keys);
    return hipGetLastError();
}

}
