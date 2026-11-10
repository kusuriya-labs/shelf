#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstddef>

namespace shelf::kernel {
namespace {

constexpr uint32_t kThreads = kQsaThreads;

__device__ __forceinline__ float bf16_value(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<std::uint32_t>(p->data) << 16);
}

__device__ __forceinline__ shelf::bf16_t bf16_from(float value) {
    return shelf::bf16_t{
        static_cast<std::uint16_t>(__bfloat16_as_ushort(__float2bfloat16(value)))};
}

__device__ __forceinline__ float block_sum(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = kThreads / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

__global__ __launch_bounds__(kThreads) void qsa_pool_kernel(QsaPoolArgs args) {
    const uint32_t block = blockIdx.x;
    const uint32_t d = threadIdx.x;
    __shared__ float scratch[kThreads];
    float mean = 0.0f;
    for (uint32_t t = 0; t < args.compress_ratio; ++t) {
        const uint32_t row = block * args.compress_ratio + t;
        mean += bf16_value(args.raw_keys + static_cast<std::uint64_t>(row) * args.head_dim + d);
    }
    mean /= static_cast<float>(args.compress_ratio);
    const float square = block_sum(scratch, mean * mean);
    __syncthreads();
    const float inv = rsqrtf(square / static_cast<float>(args.head_dim) + args.eps);
    const float weight = 1.0f + bf16_value(args.norm_weight + d);
    args.pooled[static_cast<std::uint64_t>(block) * args.head_dim + d] =
        bf16_from(mean * inv * weight);
}

__device__ __forceinline__ bool qsa_worse(float a, uint32_t ia, float b, uint32_t ib) noexcept {
    return a < b || (a == b && ia > ib);
}

__device__ __forceinline__ void qsa_swap_pair(float* scores, int32_t* indices,
                                              uint32_t a, uint32_t b) {
    const float score = scores[a];
    const int32_t index = indices[a];
    scores[a] = scores[b];
    indices[a] = indices[b];
    scores[b] = score;
    indices[b] = index;
}

__device__ __forceinline__ void qsa_sort_candidates_ascending(
    float* scores, int32_t* indices, uint32_t base, uint32_t count) {
    for (uint32_t size = 2u; size <= count; size <<= 1u) {
        for (uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
                const uint32_t j = i ^ stride;
                if (j <= i) continue;
                const bool ascending = (i & size) == 0u;
                const bool i_worse =
                    qsa_worse(scores[base + i], static_cast<uint32_t>(indices[base + i]),
                              scores[base + j], static_cast<uint32_t>(indices[base + j]));
                if (ascending ? !i_worse : i_worse)
                    qsa_swap_pair(scores, indices, base + i, base + j);
            }
            __syncthreads();
        }
    }
}

__device__ __forceinline__ void qsa_merge_descending(float* scores, int32_t* indices,
                                                     uint32_t count) {
    for (uint32_t stride = count >> 1u; stride > 0u; stride >>= 1u) {
        for (uint32_t i = threadIdx.x; i < count; i += blockDim.x) {
            const uint32_t j = i ^ stride;
            if (j <= i) continue;
            const bool i_worse =
                qsa_worse(scores[i], static_cast<uint32_t>(indices[i]),
                          scores[j], static_cast<uint32_t>(indices[j]));
            if (i_worse) qsa_swap_pair(scores, indices, i, j);
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kQsaThreads) void qsa_select_batch_kernel(
    QsaSelectBatchArgs args) {
    constexpr uint32_t kWarps = kQsaThreads / 32u;
    constexpr uint32_t kBlocksPerWarp = kQsaSelectChunk / kWarps;
    constexpr uint32_t kDimsPerLane = kQsaHeadDim / 32u;

    __shared__ bf16_t q_stage[kQsaSelectMaxHeads][kQsaHeadDim];
    __shared__ float block_score[kQsaSelectChunk];
    __shared__ uint32_t candidate_count;

    extern __shared__ uint8_t qsa_select_dynamic[];
    const uint32_t row = blockIdx.x;
    const uint32_t t = threadIdx.x;
    const uint32_t heads = args.n_heads;
    const bf16_t* q_row = args.q + static_cast<uint64_t>(row) * args.q_row_stride;
    for (uint32_t i = t; i < heads * kQsaHeadDim; i += kQsaThreads) {
        q_stage[i / kQsaHeadDim][i % kQsaHeadDim] = q_row[i];
    }

    uint32_t* mask_words_ptr = nullptr;
    const uint32_t mask_words = (args.blocks + 31u) / 32u;
    if (args.block_mask != nullptr) {
        mask_words_ptr = reinterpret_cast<uint32_t*>(
            args.block_mask + static_cast<uint64_t>(row) * args.mask_row_bytes);
        for (uint32_t i = t; i < mask_words; i += kQsaThreads) mask_words_ptr[i] = 0u;
    }
    int32_t* out = args.selected != nullptr
                       ? args.selected + static_cast<uint64_t>(row) * args.selected_row_stride
                       : nullptr;

    const uint32_t slot = args.row_sequence_slots[row];
    const uint32_t complete = (args.positions[row] + 1u) / args.compress_ratio;
    const uint32_t visible = complete < args.blocks ? complete : args.blocks;
    const uint32_t top_k = args.top_k < kQsaSelectMaxTopK ? args.top_k : kQsaSelectMaxTopK;
    uint32_t half = kQsaSelectChunk;
    while (half < top_k) half <<= 1u;
    const uint32_t count = half * 2u;
    float* scores = reinterpret_cast<float*>(qsa_select_dynamic);
    int32_t* indices =
        reinterpret_cast<int32_t*>(qsa_select_dynamic + count * sizeof(float));
    __syncthreads();
    if (slot >= args.max_sequences) {
        if (out != nullptr) {
            for (uint32_t i = t; i < args.top_k; i += kQsaThreads) out[i] = -1;
        }
        return;
    }

    const bf16_t* pooled = args.pooled_base +
                           static_cast<uint64_t>(slot) * args.pooled_slot_stride +
                           static_cast<uint64_t>(args.state_index) *
                               args.pooled_layer_stride;

    const uint32_t warp = t >> 5u;
    const uint32_t lane = t & 31u;
    const uint32_t lane_dim = lane * kDimsPerLane;
    for (uint32_t i = t; i < count; i += kQsaThreads) {
        scores[i] = -INFINITY;
        indices[i] = -1;
    }
    if (t == 0u) candidate_count = 0u;
    __syncthreads();

    for (uint32_t base = 0u; base < visible; base += kQsaSelectChunk) {
        const uint32_t n =
            visible - base < kQsaSelectChunk ? visible - base : kQsaSelectChunk;
        for (uint32_t slot_in_warp = 0u; slot_in_warp < kBlocksPerWarp; ++slot_in_warp) {
            const uint32_t b = warp * kBlocksPerWarp + slot_in_warp;
            if (b >= n) continue;
            const bf16_t* k_row =
                pooled + static_cast<uint64_t>(base + b) * kQsaHeadDim + lane_dim;
            const float k0 = bf16_value(k_row + 0u);
            const float k1 = bf16_value(k_row + 1u);
            const float k2 = bf16_value(k_row + 2u);
            const float k3 = bf16_value(k_row + 3u);
            float total = 0.0f;
            for (uint32_t head = 0u; head < heads; ++head) {
                const bf16_t* qp = &q_stage[head][lane_dim];
                float dot = bf16_value(qp + 0u) * k0 + bf16_value(qp + 1u) * k1 +
                            bf16_value(qp + 2u) * k2 + bf16_value(qp + 3u) * k3;
                dot += __shfl_xor(dot, 16u);
                dot += __shfl_xor(dot, 8u);
                dot += __shfl_xor(dot, 4u);
                dot += __shfl_xor(dot, 2u);
                dot += __shfl_xor(dot, 1u);
                total += fmaxf(dot, 0.0f);
            }
            if (lane == 0u) block_score[b] = total * args.scale;
        }
        __syncthreads();
        if (t < n) {
            scores[half + candidate_count + t] = block_score[t];
            indices[half + candidate_count + t] = static_cast<int32_t>(base + t);
        }
        __syncthreads();
        if (t == 0u) candidate_count += n;
        __syncthreads();
        if (candidate_count == half) {
            qsa_sort_candidates_ascending(scores, indices, half, half);
            qsa_merge_descending(scores, indices, count);
            if (t == 0u) candidate_count = 0u;
            for (uint32_t i = t; i < half; i += kQsaThreads) {
                scores[half + i] = -INFINITY;
                indices[half + i] = -1;
            }
            __syncthreads();
        }
    }

    if (candidate_count != 0u) {
        qsa_sort_candidates_ascending(scores, indices, half, half);
        qsa_merge_descending(scores, indices, count);
    }

    const uint32_t selected = visible < top_k ? visible : top_k;
    if (out != nullptr) {
        if (t == 0u) {
            for (uint32_t i = 1u; i < selected; ++i) {
                const int32_t value = indices[i];
                uint32_t j = i;
                while (j > 0u && indices[j - 1u] > value) {
                    indices[j] = indices[j - 1u];
                    --j;
                }
                indices[j] = value;
            }
        }
        __syncthreads();
        for (uint32_t i = t; i < args.top_k; i += kQsaThreads)
            out[i] = i < selected ? indices[i] : -1;
    }
    if (mask_words_ptr != nullptr) {
        for (uint32_t i = t; i < selected; i += kQsaThreads) {
            const uint32_t block = static_cast<uint32_t>(indices[i]);
            atomicOr(&mask_words_ptr[block / 32u], 1u << (block % 32u));
        }
        if (t == 0u && complete < args.blocks) {
            atomicOr(&mask_words_ptr[complete / 32u], 1u << (complete % 32u));
        }
    }
}

__global__ __launch_bounds__(kQsaHeadDim) void qsa_rope_kernel(QsaRopeArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t half = args.rotary_dim / 2u;
    const float position = static_cast<float>(args.positions[row]);
    for (uint32_t head = 0; head < args.heads; ++head) {
        shelf::bf16_t* base =
            args.x + (static_cast<std::uint64_t>(row) * args.heads + head) * args.head_dim;
        for (uint32_t i = threadIdx.x; i < half; i += blockDim.x) {
            const float exponent = -2.0f * static_cast<float>(i) /
                                   static_cast<float>(args.rotary_dim);
            const float frequency = args.inv_freq != nullptr
                                        ? static_cast<float>(args.inv_freq[i])
                                        : powf(args.theta, exponent);
            const float angle = (position / args.position_scale) * frequency;
            const float cosine = cosf(angle) * args.attention_factor;
            const float sine = sinf(angle) * args.attention_factor;
            const float x0 = bf16_value(base + i);
            const float x1 = bf16_value(base + i + half);
            base[i] = bf16_from(x0 * cosine - x1 * sine);
            base[i + half] = bf16_from(x1 * cosine + x0 * sine);
        }
    }
}

__global__ __launch_bounds__(kQsaHeadDim) void qsa_key_cache_batch_kernel(
    QsaKeyCacheBatchArgs args) {
    const uint32_t d = threadIdx.x;
    __shared__ float scratch[kQsaHeadDim];
    __shared__ float normed[kQsaHeadDim];
    const ::shelf::runtime::DeviceRequestDescriptor request = args.requests[blockIdx.x];
    const uint32_t slot = request.request_handle.slot;
    if (slot >= args.max_sequences) return;
    bf16_t* pooled = args.pooled_base +
                     static_cast<std::uint64_t>(slot) * args.pooled_slot_stride +
                     static_cast<std::uint64_t>(args.state_index) *
                         args.pooled_layer_stride;
    bf16_t* partial = args.partial_base +
                      static_cast<std::uint64_t>(slot) * args.partial_slot_stride +
                      static_cast<std::uint64_t>(args.state_index) *
                          args.partial_layer_stride;
    uint32_t* pool_count_ptr =
        args.pool_count_base +
        static_cast<std::uint64_t>(slot) * args.count_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.count_layer_stride;
    uint32_t* partial_count_ptr =
        args.partial_count_base +
        static_cast<std::uint64_t>(slot) * args.count_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.count_layer_stride;

    uint32_t partial_count = *partial_count_ptr;
    uint32_t pool_count = *pool_count_ptr;
    for (uint32_t t = 0; t < request.row_count; ++t) {
        const uint32_t row = request.row_begin + t;
        if (row >= args.actual_rows) break;
        if (args.partial_history != nullptr && row < args.history_rows) {
            bf16_t* hp = args.partial_history +
                static_cast<std::uint64_t>(args.state_index) *
                    args.history_partial_state_stride +
                static_cast<std::uint64_t>(row) * args.compress_ratio * args.head_dim;
            for (uint32_t i = 0; i < args.compress_ratio; ++i) {
                hp[static_cast<std::uint64_t>(i) * args.head_dim + d] =
                    partial[static_cast<std::uint64_t>(i) * args.head_dim + d];
            }
            if (d == 0u && args.count_history != nullptr) {
                uint32_t* hc = args.count_history +
                    static_cast<std::uint64_t>(args.state_index) *
                        args.history_count_state_stride +
                    static_cast<std::uint64_t>(row) * 2u;
                hc[0] = pool_count;
                hc[1] = partial_count;
            }
        }
        const bf16_t* raw =
            args.raw_keys + static_cast<std::uint64_t>(row) * args.raw_row_stride;
        partial[static_cast<std::uint64_t>(partial_count) * args.head_dim + d] = raw[d];
        ++partial_count;
        __syncthreads();
        if (partial_count == args.compress_ratio) {
            float mean = 0.0f;
            for (uint32_t i = 0; i < args.compress_ratio; ++i) {
                mean += bf16_value(partial +
                                   static_cast<std::uint64_t>(i) * args.head_dim + d);
            }
            mean /= static_cast<float>(args.compress_ratio);
            const float square = block_sum(scratch, mean * mean);
            __syncthreads();
            const float inv =
                rsqrtf(square / static_cast<float>(args.head_dim) + args.eps);
            normed[d] = mean * inv * (1.0f + bf16_value(args.norm_weight + d));
            __syncthreads();
            if (pool_count < args.max_blocks) {
                const uint32_t position = args.positions[row];
                const uint32_t start = position >= args.compress_ratio - 1u
                                           ? position - (args.compress_ratio - 1u)
                                           : 0u;
                bf16_t* dst =
                    pooled + static_cast<std::uint64_t>(pool_count) * args.head_dim;
                const uint32_t half = args.rotary_dim / 2u;
                if (d < half) {
                    const float exponent = -2.0f * static_cast<float>(d) /
                                           static_cast<float>(args.rotary_dim);
                    const float frequency =
                        args.inv_freq != nullptr
                            ? static_cast<float>(args.inv_freq[d])
                            : powf(args.theta, exponent);
                    const float angle =
                        (static_cast<float>(start) / args.position_scale) * frequency;
                    const float cosine = cosf(angle);
                    const float sine = sinf(angle);
                    const float x0 = normed[d];
                    const float x1 = normed[d + half];
                    dst[d] = bf16_from(x0 * cosine - x1 * sine);
                } else if (d < args.rotary_dim) {
                    const uint32_t i = d - half;
                    const float exponent = -2.0f * static_cast<float>(i) /
                                           static_cast<float>(args.rotary_dim);
                    const float frequency =
                        args.inv_freq != nullptr
                            ? static_cast<float>(args.inv_freq[i])
                            : powf(args.theta, exponent);
                    const float angle =
                        (static_cast<float>(start) / args.position_scale) * frequency;
                    const float cosine = cosf(angle);
                    const float sine = sinf(angle);
                    const float x0 = normed[i];
                    const float x1 = normed[d];
                    dst[d] = bf16_from(x1 * cosine + x0 * sine);
                } else {
                    dst[d] = bf16_from(normed[d]);
                }
                ++pool_count;
            }
            partial_count = 0;
            __syncthreads();
        }
    }
    if (d == 0u) {
        *partial_count_ptr = partial_count;
        *pool_count_ptr = pool_count;
    }
}

__global__ __launch_bounds__(kQsaHeadDim) void qsa_key_cache_kernel(
    QsaKeyCacheArgs args) {
    const uint32_t d = threadIdx.x;
    __shared__ float scratch[kQsaHeadDim];
    __shared__ float normed[kQsaHeadDim];
    uint32_t partial_count = *args.partial_count;
    uint32_t pool_count = *args.pool_count;
    for (uint32_t row = 0; row < args.rows; ++row) {
        const shelf::bf16_t* raw =
            args.raw_keys + static_cast<std::uint64_t>(row) * args.head_dim;
        args.partial[static_cast<std::uint64_t>(partial_count) * args.head_dim + d] =
            raw[d];
        ++partial_count;
        __syncthreads();
        if (partial_count == args.compress_ratio) {
            float mean = 0.0f;
            for (uint32_t t = 0; t < args.compress_ratio; ++t) {
                mean += bf16_value(args.partial +
                                   static_cast<std::uint64_t>(t) * args.head_dim + d);
            }
            mean /= static_cast<float>(args.compress_ratio);
            const float square = block_sum(scratch, mean * mean);
            __syncthreads();
            const float inv = rsqrtf(square / static_cast<float>(args.head_dim) + args.eps);
            normed[d] = mean * inv * (1.0f + bf16_value(args.norm_weight + d));
            __syncthreads();
            if (pool_count < args.max_blocks) {
                const uint32_t start =
                    args.positions[row] >= args.compress_ratio - 1u
                        ? args.positions[row] - (args.compress_ratio - 1u)
                        : 0u;
                shelf::bf16_t* dst =
                    args.pooled + static_cast<std::uint64_t>(pool_count) * args.head_dim;
                const uint32_t half = args.rotary_dim / 2u;
                if (d < half) {
                    const float exponent = -2.0f * static_cast<float>(d) /
                                           static_cast<float>(args.rotary_dim);
                    const float frequency = args.inv_freq != nullptr
                                                ? static_cast<float>(args.inv_freq[d])
                                                : powf(args.theta, exponent);
                    const float angle = (static_cast<float>(start) /
                                         args.position_scale) * frequency;
                    const float cosine = cosf(angle);
                    const float sine = sinf(angle);
                    const float x0 = normed[d];
                    const float x1 = normed[d + half];
                    dst[d] = bf16_from(x0 * cosine - x1 * sine);
                } else if (d < args.rotary_dim) {
                    const uint32_t i = d - half;
                    const float exponent = -2.0f * static_cast<float>(i) /
                                           static_cast<float>(args.rotary_dim);
                    const float frequency = args.inv_freq != nullptr
                                                ? static_cast<float>(args.inv_freq[i])
                                                : powf(args.theta, exponent);
                    const float angle = (static_cast<float>(start) /
                                         args.position_scale) * frequency;
                    const float cosine = cosf(angle);
                    const float sine = sinf(angle);
                    const float x0 = normed[i];
                    const float x1 = normed[d];
                    dst[d] = bf16_from(x1 * cosine + x0 * sine);
                } else {
                    dst[d] = bf16_from(normed[d]);
                }
                ++pool_count;
            }
            partial_count = 0;
            __syncthreads();
        }
    }
    if (d == 0u) {
        *args.partial_count = partial_count;
        *args.pool_count = pool_count;
    }
}

}  // namespace

hipError_t launch_qsa_pool(const QsaPoolArgs& args, hipStream_t stream) {    if (args.raw_keys == nullptr || args.norm_weight == nullptr || args.pooled == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.blocks * args.compress_ratio > args.rows) {
        return hipErrorInvalidValue;
    }
    qsa_pool_kernel<<<args.blocks, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_key_cache(const QsaKeyCacheArgs& args, hipStream_t stream) {
    if (args.raw_keys == nullptr || args.norm_weight == nullptr ||
        args.positions == nullptr || args.pooled == nullptr || args.partial == nullptr ||
        args.pool_count == nullptr || args.partial_count == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.max_blocks == 0u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.compress_ratio > 4u ||
        args.rotary_dim == 0u || (args.rotary_dim & 1u) != 0u ||
        args.rotary_dim > args.head_dim || args.position_scale < 1.0f ||
        args.position_scale > 2.0f) {
        return hipErrorInvalidValue;
    }
    qsa_key_cache_kernel<<<1, kQsaHeadDim, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_key_cache_batch(const QsaKeyCacheBatchArgs& args,
                                      hipStream_t stream) {
    if (args.raw_keys == nullptr || args.norm_weight == nullptr ||
        args.positions == nullptr || args.requests == nullptr ||
        args.pooled_base == nullptr || args.partial_base == nullptr ||
        args.pool_count_base == nullptr || args.partial_count_base == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.num_requests == 0u || args.actual_rows == 0u || args.max_sequences == 0u ||
        args.max_blocks == 0u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.compress_ratio > 4u ||
        args.rotary_dim == 0u || (args.rotary_dim & 1u) != 0u ||
        args.rotary_dim > args.head_dim || args.position_scale < 1.0f ||
        args.position_scale > 2.0f || args.raw_row_stride < args.head_dim ||
        args.pooled_slot_stride < args.pooled_layer_stride ||
        args.pooled_layer_stride < static_cast<std::uint64_t>(args.max_blocks) * args.head_dim ||
        args.partial_slot_stride < args.partial_layer_stride ||
        args.partial_layer_stride <
            static_cast<std::uint64_t>(args.compress_ratio) * args.head_dim ||
        args.count_slot_stride < args.count_layer_stride) {
        return hipErrorInvalidValue;
    }
    qsa_key_cache_batch_kernel<<<args.num_requests, kQsaHeadDim, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_select_batch(const QsaSelectBatchArgs& args, hipStream_t stream) {
    if (args.q == nullptr || args.pooled_base == nullptr || args.block_mask == nullptr ||
        args.row_sequence_slots == nullptr || args.positions == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.top_k == 0u ||
        args.n_heads == 0u || args.n_heads > kQsaSelectMaxHeads ||
        args.top_k > kQsaSelectMaxTopK || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.max_sequences == 0u ||
        args.q_row_stride < args.n_heads * args.head_dim ||
        args.pooled_slot_stride < args.pooled_layer_stride ||
        args.pooled_layer_stride <
            static_cast<uint64_t>(args.blocks) * args.head_dim ||
        args.scale == 0.0f) {
        return hipErrorInvalidValue;
    }
    const uint32_t mask_words = (args.blocks + 31u) / 32u;
    if (args.mask_row_bytes < mask_words * 4u || (args.mask_row_bytes % 4u) != 0u ||
        (reinterpret_cast<uintptr_t>(args.block_mask) % 4u) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.selected != nullptr && args.selected_row_stride < args.top_k) {
        return hipErrorInvalidValue;
    }
    uint32_t half = kQsaSelectChunk;
    while (half < args.top_k) half <<= 1u;
    const std::size_t dynamic_bytes =
        static_cast<std::size_t>(half * 2u) * (sizeof(float) + sizeof(std::int32_t));
    qsa_select_batch_kernel<<<args.rows, kQsaThreads, dynamic_bytes, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_rope(const QsaRopeArgs& args, hipStream_t stream) {
    if (args.x == nullptr || args.positions == nullptr) return hipErrorInvalidValue;
    if (args.rows == 0u || args.heads == 0u || args.head_dim == 0u ||
        args.rotary_dim == 0u || (args.rotary_dim & 1u) != 0u ||
        args.rotary_dim > args.head_dim || args.head_dim != kQsaHeadDim ||
        args.position_scale < 1.0f || args.position_scale > 2.0f) {
        return hipErrorInvalidValue;
    }
    qsa_rope_kernel<<<args.rows, kQsaHeadDim, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
