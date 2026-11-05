#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::kernel {
namespace {

constexpr uint32_t kThreads = kQsaThreads;
constexpr uint32_t kQsaFastPathBlocks = 512u;

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

__global__ __launch_bounds__(kThreads) void qsa_scores_kernel(QsaScoresArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t d = threadIdx.x;
    __shared__ float scratch[kThreads];
    const uint32_t visible = args.visible[row] < args.blocks ? args.visible[row] : args.blocks;
    const shelf::bf16_t* q_row =
        args.q + static_cast<std::uint64_t>(row) * args.n_heads * args.head_dim;
    for (uint32_t block = 0; block < args.blocks; ++block) {
        if (block >= visible) {
            if (d == 0u) args.scores[static_cast<std::uint64_t>(row) * args.blocks + block] =
                -INFINITY;
            continue;
        }
        float total = 0.0f;
        for (uint32_t head = 0; head < args.n_heads; ++head) {
            const float qv = bf16_value(q_row + head * args.head_dim + d);
            const float kv =
                bf16_value(args.pooled + static_cast<std::uint64_t>(block) * args.head_dim + d);
            const float dot = block_sum(scratch, qv * kv);
            __syncthreads();
            if (d == 0u) total += fmaxf(dot, 0.0f);
            __syncthreads();
        }
        if (d == 0u) {
            args.scores[static_cast<std::uint64_t>(row) * args.blocks + block] =
                total * args.scale;
        }
        __syncthreads();
    }
}

__global__ void __launch_bounds__(1) qsa_topk_kernel(QsaTopkArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t visible =
        args.visible[row] < args.blocks ? args.visible[row] : args.blocks;
    int32_t* out = args.indices + static_cast<std::uint64_t>(row) * args.k;
    for (uint32_t j = 0; j < args.k; ++j) out[j] = -1;
    // Every visible block fits in the budget: one selection pass over `visible` blocks
    // (instead of `k` passes) yields the same score-descending order.
    if (visible <= args.k && visible <= kQsaFastPathBlocks) {
        bool used[kQsaFastPathBlocks] = {};
        for (uint32_t j = 0; j < visible; ++j) {
            int32_t best = -1;
            float best_score = -INFINITY;
            for (uint32_t block = 0; block < visible; ++block) {
                if (used[block]) continue;
                const float score =
                    args.scores[static_cast<std::uint64_t>(row) * args.blocks + block];
                if (score > best_score ||
                    (score == best_score && static_cast<int32_t>(block) < best)) {
                    best_score = score;
                    best = static_cast<int32_t>(block);
                }
            }
            out[j] = best;
            used[best] = true;
        }
        return;
    }
    for (uint32_t j = 0; j < args.k && j < visible; ++j) {
        int32_t best = -1;
        float best_score = -INFINITY;
        for (uint32_t block = 0; block < visible; ++block) {
            bool used = false;
            for (uint32_t p = 0; p < j; ++p) {
                if (out[p] == static_cast<int32_t>(block)) {
                    used = true;
                    break;
                }
            }
            if (used) continue;
            const float score =
                args.scores[static_cast<std::uint64_t>(row) * args.blocks + block];
            if (score > best_score) {
                best_score = score;
                best = static_cast<int32_t>(block);
            }
        }
        out[j] = best;
    }
}

__global__ void __launch_bounds__(1) qsa_topk_batch_kernel(QsaTopkBatchArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t complete = (args.positions[row] + 1u) / args.compress_ratio;
    const uint32_t visible = complete < args.blocks ? complete : args.blocks;
    int32_t* out = args.indices + static_cast<std::uint64_t>(row) * args.indices_row_stride;
    for (uint32_t j = 0; j < args.k; ++j) out[j] = -1;
    if (visible <= args.k && visible <= kQsaFastPathBlocks) {
        bool used[kQsaFastPathBlocks] = {};
        for (uint32_t j = 0; j < visible; ++j) {
            int32_t best = -1;
            float best_score = -INFINITY;
            for (uint32_t block = 0; block < visible; ++block) {
                if (used[block]) continue;
                const float score =
                    args.scores[static_cast<std::uint64_t>(row) * args.scores_row_stride +
                                block];
                if (score > best_score ||
                    (score == best_score && static_cast<int32_t>(block) < best)) {
                    best_score = score;
                    best = static_cast<int32_t>(block);
                }
            }
            out[j] = best;
            used[best] = true;
        }
        return;
    }
    for (uint32_t j = 0; j < args.k && j < visible; ++j) {
        int32_t best = -1;
        float best_score = -INFINITY;
        for (uint32_t block = 0; block < visible; ++block) {
            bool used = false;
            for (uint32_t p = 0; p < j; ++p) {
                if (out[p] == static_cast<int32_t>(block)) {
                    used = true;
                    break;
                }
            }
            if (used) continue;
            const float score =
                args.scores[static_cast<std::uint64_t>(row) * args.scores_row_stride +
                            block];
            if (score > best_score) {
                best_score = score;
                best = static_cast<int32_t>(block);
            }
        }
        out[j] = best;
    }
}

__global__ __launch_bounds__(kQsaThreads) void qsa_scores_batch_kernel(
    QsaScoresBatchArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t d = threadIdx.x;
    __shared__ float scratch[kQsaThreads];
    const uint32_t slot = args.row_sequence_slots[row];
    const uint32_t complete = (args.positions[row] + 1u) / args.compress_ratio;
    const uint32_t visible = complete < args.blocks ? complete : args.blocks;
    if (slot >= args.max_sequences) {
        if (d == 0u) {
            for (uint32_t block = 0; block < args.blocks; ++block) {
                args.scores[static_cast<std::uint64_t>(row) * args.scores_row_stride +
                            block] = -INFINITY;
            }
        }
        return;
    }
    const bf16_t* pooled =
        args.pooled_base +
        static_cast<std::uint64_t>(slot) * args.pooled_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.pooled_layer_stride;
    const bf16_t* q_row =
        args.q + static_cast<std::uint64_t>(row) * args.q_row_stride;
    for (uint32_t block = 0; block < args.blocks; ++block) {
        if (block >= visible) {
            if (d == 0u) {
                args.scores[static_cast<std::uint64_t>(row) * args.scores_row_stride +
                            block] = -INFINITY;
            }
            continue;
        }
        float total = 0.0f;
        for (uint32_t head = 0; head < args.n_heads; ++head) {
            const float qv = bf16_value(q_row + head * args.head_dim + d);
            const float kv =
                bf16_value(pooled + static_cast<std::uint64_t>(block) * args.head_dim + d);
            const float dot = block_sum(scratch, qv * kv);
            __syncthreads();
            if (d == 0u) total += fmaxf(dot, 0.0f);
            __syncthreads();
        }
        if (d == 0u) {
            args.scores[static_cast<std::uint64_t>(row) * args.scores_row_stride +
                        block] = total * args.scale;
        }
        __syncthreads();
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

__global__ __launch_bounds__(kQsaHeadDim) void qsa_expand_mask_kernel(
    QsaExpandMaskArgs args) {
    const uint32_t row = blockIdx.x;
    const uint32_t position = args.positions[row];
    const uint32_t complete_blocks = (position + 1u) / args.compress_ratio;
    const int32_t* selected =
        args.selected_blocks + static_cast<std::uint64_t>(row) * args.selected_row_stride;
    uint8_t* mask = args.mask + static_cast<std::uint64_t>(row) * args.mask_row_stride;
    for (uint32_t token = threadIdx.x; token < args.total_len; token += blockDim.x) {
        bool visible = false;
        if (token <= position) {
            const uint32_t block = token / args.compress_ratio;
            if (token >= complete_blocks * args.compress_ratio) {
                visible = true;
            } else {
                for (uint32_t i = 0; i < args.selected_per_row; ++i) {
                    if (selected[i] == static_cast<int32_t>(block)) {
                        visible = true;
                        break;
                    }
                }
            }
        }
        mask[token] = visible ? 1u : 0u;
    }
}

}  // namespace

hipError_t launch_qsa_pool(const QsaPoolArgs& args, hipStream_t stream) {
    if (args.raw_keys == nullptr || args.norm_weight == nullptr || args.pooled == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.head_dim != kQsaHeadDim ||
        args.compress_ratio == 0u || args.blocks * args.compress_ratio > args.rows) {
        return hipErrorInvalidValue;
    }
    qsa_pool_kernel<<<args.blocks, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_expand_mask(const QsaExpandMaskArgs& args, hipStream_t stream) {
    if (args.selected_blocks == nullptr || args.positions == nullptr ||
        args.mask == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.selected_per_row == 0u ||
        args.compress_ratio == 0u || args.total_len == 0u ||
        args.selected_row_stride < args.selected_per_row ||
        args.mask_row_stride < args.total_len) {
        return hipErrorInvalidValue;
    }
    qsa_expand_mask_kernel<<<args.rows, kQsaHeadDim, 0, stream>>>(args);
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

hipError_t launch_qsa_scores_batch(const QsaScoresBatchArgs& args,
                                   hipStream_t stream) {
    if (args.q == nullptr || args.pooled_base == nullptr || args.scores == nullptr ||
        args.row_sequence_slots == nullptr || args.positions == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.n_heads == 0u ||
        args.head_dim != kQsaHeadDim || args.compress_ratio == 0u ||
        args.scale == 0.0f || args.max_sequences == 0u ||
        args.q_row_stride < args.n_heads * args.head_dim ||
        args.scores_row_stride < args.blocks ||
        args.pooled_slot_stride < args.pooled_layer_stride) {
        return hipErrorInvalidValue;
    }
    qsa_scores_batch_kernel<<<args.rows, kQsaThreads, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_topk_batch(const QsaTopkBatchArgs& args, hipStream_t stream) {
    if (args.scores == nullptr || args.positions == nullptr || args.indices == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.k == 0u ||
        args.compress_ratio == 0u || args.scores_row_stride < args.blocks ||
        args.indices_row_stride < args.k) {
        return hipErrorInvalidValue;
    }
    qsa_topk_batch_kernel<<<args.rows, 1, 0, stream>>>(args);
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

hipError_t launch_qsa_topk(const QsaTopkArgs& args, hipStream_t stream) {
    if (args.scores == nullptr || args.visible == nullptr || args.indices == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.k == 0u) {
        return hipErrorInvalidValue;
    }
    qsa_topk_kernel<<<args.rows, 1, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_qsa_scores(const QsaScoresArgs& args, hipStream_t stream) {
    if (args.q == nullptr || args.pooled == nullptr || args.scores == nullptr ||
        args.visible == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.blocks == 0u || args.n_heads == 0u ||
        args.head_dim != kQsaHeadDim || args.scale == 0.0f) {
        return hipErrorInvalidValue;
    }
    qsa_scores_kernel<<<args.rows, kThreads, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
