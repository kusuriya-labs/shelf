#include <shelf/kernels/optimized/qsa.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

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
        args.selected_blocks + static_cast<std::uint64_t>(row) * args.selected_per_row;
    uint8_t* mask = args.mask + static_cast<std::uint64_t>(row) * args.total_len;
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
        args.compress_ratio == 0u || args.total_len == 0u) {
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
