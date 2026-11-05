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
