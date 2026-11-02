#include <phaseshift/models/qwen35/kernels/eagle4/ops.h>
#include <phaseshift/kernels/optimized/detail/vector_io.h>
#include "../dflash2/detail/rope_device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ps::kernel {

namespace {

using bf16 = ::ps::kernel::detail::bf16;

constexpr uint32_t kEagle4NormThreads = 128u;

__device__ __forceinline__ float eagle4_block_reduce_sum(float value, float* scratch) {
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

__device__ __forceinline__ void eagle4_norm_body(
    const bf16* x,
    const bf16* weight,
    bf16* out,
    uint32_t features,
    float eps,
    float* scratch) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < features; i += blockDim.x) {
        const float v = __bfloat162float(x[i]);
        sum += v * v;
    }
    const float mean = eagle4_block_reduce_sum(sum, scratch) / static_cast<float>(features);
    const float inv = rsqrtf(mean + eps);
    for (uint32_t i = threadIdx.x; i < features; i += blockDim.x) {
        const float norm = __bfloat162float(x[i]) * inv;
        const float norm_rounded = __bfloat162float(__float2bfloat16(norm));
        out[i] = __float2bfloat16(__bfloat162float(weight[i]) * norm_rounded);
    }
}

__global__ void eagle4_tap_norm_concat_kernel(
    const bf16* tap0,
    const bf16* tap1,
    const bf16* tap2,
    const bf16* weight0,
    const bf16* weight1,
    const bf16* weight2,
    uint32_t tap_features,
    uint32_t rows,
    bf16* output,
    float eps) {
    __shared__ float scratch[kEagle4NormThreads];
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const std::size_t in_base = static_cast<std::size_t>(row) * tap_features;
    const std::size_t out_base =
        static_cast<std::size_t>(row) * tap_features * 3u;
    const bf16* taps[3] = {tap0, tap1, tap2};
    const bf16* weights[3] = {weight0, weight1, weight2};
    for (uint32_t t = 0; t < 3u; ++t) {
        eagle4_norm_body(taps[t] + in_base, weights[t],
                         output + out_base + static_cast<std::size_t>(t) * tap_features,
                         tap_features, eps, scratch);
        __syncthreads();
    }
}

__global__ void eagle4_layer_input_norm_kernel(
    const bf16* embed,
    const bf16* hidden,
    const bf16* embed_weight,
    const bf16* hidden_weight,
    uint32_t features,
    uint32_t rows,
    bf16* output,
    float eps) {
    __shared__ float scratch[kEagle4NormThreads];
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const std::size_t offset = static_cast<std::size_t>(row) * features;
    eagle4_norm_body(embed + offset, embed_weight, output + offset * 2u, features, eps,
                     scratch);
    __syncthreads();
    eagle4_norm_body(hidden + offset, hidden_weight, output + offset * 2u + features,
                     features, eps, scratch);
}

__global__ void eagle4_add_kernel(
    const bf16* lhs,
    const bf16* rhs,
    bf16* output,
    std::size_t count) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = __float2bfloat16(__bfloat162float(lhs[index]) +
                                     __bfloat162float(rhs[index]));
}

__device__ __forceinline__ float eagle4_block_reduce_max(float value, float* scratch) {
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

__global__ void eagle4_gemm_fallback_kernel(
    const bf16* weight,
    uint32_t out_features,
    uint32_t k,
    const bf16* input,
    uint32_t input_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t rows) {
    const uint32_t row = blockIdx.y;
    const uint32_t feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows || feature >= out_features) return;
    const bf16* weight_row = weight + static_cast<std::size_t>(feature) * k;
    const bf16* input_row = input + static_cast<std::size_t>(row) * input_row_stride;
    float acc = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        acc += __bfloat162float(weight_row[i]) * __bfloat162float(input_row[i]);
    }
    output[static_cast<std::size_t>(row) * output_row_stride + feature] =
        __float2bfloat16(acc);
}

__global__ void eagle4_argmax_kernel(
    const bf16* logits,
    uint32_t rows,
    uint32_t vocab,
    int32_t* out_ids) {
    __shared__ float scratch[kEagle4OpsThreads];
    __shared__ float s_best_value;
    __shared__ int32_t s_best_id;
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const bf16* rowp = logits + static_cast<std::size_t>(row) * vocab;
    float best = -INFINITY;
    int32_t best_id = 0;
    for (uint32_t v = threadIdx.x; v < vocab; v += blockDim.x) {
        const float value = __bfloat162float(rowp[v]);
        if (value > best) {
            best = value;
            best_id = static_cast<int32_t>(v);
        }
    }
    const float max_value = eagle4_block_reduce_max(best, scratch);
    if (threadIdx.x == 0u) {
        s_best_value = max_value;
        s_best_id = std::numeric_limits<int32_t>::max();
    }
    __syncthreads();
    if (best == s_best_value && best_id < s_best_id) {
        atomicMin(reinterpret_cast<unsigned int*>(&s_best_id),
                  static_cast<unsigned int>(best_id));
    }
    __syncthreads();
    if (threadIdx.x == 0u) out_ids[row] = s_best_id;
}

}  // namespace

hipError_t launch_eagle4_tap_norm_concat_bf16(
    const bf16_t* tap0,
    const bf16_t* tap1,
    const bf16_t* tap2,
    const bf16_t* weight0,
    const bf16_t* weight1,
    const bf16_t* weight2,
    uint32_t tap_features,
    uint32_t rows,
    bf16_t* output,
    float eps,
    hipStream_t stream) {
    if (tap0 == nullptr || tap1 == nullptr || tap2 == nullptr || weight0 == nullptr ||
        weight1 == nullptr || weight2 == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || tap_features == 0u || tap_features % kEagle4NormThreads != 0u) {
        return hipErrorInvalidValue;
    }
    eagle4_tap_norm_concat_kernel<<<rows, kEagle4NormThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(tap0), reinterpret_cast<const bf16*>(tap1),
        reinterpret_cast<const bf16*>(tap2), reinterpret_cast<const bf16*>(weight0),
        reinterpret_cast<const bf16*>(weight1), reinterpret_cast<const bf16*>(weight2),
        tap_features, rows, reinterpret_cast<bf16*>(output), eps);
    return hipGetLastError();
}

hipError_t launch_eagle4_layer_input_norm_bf16(
    const bf16_t* embed,
    const bf16_t* hidden,
    const bf16_t* embed_weight,
    const bf16_t* hidden_weight,
    uint32_t features,
    uint32_t rows,
    bf16_t* output,
    float eps,
    hipStream_t stream) {
    if (embed == nullptr || hidden == nullptr || embed_weight == nullptr ||
        hidden_weight == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || features == 0u || features % kEagle4NormThreads != 0u) {
        return hipErrorInvalidValue;
    }
    eagle4_layer_input_norm_kernel<<<rows, kEagle4NormThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(embed), reinterpret_cast<const bf16*>(hidden),
        reinterpret_cast<const bf16*>(embed_weight),
        reinterpret_cast<const bf16*>(hidden_weight), features, rows,
        reinterpret_cast<bf16*>(output), eps);
    return hipGetLastError();
}

hipError_t launch_eagle4_add_bf16(
    const bf16_t* lhs,
    const bf16_t* rhs,
    bf16_t* output,
    std::size_t count,
    hipStream_t stream) {
    if (lhs == nullptr || rhs == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (count == 0u) return hipSuccess;
    const uint32_t block = 256u;
    const uint32_t grid = static_cast<uint32_t>((count + block - 1u) / block);
    eagle4_add_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const bf16*>(lhs), reinterpret_cast<const bf16*>(rhs),
        reinterpret_cast<bf16*>(output), count);
    return hipGetLastError();
}

hipError_t launch_eagle4_gemm_fallback_bf16(
    const bf16_t* weight,
    uint32_t out_features,
    uint32_t k,
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    hipStream_t stream) {
    if (weight == nullptr || input == nullptr || output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || out_features == 0u || k == 0u) return hipErrorInvalidValue;
    if (input_row_stride < k || output_row_stride < out_features) {
        return hipErrorInvalidValue;
    }
    const dim3 grid((out_features + kEagle4OpsThreads - 1u) / kEagle4OpsThreads, rows, 1u);
    eagle4_gemm_fallback_kernel<<<grid, kEagle4OpsThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(weight), out_features, k,
        reinterpret_cast<const bf16*>(input), input_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride, rows);
    return hipGetLastError();
}

hipError_t launch_eagle4_argmax_bf16(
    const bf16_t* logits,
    uint32_t rows,
    uint32_t vocab,
    int32_t* out_ids,
    hipStream_t stream) {
    if (logits == nullptr || out_ids == nullptr) return hipErrorInvalidValue;
    if (rows == 0u || vocab == 0u) return hipErrorInvalidValue;
    eagle4_argmax_kernel<<<rows, kEagle4OpsThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(logits), rows, vocab, out_ids);
    return hipGetLastError();
}

namespace {

__global__ void eagle4_rope_uniform_kernel(
    const bf16* input,
    uint32_t input_row_stride,
    bf16* output,
    uint32_t output_row_stride,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t position,
    float theta) {
    const uint32_t row = blockIdx.y;
    const uint32_t head = blockIdx.x;
    const uint32_t pair = threadIdx.x;
    const uint32_t half = head_dim / 2u;
    if (row >= rows || pair >= half) return;
    const uint32_t base = head * head_dim;
    const bf16* src = input + static_cast<std::size_t>(row) * input_row_stride + base;
    bf16* dst = output + static_cast<std::size_t>(row) * output_row_stride + base;
    detail::dflash2_rope_pair_body(src, dst, pair, head_dim, position, theta);
}

__global__ void eagle4_round_kv_append_kernel(
    const bf16* k_src,
    const bf16* v_src,
    bf16* k_round,
    bf16* v_round,
    uint32_t rows,
    uint32_t kv_features,
    uint32_t round_capacity,
    const int32_t* row_nodes) {
    const std::size_t total = static_cast<std::size_t>(rows) * kv_features;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint32_t row = static_cast<uint32_t>(i / kv_features);
        const uint32_t feature = static_cast<uint32_t>(i % kv_features);
        const int32_t node = row_nodes[row];
        if (node < 0 || static_cast<uint32_t>(node) >= round_capacity) continue;
        const std::size_t dst = static_cast<std::size_t>(node) * kv_features + feature;
        k_round[dst] = k_src[i];
        v_round[dst] = v_src[i];
    }
}

__global__ void eagle4_gather_rows_kernel(
    const bf16* src,
    uint32_t src_row_stride,
    bf16* dst,
    uint32_t dst_row_stride,
    const int32_t* src_rows,
    uint32_t rows,
    uint32_t features) {
    const std::size_t total = static_cast<std::size_t>(rows) * features;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint32_t row = static_cast<uint32_t>(i / features);
        const uint32_t feature = static_cast<uint32_t>(i % features);
        const int32_t src_row = src_rows[row];
        if (src_row < 0) continue;
        dst[static_cast<std::size_t>(row) * dst_row_stride + feature] =
            src[static_cast<std::size_t>(src_row) * src_row_stride + feature];
    }
}

}  // namespace

hipError_t launch_eagle4_rope_uniform_bf16(
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t position,
    float theta,
    hipStream_t stream) {
    if (input == nullptr || output == nullptr) return hipErrorInvalidValue;
    if (rows == 0u || head_dim == 0u) return hipErrorInvalidValue;
    if (input_row_stride % head_dim != 0u || output_row_stride % head_dim != 0u) {
        return hipErrorInvalidValue;
    }
    const dim3 grid(input_row_stride / head_dim, rows, 1u);
    eagle4_rope_uniform_kernel<<<grid, head_dim / 2u, 0, stream>>>(
        reinterpret_cast<const bf16*>(input), input_row_stride,
        reinterpret_cast<bf16*>(output), output_row_stride, rows, head_dim, position, theta);
    return hipGetLastError();
}

hipError_t launch_eagle4_round_kv_append_bf16(
    const bf16_t* k_src,
    const bf16_t* v_src,
    bf16_t* k_round,
    bf16_t* v_round,
    uint32_t rows,
    uint32_t kv_features,
    uint32_t round_capacity,
    const int32_t* row_nodes,
    hipStream_t stream) {
    if (k_src == nullptr || v_src == nullptr || k_round == nullptr || v_round == nullptr ||
        row_nodes == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u || kv_features == 0u || round_capacity == 0u) return hipErrorInvalidValue;
    const std::size_t total = static_cast<std::size_t>(rows) * kv_features;
    const uint32_t blocks = static_cast<uint32_t>(
        std::min<std::size_t>((total + kEagle4OpsThreads - 1u) / kEagle4OpsThreads, 4096u));
    eagle4_round_kv_append_kernel<<<blocks, kEagle4OpsThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(k_src), reinterpret_cast<const bf16*>(v_src),
        reinterpret_cast<bf16*>(k_round), reinterpret_cast<bf16*>(v_round), rows, kv_features,
        round_capacity, row_nodes);
    return hipGetLastError();
}

hipError_t launch_eagle4_gather_rows_bf16(
    const bf16_t* src,
    uint32_t src_row_stride,
    bf16_t* dst,
    uint32_t dst_row_stride,
    const int32_t* src_rows,
    uint32_t rows,
    uint32_t features,
    hipStream_t stream) {
    if (src == nullptr || dst == nullptr || src_rows == nullptr) return hipErrorInvalidValue;
    if (rows == 0u || features == 0u) return hipErrorInvalidValue;
    if (src_row_stride < features || dst_row_stride < features) return hipErrorInvalidValue;
    const std::size_t total = static_cast<std::size_t>(rows) * features;
    const uint32_t blocks = static_cast<uint32_t>(
        std::min<std::size_t>((total + kEagle4OpsThreads - 1u) / kEagle4OpsThreads, 4096u));
    eagle4_gather_rows_kernel<<<blocks, kEagle4OpsThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(src), src_row_stride, reinterpret_cast<bf16*>(dst),
        dst_row_stride, src_rows, rows, features);
    return hipGetLastError();
}

}  // namespace ps::kernel
