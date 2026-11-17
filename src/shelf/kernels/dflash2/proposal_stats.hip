#include <shelf/kernels/dflash2/proposal_stats.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

namespace {

constexpr uint32_t kStatsThreads = 256u;

__global__ void dflash2_row_logsumexp_kernel(
    const float* __restrict__ logits,
    uint32_t vocab,
    uint32_t row_stride,
    float* __restrict__ stats) {
    __shared__ float partial[kStatsThreads];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float* row_logits = logits + static_cast<std::size_t>(row) * row_stride;

    float maximum = -INFINITY;
    for (uint32_t v = tid; v < vocab; v += kStatsThreads) {
        maximum = fmaxf(maximum, row_logits[v]);
    }
    partial[tid] = maximum;
    __syncthreads();
    for (uint32_t step = kStatsThreads / 2u; step > 0u; step >>= 1u) {
        if (tid < step) partial[tid] = fmaxf(partial[tid], partial[tid + step]);
        __syncthreads();
    }
    maximum = partial[0];

    float sum = 0.0f;
    for (uint32_t v = tid; v < vocab; v += kStatsThreads) {
        sum += expf(row_logits[v] - maximum);
    }
    __syncthreads();
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t step = kStatsThreads / 2u; step > 0u; step >>= 1u) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0u) {
        stats[static_cast<std::size_t>(row) * 2u] = maximum;
        stats[static_cast<std::size_t>(row) * 2u + 1u] = partial[0];
    }
}


__global__ void dflash2_combine_denominators_kernel(
    const float* __restrict__ rank_stats,
    uint32_t ranks,
    uint32_t rank_stride,
    float* __restrict__ out) {
    const uint32_t row = blockIdx.x;
    if (threadIdx.x != 0u) return;
    float maximum = -INFINITY;
    for (uint32_t rank = 0u; rank < ranks; ++rank) {
        const float local_max =
            rank_stats[static_cast<std::size_t>(rank) * rank_stride +
                       static_cast<std::size_t>(row) * 2u];
        maximum = fmaxf(maximum, local_max);
    }
    float total = 0.0f;
    for (uint32_t rank = 0u; rank < ranks; ++rank) {
        const std::size_t base = static_cast<std::size_t>(rank) * rank_stride +
                                 static_cast<std::size_t>(row) * 2u;
        total += expf(rank_stats[base] - maximum) * rank_stats[base + 1u];
    }
    out[static_cast<std::size_t>(row) * 2u] = maximum;
    out[static_cast<std::size_t>(row) * 2u + 1u] = total;
}

}  // namespace

hipError_t launch_dflash2_row_logsumexp(
    const float* logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t row_stride,
    float* stats,
    hipStream_t stream) {
    if (logits == nullptr || stats == nullptr) return hipErrorInvalidValue;
    if (rows == 0u) return hipSuccess;
    if (vocab == 0u || row_stride < vocab) return hipErrorInvalidValue;
    dflash2_row_logsumexp_kernel<<<rows, kStatsThreads, 0, stream>>>(
        logits, vocab, row_stride, stats);
    return hipGetLastError();
}

hipError_t launch_dflash2_combine_denominators(
    const float* rank_stats,
    uint32_t ranks,
    uint32_t rank_stride,
    uint32_t rows,
    float* out,
    hipStream_t stream) {
    if (rank_stats == nullptr || out == nullptr) return hipErrorInvalidValue;
    if (rows == 0u) return hipSuccess;
    if (ranks == 0u || rank_stride < rows * 2u) return hipErrorInvalidValue;
    dflash2_combine_denominators_kernel<<<rows, 1u, 0, stream>>>(
        rank_stats, ranks, rank_stride, out);
    return hipGetLastError();
}

}  // namespace shelf::kernel
