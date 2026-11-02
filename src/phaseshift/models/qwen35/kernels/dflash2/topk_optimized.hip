#include <phaseshift/models/qwen35/kernels/dflash2/topk.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

namespace {

constexpr uint32_t kK = kDflash2TopKMaxK;

__device__ __forceinline__ bool topk_better(float la, int32_t ia, float lb, int32_t ib) {
    if (la > lb) return true;
    if (la < lb) return false;
    return ia < ib;
}

struct TopKBest {
    float value;
    int32_t id;
    int32_t index;
};

__device__ __forceinline__ TopKBest topk_pick(TopKBest a, TopKBest b) {
    return topk_better(b.value, b.id, a.value, a.id) ? b : a;
}

__device__ __forceinline__ TopKBest topk_warp_best(TopKBest b) {
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        TopKBest o;
        o.value = __shfl_down(b.value, off);
        o.id = __shfl_down(b.id, off);
        o.index = __shfl_down(b.index, off);
        b = topk_pick(b, o);
    }
    return b;
}

template <uint32_t THREADS>
__device__ __forceinline__ TopKBest topk_block_best(
    TopKBest b,
    float* shared_value,
    int32_t* shared_id,
    int32_t* shared_index) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    constexpr uint32_t WARPS = THREADS / 32u;
    b = topk_warp_best(b);
    if (lane == 0u) {
        shared_value[warp] = b.value;
        shared_id[warp] = b.id;
        shared_index[warp] = b.index;
    }
    __syncthreads();
    TopKBest r{-INFINITY, 0x7FFFFFFF, -1};
    if (warp == 0u) {
        if (lane < WARPS) {
            r.value = shared_value[lane];
            r.id = shared_id[lane];
            r.index = shared_index[lane];
        }
        r = topk_warp_best(r);
        if (lane == 0u) {
            shared_value[0] = r.value;
            shared_id[0] = r.id;
            shared_index[0] = r.index;
        }
    }
    __syncthreads();
    r.value = shared_value[0];
    r.id = shared_id[0];
    r.index = shared_index[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ void topk_insert(
    float* values,
    int32_t* ids,
    float value,
    int32_t id) {
    if (!topk_better(value, id, values[kK - 1u], ids[kK - 1u])) return;
    uint32_t pos = kK - 1u;
    while (pos > 0u && topk_better(value, id, values[pos - 1u], ids[pos - 1u])) {
        values[pos] = values[pos - 1u];
        ids[pos] = ids[pos - 1u];
        --pos;
    }
    values[pos] = value;
    ids[pos] = id;
}

template <uint32_t THREADS>
__global__ void dflash2_topk_partition_hier_kernel(
    const float* logits,
    uint32_t vocab_size,
    uint32_t row_stride,
    uint32_t partitions,
    uint32_t partition_size,
    uint32_t k,
    int32_t* scratch_ids,
    float* scratch_logits,
    uint32_t scratch_row_stride) {
    const uint32_t row = blockIdx.x / partitions;
    const uint32_t part = blockIdx.x % partitions;
    const uint32_t tid = threadIdx.x;
    const float* row_logits = logits + static_cast<std::size_t>(row) * row_stride;

    const uint32_t start = part * partition_size;
    const uint32_t end = start + partition_size < vocab_size ? start + partition_size
                                                             : vocab_size;
    float local_values[kK];
    int32_t local_ids[kK];
#pragma unroll
    for (uint32_t i = 0u; i < kK; ++i) {
        local_values[i] = -INFINITY;
        local_ids[i] = 0x7FFFFFFF;
    }
    for (uint32_t v = start + tid; v < end; v += THREADS) {
        const float value = row_logits[v];
        if (value == -INFINITY) continue;
        topk_insert(local_values, local_ids, value, static_cast<int32_t>(v));
    }

    __shared__ float shared_value[THREADS / 32u];
    __shared__ int32_t shared_id[THREADS / 32u];
    __shared__ int32_t shared_index[THREADS / 32u];

    int32_t* out_ids =
        scratch_ids + static_cast<std::size_t>(row) * scratch_row_stride + part * k;
    float* out_logits =
        scratch_logits + static_cast<std::size_t>(row) * scratch_row_stride + part * k;

    for (uint32_t o = 0u; o < k; ++o) {
        TopKBest b{-INFINITY, 0x7FFFFFFF, -1};
#pragma unroll
        for (uint32_t i = 0u; i < kK; ++i) {
            if (topk_better(local_values[i], local_ids[i], b.value, b.id)) {
                b.value = local_values[i];
                b.id = local_ids[i];
                b.index = static_cast<int32_t>(tid * kK + i);
            }
        }
        b = topk_block_best<THREADS>(b, shared_value, shared_id, shared_index);
        if (tid == 0u) {
            out_logits[o] = b.value;
            out_ids[o] = b.id;
        }
        if (b.index >= 0) {
            const uint32_t owner = static_cast<uint32_t>(b.index) / kK;
            const uint32_t slot = static_cast<uint32_t>(b.index) % kK;
            if (owner == tid) {
                local_values[slot] = -INFINITY;
                local_ids[slot] = 0x7FFFFFFF;
            }
        }
    }
}

template <uint32_t THREADS>
__global__ void dflash2_topk_merge_hier_kernel(
    const int32_t* scratch_ids,
    float* scratch_logits,
    uint32_t partitions,
    uint32_t k,
    uint32_t scratch_row_stride,
    int32_t* out_ids,
    float* out_logits) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t count = partitions * k;
    const int32_t* row_ids =
        scratch_ids + static_cast<std::size_t>(row) * scratch_row_stride;
    float* row_logits =
        scratch_logits + static_cast<std::size_t>(row) * scratch_row_stride;

    __shared__ float shared_value[THREADS / 32u];
    __shared__ int32_t shared_id[THREADS / 32u];
    __shared__ int32_t shared_index[THREADS / 32u];

    for (uint32_t o = 0u; o < k; ++o) {
        TopKBest b{-INFINITY, 0x7FFFFFFF, -1};
        for (uint32_t i = tid; i < count; i += THREADS) {
            const float value = row_logits[i];
            if (value == -INFINITY) continue;
            const int32_t id = row_ids[i];
            if (topk_better(value, id, b.value, b.id)) {
                b.value = value;
                b.id = id;
                b.index = static_cast<int32_t>(i);
            }
        }
        b = topk_block_best<THREADS>(b, shared_value, shared_id, shared_index);
        if (tid == 0u) {
            const bool found = b.index >= 0;
            out_ids[static_cast<std::size_t>(row) * k + o] = found ? b.id : 0;
            out_logits[static_cast<std::size_t>(row) * k + o] =
                found ? b.value : -INFINITY;
        }
        if (b.index >= 0) {
            row_logits[b.index] = -INFINITY;
        }
    }
}

template <uint32_t THREADS>
hipError_t launch_optimized(
    const float* logits,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t row_stride,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    int32_t* scratch_ids,
    float* scratch_logits,
    uint32_t partitions,
    hipStream_t stream) {
    const uint32_t partition_size = (vocab_size + partitions - 1u) / partitions;
    dflash2_topk_partition_hier_kernel<THREADS>
        <<<rows * partitions, THREADS, 0, stream>>>(
            logits, vocab_size, row_stride, partitions, partition_size, k, scratch_ids,
            scratch_logits, kDflash2TopKScratchPerRow);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return err;
    dflash2_topk_merge_hier_kernel<THREADS><<<rows, THREADS, 0, stream>>>(
        scratch_ids, scratch_logits, partitions, k, kDflash2TopKScratchPerRow, out_ids,
        out_logits);
    return hipGetLastError();
}

}  // namespace

DFlash2TopKPlan dflash2_topk_default_plan() {
    DFlash2TopKPlan plan;
    plan.partitions = 32u;
    plan.threads = 256u;
    return plan;
}

hipError_t launch_dflash2_topk_f32_optimized(
    const float* logits,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t row_stride,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    int32_t* scratch_ids,
    float* scratch_logits,
    const DFlash2TopKPlan& plan,
    hipStream_t stream) {
    if (logits == nullptr || out_ids == nullptr || out_logits == nullptr ||
        scratch_ids == nullptr || scratch_logits == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u) return hipSuccess;
    if (k == 0u || k > kK) return hipErrorInvalidValue;
    if (vocab_size == 0u || row_stride < vocab_size) return hipErrorInvalidValue;
    if (plan.partitions == 0u || plan.partitions > kDflash2TopKMaxPartitions) {
        return hipErrorInvalidValue;
    }
    switch (plan.threads) {
        case 64u:
            return launch_optimized<64u>(logits, rows, vocab_size, row_stride, k, out_ids,
                                         out_logits, scratch_ids, scratch_logits,
                                         plan.partitions, stream);
        case 128u:
            return launch_optimized<128u>(logits, rows, vocab_size, row_stride, k, out_ids,
                                          out_logits, scratch_ids, scratch_logits,
                                          plan.partitions, stream);
        case 256u:
            return launch_optimized<256u>(logits, rows, vocab_size, row_stride, k, out_ids,
                                          out_logits, scratch_ids, scratch_logits,
                                          plan.partitions, stream);
        default:
            return hipErrorInvalidValue;
    }
}

}  // namespace ps::kernel
