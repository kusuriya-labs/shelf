#include <shelf/kernels/dflash2/topk.h>
#include <shelf/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

using namespace ::shelf::kernel::detail;

namespace {

constexpr uint32_t kTopK = kDflash2TopKMaxK;
constexpr uint32_t kTopKThreads = kDflash2TopKThreads;

__device__ __forceinline__ bool topk_better(float la, int32_t ia, float lb, int32_t ib) {
    if (la > lb) return true;
    if (la < lb) return false;
    return ia < ib;
}

__device__ __forceinline__ void topk_insert(
    float* logits,
    int32_t* ids,
    float value,
    int32_t id) {
    if (!topk_better(value, id, logits[kTopK - 1u], ids[kTopK - 1u])) {
        return;
    }
    uint32_t pos = kTopK - 1u;
    while (pos > 0u &&
           topk_better(value, id, logits[pos - 1u], ids[pos - 1u])) {
        logits[pos] = logits[pos - 1u];
        ids[pos] = ids[pos - 1u];
        --pos;
    }
    logits[pos] = value;
    ids[pos] = id;
}

__global__ void dflash2_topk_partition_kernel(
    const float* logits,
    uint32_t vocab_size,
    uint32_t row_stride,
    int32_t* scratch_ids,
    float* scratch_logits) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const float* row_logits = logits + static_cast<std::size_t>(row) * row_stride;

    float local_logits[kTopK];
    int32_t local_ids[kTopK];
#pragma unroll
    for (uint32_t i = 0u; i < kTopK; ++i) {
        local_logits[i] = -INFINITY;
        local_ids[i] = 0x7FFFFFFF;
    }
    for (uint32_t v = tid; v < vocab_size; v += kTopKThreads) {
        const float value = row_logits[v];
        if (value == -INFINITY) continue;
        topk_insert(local_logits, local_ids, value, static_cast<int32_t>(v));
    }

    int32_t* slot_ids = scratch_ids +
        static_cast<std::size_t>(row) * kDflash2TopKScratchPerRow + tid * kTopK;
    float* slot_logits = scratch_logits +
        static_cast<std::size_t>(row) * kDflash2TopKScratchPerRow + tid * kTopK;
#pragma unroll
    for (uint32_t i = 0u; i < kTopK; ++i) {
        slot_logits[i] = local_logits[i];
        slot_ids[i] = local_ids[i];
    }
}

__global__ void dflash2_topk_merge_kernel(
    const int32_t* scratch_ids,
    float* scratch_logits,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    int32_t id_offset) {
    const uint32_t row = blockIdx.x;
    if (threadIdx.x != 0u) return;
    const int32_t* row_ids =
        scratch_ids + static_cast<std::size_t>(row) * kDflash2TopKScratchPerRow;
    float* row_logits =
        scratch_logits + static_cast<std::size_t>(row) * kDflash2TopKScratchPerRow;

    for (uint32_t slot = 0u; slot < k; ++slot) {
        float best_logit = -INFINITY;
        int32_t best_id = 0x7FFFFFFF;
        int32_t best_index = -1;
        for (uint32_t i = 0u; i < kDflash2TopKScratchPerRow; ++i) {
            const float value = row_logits[i];
            if (value == -INFINITY) continue;
            const int32_t id = row_ids[i];
            if (topk_better(value, id, best_logit, best_id)) {
                best_logit = value;
                best_id = id;
                best_index = static_cast<int32_t>(i);
            }
        }
        if (best_index < 0) {
            out_ids[static_cast<std::size_t>(row) * k + slot] = 0;
            out_logits[static_cast<std::size_t>(row) * k + slot] = -INFINITY;
            continue;
        }
        out_ids[static_cast<std::size_t>(row) * k + slot] = best_id + id_offset;
        out_logits[static_cast<std::size_t>(row) * k + slot] = best_logit;
        row_logits[best_index] = -INFINITY;
    }
}


__global__ void dflash2_merge_topk_kernel(
    const int32_t* candidate_ids,
    const float* candidate_logits,
    uint32_t candidates,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits) {
    const uint32_t row = blockIdx.x;
    if (threadIdx.x != 0u) return;
    const int32_t* row_ids =
        candidate_ids + static_cast<std::size_t>(row) * candidates;
    const float* row_logits =
        candidate_logits + static_cast<std::size_t>(row) * candidates;

    float best_logits[kTopK];
    int32_t best_ids[kTopK];
    for (uint32_t i = 0u; i < k; ++i) {
        best_logits[i] = -INFINITY;
        best_ids[i] = 0x7FFFFFFF;
    }
    for (uint32_t i = 0u; i < candidates; ++i) {
        const float value = row_logits[i];
        if (value == -INFINITY) continue;
        topk_insert(best_logits, best_ids, value, row_ids[i]);
    }
    for (uint32_t i = 0u; i < k; ++i) {
        out_ids[static_cast<std::size_t>(row) * k + i] = best_ids[i];
        out_logits[static_cast<std::size_t>(row) * k + i] = best_logits[i];
    }
}

}  // namespace

hipError_t launch_dflash2_merge_topk(
    const int32_t* candidate_ids,
    const float* candidate_logits,
    uint32_t rows,
    uint32_t candidates,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    hipStream_t stream) {
    if (candidate_ids == nullptr || candidate_logits == nullptr || out_ids == nullptr ||
        out_logits == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u) return hipSuccess;
    if (k == 0u || k > kDflash2TopKMaxK) return hipErrorInvalidValue;
    if (candidates < k || candidates > kDflash2MergeMaxCandidates) {
        return hipErrorInvalidValue;
    }
    dflash2_merge_topk_kernel<<<rows, 1u, 0, stream>>>(
        candidate_ids, candidate_logits, candidates, k, out_ids, out_logits);
    return hipGetLastError();
}

hipError_t launch_dflash2_topk_f32(
    const float* logits,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t row_stride,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    int32_t* scratch_ids,
    float* scratch_logits,
    int32_t id_offset,
    hipStream_t stream) {
    if (logits == nullptr || out_ids == nullptr || out_logits == nullptr ||
        scratch_ids == nullptr || scratch_logits == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u) return hipSuccess;
    if (k == 0u || k > kTopK) return hipErrorInvalidValue;
    if (vocab_size == 0u || row_stride < vocab_size) return hipErrorInvalidValue;

    dflash2_topk_partition_kernel<<<rows, kTopKThreads, 0, stream>>>(
        logits, vocab_size, row_stride, scratch_ids, scratch_logits);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return err;
    dflash2_topk_merge_kernel<<<rows, 1u, 0, stream>>>(
        scratch_ids, scratch_logits, k, out_ids, out_logits, id_offset);
    return hipGetLastError();
}

}  // namespace shelf::kernel
