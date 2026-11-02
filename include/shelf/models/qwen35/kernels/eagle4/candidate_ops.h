#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/kernels/optimized/radix_topn.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kEagle4CandidateThreads = 128u;
inline constexpr uint32_t kEagle4CandidateMaxK = 16u;

hipError_t launch_eagle4_rank_project_bf16(
    const bf16_t* hidden,
    uint32_t hidden_row_stride,
    const bf16_t* down_proj,
    bf16_t* projected,
    uint32_t rows,
    uint32_t rank,
    uint32_t hidden_size,
    hipStream_t stream);

hipError_t launch_eagle4_rough_scores_bf16(
    const bf16_t* projected,
    uint32_t projected_row_stride,
    const bf16_t* score_head,
    float* scores,
    uint32_t rows,
    uint32_t rank,
    uint32_t vocab,
    hipStream_t stream);

hipError_t launch_eagle4_candidate_logits(
    const bf16_t* lm_head,
    uint32_t lm_head_row_stride,
    const bf16_t* hidden,
    uint32_t hidden_row_stride,
    const int32_t* candidate_ids,
    uint32_t rows,
    uint32_t candidates,
    uint32_t hidden_size,
    float* logits,
    hipStream_t stream);

hipError_t launch_eagle4_candidate_topk(
    const float* logits,
    const int32_t* candidate_ids,
    uint32_t rows,
    uint32_t candidates,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    float* out_probs,
    hipStream_t stream);

}  // namespace ps::kernel
