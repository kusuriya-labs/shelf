#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::runtime::dflash2 {

// Per-rank state for a vocabulary-sharded proposal head. The head holds only
// `local_vocab` rows starting at `vocab_offset`, so the logits buffer, the
// local top-k and the row normalizer stay local; only the top-k candidates and
// the two normalizer floats have to cross ranks.
struct ProposalShardBuffers {
    uint32_t max_rows = 0;
    uint32_t local_vocab = 0;
    uint32_t top_k = 0;
    gpu::Tensor logits;
    gpu::Tensor topk_ids;
    gpu::Tensor topk_logits;
    gpu::Tensor scratch_ids;
    gpu::Tensor scratch_logits;
    // Row maximum and sum(exp(x - max)) over the local vocabulary.
    gpu::Tensor stats;
};

Result<ProposalShardBuffers> create_proposal_shard_buffers(
    gpu::GpuArena& arena,
    uint32_t max_rows,
    uint32_t local_vocab,
    uint32_t top_k);

// Computes the shard logits and the local top-k. Ids are returned as global
// vocabulary ids by adding `vocab_offset`.
Status proposal_shard_compute(
    const shelf::weights::MatrixWeight& head,
    const ProposalShardBuffers& buffers,
    const bf16_t* hidden,
    uint32_t hidden_stride,
    uint32_t rows,
    uint32_t vocab_offset,
    hipStream_t stream);

// Merges per-rank candidate lists. Rank q's list for row r starts at
// q * rank_stride + r * k.
Status proposal_merge_candidates(
    const int32_t* candidate_ids,
    const float* candidate_logits,
    uint32_t ranks,
    uint32_t rank_stride,
    uint32_t rows,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    hipStream_t stream);

// Normalizes the sharded softmax on device. `rank_stats` holds [max, sum] for
// every rank and row, rank-major: rank q at q * rank_stride. `out` receives
// {global max, full-vocabulary denominator} per row.
Status proposal_combine_denominators(
    const float* rank_stats,
    uint32_t ranks,
    uint32_t rank_stride,
    uint32_t rows,
    float* out,
    hipStream_t stream);

}  // namespace shelf::runtime::dflash2
