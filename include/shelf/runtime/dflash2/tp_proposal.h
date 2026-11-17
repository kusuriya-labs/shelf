#pragma once

#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/runtime/dflash2/executor.h>
#include <shelf/runtime/dflash2/proposal_shard.h>
#include <shelf/runtime/tp/tensor_parallel.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace shelf::runtime::dflash2 {

// Runs the target head of a DFlash2 drafter across tensor-parallel ranks. Every
// rank holds one vocabulary shard of the target head, so no rank keeps a full
// copy. The drafter itself stays on rank 0: it hands over the final hidden rows,
// each rank computes its shard logits and local top-k, and rank 0 merges the
// candidates and the full-vocabulary normalizer.
struct DFlash2TpProposalConfig {
    TpCoordinator* coordinator = nullptr;
    std::vector<const shelf::weights::MatrixWeight*> heads;
    std::vector<uint32_t> vocab_offsets;
    uint32_t local_vocab = 0;
    uint32_t top_k = 0;
    uint32_t hidden_stride = 0;
    uint32_t max_rows = 0;
};

class DFlash2TpProposal final : public DFlash2ProposalDistributor {
public:
    static Result<DFlash2TpProposal> create(const DFlash2TpProposalConfig& config);
    ~DFlash2TpProposal() noexcept;

    DFlash2TpProposal(const DFlash2TpProposal&) = delete;
    DFlash2TpProposal& operator=(const DFlash2TpProposal&) = delete;
    DFlash2TpProposal(DFlash2TpProposal&& other) noexcept;
    DFlash2TpProposal& operator=(DFlash2TpProposal&& other) noexcept;

    Status distribute(
        const bf16_t* hidden,
        uint32_t hidden_stride,
        uint32_t rows,
        int32_t* topk_ids,
        float* topk_logits,
        float* normalizers,
        hipStream_t stream) override;

    void shutdown() noexcept;

private:
    DFlash2TpProposal() = default;

    DFlash2TpProposalConfig config_{};
    std::vector<int> devices_;
    std::vector<hipStream_t> streams_;
    std::vector<hipStream_t> proposal_streams_;
    std::vector<ProposalShardBuffers> shards_;
    std::vector<gpu::Tensor> peer_hidden_;
    gpu::Tensor ranked_ids_;
    gpu::Tensor ranked_logits_;
    gpu::Tensor ranked_stats_;
    uint32_t ranked_stride_ = 0;
    uint32_t stats_stride_ = 0;
    std::vector<hipEvent_t> hidden_sent_;
    std::vector<hipEvent_t> peer_done_;
    bool initialized_ = false;
};

}  // namespace shelf::runtime::dflash2
