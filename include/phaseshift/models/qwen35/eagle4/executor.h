#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/eagle4/config.h>
#include <phaseshift/models/qwen35/eagle4/weights.h>
#include <phaseshift/runtime/tp/tp_execution.h>
#include <phaseshift/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace ps::qwen35 {
struct Executor;
}

namespace ps::qwen35::runtime {
class TpBatchHook;
}

namespace ps::qwen35::eagle4 {

struct Eagle4ContextState;
struct Eagle4RankDraft;

inline constexpr uint32_t kEagle4ExtendRowsMax = 256u;
inline constexpr uint32_t kEagle4CandidateRowsMax = 16u;

struct Eagle4ExecutorConfig {
    uint32_t max_rows = kEagle4ExtendRowsMax;
    uint32_t max_candidate_rows = kEagle4CandidateRowsMax;
    uint32_t num_candidates = 0u;
    const ps::weights::MatrixWeight* target_embed_tokens = nullptr;
    const ps::weights::MatrixWeight* target_lm_head = nullptr;
    bool use_lowrank_candidate_head = true;
    uint32_t tp_rank = 0u;
    uint32_t tp_size = 1u;
    std::function<Status(const ps::runtime::TpSumTarget&, hipStream_t, uint32_t)>
        tp_sum;
};

struct Eagle4Executor {
    const Eagle4Config* config = nullptr;
    const Eagle4Weights* weights = nullptr;
    uint32_t max_rows = 0u;
    uint32_t max_candidate_rows = 0u;
    uint32_t num_candidates = 0u;
    uint32_t max_root_candidates = 0u;
    const ps::weights::MatrixWeight* embed_tokens = nullptr;
    const ps::weights::MatrixWeight* lm_head = nullptr;

    gpu::Tensor token_ids;
    gpu::Tensor tap_concat;
    gpu::Tensor fc_out;
    gpu::Tensor embed_out;
    gpu::Tensor embed_norm;
    gpu::Tensor hidden_norm;
    gpu::Tensor layer_input;
    gpu::Tensor q_raw;
    gpu::Tensor q_rope;
    gpu::Tensor k_raw;
    gpu::Tensor k_rope;
    gpu::Tensor v_raw;
    gpu::Tensor attn_out;
    gpu::Tensor attn_o;
    gpu::Tensor residual;
    gpu::Tensor post_norm;
    gpu::Tensor mlp_gate;
    gpu::Tensor mlp_up;
    gpu::Tensor mlp_swiglu;
    gpu::Tensor mlp_down;
    gpu::Tensor summed;
    gpu::Tensor normed;

    gpu::Tensor root_hidden;
    gpu::Tensor step_hidden_a;
    gpu::Tensor step_hidden_b;

    gpu::Tensor tree_hidden_in;
    gpu::Tensor round_k;
    gpu::Tensor round_v;

    gpu::Tensor draft_tokens;
    gpu::Tensor tree_tokens;
    gpu::Tensor tree_row_nodes;
    gpu::Tensor tree_src_rows;
    gpu::Tensor tree_ancestor_offsets;
    gpu::Tensor tree_ancestor_nodes;
    gpu::Tensor tap_rows;
    gpu::Tensor gathered_taps;

    gpu::Tensor projected;
    gpu::Tensor rough_scores;
    gpu::Tensor candidate_ids;
    gpu::Tensor candidate_logits;
    gpu::Tensor candidate_scratch_ids;
    gpu::Tensor candidate_scratch_logits;
    gpu::Tensor radix_scratch;
    gpu::Tensor topk_ids;
    gpu::Tensor topk_logits;
    gpu::Tensor topk_probs;
    gpu::Tensor gemv_logits;
    gpu::Tensor gemv_ids;
    gpu::Tensor error_word;

    bool use_lowrank_candidate_head = true;
    uint32_t tp_rank = 0u;
    uint32_t tp_size = 1u;
    int device = -1;
    std::function<Status(const ps::runtime::TpSumTarget&, hipStream_t, uint32_t)> tp_sum;
    ::ps::qwen35::runtime::TpBatchHook* tp_hook = nullptr;
    bool initialized = false;
};

struct Eagle4RankDraft {
    Eagle4Executor* executor = nullptr;
    Eagle4ContextState* context = nullptr;
    hipStream_t stream = nullptr;
    int device = 0;
    ::ps::qwen35::Executor* target = nullptr;
    const ::ps::bf16_t* const* taps = nullptr;
    std::uint32_t tap_count = 0;
};

struct Eagle4DraftJob {
    enum class Op : std::uint8_t { TreeStep, ExtendRows, DraftSteps };

    Op op = Op::TreeStep;
    std::vector<Eagle4RankDraft> ranks;

    const ::ps::bf16_t* const* taps = nullptr;
    std::uint32_t tap_count = 0;
    std::uint32_t tap_row_offset = 0;
    const std::int32_t* tap_gather_rows_host = nullptr;
    std::uint32_t tap_gather_count = 0;
    std::uint32_t extend_rows = 0;
    std::uint32_t num_root_candidates = 0;
    std::int32_t* root_tokens = nullptr;

    const std::int32_t* tokens_host = nullptr;
    const std::int32_t* row_nodes_host = nullptr;
    const std::int32_t* src_rows_host = nullptr;
    bool src_is_root = false;
    const std::int32_t* ancestor_offsets_host = nullptr;
    const std::int32_t* ancestor_nodes_host = nullptr;
    std::uint32_t ancestor_offset_count = 0;
    std::uint32_t ancestor_node_count = 0;

    std::uint32_t rows = 0;
    std::uint32_t position = 0;
    std::uint32_t committed_end = 0;
    std::uint32_t candidate_k = 0;

    std::uint32_t first_position = 0;
    std::uint32_t num_drafts = 0;
};


Result<Eagle4Executor> create_eagle4_executor(
    const Eagle4Config& config,
    const Eagle4Weights& weights,
    gpu::GpuArena& arena,
    const Eagle4ExecutorConfig& exec_config);

Status eagle4_executor_shutdown(Eagle4Executor& executor) noexcept;

Status eagle4_extend_rows(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const bf16_t* const* taps,
    uint32_t tap_count,
    const int32_t* token_ids,
    uint32_t rows,
    uint32_t position_start,
    uint32_t num_root_candidates,
    int32_t* root_tokens,
    hipStream_t stream);

Status eagle4_draft_steps(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    uint32_t first_position,
    uint32_t num_drafts,
    int32_t* drafts,
    hipStream_t stream,
    int32_t* step_out = nullptr);

Status eagle4_tree_draft_step(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const int32_t* tokens_device,
    const int32_t* row_nodes_device,
    const int32_t* src_rows_device,
    bool src_is_root,
    const int32_t* ancestor_offsets_device,
    const int32_t* ancestor_nodes_device,
    uint32_t rows,
    uint32_t position,
    uint32_t committed_end,
    uint32_t candidate_k,
    hipStream_t stream);

Status eagle4_forward_rows(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const int32_t* token_ids,
    const bf16_t* const* taps,
    const bf16_t* hidden_in,
    uint32_t rows,
    uint32_t position_start,
    bf16_t* normed_out,
    bf16_t* summed_out,
    hipStream_t stream);


Status run_eagle4_rank_job(const Eagle4DraftJob& job, std::size_t rank);

Status eagle4_extend_rows_tp(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const std::vector<Eagle4RankDraft>* ranks,
    const bf16_t* const* taps,
    uint32_t tap_count,
    uint32_t tap_row_offset,
    const int32_t* token_ids_host,
    uint32_t rows,
    uint32_t position_start,
    uint32_t num_root_candidates,
    int32_t* root_tokens,
    hipStream_t stream,
    const int32_t* tap_gather_rows_host = nullptr,
    uint32_t tap_gather_count = 0u);

Status eagle4_draft_steps_tp(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const std::vector<Eagle4RankDraft>* ranks,
    uint32_t first_position,
    uint32_t num_drafts,
    int32_t* drafts,
    hipStream_t stream);

Status eagle4_tree_draft_step_tp(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const std::vector<Eagle4RankDraft>* ranks,
    const int32_t* tokens_host,
    const int32_t* row_nodes_host,
    const int32_t* src_rows_host,
    bool src_is_root,
    const int32_t* ancestor_offsets_host,
    const int32_t* ancestor_nodes_host,
    uint32_t ancestor_offset_count,
    uint32_t ancestor_node_count,
    uint32_t rows,
    uint32_t position,
    uint32_t committed_end,
    uint32_t candidate_k,
    hipStream_t stream);

}  // namespace ps::qwen35::eagle4
