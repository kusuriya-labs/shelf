#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/eagle4/context_state.h>
#include <shelf/models/qwen35/eagle4/executor.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/eagle4_cost.h>
#include <shelf/models/qwen35/runtime/gdn_spec_history.h>
#include <shelf/models/qwen35/runtime/tp_batch_hook.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/stop_tokens.h>
#include <shelf/runtime/execution/execution_types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include <array>
#include <span>
#include <vector>

namespace ps::qwen35::runtime {

constexpr uint32_t kDFlash2SpecMaxVerifyRows = 512u;
constexpr uint64_t kDFlash2SpecHistoryBytesMax = (9ull * 1024ull * 1024ull * 1024ull) / 4ull;
constexpr uint32_t kDFlash2NoSessionSlot = 0xFFFFFFFFu;

struct DFlash2SpecDecoderConfig {
    uint32_t num_drafts = 7u;
    uint32_t eagle4_tree_width = 0u;
    uint32_t eagle4_tree_dynamic = 0u;
    StopTokens eos_tokens;
};

struct DFlash2SpecTiming;

struct DFlash2GdnRankBinding {
    GdnStatePool* pool = nullptr;
    gpu::GpuArena* arena = nullptr;
    hipStream_t stream = nullptr;
    int device = -1;
    const ::ps::kernel::ModelDispatchStateView* model_state = nullptr;
    Executor* executor = nullptr;
};

struct DFlash2GdnRankState {
    GdnStatePool* pool = nullptr;
    hipStream_t stream = nullptr;
    int device = -1;
    void* conv_snapshot = nullptr;
    void* rec_snapshot = nullptr;
    std::size_t conv_bytes = 0u;
    std::size_t rec_bytes = 0u;
    GdnSpecHistory history;
    float* compact_delta = nullptr;
    float* compact_k = nullptr;
    float* compact_a = nullptr;
    uint64_t compact_delta_layer_stride = 0u;
    uint64_t compact_k_layer_stride = 0u;
    uint64_t compact_a_layer_stride = 0u;
    ::ps::runtime::DeviceRequestDescriptor* commit_request = nullptr;
    const ::ps::kernel::ModelDispatchStateView* model_state = nullptr;
    std::uint32_t* tree_rope_positions = nullptr;
    int32_t* tree_row_parent = nullptr;
    int32_t* tree_anc_offsets = nullptr;
    std::uint32_t* tree_anc_positions = nullptr;
    int32_t* tree_commit_rows = nullptr;
    std::uint32_t* tree_kv_src = nullptr;
    std::uint32_t* tree_kv_dst = nullptr;
};

struct DFlash2SpecWorkspace {
    std::vector<DFlash2GdnRankState> gdn_ranks;
    uint64_t gdn_snapshot_bytes = 0u;
    uint64_t gdn_compact_bytes = 0u;
    uint32_t gdn_key_heads = 0u;
    int32_t* verify_token_ids_device = nullptr;
    int32_t* decision_staging_device = nullptr;
    hipEvent_t draft_start_event = nullptr;
    hipEvent_t draft_stop_event = nullptr;
    hipEvent_t verify_start_event = nullptr;
    hipEvent_t verify_stop_event = nullptr;
    bool initialized = false;
};

struct DFlash2SpecDecoder {
    Executor* target = nullptr;
    dflash2::DFlash2Executor* draft = nullptr;
    dflash2::DFlash2ContextState* context = nullptr;
    eagle4::Eagle4Executor* eagle = nullptr;
    eagle4::Eagle4ContextState* eagle_context = nullptr;
    std::vector<eagle4::Eagle4RankDraft> eagle_ranks;
    int32_t* eagle_draft_tokens_device = nullptr;
    ::ps::qwen35::runtime::Eagle4CostTable tree_cost{};
    uint32_t eagle_num_candidates = 0u;
    uint32_t eagle_candidate_topk = 1u;
    bool eagle_root_ready = false;
    PagedSequenceState* sequence = nullptr;
    hipStream_t stream = nullptr;
    DFlash2SpecDecoderConfig config;

    std::vector<DFlash2GdnRankState> gdn_ranks;
    TpBatchHook* tp_hook = nullptr;
    uint64_t gdn_snapshot_bytes = 0u;
    uint64_t gdn_compact_bytes = 0u;

    uint32_t gdn_key_heads = 0u;

    uint32_t hidden_size = 0u;
    int32_t* verify_token_ids_device = nullptr;
    int32_t* decision_staging_device = nullptr;
    bool device_token_bridge = false;
    bool host_proposal_visibility = false;
    std::vector<int32_t> token_history;
    hipEvent_t draft_start_event = nullptr;
    hipEvent_t draft_stop_event = nullptr;
    hipEvent_t verify_start_event = nullptr;
    hipEvent_t verify_stop_event = nullptr;
    bool initialized = false;
    DFlash2SpecTiming* timing = nullptr;
    bool owns_workspace = false;


    SamplingConfig sampling{};
    uint64_t sample_index = 0u;
};

Result<DFlash2SpecWorkspace> create_dflash2_spec_workspace_for_drafts(
    Executor& target,
    uint32_t draft_max_drafts,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks = nullptr);

Result<DFlash2SpecWorkspace> create_dflash2_spec_workspace(
    Executor& target,
    dflash2::DFlash2Executor& draft,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks = nullptr);

Status shutdown_dflash2_spec_workspace(DFlash2SpecWorkspace& workspace) noexcept;

Status bind_dflash2_spec_workspace(
    DFlash2SpecDecoder& decoder,
    const DFlash2SpecWorkspace& workspace);

Result<DFlash2SpecDecoder> create_dflash2_spec_decoder(
    Executor& target,
    dflash2::DFlash2Executor& draft,
    dflash2::DFlash2ContextState& context,
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks = nullptr,
    TpBatchHook* tp_hook = nullptr,
    DFlash2SpecWorkspace* workspace = nullptr);

Result<DFlash2SpecDecoder> create_eagle4_spec_decoder(
    Executor& target,
    eagle4::Eagle4Executor& draft,
    eagle4::Eagle4ContextState& context,
    PagedSequenceState& sequence,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* tp_ranks = nullptr,
    TpBatchHook* tp_hook = nullptr,
    DFlash2SpecWorkspace* workspace = nullptr,
    const std::vector<eagle4::Eagle4RankDraft>* eagle_ranks = nullptr);

Status dflash2_spec_decoder_shutdown(DFlash2SpecDecoder& decoder) noexcept;

void dflash2_spec_decoder_set_sampling(DFlash2SpecDecoder& decoder,
                                       const SamplingConfig& sampling);

struct DFlash2PrefillOutput {
    int32_t pending_token = -1;
};

Result<DFlash2PrefillOutput> dflash2_spec_prefill(
    DFlash2SpecDecoder& decoder,
    const int32_t* prompt_tokens,
    uint32_t prompt_count,
    uint32_t prefix_tokens = 0u);

struct DFlash2TargetSnapshot {
    uint32_t position = 0u;
    uint32_t block_count = 0u;
    bool valid = false;
};

struct DFlash2SpecTiming {
    double prompt_sync_ms = 0.0;
    double draft_ms = 0.0;
    double draft_d2h_ms = 0.0;
    double draft_gpu_ms = 0.0;
    double proposal_wait_ms = 0.0;
    double proposal_copy_ms = 0.0;
    double gdn_snapshot_ms = 0.0;
    double verify_ms = 0.0;
    double verify_gpu_ms = 0.0;
    double decision_wait_ms = 0.0;
    double decision_copy_ms = 0.0;
    double gdn_restore_ms = 0.0;
    double rerun_ms = 0.0;
    double dflash_commit_ms = 0.0;
    double round_ms = 0.0;
    uint32_t rounds = 0u;
    uint32_t full_accepts = 0u;
    uint32_t partial_accepts = 0u;
    uint32_t accepted_drafts = 0u;
    uint32_t generated_tokens = 0u;
    uint32_t dflash_prefix_full_accepts = 0u;
    uint64_t verify_rows_total = 0u;
    uint64_t gdn_history_bytes = 0u;
};

struct DFlash2SpecIterationOutput {
    std::array<int32_t, kDFlash2SpecMaxVerifyRows> emitted{};
    uint32_t emitted_count = 0u;
    int32_t pending_token = -1;
    uint32_t num_drafts = 0u;
    uint32_t num_dflash_drafts = 0u;
    uint32_t num_accepted = 0u;
    bool finished = false;
};

Result<DFlash2SpecIterationOutput> dflash2_spec_step(
    DFlash2SpecDecoder& decoder,
    int32_t pending_token,
    uint32_t remaining_tokens);

struct DFlash2SpecBatchCompact {
    float* delta = nullptr;
    float* k = nullptr;
    float* a = nullptr;
    uint64_t delta_layer_stride = 0u;
    uint64_t k_layer_stride = 0u;
    uint64_t a_layer_stride = 0u;
};

struct DFlash2SpecBatchContext {
    std::vector<GdnSpecHistory> history;
    std::vector<DFlash2SpecBatchCompact> compact;
    uint32_t max_rows = 0u;
    uint64_t history_bytes = 0u;
    uint64_t compact_bytes = 0u;
    bool initialized = false;
    int32_t* verify_token_ids_device = nullptr;
    int32_t* decision_staging_device = nullptr;
    std::vector<uint32_t*> tree_rope_positions_device;
    std::vector<int32_t*> tree_row_parent_device;
    std::vector<int32_t*> tree_anc_offsets_device;
    std::vector<uint32_t*> tree_anc_positions_device;
    std::vector<uint32_t*> tree_committed_end_rows_device;
    std::vector<uint32_t> tree_rope_positions_host;
    std::vector<int32_t> tree_row_parent_host;
    std::vector<int32_t> tree_anc_offsets_host;
    std::vector<uint32_t> tree_anc_positions_host;
    std::vector<uint32_t> tree_committed_end_host;
};

Result<DFlash2SpecBatchContext> create_dflash2_spec_batch_context(
    const std::vector<DFlash2GdnRankBinding>& bindings,
    uint32_t max_rows);

Status shutdown_dflash2_spec_batch_context(DFlash2SpecBatchContext& context) noexcept;

struct DFlash2SpecBatchSlot {
    DFlash2SpecDecoder* decoder = nullptr;
    int32_t anchor_token = 0;
    uint32_t num_drafts = 0u;
    uint32_t remaining_tokens = 0u;
};

struct DFlash2SpecBatchOutput {
    std::vector<DFlash2SpecIterationOutput> slots;
    uint32_t total_rows = 0u;
    uint32_t rounds = 0u;
    double round_ms = 0.0;
    double draft_gpu_ms = 0.0;
    double verify_gpu_ms = 0.0;
};

using Eagle4SpecBatchSlot = DFlash2SpecBatchSlot;
using Eagle4SpecBatchContext = DFlash2SpecBatchContext;
using Eagle4SpecBatchOutput = DFlash2SpecBatchOutput;

Result<DFlash2SpecBatchOutput> eagle4_tree_spec_step_batch(
    DFlash2SpecBatchContext& context,
    std::span<const DFlash2SpecBatchSlot> slots);

Result<DFlash2SpecBatchOutput> eagle4_spec_step_batch(
    DFlash2SpecBatchContext& context,
    std::span<const DFlash2SpecBatchSlot> slots);

Result<DFlash2SpecBatchOutput> dflash2_spec_step_batch(
    DFlash2SpecBatchContext& context,
    std::span<const DFlash2SpecBatchSlot> slots);

}  // namespace ps::qwen35::runtime
