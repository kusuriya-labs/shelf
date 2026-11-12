#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/common/qwen_hybrid/lowered_primitives.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/gdn_spec_history.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace shelf {
namespace qwen35 {
namespace runtime {

constexpr uint32_t kMtpMaxDrafts = 31u;
constexpr uint32_t kMtpMaxPrefillRows = 512u;

struct MtpDrafterConfig {
    uint32_t num_drafts = 7;
    uint32_t max_seq_len = 0;
    uint32_t page_tokens = 16;
    uint32_t max_batch_rows = 0;
    kernel::KVCacheDType kv_dtype = kernel::KVCacheDType::BF16;
    EffectiveRopeProfile rope_profile;
    // A batched context shares one history / compact log per rank, so its per-request
    // drafters leave those planes to the context.
    bool allocate_spec_buffers = true;
};

struct MtpPrefillOutput {
    int32_t pending_token = -1;
};

struct MtpPrefillProgress {
    void* context = nullptr;
    void (*report)(void* context, uint32_t processed_tokens,
                   double elapsed_ms) = nullptr;
    // Polled between prefill chunks. Returning true aborts the prefill before the
    // next target forward, so a cancelled request does not finish the whole prompt.
    bool (*should_stop)(void* stop_context) = nullptr;
    void* stop_context = nullptr;
};

struct MtpIterationOutput {
    std::array<int32_t, kMtpMaxDrafts + 1u> emitted{};
    uint32_t emitted_count = 0;
    int32_t pending_token = -1;
    uint32_t num_accepted = 0;
    bool finished = false;
};

struct MtpTiming {
    uint32_t rounds = 0;
    uint32_t full_accepts = 0;
    uint32_t partial_accepts = 0;
    uint32_t accepted_drafts = 0;
    uint32_t generated_tokens = 0;
    uint32_t verify_rows = 0;
    uint32_t gdn_restores = 0;
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    double gdn_restore_ms = 0.0;
    double compact_replay_ms = 0.0;
    double round_ms = 0.0;
};

class TpBatchHook;

struct MtpGdnRankBinding {
    const state::GdnStatePool* pool = nullptr;
    gpu::GpuArena* arena = nullptr;
    hipStream_t stream = nullptr;
    int device = -1;
    Executor* executor = nullptr;
};

struct MtpGdnRankState {
    const state::GdnStatePool* pool = nullptr;
    hipStream_t stream = nullptr;
    int device = -1;
    void* conv_snapshot = nullptr;
    void* recurrent_snapshot = nullptr;
    std::size_t conv_bytes = 0;
    std::size_t recurrent_bytes = 0;
    GdnSpecHistory history;
    float* compact_delta = nullptr;
    float* compact_k = nullptr;
    float* compact_a = nullptr;
    std::uint64_t compact_delta_layer_stride = 0;
    std::uint64_t compact_k_layer_stride = 0;
    std::uint64_t compact_a_layer_stride = 0;
    ::shelf::runtime::DeviceRequestDescriptor* commit_request = nullptr;
    ::shelf::runtime::DeviceRequestDescriptor commit_request_host{};
    std::uint32_t gdn_key_heads = 0;
    Executor* target = nullptr;
    bf16_t* qsa_partial_history = nullptr;
    std::uint32_t* qsa_count_history = nullptr;
    std::uint64_t qsa_partial_state_stride = 0;
    std::uint64_t qsa_count_state_stride = 0;
    std::uint32_t qsa_history_rows = 0;
    bf16_t* ple_conv_history = nullptr;
    std::uint64_t ple_conv_state_stride = 0;
    std::uint32_t ple_conv_history_rows = 0;
};

struct MtpDrafter {
    shelf::qwen_hybrid::QwenHybridConfig config;
    std::unique_ptr<state::SequenceSlotPool> slot_pool;
    std::unique_ptr<state::GdnStatePool> gdn_pool;
    std::unique_ptr<state::PagedKVPool> kv_pool;
    std::unique_ptr<state::PagedSequenceState> sequence;
    Executor executor;

    void* gdn_conv_snapshot = nullptr;
    void* gdn_recurrent_snapshot = nullptr;
    std::size_t gdn_conv_bytes = 0;
    std::size_t gdn_recurrent_bytes = 0;

    GdnSpecHistory gdn_history;
    float* compact_delta = nullptr;
    float* compact_k = nullptr;
    float* compact_a = nullptr;
    std::uint64_t compact_delta_layer_stride = 0;
    std::uint64_t compact_k_layer_stride = 0;
    std::uint64_t compact_a_layer_stride = 0;
    ::shelf::runtime::DeviceRequestDescriptor* commit_request = nullptr;
    ::shelf::runtime::DeviceRequestDescriptor commit_request_host{};
    std::uint32_t gdn_key_heads = 0;

    const state::GdnStatePool* target_pool = nullptr;
    hipStream_t target_stream = nullptr;
    int target_device = -1;
    std::vector<MtpGdnRankState> extra_ranks;
    TpBatchHook* tp_hook = nullptr;

    // The target hidden of the tokens the drafter has committed, copied out of the target
    // batch so a request that sits out a round still has its input.
    gpu::Tensor hidden_cache;
    Executor* target_executor = nullptr;
    bf16_t* qsa_partial_history = nullptr;
    std::uint32_t* qsa_count_history = nullptr;
    std::uint64_t qsa_partial_state_stride = 0;
    std::uint64_t qsa_count_state_stride = 0;
    std::uint32_t qsa_history_rows = 0;
    bf16_t* ple_conv_history = nullptr;
    std::uint64_t ple_conv_state_stride = 0;
    std::uint32_t ple_conv_history_rows = 0;

    std::vector<int32_t> last_tokens;
    uint32_t last_position = 0;
    uint32_t last_hidden_offset = 0;
    uint64_t sample_index = 0;
    SamplingConfig sampling;
    uint32_t max_seq_len = 0;
    uint32_t num_drafts = 0;
    uint32_t hidden_width = 0;
    bool wide_hidden = false;
    MtpTiming timing;
    bool initialized = false;
};

// What a drafter needs from the model it drafts for: the target's geometry, the embedding
// its graph reads (the model's own, not a copy), and the drafter block already lowered by
// that model's family. The drafter does not know how a layer's FFN is built; a model that
// runs on grouped expert rows also names the workspace term those rows add, because the
// shared executor estimate cannot take it from a dense intermediate width.
struct MtpDrafterInput {
    const shelf::qwen_hybrid::QwenHybridConfig* config = nullptr;
    const shelf::weights::MatrixWeight* embed_tokens = nullptr;
    shelf::qwen_hybrid::HybridLoweredPrimitives lowered;
    std::size_t token_row_intermediate = 0;
    std::size_t grouped_executor_bytes = 0;
    // Largest rotated-domain linear the draft block runs. An empty shape keeps the
    // model-independent upper bound.
    shelf::weights::Exl3LinearShape exl3_linear_shape;
    // The target hidden the drafter consumes. Qwen4Exp feeds the pre-mixer wide residual
    // (hc * hidden); the Qwen35 families feed the final normed hidden.
    std::uint32_t hidden_width = 0;
    bool wide_hidden = false;
};

// The row count the drafter's executor runs at: its own batch limit, widened to hold the
// drafts plus the token the target verifies.
inline uint32_t mtp_drafter_rows(const MtpDrafterConfig& options) noexcept {
    const uint32_t verify_rows = options.num_drafts + 1u;
    return options.max_batch_rows > verify_rows ? options.max_batch_rows : verify_rows;
}

// Arena bytes the drafter reserves: its own KV cache and executor workspace, plus the
// target's GDN state snapshots and speculative history it drives. The two executor terms
// are what the model's family adds at `mtp_drafter_rows(options)` rows.
std::size_t mtp_drafter_reserve_bytes(const shelf::qwen_hybrid::QwenHybridConfig& config,
                                      const MtpDrafterConfig& options,
                                      std::size_t token_row_intermediate,
                                      std::size_t grouped_executor_bytes,
                                      bool exl3_linear_workspace,
                                      const shelf::weights::Exl3LinearShape&
                                          exl3_linear_shape = {},
                                      const state::GdnStatePoolLayout* target_gdn =
                                          nullptr) noexcept;

Result<std::unique_ptr<MtpDrafter>> create_mtp_drafter(
    const MtpDrafterInput& input,
    Executor* target,
    const state::GdnStatePool& target_gdn_pool,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const MtpDrafterConfig& options,
    const std::vector<MtpGdnRankBinding>* extra_ranks = nullptr,
    TpBatchHook* tp_hook = nullptr);

Status mtp_drafter_shutdown(MtpDrafter& drafter, hipStream_t stream) noexcept;

Status mtp_drafter_reset_sequence(MtpDrafter& drafter, hipStream_t stream) noexcept;

Result<MtpPrefillOutput> mtp_spec_prefill(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    const int32_t* prompt_tokens,
    uint32_t prompt_count,
    const SamplingConfig& sampling,
    const MtpPrefillProgress* progress = nullptr);

Result<MtpIterationOutput> mtp_spec_step(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    int32_t pending_token,
    uint32_t remaining_tokens);

// The draft phase of one MTP round: it advances the drafter's own 1-layer state over the
// tokens the target committed last round, samples the draft proposals, and snapshots the
// target GDN state so a partial accept can be rolled back. It does not run the target.
struct MtpDraftResult {
    std::array<int32_t, kMtpMaxDrafts> proposals{};
    uint32_t drafts = 0u;
    uint32_t position = 0u;
    bool decode_only = false;
};

Result<MtpDraftResult> mtp_draft(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    int32_t pending_token,
    uint32_t remaining_tokens);

// One request's place in a batched MTP round. The drafter and the target sequence are the
// request's own; the batch context supplies the shared verify buffer and the target.
struct MtpSpecBatchSlot {
    MtpDrafter* drafter = nullptr;
    state::PagedSequenceState* sequence = nullptr;
    int32_t pending_token = 0;
    uint32_t num_drafts = 0u;
    uint32_t remaining_tokens = 0u;
    uint32_t page_tokens = 0u;
};

struct MtpSpecBatchOutput {
    std::vector<MtpIterationOutput> slots;
    uint32_t total_rows = 0u;
    uint32_t rounds = 0u;
    double round_ms = 0.0;
};

// Shared state for a batched MTP round: the target executor the verifies run on, the
// per-request drafters, and the per-rank GDN history / compact log sized for the whole
// batch. The draft and accept phases stay per-request; only the target verify is fused
// into one forward.
struct MtpSpecBatchContext {
    Executor* target = nullptr;
    std::vector<std::unique_ptr<MtpDrafter>> drafters;
    // One entry per rank: pool / stream / device, the shared history and compact log, and
    // the commit descriptor this rank's accept path uploads.
    std::vector<MtpGdnRankState> ranks;
    uint32_t max_rows = 0u;
    uint32_t max_requests = 0u;
    uint32_t num_drafts = 0u;
    bool initialized = false;
};

// The rows one fused verify round can produce: every request contributes its drafts plus
// the token the target verifies. The batch history / compact log is sized from this, not
// from the drafter's prefill row limit.
inline uint32_t mtp_spec_rows(uint32_t max_requests, uint32_t num_drafts) noexcept {
    const std::uint64_t drafts = static_cast<std::uint64_t>(num_drafts) + 1u;
    const std::uint64_t rows = static_cast<std::uint64_t>(max_requests) * drafts;
    return rows < 0xffffffffull ? static_cast<uint32_t>(rows) : 0xffffffffu;
}

// The arena bytes a rank reserves for one fused verify batch: the shared target GDN history
// and compact log, plus one target GDN snapshot per drafter. Every rank allocates its own
// snapshot for each drafter in the batch, so the caller passes the request count. The caller
// adds the per-request drafters themselves, which only the drafter's rank allocates (see
// mtp_drafter_reserve_bytes).
std::size_t mtp_batch_rank_state_reserve_bytes(
    const state::GdnStatePoolLayout& rank_gdn,
    std::uint32_t spec_rows,
    std::uint32_t drafter_count) noexcept;

Result<MtpSpecBatchContext> create_mtp_spec_batch_context(
    Executor& target,
    const MtpDrafterInput& input,
    const state::GdnStatePool& target_gdn_pool,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const MtpDrafterConfig& options,
    uint32_t max_requests,
    const std::vector<MtpGdnRankBinding>* extra_ranks = nullptr,
    TpBatchHook* tp_hook = nullptr);

Status shutdown_mtp_spec_batch_context(MtpSpecBatchContext& context) noexcept;

// Runs one batched round: every slot drafts on its own drafter, the target verifies all
// proposals in one forward, and each slot accepts its own prefix. Slots whose room or
// budget leaves no draft fall back to a single decode on their own sequence.
Result<MtpSpecBatchOutput> mtp_spec_step_batch(
    MtpSpecBatchContext& context,
    std::span<const MtpSpecBatchSlot> slots);

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
