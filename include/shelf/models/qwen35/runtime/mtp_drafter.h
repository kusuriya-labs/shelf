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
};

struct MtpPrefillOutput {
    int32_t pending_token = -1;
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
                                      bool exl3_linear_workspace) noexcept;

Result<std::unique_ptr<MtpDrafter>> create_mtp_drafter(
    const MtpDrafterInput& input,
    const state::GdnStatePool& target_gdn_pool,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const MtpDrafterConfig& options);

Status mtp_drafter_shutdown(MtpDrafter& drafter, hipStream_t stream) noexcept;

Result<MtpPrefillOutput> mtp_spec_prefill(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    const int32_t* prompt_tokens,
    uint32_t prompt_count,
    const SamplingConfig& sampling);

Result<MtpIterationOutput> mtp_spec_step(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    int32_t pending_token,
    uint32_t remaining_tokens);

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
