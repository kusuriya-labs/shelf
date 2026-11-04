#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace shelf {
namespace qwen35 {
namespace runtime {

constexpr uint32_t kMtpMaxDrafts = 31u;

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
    uint32_t reruns = 0;
    double draft_ms = 0.0;
    double verify_ms = 0.0;
    double rerun_ms = 0.0;
    double round_ms = 0.0;
};

struct MtpDrafter {
    Qwen35TextConfig config;
    std::unique_ptr<state::SequenceSlotPool> slot_pool;
    std::unique_ptr<state::GdnStatePool> gdn_pool;
    std::unique_ptr<state::PagedKVPool> kv_pool;
    std::unique_ptr<state::PagedSequenceState> sequence;
    Executor executor;

    void* gdn_conv_snapshot = nullptr;
    void* gdn_recurrent_snapshot = nullptr;
    std::size_t gdn_conv_bytes = 0;
    std::size_t gdn_recurrent_bytes = 0;

    std::vector<int32_t> last_tokens;
    uint32_t last_position = 0;
    uint64_t sample_index = 0;
    SamplingConfig sampling;
    uint32_t max_seq_len = 0;
    uint32_t num_drafts = 0;
    MtpTiming timing;
    bool initialized = false;
};

std::size_t mtp_drafter_reserve_bytes(const Qwen35TextConfig& config,
                                      const MtpDrafterConfig& options) noexcept;

Result<std::unique_ptr<MtpDrafter>> create_mtp_drafter(
    const Qwen35Model& model,
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
