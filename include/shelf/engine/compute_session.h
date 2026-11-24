#pragma once

#include <shelf/core/status.h>
#include <shelf/models/model_info.h>
#include <shelf/models/model_policy.h>
#include <shelf/runtime/dflash2_config.h>
#include <shelf/runtime/decode_backend.h>
#include <shelf/runtime/rope_profile.h>
#include <shelf/runtime/scheduling_types.h>
#include <shelf/runtime/state_types.h>
#include <shelf/runtime/runtime_request.h>
#include <shelf/runtime/prefix_cache.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace shelf::engine {

enum class ArenaBudgetMode {
    // Free VRAM at startup minus a small margin.
    Auto,
    // Exactly `arena_bytes`.
    Fixed,
    // Total VRAM minus `arena_bytes` (a margin).
    TotalMinus,
};

// Model-neutral input for the execution layer. Requested options are resolved by
// shelf::models before execution; no family type is part of this header.
struct ComputeConfig {
    std::string model_dir;
    models::ModelRuntimeRequest requested;
    models::ResolvedModelOptions resolved;
    bool resolved_ready = false;

    std::string dflash2_model_dir;
    ::shelf::runtime::DFlash2Config draft_config;
    bool draft_config_ready = false;
    uint32_t dflash2_drafts = 7;
    bool dflash2_tree = false;
    uint32_t dflash2_tree_extra = 0;
    uint32_t dflash2_tree_budget = 0;
    bool dflash2_tree_budget_auto = false;
    uint32_t dflash2_max_sessions = 0;
    uint32_t dflash2_sessions() const noexcept {
        const uint32_t cap = max_concurrent_requests == 0u ? 1u : max_concurrent_requests;
        const uint32_t want = dflash2_max_sessions != 0u ? dflash2_max_sessions
                                                         : (cap < 8u ? cap : 8u);
        return want < cap ? want : cap;
    }
    std::size_t dflash2_coarse_bytes = 0;
    uint32_t dflash2_prefill_rows = 0;
    bool dflash2_host_token_ids = false;
    bool mtp = false;
    bool mtp_serve = false;
    uint32_t mtp_drafts = 7;
    uint32_t mtp_window = 4096;
    uint32_t mtp_sink = 64;
    uint32_t max_seq_len = 512;
    uint32_t max_num_batched_tokens = 0;
    uint32_t max_concurrent_requests = 4;
    uint32_t prefill_decode_interval = 0;
    bool sampled_ids_only = false;
    ArenaBudgetMode arena_mode = ArenaBudgetMode::Auto;
    std::size_t arena_bytes = 0;
    std::size_t draft_bytes = 0;
    bool private_target_weights = false;
    bool private_target_lm_head = true;
    uint32_t page_tokens = 16;
    std::vector<int> devices = {0};
    uint32_t tp = 1;
    uint32_t expert_parallel = 1;
    uint32_t pipeline_parallel = 1;
    kernel::KVCacheDType kv_cache_dtype = kernel::KVCacheDType::BF16;
    kernel::OctaveKVCacheFormat octave_kv_format{};
    bool verify_weights = false;
    uint32_t max_scheduled_output_rows = 0;
    std::array<uint32_t, runtime::kMaxTargetHiddenTaps> target_hidden_taps{};
    uint32_t target_hidden_tap_count = 0;
    uint32_t tp_combine_f32 = 0;
    ::shelf::runtime::DecodeBackend decode_backend =
        ::shelf::runtime::DecodeBackend::Hip;
    bool prefix_cache = true;
    uint32_t prefix_cache_checkpoints = 16;
    uint32_t prefix_cache_spec_ring_bytes = 0;
    runtime::PrefixCacheStrategy prefix_cache_strategy =
        runtime::PrefixCacheStrategy::Radix;
    uint32_t prefix_cache_slot_count = 0;
    uint32_t prefix_cache_save_interval = 16384;
    std::size_t working_set_reserve_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    std::optional<runtime::EffectiveRopeProfile> rope_profile;
};

struct RequestView {
    bool found = false;
    runtime::RequestState state = runtime::RequestState::Queued;
    runtime::FinishReason finish_reason = runtime::FinishReason::None;
    const int32_t* generated = nullptr;
    std::size_t generated_size = 0;
    const int32_t* input_tokens = nullptr;
    std::size_t input_tokens_size = 0;
    uint32_t sequence_position = 0;
    uint32_t restored_tokens = 0;
    uint32_t generated_tokens = 0;
    bool dflash_used = false;
    bool structured_dflash = false;
    runtime::DraftMaskState draft_mask_state = runtime::DraftMaskState::Idle;
    uint64_t draft_round_id = 0;
    const int32_t* pending_drafts = nullptr;
    std::size_t pending_drafts_size = 0;
    double prefill_ms = 0.0;
    const char* error_message = nullptr;
};

struct PoolUsage {
    uint64_t arena_used_bytes = 0;
    uint64_t arena_committed_bytes = 0;
    uint64_t arena_capacity_bytes = 0;
    bool arena_vmm = false;
    uint64_t kv_reserved_bytes = 0;
    uint32_t kv_used_pages = 0;
    uint32_t kv_total_pages = 0;
    uint32_t sequence_used_slots = 0;
    uint32_t gdn_used_slots = 0;
};

struct MemoryDebugInfo {
    models::DenseWeightBytes dense_weights;
    bool has_dense_weight_view = false;
    uint64_t staging_dispatch_count = 0;
    uint64_t staging_max_image_bytes = 0;
    uint64_t staging_shared_program_bytes = 0;
    uint64_t staging_device_bytes = 0;
    uint64_t staging_host_bytes = 0;
    uint64_t execution_workspace_bytes = 0;
    uint64_t gdn_state_bytes = 0;
    std::vector<uint64_t> row_bucket_workspace_bytes;
};

class ComputeSession {
 public:
    static Result<std::unique_ptr<ComputeSession>> create(const ComputeConfig& config);

    ComputeSession(const ComputeSession&) = delete;
    ComputeSession& operator=(const ComputeSession&) = delete;
    ~ComputeSession() noexcept;

    Result<uint64_t> submit(
        std::vector<int32_t> input_ids,
        uint32_t max_new_tokens,
        const runtime::SamplingConfig& sampling);

    Result<uint64_t> submit(
        std::vector<int32_t> input_ids,
        uint32_t max_new_tokens,
        const runtime::SamplingConfig& sampling,
        const runtime::ThinkingBudget& thinking);

    Result<uint64_t> submit_unconstrained_decode(
        std::vector<int32_t> input_ids, uint32_t max_new_tokens,
        const runtime::SamplingConfig& sampling,
        const runtime::ThinkingBudget& thinking);

    Result<uint64_t> submit_structured_dflash(
        std::vector<int32_t> input_ids, uint32_t max_new_tokens,
        const runtime::SamplingConfig& sampling,
        const runtime::ThinkingBudget& thinking);

    Status set_token_bitmask(uint64_t id, const uint32_t* mask, bool waiting);
    Status set_structured_dflash(uint64_t id, bool delayed, int32_t close_token);
    Status activate_grammar(uint64_t id);
    Status set_draft_masks(uint64_t id, uint64_t round, const uint32_t* mask,
                           uint32_t stride, uint32_t valid_prefix);

    Result<runtime::StepResult> step();
    bool has_pending() const;
    bool has_active_requests() const;

    RequestView request(uint64_t id) const;

    Status set_external_request_id(uint64_t id, int64_t external_request_id);
    Status set_ignore_eos(uint64_t id, bool ignore_eos);
    Status cancel(uint64_t id);
    Status retire(uint64_t id);
    Status cancel_all();
    Status shutdown();

    bool is_shutdown() const noexcept;
    const ComputeConfig& config() const noexcept;
    double model_load_ms() const noexcept;
    uint32_t max_num_batched_tokens() const noexcept;
    bool dflash2_host_token_ids() const noexcept;
    bool dflash2_batcher_active() const noexcept;
    bool prefix_cache_active() const noexcept;
    std::string prefix_cache_error() const;
    runtime::KVCapacitySnapshot kv_capacity_snapshot() const;
    std::size_t queued_request_count() const noexcept;

    models::ModelInfo model_info() const;
    PoolUsage pool_usage() const;
    MemoryDebugInfo memory_debug();

    // Diagnostics only. The default path performs no device-to-host copy.
    bool has_target_hidden_taps() const noexcept;
    Status synchronize() const;
    Status copy_logits(std::vector<float>& out, std::size_t& rows);
    Status copy_hidden(std::size_t which, std::vector<float>& out, std::size_t& rows,
                       std::size_t& cols);
    Status copy_target_hidden(std::size_t index, std::vector<float>& out,
                              std::size_t& rows, std::size_t& cols,
                              std::size_t& layer);

 private:
    friend class ServeController;
    friend class SpeculationRunner;

    ComputeSession() = default;
    Status initialize(const ComputeConfig& config);
    Status initialize_tensor_parallel();
    Status initialize_pipeline_parallel();
    std::size_t drafter_reserve_bytes(const ::shelf::state::GdnStatePool& pool) const;
    std::size_t mtp_batch_requests(uint32_t num_drafts) const;
    void apply_auto_working_set_reserve();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace shelf::engine
