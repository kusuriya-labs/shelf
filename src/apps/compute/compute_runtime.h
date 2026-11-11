#pragma once

#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35/runtime/continuous_batcher.h>
#include <shelf/models/qwen35/runtime/decode_backend.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/prefix_cache.h>
#include <shelf/models/qwen35/runtime/pipeline_parallel.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace shelf {
namespace app {

enum class ArenaBudgetMode {
    // Free VRAM at startup minus a small margin.
    Auto,
    // Exactly `arena_bytes`.
    Fixed,
    // Total VRAM minus `arena_bytes` (a margin).
    TotalMinus,
};

struct Qwen35RuntimeConfig {
    std::string model_dir;
    std::string dflash2_model_dir;
    qwen35::dflash2::DFlash2Config draft_config;
    bool draft_config_ready = false;
    uint32_t dflash2_drafts = 7;
    bool dflash2_tree = false;
    uint32_t dflash2_tree_extra = 0;
    uint32_t dflash2_tree_budget = 0;
    bool dflash2_tree_budget_auto = false;
    // 0 = min(max_concurrent_requests, 8). Requests beyond this run without speculation.
    uint32_t dflash2_max_sessions = 0;
    uint32_t dflash2_sessions() const noexcept {
        const uint32_t cap = max_concurrent_requests == 0u ? 1u : max_concurrent_requests;
        const uint32_t want = dflash2_max_sessions != 0u ? dflash2_max_sessions
                                                         : (cap < 8u ? cap : 8u);
        return want < cap ? want : cap;
    }
    uint32_t dflash2_prefill_rows = 0;
    bool mtp = false;
    bool mtp_serve = false;
    uint32_t mtp_drafts = 7;
    uint32_t max_seq_len = 512;
    uint32_t max_num_batched_tokens = 0;
    uint32_t max_concurrent_requests = 4;
    ArenaBudgetMode arena_mode = ArenaBudgetMode::Auto;
    std::size_t arena_bytes = 0;
    std::size_t draft_bytes = 0;
    bool private_target_weights = false;
    bool private_target_lm_head = true;
    uint32_t page_tokens = 16;
    std::vector<int> devices = {0};
    uint32_t tp = 1;
    uint32_t pipeline_parallel = 1;
    kernel::KVCacheDType kv_cache_dtype = kernel::KVCacheDType::BF16;
    kernel::OctaveKVCacheFormat octave_kv_format{};
    bool verify_weights = false;
    uint32_t max_scheduled_output_rows = 0;
    std::array<uint32_t, qwen35::kMaxTargetHiddenTaps> target_hidden_taps{};
    uint32_t target_hidden_tap_count = 0;
    uint32_t tp_combine_f32 = 0;
    qwen35::runtime::DecodeBackend decode_backend =
        qwen35::runtime::DecodeBackend::Hip;
    bool prefix_cache = true;
    uint32_t prefix_cache_concurrency = 4;
    uint32_t prefix_cache_spec_ring_bytes = 0;
    std::size_t working_set_reserve_bytes = 4ull * 1024ull * 1024ull * 1024ull;
    std::optional<qwen35::EffectiveRopeProfile> rope_profile;
};

class Qwen35ComputeRuntime {
 public:
    static Result<std::unique_ptr<Qwen35ComputeRuntime>> create(
        const Qwen35RuntimeConfig& config);

    Qwen35ComputeRuntime(const Qwen35ComputeRuntime&) = delete;
    Qwen35ComputeRuntime& operator=(const Qwen35ComputeRuntime&) = delete;
    ~Qwen35ComputeRuntime() noexcept;

    Result<uint64_t> submit(
        std::vector<int32_t> input_ids,
        uint32_t max_new_tokens,
        const qwen35::runtime::SamplingConfig& sampling);

    Result<uint64_t> submit(
        std::vector<int32_t> input_ids,
        uint32_t max_new_tokens,
        const qwen35::runtime::SamplingConfig& sampling,
        const qwen35::runtime::ThinkingBudget& thinking);

    Result<uint64_t> submit_unconstrained_decode(
        std::vector<int32_t> input_ids, uint32_t max_new_tokens,
        const qwen35::runtime::SamplingConfig& sampling,
        const qwen35::runtime::ThinkingBudget& thinking);

    Status set_token_bitmask(uint64_t id, const uint32_t* mask, bool waiting);

    Result<qwen35::runtime::StepResult> step();

    bool has_pending() const;

    const qwen35::runtime::RuntimeRequest* request(uint64_t id) const;

    Status set_external_request_id(uint64_t id, int64_t external_request_id);

    Status cancel(uint64_t id);

    Status retire(uint64_t id);

    Status cancel_all();

    Status shutdown();

    bool is_shutdown() const noexcept { return shutdown_; }

    const Qwen35RuntimeConfig& config() const noexcept { return config_; }

    const shelf::models::LoadedModel& model() const
    {
        if (pp_model_ != nullptr) return *pp_model_;
        return tp_group_ != nullptr ? tp_group_->model() : *model_;
    }

    qwen35::Executor& executor()
    {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).executor();
        return tp_executor_ != nullptr ? *tp_executor_ : executor_;
    }

    state::SequenceSlotPool& slot_pool()
    {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).sequence_pool();
        return tp_slots_ != nullptr ? *tp_slots_ : *slot_pool_;
    }

    state::GdnStatePool& gdn_pool()
    {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).gdn_pool();
        return tp_gdn_ != nullptr ? *tp_gdn_ : *gdn_pool_;
    }

    state::PagedKVPool& kv_pool()
    {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).kv_pool();
        return tp_kv_ != nullptr ? *tp_kv_ : *kv_pool_;
    }

    gpu::GpuArena& arena() {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).arena();
        return tp_arena_ != nullptr ? *tp_arena_ : *arena_;
    }

    hipStream_t stream() const noexcept {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).stream();
        return tp_stream_ != nullptr ? tp_stream_ : stream_;
    }

    double model_load_ms() const noexcept { return model_load_ms_; }

    Result<qwen35::dflash2::DFlash2Weights> load_dflash2_weights(
        const qwen35::dflash2::DFlash2Config& config,
        const shelf::weights::WeightLoadOptions& options = {});

    Status set_dflash2_batcher(const qwen35::runtime::DFlash2BatcherBinding& binding);

    bool dflash2_batcher_active() const noexcept {
        return batcher_ != nullptr && batcher_->dflash2_available();
    }

    uint32_t max_num_batched_tokens() const noexcept { return exec_max_tokens_; }

    qwen35::runtime::TpCoordinator* tp_coordinator() noexcept { return tp_coordinator_; }

    bool prefix_cache_active() const noexcept { return prefix_cache_.has_value(); }

    qwen35::runtime::KVCapacitySnapshot kv_capacity_snapshot() const;

    std::size_t queued_request_count() const noexcept;

    qwen35::runtime::PrefixCache* prefix_cache() noexcept {
        return prefix_cache_.has_value() ? &prefix_cache_.value() : nullptr;
    }

    const std::string& prefix_cache_error() const noexcept { return prefix_cache_error_; }

 private:
    Qwen35ComputeRuntime() = default;

    Status initialize(const Qwen35RuntimeConfig& config);

    Status initialize_tensor_parallel();

    Status initialize_pipeline_parallel();

    std::size_t drafter_reserve_bytes(const ::shelf::state::GdnStatePool& pool) const;

    std::size_t mtp_batch_requests(
        const qwen35::runtime::MtpDrafterConfig& options) const;

    void apply_auto_working_set_reserve();

    Qwen35RuntimeConfig config_;
    uint32_t exec_max_tokens_ = 0;
    double model_load_ms_ = 0.0;
    bool shutdown_ = false;
    bool auto_arena_ = false;
    std::size_t startup_available_bytes_ = 0;

    // The tensor parallel group owns the ranks, their weights and the loaded-model handle.
    // tp_coordinator_ is a handle into it, so it is cleared before the group is.
    std::unique_ptr<shelf::models::TpGroup> tp_group_;
    qwen35::runtime::TpCoordinator* tp_coordinator_ = nullptr;
    std::unique_ptr<qwen35::runtime::PipelineParallelCoordinator> pp_coordinator_;
    std::unique_ptr<shelf::models::LoadedModel> pp_model_;
    qwen35::Executor* tp_executor_ = nullptr;
    state::SequenceSlotPool* tp_slots_ = nullptr;
    state::GdnStatePool* tp_gdn_ = nullptr;
    state::PagedKVPool* tp_kv_ = nullptr;
    gpu::GpuArena* tp_arena_ = nullptr;
    hipStream_t tp_stream_ = nullptr;

    hipStream_t stream_ = nullptr;
    std::optional<gpu::GpuArena> arena_;
    std::unique_ptr<shelf::models::LoadedModel> model_;
    std::optional<state::SequenceSlotPool> slot_pool_;
    std::optional<state::GdnStatePool> gdn_pool_;
    std::optional<state::PagedKVPool> kv_pool_;
    qwen35::Executor executor_;
    std::unique_ptr<qwen35::runtime::ContinuousBatcher> batcher_;
    std::optional<qwen35::runtime::PrefixCache> prefix_cache_;
    std::string prefix_cache_error_;
};

}  // namespace app
}  // namespace shelf
