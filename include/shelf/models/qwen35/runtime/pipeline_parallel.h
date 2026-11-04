#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35/model/stage_range.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/tp_batch_hook.h>
#include <shelf/models/qwen35/state/state_types.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace shelf {
namespace qwen35 {
namespace runtime {

struct PipelineStageRuntimeConfig {
    std::string model_dir;
    PipelineStage stage;
    int device_id = 0;
    std::uint32_t max_batch_rows = 0;
    std::uint32_t max_scheduled_requests = 0;
    std::uint32_t max_scheduled_output_rows = 0;
    std::uint32_t max_seq_len = 16;
    std::uint32_t page_tokens = 16;
    KVCacheDType kv_cache_dtype = KVCacheDType::BF16;
    std::size_t arena_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    std::size_t working_set_reserve_bytes = 0;
    std::uint32_t kv_pool_pages = 0;
    EffectiveRopeProfile rope_profile;
    Qwen35LoadOptions load_options;
};

class PipelineStageRuntime {
public:
    static Result<std::unique_ptr<PipelineStageRuntime>> create(
        const PipelineStageRuntimeConfig& config);
    ~PipelineStageRuntime() noexcept;

    PipelineStageRuntime(const PipelineStageRuntime&) = delete;
    PipelineStageRuntime& operator=(const PipelineStageRuntime&) = delete;

    int device_id() const noexcept { return device_id_; }
    hipStream_t stream() const noexcept { return stream_; }
    gpu::GpuArena& arena() { return arena_.value(); }
    const Qwen35Model& model() const { return model_.value(); }
    SequenceSlotPool& sequence_pool() { return seq_pool_.value(); }
    GdnStatePool& gdn_pool() { return gdn_pool_.value(); }
    PagedKVPool& kv_pool() { return kv_pool_.value(); }
    Executor& executor() { return executor_; }
    const PipelineStage& stage() const noexcept { return stage_; }
    std::uint32_t kv_pool_pages() const;

private:
    PipelineStageRuntime() = default;

    int device_id_ = -1;
    hipStream_t stream_ = nullptr;
    std::optional<gpu::GpuArena> arena_;
    std::optional<Qwen35Model> model_;
    std::optional<SequenceSlotPool> seq_pool_;
    std::optional<GdnStatePool> gdn_pool_;
    std::optional<PagedKVPool> kv_pool_;
    Executor executor_;
    bool executor_ready_ = false;
    PipelineStage stage_;
};

struct PipelineParallelConfig {
    std::string model_dir;
    std::vector<int> devices;
    std::vector<StageRange> stages;
    std::uint32_t max_batch_rows = 0;
    std::uint32_t max_scheduled_requests = 0;
    std::uint32_t max_scheduled_output_rows = 0;
    std::uint32_t max_seq_len = 16;
    std::uint32_t page_tokens = 16;
    KVCacheDType kv_cache_dtype = KVCacheDType::BF16;
    std::size_t arena_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    std::size_t working_set_reserve_bytes = 0;
    EffectiveRopeProfile rope_profile;
    Qwen35LoadOptions load_options;
};

class PipelineParallelCoordinator final : public TpBatchHook {
public:
    static Result<std::unique_ptr<PipelineParallelCoordinator>> create(
        const PipelineParallelConfig& config);
    ~PipelineParallelCoordinator() noexcept;

    PipelineParallelCoordinator(const PipelineParallelCoordinator&) = delete;
    PipelineParallelCoordinator& operator=(const PipelineParallelCoordinator&) = delete;

    std::size_t stage_count() const noexcept { return stages_.size(); }
    PipelineStageRuntime& stage(std::size_t index) { return *stages_[index]; }
    const PipelineStageRuntime& stage(std::size_t index) const { return *stages_[index]; }

    Status on_sequence_created(const PagedSequenceState& source) override;
    Status on_sequence_released(const PagedSequenceState& source) override;
    Result<BatchExecutionOutput> on_execute(const ScheduledBatch& batch) override;
    Result<BatchExecutionOutput> on_execute(
        const ScheduledBatch& batch, const ExecuteBatchOptions* per_rank) override;

    std::string debug_report() const;

private:
    struct MirrorSequences {
        std::vector<PagedSequenceState> states;
    };

    PipelineParallelCoordinator() = default;

    Status initialize(const PipelineParallelConfig& config);
    Status sync_mirror_state(const ScheduledBatch& batch);
    Status run_stage(std::size_t index, const ScheduledBatch& batch,
                     const uint16_t* input_hidden, std::size_t rows,
                     BatchExecutionOutput& output);

    std::vector<int> devices_;
    std::vector<std::unique_ptr<PipelineStageRuntime>> stages_;
    std::unordered_map<std::uint64_t, MirrorSequences> mirrors_;
    void* host_hidden_ = nullptr;
    std::size_t host_hidden_bytes_ = 0;
    void* host_sampled_ = nullptr;
    std::size_t host_sampled_bytes_ = 0;
    std::uint32_t hidden_size_ = 0;
    std::uint32_t max_rows_ = 0;
    std::uint32_t max_outputs_ = 0;
};

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
