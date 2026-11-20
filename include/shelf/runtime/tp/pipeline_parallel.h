#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/gpu_memory_manager.h>
#include <shelf/runtime/rope_profile.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/runtime/executor.h>
#include <shelf/runtime/tp/tensor_parallel.h>
#include <shelf/runtime/tp/tp_batch_hook.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace shelf::models {
class PipelineStageModel;
}  // namespace shelf::models

namespace shelf {
namespace runtime {

using ::shelf::qwen_hybrid::PipelineStage;
using ::shelf::qwen_hybrid::StageRange;

// Loads one stage's weights and hands back the family-specific model behind them. The stage
// runtime does not know which family a checkpoint is, so the caller supplies the builder;
// shelf::models chooses between the families (shelf::models::select_pipeline_stage_model).
using PipelineStageModelBuilder =
    ::shelf::Result<std::unique_ptr<shelf::models::PipelineStageModel>> (*)(
        const std::string& model_dir,
        const PipelineStage& stage,
        gpu::GpuArena& arena,
        hipStream_t stream,
        const ::shelf::qwen_hybrid::HybridLoadOptions& load_options);

// The dense family's builder.


struct PipelineStageRuntimeConfig {
    std::string model_dir;
    PipelineStage stage;
    int device_id = 0;
    std::uint32_t max_batch_rows = 0;
    std::uint32_t max_scheduled_requests = 0;
    std::uint32_t max_scheduled_output_rows = 0;
    std::uint32_t max_seq_len = 16;
    std::uint32_t page_tokens = 16;
    std::uint32_t pipeline_hidden_width = 0;
    KVCacheDType kv_cache_dtype = KVCacheDType::BF16;
    ::shelf::kernel::OctaveKVCacheFormat octave_kv_format{};
    gpu::GpuArena* arena = nullptr;
    std::size_t working_set_reserve_bytes = 0;
    std::uint32_t kv_pool_pages = 0;
    ::shelf::runtime::EffectiveRopeProfile rope_profile;
    ::shelf::qwen_hybrid::HybridLoadOptions load_options;
    PipelineStageModelBuilder build_stage_model = nullptr;
    // Empty devices keeps the stage on a single executor. A populated config runs the stage
    // as a tensor-parallel group; the group width is tp.devices.size().
    TpCoordinatorConfig tp;
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
    gpu::GpuArena& arena() { return tp_ ? tp_->rank(0).arena() : *arena_; }
    const shelf::models::PipelineStageModel& model() const { return *model_; }
    const shelf::qwen_hybrid::QwenHybridConfig& hybrid_config() const;
    std::size_t host_pinned_bytes() const;
    SequenceSlotPool& sequence_pool() {
        return tp_ ? tp_->rank(0).sequence_pool() : seq_pool_.value();
    }
    GdnStatePool& gdn_pool() { return tp_ ? tp_->rank(0).gdn_pool() : gdn_pool_.value(); }
    PagedKVPool& kv_pool() { return tp_ ? tp_->rank(0).kv_pool() : kv_pool_.value(); }
    Executor& executor() { return tp_ ? tp_->rank(0).executor() : executor_; }
    TpCoordinator* tp() { return tp_.get(); }
    const TpCoordinator* tp() const { return tp_.get(); }
    const PipelineStage& stage() const noexcept { return stage_; }
    std::uint32_t kv_pool_pages() const;

private:
    PipelineStageRuntime() = default;

    int device_id_ = -1;
    hipStream_t stream_ = nullptr;
    gpu::GpuArena* arena_ = nullptr;
    std::unique_ptr<shelf::models::PipelineStageModel> model_;
    std::optional<SequenceSlotPool> seq_pool_;
    std::optional<GdnStatePool> gdn_pool_;
    std::optional<PagedKVPool> kv_pool_;
    Executor executor_;
    std::unique_ptr<TpCoordinator> tp_;
    bool executor_ready_ = false;
    PipelineStage stage_;
};

struct PipelineParallelConfig {
    std::string model_dir;
    std::vector<int> devices;
    std::vector<StageRange> stages;
    // One entry per stage. An empty entry keeps that stage on a single device; a populated
    // entry runs the stage as a tensor-parallel group.
    std::vector<TpCoordinatorConfig> stage_tp;
    std::uint32_t max_batch_rows = 0;
    std::uint32_t max_scheduled_requests = 0;
    std::uint32_t max_scheduled_output_rows = 0;
    std::uint32_t max_seq_len = 16;
    std::uint32_t page_tokens = 16;
    std::uint32_t pipeline_hidden_width = 0;
    KVCacheDType kv_cache_dtype = KVCacheDType::BF16;
    ::shelf::kernel::OctaveKVCacheFormat octave_kv_format{};
    gpu::GpuMemoryManager* memory = nullptr;
    std::size_t working_set_reserve_bytes = 0;
    ::shelf::runtime::EffectiveRopeProfile rope_profile;
    ::shelf::qwen_hybrid::HybridLoadOptions load_options;
    PipelineStageModelBuilder build_stage_model = nullptr;
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
}  // namespace shelf
