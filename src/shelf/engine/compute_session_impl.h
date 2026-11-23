#pragma once

#include <shelf/core/memory/gpu_memory_manager.h>
#include <shelf/engine/compute_session.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/runtime/continuous_batcher.h>
#include <shelf/runtime/executor.h>
#include <shelf/runtime/parallel/parallel_topology.h>
#include <shelf/runtime/prefix_cache.h>
#include <shelf/runtime/tp/pipeline_parallel.h>
#include <shelf/runtime/tp/tensor_parallel.h>

#include <memory>
#include <optional>
#include <string>

namespace shelf::engine {

struct ComputeSession::Impl {
    ComputeConfig config_;
    uint32_t exec_max_tokens_ = 0;
    double model_load_ms_ = 0.0;
    bool shutdown_ = false;
    bool auto_arena_ = false;
    std::size_t startup_available_bytes_ = 0;

    std::optional<runtime::ParallelTopology> topology_;
    std::optional<gpu::GpuMemoryManager> memory_;

    std::unique_ptr<models::TpGroup> tp_group_;
    runtime::TpCoordinator* tp_coordinator_ = nullptr;
    std::unique_ptr<runtime::PipelineParallelCoordinator> pp_coordinator_;
    std::unique_ptr<models::LoadedModel> pp_model_;
    runtime::Executor* tp_executor_ = nullptr;
    state::SequenceSlotPool* tp_slots_ = nullptr;
    state::GdnStatePool* tp_gdn_ = nullptr;
    state::PagedKVPool* tp_kv_ = nullptr;
    gpu::GpuArena* tp_arena_ = nullptr;
    hipStream_t tp_stream_ = nullptr;

    hipStream_t stream_ = nullptr;
    gpu::GpuArena* arena_ = nullptr;
    std::unique_ptr<models::LoadedModel> model_;
    std::optional<state::SequenceSlotPool> slot_pool_;
    std::optional<state::GdnStatePool> gdn_pool_;
    std::optional<state::PagedKVPool> kv_pool_;
    runtime::Executor executor_;
    std::unique_ptr<runtime::ContinuousBatcher> batcher_;
    std::optional<runtime::PrefixCache> prefix_cache_;
    std::string prefix_cache_error_;

    runtime::Executor& executor() {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).executor();
        return tp_executor_ != nullptr ? *tp_executor_ : executor_;
    }

    const models::LoadedModel& model() const {
        if (pp_model_ != nullptr) return *pp_model_;
        return tp_group_ != nullptr ? tp_group_->model() : *model_;
    }

    state::SequenceSlotPool& slot_pool() {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).sequence_pool();
        return tp_slots_ != nullptr ? *tp_slots_ : *slot_pool_;
    }

    state::GdnStatePool& gdn_pool() {
        if (pp_coordinator_ != nullptr) return pp_coordinator_->stage(0).gdn_pool();
        return tp_gdn_ != nullptr ? *tp_gdn_ : *gdn_pool_;
    }

    state::PagedKVPool& kv_pool() {
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

    state::SequenceSlotPool& slot_pool() const {
        return const_cast<Impl*>(this)->slot_pool();
    }
    state::GdnStatePool& gdn_pool() const {
        return const_cast<Impl*>(this)->gdn_pool();
    }
    state::PagedKVPool& kv_pool() const {
        return const_cast<Impl*>(this)->kv_pool();
    }
    gpu::GpuArena& arena() const {
        return const_cast<Impl*>(this)->arena();
    }
};

}  // namespace shelf::engine
