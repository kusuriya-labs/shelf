#include <shelf/models/qwen35/runtime/dflash2_session_pool.h>
#include <shelf/core/memory/size_format.h>

namespace ps {
namespace qwen35 {
namespace runtime {

Result<DFlash2SessionPool> DFlash2SessionPool::create(
    Executor& target,
    dflash2::DFlash2Executor* draft,
    GdnStatePool& gdn_pool,
    gpu::GpuArena& arena,
    const DFlash2SpecDecoderConfig& config,
    uint32_t max_sessions,
    hipStream_t stream,
    const std::vector<DFlash2GdnRankBinding>* rank_bindings) {
    if (max_sessions == 0u) {
        return Status::invalid_argument("dflash2 session pool needs a positive slot count",
                                        __FILE__, __LINE__);
    }
    if (draft == nullptr) {
        return Status::invalid_state("dflash2 session pool requires a draft", __FILE__,
                                     __LINE__);
    }
    if (!draft->initialized || draft->config == nullptr) {
        return Status::invalid_state("dflash2 session pool requires an initialized draft",
                                     __FILE__, __LINE__);
    }
    const std::size_t start_bytes = arena.used();
    DFlash2SessionPool pool;
    pool.slots_.resize(max_sessions);
    for (uint32_t i = 0; i < max_sessions; ++i) {
        DFlash2SessionSlot& slot = pool.slots_[i];
        if (draft != nullptr) {
            auto context_result =
                dflash2::create_dflash2_context_state(*draft->config, arena, 0u);
            if (!context_result.ok()) {
                (void)pool.shutdown();
                return context_result.status();
            }
            slot.context = context_result.release();
        }

        const uint32_t ws_draft_cap =
            static_cast<uint32_t>(draft->config->max_draft_tokens());
        auto workspace_result = create_dflash2_spec_workspace_for_drafts(
            target, ws_draft_cap, gdn_pool, arena, config, stream, rank_bindings);
        if (!workspace_result.ok()) {
            (void)pool.shutdown();
            return workspace_result.status();
        }
        slot.workspace = workspace_result.release();

        if (i == 0u) {
            const std::size_t per_slot = arena.used() - start_bytes;
            const std::size_t projected = per_slot * max_sessions;
            const std::size_t arena_free = arena.capacity() - start_bytes;
            if (projected > arena_free) {
                (void)pool.shutdown();
                return Status::insufficient_memory(
                    "dflash2 session pool needs " + gpu::format_gib(projected)
                    + " for " + std::to_string(max_sessions)
                    + " concurrent sessions but only " + gpu::format_gib(arena_free)
                    + " arena capacity is free (reduce --max-concurrent-requests or the "
                      "KV budget)",
                    __FILE__, __LINE__);
            }
        }
    }
    pool.free_list_.reserve(max_sessions);
    for (uint32_t i = max_sessions; i-- > 0;) {
        pool.free_list_.push_back(i);
    }
    pool.reserved_bytes_ = arena.used() - start_bytes;
    return pool;
}

Result<uint32_t> DFlash2SessionPool::acquire() {
    if (free_list_.empty()) {
        return Status::out_of_range("dflash2 session pool exhausted", __FILE__, __LINE__);
    }
    const uint32_t index = free_list_.back();
    free_list_.pop_back();
    DFlash2SessionSlot& target = slots_[index];
    if (target.context.initialized) {
        const Status reset_st = dflash2::dflash2_context_reset(target.context, 0u);
        if (!reset_st.ok()) {
            free_list_.push_back(index);
            return reset_st;
        }
    }
    target.in_use = true;
    ++used_;
    return index;
}

Status DFlash2SessionPool::release(uint32_t index) {
    if (index >= slots_.size()) {
        return Status::out_of_range("dflash2 session slot out of range", __FILE__, __LINE__);
    }
    DFlash2SessionSlot& target = slots_[index];
    if (!target.in_use) {
        return Status::invalid_state("dflash2 session slot is not in use", __FILE__, __LINE__);
    }
    Status reset_st = Status::make_ok();
    if (target.context.initialized) {
        reset_st = dflash2::dflash2_context_reset(target.context, 0u);
    }
    target.in_use = false;
    free_list_.push_back(index);
    --used_;
    return reset_st;
}

Status DFlash2SessionPool::shutdown() noexcept {
    Status first_error = Status::make_ok();
    for (DFlash2SessionSlot& slot : slots_) {
        const Status workspace_st = shutdown_dflash2_spec_workspace(slot.workspace);
        if (!workspace_st.ok() && first_error.ok()) first_error = workspace_st;
        if (slot.context.initialized) {
            const Status context_st = dflash2::dflash2_context_shutdown(slot.context);
            if (!context_st.ok() && first_error.ok()) first_error = context_st;
        }
        slot.in_use = false;
    }
    slots_.clear();
    free_list_.clear();
    used_ = 0u;
    reserved_bytes_ = 0u;
    return first_error;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace ps
