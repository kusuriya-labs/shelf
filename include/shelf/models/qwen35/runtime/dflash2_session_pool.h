#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ps::qwen35::runtime {

struct DFlash2SessionSlot {
    dflash2::DFlash2ContextState context;
    DFlash2SpecWorkspace workspace;
    bool in_use = false;
};

class DFlash2SessionPool {
 public:
    DFlash2SessionPool() = default;

    static Result<DFlash2SessionPool> create(
        Executor& target,
        dflash2::DFlash2Executor* draft,
        GdnStatePool& gdn_pool,
        gpu::GpuArena& arena,
        const DFlash2SpecDecoderConfig& config,
        uint32_t max_sessions,
        hipStream_t stream,
        const std::vector<DFlash2GdnRankBinding>* rank_bindings = nullptr);

    Result<uint32_t> acquire();

    Status release(uint32_t index);

    Status shutdown() noexcept;

    uint32_t capacity() const noexcept {
        return static_cast<uint32_t>(slots_.size());
    }

    uint32_t used() const noexcept { return used_; }

    uint32_t num_free() const noexcept {
        return static_cast<uint32_t>(free_list_.size());
    }

    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    DFlash2SessionSlot& slot(uint32_t index) noexcept { return slots_[index]; }

    const DFlash2SessionSlot& slot(uint32_t index) const noexcept { return slots_[index]; }

 private:
    std::vector<DFlash2SessionSlot> slots_;
    std::vector<uint32_t> free_list_;
    uint32_t used_ = 0u;
    std::size_t reserved_bytes_ = 0u;
};

}  // namespace ps::qwen35::runtime
