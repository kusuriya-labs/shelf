#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/eagle4/config.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::qwen35::eagle4 {

struct Eagle4ContextState {
    gpu::Tensor k_ring;
    gpu::Tensor v_ring;

    uint32_t num_layers = 0;
    uint32_t capacity = 0;
    uint32_t kv_features = 0;

    uint32_t next_position = 0;

    bool initialized = false;
};

Result<Eagle4ContextState> create_eagle4_context_state(
    const Eagle4Config& config,
    gpu::GpuArena& arena,
    uint32_t initial_position = 0,
    std::uint32_t tp_size = 1);

Status eagle4_context_reset(Eagle4ContextState& state, uint32_t next_position);

Status eagle4_context_shutdown(Eagle4ContextState& state) noexcept;

}  // namespace ps::qwen35::eagle4
