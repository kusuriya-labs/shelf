#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/dflash2/config.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::qwen35::dflash2 {

struct DFlash2ContextState {
    gpu::Tensor k_ring;
    gpu::Tensor v_ring;

    uint32_t num_layers = 0;
    uint32_t capacity = 0;
    uint32_t kv_features = 0;

    uint32_t length = 0;
    uint32_t next_position = 0;

    bool initialized = false;
};

Result<DFlash2ContextState> create_dflash2_context_state(
    const DFlash2Config& config,
    gpu::GpuArena& arena,
    uint32_t initial_position = 0);

Status dflash2_context_reset(
    DFlash2ContextState& state,
    uint32_t next_position);

Status dflash2_context_shutdown(DFlash2ContextState& state) noexcept;

Status dflash2_context_append(
    DFlash2Executor& executor,
    DFlash2ContextState& state,
    const bf16_t* target_feature,
    uint32_t rows,
    uint32_t position_start,
    hipStream_t stream);

}  // namespace ps::qwen35::dflash2
