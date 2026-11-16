#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/inference/execution/spec_decode_context.h>
#include <shelf/inference/state/ple_conv_state_pool.h>
#include <shelf/inference/state/qsa_state_pool.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::state {

struct SpecStatePools {
    const QsaKeyStatePool* qsa = nullptr;
    const PleConvStatePool* ple = nullptr;
};

struct SpecStateHistory {
    ::shelf::execution::QsaSpecHistoryDeviceView qsa{};
    ::shelf::execution::PleConvSpecHistoryDeviceView ple{};
};

Result<SpecStateHistory> create_spec_state_history(gpu::GpuArena& arena,
                                                   const SpecStatePools& pools,
                                                   uint32_t rows);

Status restore_spec_state(const SpecStatePools& pools, SequenceSlotId slot,
                          const SpecStateHistory& history, uint32_t row,
                          hipStream_t stream);

}  // namespace shelf::state
