#pragma once

#include <phaseshift/core/memory/arena.h>
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/core/status.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <phaseshift/models/qwen35/state/state_types.h>
#include <phaseshift/inference/execution/spec_decode_context.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::qwen35::runtime {

using ::ps::execution::GdnSpecHistoryDeviceView;
using ::ps::execution::GdnCompactLogDeviceView;

struct GdnSpecHistory {
    gpu::Tensor conv;
    gpu::Tensor recurrent;

    uint32_t rows = 0u;
    std::size_t conv_state_bytes = 0u;
    std::size_t recurrent_state_bytes = 0u;
    bool initialized = false;
};

Result<GdnSpecHistory> create_gdn_spec_history(
    gpu::GpuArena& arena,
    const GdnStatePool& pool,
    uint32_t rows,
    bool with_recurrent = true);

GdnSpecHistoryDeviceView gdn_spec_history_view(
    const GdnSpecHistory& history,
    uint32_t capture_rows) noexcept;

Status shutdown_gdn_spec_history(GdnSpecHistory& history) noexcept;

Status restore_gdn_spec_history(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnSpecHistory& history,
    uint32_t history_row,
    hipStream_t stream);

Status restore_gdn_spec_conv(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnSpecHistory& history,
    uint32_t history_row,
    hipStream_t stream);

Status commit_gdn_compact_log(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnCompactLogDeviceView& log,
    const ::ps::runtime::DeviceRequestDescriptor* requests,
    uint32_t rows,
    uint32_t key_heads,
    hipStream_t stream,
    const int32_t* row_indices = nullptr,
    uint32_t log_row_capacity = 0u);

}  // namespace ps::qwen35::runtime
