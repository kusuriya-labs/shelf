#pragma once

#include <shelf/inference/state/paged_types.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/inference/execution/spec_decode_context.h>
#include <shelf/kernels/optimized/ple_conv.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shelf {
namespace state {

inline constexpr uint32_t kPleConvHistory = 9u;

struct PleConvStatePoolDeviceView {
    bf16_t* base = nullptr;
    uint32_t max_sequences = 0;
    uint32_t channels = 0;
    uint64_t slot_stride = 0;
    uint64_t layer_stride = 0;
    uint64_t history_stride = 0;
};

struct PleConvStatePoolLayout {
    uint32_t num_ple_states = 0;
    uint32_t channels = 0;
    uint32_t history = kPleConvHistory;

    static PleConvStatePoolLayout make(uint32_t layers, uint32_t channels) {
        return PleConvStatePoolLayout{.num_ple_states = layers, .channels = channels,
                                      .history = kPleConvHistory};
    }
};

inline constexpr std::size_t kPleConvStatePoolAlignment = 256u;

std::size_t ple_conv_state_pool_bytes(uint32_t max_sequences,
                                      const PleConvStatePoolLayout& layout) noexcept;

class PleConvStatePool {
 public:
    static Result<PleConvStatePool> create(gpu::GpuArena& arena, uint32_t max_sequences,
                                           const PleConvStatePoolLayout& layout);
    Status initialize(SequenceSlotId slot, hipStream_t stream);
    Status reset(SequenceSlotId slot, hipStream_t stream);
    Status release(SequenceSlotId slot);
    Status copy_slot_from(SequenceSlotId dst_slot, const PleConvStatePool& src,
                          SequenceSlotId src_slot, hipStream_t stream);

    PleConvStatePoolDeviceView device_view() const noexcept {
        return PleConvStatePoolDeviceView{
            .base = memory_.data<bf16_t>(),
            .max_sequences = max_slots_,
            .channels = layout_.channels,
            .slot_stride = static_cast<uint64_t>(layout_.num_ple_states) *
                           layout_.history * layout_.channels,
            .layer_stride = static_cast<uint64_t>(layout_.history) * layout_.channels,
            .history_stride = layout_.channels,
        };
    }

    uint32_t max_sequences() const noexcept { return max_slots_; }
    uint32_t num_ple_states() const noexcept { return layout_.num_ple_states; }
    uint32_t channels() const noexcept { return layout_.channels; }
    const PleConvStatePoolLayout& layout() const noexcept { return layout_; }
    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

 private:
    gpu::Tensor memory_;
    int device_ = 0;
    uint32_t max_slots_ = 0;
    PleConvStatePoolLayout layout_;
    std::vector<bool> initialized_;
    std::size_t reserved_bytes_ = 0;
    uint32_t used_count_ = 0;

    PleConvStatePool() = default;
};

// Puts one slot's conv window back to the state recorded before verify row `row`.
Status restore_ple_conv_spec_state(
    const PleConvStatePool& pool,
    SequenceSlotId slot,
    const ::shelf::execution::PleConvSpecHistoryDeviceView& history,
    uint32_t row,
    hipStream_t stream);

}  // namespace state
}  // namespace shelf
