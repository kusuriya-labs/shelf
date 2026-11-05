#pragma once

#include <shelf/inference/state/paged_types.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace shelf {
namespace state {

struct QsaKeyState {
    gpu::Tensor pooled;
    gpu::Tensor partial;
    gpu::Tensor pool_count;
    gpu::Tensor partial_count;
};

struct QsaKeyStatePoolDeviceView {
    bf16_t* pooled_base = nullptr;
    bf16_t* partial_base = nullptr;
    uint32_t* pool_count_base = nullptr;
    uint32_t* partial_count_base = nullptr;

    uint32_t num_qsa_states = 0;
    uint32_t max_sequences = 0;
    uint32_t max_blocks = 0;
    uint32_t compress_ratio = 0;
    uint32_t head_dim = 0;

    uint64_t pooled_slot_stride = 0;
    uint64_t pooled_layer_stride = 0;
    uint64_t partial_slot_stride = 0;
    uint64_t partial_layer_stride = 0;
    uint64_t count_slot_stride = 0;
    uint64_t count_layer_stride = 0;
};

struct QsaKeyStatePoolLayout {
    uint32_t num_qsa_states = 0;
    uint32_t head_dim = 0;
    uint32_t compress_ratio = 0;
    uint32_t max_blocks = 0;

    static QsaKeyStatePoolLayout make(uint32_t qsa_layers, uint32_t head_dim,
                                      uint32_t compress_ratio, uint32_t max_blocks) {
        return QsaKeyStatePoolLayout{
            .num_qsa_states = qsa_layers,
            .head_dim = head_dim,
            .compress_ratio = compress_ratio,
            .max_blocks = max_blocks,
        };
    }
};

class QsaKeyStatePool {
 public:
    static Result<QsaKeyStatePool> create(
        gpu::GpuArena& arena,
        uint32_t max_sequences,
        const QsaKeyStatePoolLayout& layout);

    Status initialize(SequenceSlotId slot, hipStream_t stream);
    Status reset(SequenceSlotId slot, hipStream_t stream);
    Status release(SequenceSlotId slot);
    Status copy_slot_from(SequenceSlotId dst_slot, const QsaKeyStatePool& src,
                          SequenceSlotId src_slot, hipStream_t stream);

    QsaKeyState qsa_state(SequenceSlotId slot, uint32_t state_index) const;

    uint32_t max_sequences() const noexcept { return max_slots_; }
    uint32_t num_qsa_states() const noexcept { return layout_.num_qsa_states; }
    uint32_t max_blocks() const noexcept { return layout_.max_blocks; }
    uint32_t compress_ratio() const noexcept { return layout_.compress_ratio; }
    uint32_t head_dim() const noexcept { return layout_.head_dim; }
    const QsaKeyStatePoolLayout& layout() const noexcept { return layout_; }

    QsaKeyStatePoolDeviceView device_view() const noexcept {
        return QsaKeyStatePoolDeviceView{
            .pooled_base = pooled_memory_.data<bf16_t>(),
            .partial_base = partial_memory_.data<bf16_t>(),
            .pool_count_base = count_memory_.data<uint32_t>(),
            .partial_count_base = count_memory_.data<uint32_t>() + 1u,
            .num_qsa_states = layout_.num_qsa_states,
            .max_sequences = max_slots_,
            .max_blocks = layout_.max_blocks,
            .compress_ratio = layout_.compress_ratio,
            .head_dim = layout_.head_dim,
            .pooled_slot_stride =
                static_cast<uint64_t>(layout_.num_qsa_states) * layout_.max_blocks *
                layout_.head_dim,
            .pooled_layer_stride =
                static_cast<uint64_t>(layout_.max_blocks) * layout_.head_dim,
            .partial_slot_stride =
                static_cast<uint64_t>(layout_.num_qsa_states) * layout_.compress_ratio *
                layout_.head_dim,
            .partial_layer_stride =
                static_cast<uint64_t>(layout_.compress_ratio) * layout_.head_dim,
            .count_slot_stride = static_cast<uint64_t>(layout_.num_qsa_states) * 2u,
            .count_layer_stride = 2u,
        };
    }

    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    std::size_t bytes_per_slot() const noexcept {
        return reserved_bytes_ / static_cast<std::size_t>(max_slots_);
    }

    uint32_t num_used_slots() const noexcept { return used_count_; }
    uint32_t num_free_slots() const noexcept { return max_slots_ - used_count_; }

  private:
    gpu::Tensor pooled_memory_;
    gpu::Tensor partial_memory_;
    gpu::Tensor count_memory_;
    int device_ = 0;
    uint32_t max_slots_ = 0;
    QsaKeyStatePoolLayout layout_;
    std::vector<bool> initialized_;
    std::size_t reserved_bytes_ = 0;
    uint32_t used_count_ = 0;

    QsaKeyStatePool() = default;
};

}  // namespace state
}  // namespace shelf
