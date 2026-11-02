#pragma once

#include <shelf/inference/state/paged_types.h>
#include <shelf/runtime/request_handle.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace shelf {
namespace state {

struct SequenceBlockTableDeviceView {
    uint32_t* block_tables = nullptr;
    uint32_t max_sequences = 0;
    uint32_t block_table_stride = 0;
};

class SequenceSlotPool {
 public:
    static Result<SequenceSlotPool> create(
        gpu::GpuArena& arena,
        uint32_t max_sequences,
        uint32_t max_blocks_per_sequence);

    Result<SequenceSlotId> allocate_slot();

    Status release_slot(SequenceSlotId slot);

    Result<::shelf::runtime::RequestHandle> allocate_handle();

    Status release_handle(::shelf::runtime::RequestHandle handle);

    bool is_current(::shelf::runtime::RequestHandle handle) const noexcept;

    gpu::Tensor device_block_table(SequenceSlotId slot) const;

    SequenceBlockTableDeviceView device_view() const noexcept {
        return SequenceBlockTableDeviceView{
            .block_tables = block_table_memory_.data<uint32_t>(),
            .max_sequences = max_sequences_,
            .block_table_stride = max_blocks_,
        };
    }

    Status set_block(
        SequenceSlotId slot,
        uint32_t logical_block,
        PageId page,
        hipStream_t stream);

    Status clear_blocks(
        SequenceSlotId slot,
        uint32_t first_block,
        uint32_t block_count,
        hipStream_t stream);

    Status clear_slot(
        SequenceSlotId slot,
        hipStream_t stream);

    uint32_t max_sequences() const noexcept { return max_sequences_; }
    uint32_t max_blocks_per_sequence() const noexcept { return max_blocks_; }

    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    std::size_t bytes_per_slot() const noexcept {
        return reserved_bytes_ / static_cast<std::size_t>(max_sequences_);
    }

    uint32_t num_used_slots() const noexcept { return used_count_; }

    uint32_t num_free_slots() const noexcept { return max_sequences_ - used_count_; }

    std::size_t used_bytes() const noexcept {
        return static_cast<std::size_t>(used_count_) * bytes_per_slot();
    }

    std::size_t free_bytes() const noexcept {
        return reserved_bytes_ - used_bytes();
    }

 private:
    gpu::Tensor block_table_memory_;
    int device_;
    uint32_t max_sequences_;
    uint32_t max_blocks_;
    std::vector<bool> in_use_;
    std::vector<uint32_t> free_list_;
    std::vector<uint32_t> generations_;
    std::size_t reserved_bytes_;
    uint32_t used_count_;

    SequenceSlotPool() = default;
};

}
}
