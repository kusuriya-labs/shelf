#include <shelf/inference/state/sequence_slot_pool.h>

namespace ps {
namespace state {

Result<SequenceSlotPool> SequenceSlotPool::create(
    gpu::GpuArena& arena,
    uint32_t max_sequences,
    uint32_t max_blocks_per_sequence)
{
    if (max_sequences == 0) {
        return Status::invalid_argument("max_sequences must be > 0", __FILE__, __LINE__);
    }

    if (max_blocks_per_sequence == 0) {
        return Status::invalid_argument("max_blocks_per_sequence must be > 0", __FILE__, __LINE__);
    }

    SequenceSlotPool pool;
    pool.device_ = arena.device();
    pool.max_sequences_ = max_sequences;
    pool.max_blocks_ = max_blocks_per_sequence;
    pool.used_count_ = 0;

    auto alloc_result = arena.allocate_aligned(
        static_cast<std::size_t>(max_sequences) * max_blocks_per_sequence * sizeof(uint32_t),
        256);

    if (!alloc_result.ok()) {
        return alloc_result.status();
    }

    std::size_t shape[2] = {
        static_cast<std::size_t>(max_sequences),
        static_cast<std::size_t>(max_blocks_per_sequence),
    };
    std::size_t strides[2] = {
        static_cast<std::size_t>(max_blocks_per_sequence),
        1,
    };

    auto tensor_result = gpu::Tensor::create<uint32_t>(
        alloc_result.value(),
        shape,
        2,
        strides);

    if (!tensor_result.ok()) {
        return tensor_result.status();
    }

    pool.block_table_memory_ = tensor_result.value();
    pool.reserved_bytes_ =
        static_cast<std::size_t>(max_sequences) * max_blocks_per_sequence * sizeof(uint32_t);

    pool.in_use_.resize(max_sequences, false);
    pool.generations_.resize(max_sequences, 0u);
    pool.free_list_.reserve(max_sequences);
    for (uint32_t i = 0; i < max_sequences; ++i) {
        pool.free_list_.push_back(i);
    }

    return pool;
}

Result<::ps::runtime::RequestHandle> SequenceSlotPool::allocate_handle() {
    if (free_list_.empty()) {
        return Status::insufficient_memory("no free sequence slots", __FILE__, __LINE__);
    }

    const uint32_t slot = free_list_.back();
    free_list_.pop_back();

    if (in_use_[slot]) {
        return Status::invalid_state(
            "free list contains allocated sequence slot", __FILE__, __LINE__);
    }

    uint32_t generation = generations_[slot] + 1u;

    // generation 0 is permanently invalid.
    if (generation == 0u) {
        generation = 1u;
    }

    generations_[slot] = generation;
    in_use_[slot] = true;
    ++used_count_;

    return ::ps::runtime::RequestHandle{
        .slot = slot,
        .generation = generation,
    };
}

Result<SequenceSlotId> SequenceSlotPool::allocate_slot() {
    auto handle = allocate_handle();
    if (!handle.ok()) {
        return handle.status();
    }
    return handle.value().slot;
}

bool SequenceSlotPool::is_current(::ps::runtime::RequestHandle handle) const noexcept {
    if (!::ps::runtime::request_handle_valid(handle)) {
        return false;
    }
    if (handle.slot >= max_sequences_) {
        return false;
    }
    return in_use_[handle.slot] && generations_[handle.slot] == handle.generation;
}

Status SequenceSlotPool::release_handle(::ps::runtime::RequestHandle handle) {
    if (!is_current(handle)) {
        return Status::invalid_state("stale or invalid request handle", __FILE__, __LINE__);
    }
    return release_slot(handle.slot);
}

Status SequenceSlotPool::release_slot(SequenceSlotId slot) {
    if (slot >= max_sequences_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }

    if (!in_use_[slot]) {
        return Status::invalid_state("slot not allocated", __FILE__, __LINE__);
    }

    in_use_[slot] = false;
    free_list_.push_back(slot);
    --used_count_;

    return Status::make_ok();
}

gpu::Tensor SequenceSlotPool::device_block_table(SequenceSlotId slot) const {
    uint32_t* base_ptr = block_table_memory_.data<uint32_t>();
    uint32_t offset = slot * max_blocks_;

    std::size_t shape[1] = {static_cast<std::size_t>(max_blocks_)};
    std::size_t strides[1] = {1};

    gpu::DeviceAllocationView view = gpu::GpuArena::make_view(
        reinterpret_cast<std::byte*>(base_ptr + offset),
        max_blocks_ * sizeof(uint32_t),
        device_);

    auto result = gpu::Tensor::create<uint32_t>(view, shape, 1, strides);

    assert(result.ok() && "device_block_table slice must not fail");
    (void)result;

    return result.value();
}

Status SequenceSlotPool::set_block(
    SequenceSlotId slot,
    uint32_t logical_block,
    PageId page,
    hipStream_t stream)
{
    if (slot >= max_sequences_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }

    if (!in_use_[slot]) {
        return Status::invalid_state("slot not allocated", __FILE__, __LINE__);
    }

    if (logical_block >= max_blocks_) {
        return Status::out_of_range("logical_block out of range", __FILE__, __LINE__);
    }

    uint32_t* block_table = block_table_memory_.data<uint32_t>();
    uint32_t offset = slot * max_blocks_ + logical_block;

    hipError_t err = hipMemcpyAsync(
        block_table + offset,
        &page,
        sizeof(uint32_t),
        hipMemcpyHostToDevice,
        stream);

    if (err != hipSuccess) {
        return Status::hip_error("hipMemcpyAsync", "set_block", __FILE__, __LINE__);
    }

    return Status::make_ok();
}

Status SequenceSlotPool::clear_blocks(
    SequenceSlotId slot,
    uint32_t first_block,
    uint32_t block_count,
    hipStream_t stream)
{
    if (slot >= max_sequences_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }

    if (!in_use_[slot]) {
        return Status::invalid_state("slot not allocated", __FILE__, __LINE__);
    }

    if (first_block >= max_blocks_) {
        return Status::out_of_range("first_block out of range", __FILE__, __LINE__);
    }

    if (block_count > max_blocks_ - first_block) {
        return Status::out_of_range("block_count out of range", __FILE__, __LINE__);
    }

    if (block_count == 0) {
        return Status::make_ok();
    }

    uint32_t* block_table = block_table_memory_.data<uint32_t>();
    uint32_t offset = static_cast<uint32_t>(static_cast<std::size_t>(slot) * max_blocks_ + first_block);

    hipError_t err = hipMemsetAsync(
        block_table + offset,
        0xFF,
        block_count * sizeof(uint32_t),
        stream);

    if (err != hipSuccess) {
        return Status::hip_error("hipMemsetAsync", "clear_blocks", __FILE__, __LINE__);
    }

    return Status::make_ok();
}

Status SequenceSlotPool::clear_slot(
    SequenceSlotId slot,
    hipStream_t stream)
{
    return clear_blocks(slot, 0, max_blocks_, stream);
}

}
}
