#include <shelf/inference/state/ple_conv_state_pool.h>

#include <shelf/core/memory/alignment.h>

namespace shelf {
namespace state {

namespace {
std::size_t slot_elems(const PleConvStatePoolLayout& l) {
    return static_cast<std::size_t>(l.num_ple_states) * l.history * l.channels;
}
}  // namespace

std::size_t ple_conv_state_pool_bytes(uint32_t max_sequences,
                                      const PleConvStatePoolLayout& layout) noexcept {
    return ::shelf::gpu::align_up(static_cast<std::size_t>(max_sequences) *
                                      slot_elems(layout) * sizeof(bf16_t),
                                  kPleConvStatePoolAlignment);
}

Result<PleConvStatePool> PleConvStatePool::create(
    gpu::GpuArena& arena, uint32_t max_sequences,
    const PleConvStatePoolLayout& layout) {
    if (max_sequences == 0u || layout.num_ple_states == 0u || layout.channels == 0u ||
        layout.history == 0u) {
        return Status::invalid_argument("PLE conv state layout must be fully specified",
                                        __FILE__, __LINE__);
    }
    PleConvStatePool pool;
    pool.device_ = arena.device();
    pool.max_slots_ = max_sequences;
    pool.layout_ = layout;
    const std::size_t bytes = ple_conv_state_pool_bytes(max_sequences, layout);
    auto alloc = arena.allocate_aligned(bytes, kPleConvStatePoolAlignment);
    if (!alloc.ok()) return alloc.status();
    std::size_t shape[4] = {
        static_cast<std::size_t>(max_sequences),
        layout.num_ple_states,
        layout.history,
        layout.channels,
    };
    std::size_t strides[4] = {
        slot_elems(layout),
        static_cast<std::size_t>(layout.history) * layout.channels,
        layout.channels,
        1,
    };
    auto tensor = gpu::Tensor::create<bf16_t>(alloc.value(), shape, 4, strides);
    if (!tensor.ok()) return tensor.status();
    pool.memory_ = tensor.value();
    pool.reserved_bytes_ = bytes;
    pool.initialized_.resize(max_sequences, false);
    return pool;
}

Status PleConvStatePool::initialize(SequenceSlotId slot, hipStream_t stream) {
    if (slot >= max_slots_) {
        return Status::out_of_range("PLE conv slot out of range", __FILE__, __LINE__);
    }
    if (initialized_[slot]) {
        return Status::invalid_state("PLE conv slot already initialized", __FILE__,
                                     __LINE__);
    }
    Status st = reset(slot, stream);
    if (!st.ok()) return st;
    initialized_[slot] = true;
    ++used_count_;
    return Status::make_ok();
}

Status PleConvStatePool::reset(SequenceSlotId slot, hipStream_t stream) {
    if (slot >= max_slots_) {
        return Status::out_of_range("PLE conv slot out of range", __FILE__, __LINE__);
    }
    bf16_t* base = memory_.data<bf16_t>() +
                   static_cast<std::size_t>(slot) * slot_elems(layout_);
    const hipError_t err =
        hipMemsetAsync(base, 0, slot_elems(layout_) * sizeof(bf16_t), stream);
    if (err != hipSuccess) {
        return Status::hip_error("PLE conv reset", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status PleConvStatePool::release(SequenceSlotId slot) {
    if (slot >= max_slots_) {
        return Status::out_of_range("PLE conv slot out of range", __FILE__, __LINE__);
    }
    if (!initialized_[slot]) {
        return Status::invalid_state("PLE conv slot not initialized", __FILE__, __LINE__);
    }
    initialized_[slot] = false;
    --used_count_;
    return Status::make_ok();
}

Status PleConvStatePool::copy_slot_from(SequenceSlotId dst_slot,
                                        const PleConvStatePool& src,
                                        SequenceSlotId src_slot,
                                        hipStream_t stream) {
    if (dst_slot >= max_slots_ || src_slot >= src.max_slots_) {
        return Status::out_of_range("PLE conv copy slot out of range", __FILE__, __LINE__);
    }
    if (layout_.num_ple_states != src.layout_.num_ple_states ||
        layout_.channels != src.layout_.channels || layout_.history != src.layout_.history) {
        return Status::invalid_argument("PLE conv copy layout mismatch", __FILE__, __LINE__);
    }
    const std::size_t elems = slot_elems(layout_);
    bf16_t* dst = memory_.data<bf16_t>() + static_cast<std::size_t>(dst_slot) * elems;
    const bf16_t* from =
        src.memory_.data<bf16_t>() + static_cast<std::size_t>(src_slot) * elems;
    const hipError_t err = hipMemcpyAsync(dst, from, elems * sizeof(bf16_t),
                                          hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("PLE conv copy", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

}  // namespace state
}  // namespace shelf
