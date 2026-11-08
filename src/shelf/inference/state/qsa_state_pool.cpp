#include <shelf/inference/state/qsa_state_pool.h>

#include <shelf/core/memory/alignment.h>

#include <cassert>

namespace shelf {
namespace state {

namespace {

std::size_t pooled_slot_elems(const QsaKeyStatePoolLayout& layout) {
    return static_cast<std::size_t>(layout.num_qsa_states) * layout.max_blocks *
           layout.head_dim;
}

std::size_t partial_slot_elems(const QsaKeyStatePoolLayout& layout) {
    return static_cast<std::size_t>(layout.num_qsa_states) * layout.compress_ratio *
           layout.head_dim;
}

std::size_t count_slot_elems(const QsaKeyStatePoolLayout& layout) {
    return static_cast<std::size_t>(layout.num_qsa_states) * 2u;
}

}  // namespace

QsaKeyStatePoolBytes qsa_key_state_pool_bytes(uint32_t max_sequences,
                                              const QsaKeyStatePoolLayout& layout) noexcept {
    const std::size_t slots = static_cast<std::size_t>(max_sequences);
    return QsaKeyStatePoolBytes{
        .pooled = ::shelf::gpu::align_up(slots * pooled_slot_elems(layout) * sizeof(bf16_t),
                                         kQsaKeyStatePoolAlignment),
        .partial =
            ::shelf::gpu::align_up(slots * partial_slot_elems(layout) * sizeof(bf16_t),
                                   kQsaKeyStatePoolAlignment),
        .counts = ::shelf::gpu::align_up(slots * count_slot_elems(layout) * sizeof(uint32_t),
                                         kQsaKeyStatePoolAlignment),
    };
}

Status restore_qsa_spec_state(
    const QsaKeyStatePool& pool,
    SequenceSlotId slot,
    const ::shelf::execution::QsaSpecHistoryDeviceView& history,
    uint32_t row,
    hipStream_t stream) {
    if (history.partial == nullptr || history.counts == nullptr ||
        history.partial_state_stride == 0u || history.count_state_stride == 0u) {
        return Status::invalid_argument("restore_qsa_spec_state: history is empty",
                                        __FILE__, __LINE__);
    }
    if (row >= history.rows) {
        return Status::out_of_range("restore_qsa_spec_state: row out of range", __FILE__,
                                    __LINE__);
    }
    const QsaKeyStatePoolDeviceView view = pool.device_view();
    if (slot >= view.max_sequences) {
        return Status::out_of_range("restore_qsa_spec_state: slot out of range", __FILE__,
                                    __LINE__);
    }
    const std::size_t partial_elems =
        static_cast<std::size_t>(view.compress_ratio) * view.head_dim;
    for (uint32_t state = 0u; state < view.num_qsa_states; ++state) {
        const ::shelf::bf16_t* src_partial =
            history.partial +
            static_cast<std::uint64_t>(state) * history.partial_state_stride +
            static_cast<std::uint64_t>(row) * partial_elems;
        ::shelf::bf16_t* dst_partial =
            view.partial_base +
            static_cast<std::uint64_t>(slot) * view.partial_slot_stride +
            static_cast<std::uint64_t>(state) * view.partial_layer_stride;
        hipError_t err = hipMemcpyAsync(dst_partial, src_partial,
                                        partial_elems * sizeof(::shelf::bf16_t),
                                        hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("restore_qsa_spec_state partial",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        const std::uint32_t* src_counts =
            history.counts +
            static_cast<std::uint64_t>(state) * history.count_state_stride +
            static_cast<std::uint64_t>(row) * 2u;
        std::uint32_t* dst_pool =
            view.pool_count_base +
            static_cast<std::uint64_t>(slot) * view.count_slot_stride +
            static_cast<std::uint64_t>(state) * view.count_layer_stride;
        std::uint32_t* dst_partial_count =
            view.partial_count_base +
            static_cast<std::uint64_t>(slot) * view.count_slot_stride +
            static_cast<std::uint64_t>(state) * view.count_layer_stride;
        err = hipMemcpyAsync(dst_pool, src_counts, sizeof(std::uint32_t),
                             hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("restore_qsa_spec_state pool count",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        err = hipMemcpyAsync(dst_partial_count, src_counts + 1, sizeof(std::uint32_t),
                             hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("restore_qsa_spec_state partial count",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Result<QsaKeyStatePool> QsaKeyStatePool::create(
    gpu::GpuArena& arena,
    uint32_t max_sequences,
    const QsaKeyStatePoolLayout& layout) {
    if (max_sequences == 0u) {
        return Status::invalid_argument("max_sequences must be > 0", __FILE__, __LINE__);
    }
    if (layout.num_qsa_states == 0u || layout.head_dim == 0u ||
        layout.compress_ratio == 0u || layout.max_blocks == 0u) {
        return Status::invalid_argument("QsaKeyStatePoolLayout must be fully specified",
                                        __FILE__, __LINE__);
    }

    QsaKeyStatePool pool;
    pool.device_ = arena.device();
    pool.max_slots_ = max_sequences;
    pool.layout_ = layout;
    pool.used_count_ = 0u;

    const auto pool_bytes = qsa_key_state_pool_bytes(max_sequences, layout);
    const std::size_t pooled_bytes = pool_bytes.pooled;
    const std::size_t partial_bytes = pool_bytes.partial;
    const std::size_t count_bytes = pool_bytes.counts;

    auto pooled_alloc = arena.allocate_aligned(pooled_bytes, kQsaKeyStatePoolAlignment);
    if (!pooled_alloc.ok()) return pooled_alloc.status();
    auto partial_alloc = arena.allocate_aligned(partial_bytes, kQsaKeyStatePoolAlignment);
    if (!partial_alloc.ok()) return partial_alloc.status();
    auto count_alloc = arena.allocate_aligned(count_bytes, kQsaKeyStatePoolAlignment);
    if (!count_alloc.ok()) return count_alloc.status();

    std::size_t pooled_shape[4] = {
        static_cast<std::size_t>(max_sequences),
        layout.num_qsa_states,
        layout.max_blocks,
        layout.head_dim,
    };
    std::size_t pooled_strides[4] = {
        pooled_slot_elems(layout),
        static_cast<std::size_t>(layout.max_blocks) * layout.head_dim,
        layout.head_dim,
        1,
    };
    auto pooled = gpu::Tensor::create<bf16_t>(pooled_alloc.value(), pooled_shape, 4,
                                              pooled_strides);
    if (!pooled.ok()) return pooled.status();

    std::size_t partial_shape[4] = {
        static_cast<std::size_t>(max_sequences),
        layout.num_qsa_states,
        layout.compress_ratio,
        layout.head_dim,
    };
    std::size_t partial_strides[4] = {
        partial_slot_elems(layout),
        static_cast<std::size_t>(layout.compress_ratio) * layout.head_dim,
        layout.head_dim,
        1,
    };
    auto partial = gpu::Tensor::create<bf16_t>(partial_alloc.value(), partial_shape, 4,
                                               partial_strides);
    if (!partial.ok()) return partial.status();

    std::size_t count_shape[3] = {
        static_cast<std::size_t>(max_sequences),
        layout.num_qsa_states,
        2,
    };
    std::size_t count_strides[3] = {
        count_slot_elems(layout),
        2,
        1,
    };
    auto counts = gpu::Tensor::create<uint32_t>(count_alloc.value(), count_shape, 3,
                                                count_strides);
    if (!counts.ok()) return counts.status();

    pool.pooled_memory_ = pooled.value();
    pool.partial_memory_ = partial.value();
    pool.count_memory_ = counts.value();
    pool.reserved_bytes_ = pooled_bytes + partial_bytes + count_bytes;
    pool.initialized_.resize(max_sequences, false);
    return pool;
}

Status QsaKeyStatePool::initialize(SequenceSlotId slot, hipStream_t stream) {
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (initialized_[slot]) {
        return Status::invalid_state("slot already initialized", __FILE__, __LINE__);
    }
    Status st = reset(slot, stream);
    if (!st.ok()) return st;
    initialized_[slot] = true;
    ++used_count_;
    return Status::make_ok();
}

Status QsaKeyStatePool::reset(SequenceSlotId slot, hipStream_t stream) {
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    // reset is also the body of initialize, so it clears the slot directly.
    bf16_t* pooled =
        pooled_memory_.data<bf16_t>() + static_cast<std::size_t>(slot) *
                                            pooled_slot_elems(layout_);
    const std::size_t pooled_bytes = pooled_slot_elems(layout_) * sizeof(bf16_t);
    hipError_t err = hipMemsetAsync(pooled, 0, pooled_bytes, stream);
    if (err != hipSuccess) {
        return Status::hip_error("reset pooled keys", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }

    bf16_t* partial =
        partial_memory_.data<bf16_t>() + static_cast<std::size_t>(slot) *
                                             partial_slot_elems(layout_);
    const std::size_t partial_bytes = partial_slot_elems(layout_) * sizeof(bf16_t);
    err = hipMemsetAsync(partial, 0, partial_bytes, stream);
    if (err != hipSuccess) {
        return Status::hip_error("reset partial keys", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }

    uint32_t* counts =
        count_memory_.data<uint32_t>() + static_cast<std::size_t>(slot) *
                                             count_slot_elems(layout_);
    const std::size_t count_bytes = count_slot_elems(layout_) * sizeof(uint32_t);
    err = hipMemsetAsync(counts, 0, count_bytes, stream);
    if (err != hipSuccess) {
        return Status::hip_error("reset counts", hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status QsaKeyStatePool::release(SequenceSlotId slot) {
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }
    if (!initialized_[slot]) {
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }
    initialized_[slot] = false;
    --used_count_;
    return Status::make_ok();
}

Status QsaKeyStatePool::copy_slot_from(SequenceSlotId dst_slot,
                                       const QsaKeyStatePool& src,
                                       SequenceSlotId src_slot,
                                       hipStream_t stream) {
    if (dst_slot >= max_slots_) {
        return Status::out_of_range("copy_slot_from destination out of range", __FILE__,
                                    __LINE__);
    }
    if (src_slot >= src.max_slots_) {
        return Status::out_of_range("copy_slot_from source out of range", __FILE__,
                                    __LINE__);
    }
    const QsaKeyStatePoolLayout& a = layout_;
    const QsaKeyStatePoolLayout& b = src.layout_;
    if (a.num_qsa_states != b.num_qsa_states || a.head_dim != b.head_dim ||
        a.compress_ratio != b.compress_ratio || a.max_blocks != b.max_blocks) {
        return Status::invalid_argument("QSA slot copy layout mismatch", __FILE__,
                                        __LINE__);
    }

    const std::size_t pooled_elems = pooled_slot_elems(layout_);
    bf16_t* dst_pooled = pooled_memory_.data<bf16_t>() +
                         static_cast<std::size_t>(dst_slot) * pooled_elems;
    const bf16_t* src_pooled = src.pooled_memory_.data<bf16_t>() +
                               static_cast<std::size_t>(src_slot) * pooled_elems;
    hipError_t err = hipMemcpyAsync(dst_pooled, src_pooled,
                                    pooled_elems * sizeof(bf16_t),
                                    hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("copy_slot_from pooled", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    const std::size_t partial_elems = partial_slot_elems(layout_);
    bf16_t* dst_partial = partial_memory_.data<bf16_t>() +
                          static_cast<std::size_t>(dst_slot) * partial_elems;
    const bf16_t* src_partial = src.partial_memory_.data<bf16_t>() +
                                static_cast<std::size_t>(src_slot) * partial_elems;
    err = hipMemcpyAsync(dst_partial, src_partial,
                         partial_elems * sizeof(bf16_t), hipMemcpyDeviceToDevice,
                         stream);
    if (err != hipSuccess) {
        return Status::hip_error("copy_slot_from partial", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    const std::size_t count_elems = count_slot_elems(layout_);
    uint32_t* dst_counts = count_memory_.data<uint32_t>() +
                           static_cast<std::size_t>(dst_slot) * count_elems;
    const uint32_t* src_counts = src.count_memory_.data<uint32_t>() +
                                 static_cast<std::size_t>(src_slot) * count_elems;
    err = hipMemcpyAsync(dst_counts, src_counts, count_elems * sizeof(uint32_t),
                         hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("copy_slot_from counts", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

QsaKeyState QsaKeyStatePool::qsa_state(SequenceSlotId slot,
                                       uint32_t state_index) const {
    if (slot >= max_slots_ || state_index >= layout_.num_qsa_states) {
        return QsaKeyState{};
    }

    const std::size_t pooled_offset =
        static_cast<std::size_t>(slot) * pooled_slot_elems(layout_) +
        static_cast<std::size_t>(state_index) * layout_.max_blocks * layout_.head_dim;
    std::size_t pooled_shape[2] = {layout_.max_blocks, layout_.head_dim};
    std::size_t pooled_strides[2] = {layout_.head_dim, 1};
    gpu::DeviceAllocationView pooled_view = gpu::GpuArena::make_view(
        reinterpret_cast<std::byte*>(pooled_memory_.data<bf16_t>() + pooled_offset),
        static_cast<std::size_t>(layout_.max_blocks) * layout_.head_dim * sizeof(bf16_t),
        device_);
    auto pooled = gpu::Tensor::create<bf16_t>(pooled_view, pooled_shape, 2, pooled_strides);
    assert(pooled.ok());
    (void)pooled;

    const std::size_t partial_offset =
        static_cast<std::size_t>(slot) * partial_slot_elems(layout_) +
        static_cast<std::size_t>(state_index) * layout_.compress_ratio * layout_.head_dim;
    std::size_t partial_shape[2] = {layout_.compress_ratio, layout_.head_dim};
    std::size_t partial_strides[2] = {layout_.head_dim, 1};
    gpu::DeviceAllocationView partial_view = gpu::GpuArena::make_view(
        reinterpret_cast<std::byte*>(partial_memory_.data<bf16_t>() + partial_offset),
        static_cast<std::size_t>(layout_.compress_ratio) * layout_.head_dim *
            sizeof(bf16_t),
        device_);
    auto partial =
        gpu::Tensor::create<bf16_t>(partial_view, partial_shape, 2, partial_strides);
    assert(partial.ok());
    (void)partial;

    const std::size_t count_offset = static_cast<std::size_t>(slot) *
                                         count_slot_elems(layout_) +
                                     static_cast<std::size_t>(state_index) * 2u;
    std::size_t count_shape[1] = {1};
    std::size_t count_strides[1] = {1};
    gpu::DeviceAllocationView pool_count_view = gpu::GpuArena::make_view(
        reinterpret_cast<std::byte*>(count_memory_.data<uint32_t>() + count_offset),
        sizeof(uint32_t), device_);
    auto pool_count =
        gpu::Tensor::create<uint32_t>(pool_count_view, count_shape, 1, count_strides);
    assert(pool_count.ok());
    (void)pool_count;
    gpu::DeviceAllocationView partial_count_view = gpu::GpuArena::make_view(
        reinterpret_cast<std::byte*>(count_memory_.data<uint32_t>() + count_offset + 1u),
        sizeof(uint32_t), device_);
    auto partial_count = gpu::Tensor::create<uint32_t>(partial_count_view, count_shape, 1,
                                                       count_strides);
    assert(partial_count.ok());
    (void)partial_count;

    return QsaKeyState{
        .pooled = pooled.value(),
        .partial = partial.value(),
        .pool_count = pool_count.value(),
        .partial_count = partial_count.value(),
    };
}

}  // namespace state
}  // namespace shelf
