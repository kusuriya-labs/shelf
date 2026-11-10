#include <shelf/inference/state/gdn_state_pool.h>

namespace shelf {
namespace state {

Result<GdnStatePool> GdnStatePool::create(
    gpu::GpuArena& arena,
    uint32_t max_sequences,
    const GdnStatePoolLayout& layout)
{
    if (max_sequences == 0) {
        return Status::invalid_argument("max_sequences must be > 0", __FILE__, __LINE__);
    }
    if (layout.num_gdn_states != 0 && (layout.conv_dim == 0 ||
        layout.conv_dim_padded == 0 || layout.conv_history == 0 ||
        layout.conv_history_stride == 0 || layout.num_v_heads == 0 ||
        layout.head_k == 0 || layout.head_v == 0)) {
        return Status::invalid_argument(
            "GdnStatePoolLayout must be fully specified", __FILE__, __LINE__);
    }

    GdnStatePool pool;
    pool.device_ = arena.device();
    pool.max_slots_ = max_sequences;
    pool.layout_ = layout;
    pool.used_count_ = 0;
    pool.reserved_bytes_ = 0;
    pool.initialized_.resize(max_sequences, false);
    if (layout.num_gdn_states == 0) return pool;

    std::size_t conv_bytes =
        static_cast<std::size_t>(max_sequences)
        * layout.num_gdn_states
        * layout.conv_history
        * layout.conv_history_stride
        * sizeof(bf16_t);

    auto conv_alloc = arena.allocate_aligned(conv_bytes, 256);
    if (!conv_alloc.ok()) {
        return conv_alloc.status();
    }

    std::size_t rec_bytes =
        static_cast<std::size_t>(max_sequences)
        * layout.num_gdn_states
        * layout.num_v_heads
        * layout.head_k
        * layout.head_v
        * sizeof(fp16_t);

    auto rec_alloc = arena.allocate_aligned(rec_bytes, 256);
    if (!rec_alloc.ok()) {
        return rec_alloc.status();
    }

    std::size_t conv_shape[4] = {
        static_cast<std::size_t>(max_sequences),
        layout.num_gdn_states,
        layout.conv_history,
        layout.conv_history_stride,
    };
    std::size_t conv_strides[4] = {
        layout.num_gdn_states * layout.conv_history * layout.conv_history_stride,
        layout.conv_history * layout.conv_history_stride,
        layout.conv_history_stride,
        1,
    };

    auto conv_tensor = gpu::Tensor::create<bf16_t>(
        conv_alloc.value(),
        conv_shape,
        4,
        conv_strides);

    if (!conv_tensor.ok()) {
        return conv_tensor.status();
    }

    std::size_t rec_shape[5] = {
        static_cast<std::size_t>(max_sequences),
        layout.num_gdn_states,
        layout.num_v_heads,
        layout.head_k,
        layout.head_v,
    };
    std::size_t rec_strides[5] = {
        layout.num_gdn_states * layout.num_v_heads * layout.head_k * layout.head_v,
        layout.num_v_heads * layout.head_k * layout.head_v,
        layout.head_k * layout.head_v,
        layout.head_v,
        1,
    };

    auto rec_tensor = gpu::Tensor::create<fp16_t>(
        rec_alloc.value(),
        rec_shape,
        5,
        rec_strides);

    if (!rec_tensor.ok()) {
        return rec_tensor.status();
    }

    pool.conv_memory_ = conv_tensor.value();
    pool.recurrent_memory_ = rec_tensor.value();
    pool.reserved_bytes_ = conv_bytes + rec_bytes;

    return pool;
}

Status GdnStatePool::initialize(
    SequenceSlotId slot,
    hipStream_t stream)
{
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }

    if (initialized_[slot]) {
        return Status::invalid_state("slot already initialized", __FILE__, __LINE__);
    }

    if (layout_.num_gdn_states == 0) {
        initialized_[slot] = true;
        ++used_count_;
        return Status::make_ok();
    }

    bf16_t* conv_base = conv_memory_.data<bf16_t>();
    std::size_t slot_offset_conv =
        slot * layout_.num_gdn_states * layout_.conv_history * layout_.conv_history_stride;

    std::size_t conv_bytes =
        layout_.num_gdn_states
        * layout_.conv_history
        * layout_.conv_history_stride
        * sizeof(bf16_t);

    hipError_t err = hipMemsetAsync(
        conv_base + slot_offset_conv,
        0,
        conv_bytes,
        stream);

    if (err != hipSuccess) {
        return Status::hip_error("hipMemsetAsync", "initialize conv", __FILE__, __LINE__);
    }

    fp16_t* rec_base = recurrent_memory_.data<fp16_t>();
    std::size_t slot_offset_rec =
        slot * layout_.num_gdn_states * layout_.num_v_heads * layout_.head_k * layout_.head_v;

    std::size_t rec_bytes =
        layout_.num_gdn_states
        * layout_.num_v_heads
        * layout_.head_k
        * layout_.head_v
        * sizeof(fp16_t);

    err = hipMemsetAsync(
        rec_base + slot_offset_rec,
        0,
        rec_bytes,
        stream);

    if (err != hipSuccess) {
        return Status::hip_error("hipMemsetAsync", "initialize recurrent", __FILE__, __LINE__);
    }

    initialized_[slot] = true;
    ++used_count_;

    return Status::make_ok();
}

Status GdnStatePool::reset(
    SequenceSlotId slot,
    hipStream_t stream)
{
    if (slot >= max_slots_) {
        return Status::out_of_range("slot out of range", __FILE__, __LINE__);
    }

    if (!initialized_[slot]) {
        return Status::invalid_state("slot not initialized", __FILE__, __LINE__);
    }

    if (layout_.num_gdn_states == 0) return Status::make_ok();

    bf16_t* conv_base = conv_memory_.data<bf16_t>();
    std::size_t slot_offset_conv =
        slot * layout_.num_gdn_states * layout_.conv_history * layout_.conv_history_stride;

    std::size_t conv_bytes =
        layout_.num_gdn_states
        * layout_.conv_history
        * layout_.conv_history_stride
        * sizeof(bf16_t);

    hipError_t err = hipMemsetAsync(
        conv_base + slot_offset_conv,
        0,
        conv_bytes,
        stream);

    if (err != hipSuccess) {
        return Status::hip_error("hipMemsetAsync", "reset conv", __FILE__, __LINE__);
    }

    fp16_t* rec_base = recurrent_memory_.data<fp16_t>();
    std::size_t slot_offset_rec =
        slot * layout_.num_gdn_states * layout_.num_v_heads * layout_.head_k * layout_.head_v;

    std::size_t rec_bytes =
        layout_.num_gdn_states
        * layout_.num_v_heads
        * layout_.head_k
        * layout_.head_v
        * sizeof(fp16_t);

    err = hipMemsetAsync(
        rec_base + slot_offset_rec,
        0,
        rec_bytes,
        stream);

    if (err != hipSuccess) {
        return Status::hip_error("hipMemsetAsync", "reset recurrent", __FILE__, __LINE__);
    }

    return Status::make_ok();
}

Status GdnStatePool::release(SequenceSlotId slot) {
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

GDNState GdnStatePool::gdn_state(
    SequenceSlotId slot,
    uint32_t state_index) const
{
    if (state_index >= layout_.num_gdn_states) {
        return GDNState{};
    }

    std::size_t slot_conv_offset =
        slot * layout_.num_gdn_states * layout_.conv_history * layout_.conv_history_stride;
    std::size_t state_conv_offset =
        state_index * layout_.conv_history * layout_.conv_history_stride;

    std::size_t conv_shape[2] = {layout_.conv_history, layout_.conv_history_stride};
    std::size_t conv_strides[2] = {layout_.conv_history_stride, 1};

    gpu::DeviceAllocationView conv_view = gpu::GpuArena::make_view(
        reinterpret_cast<std::byte*>(
            conv_memory_.data<bf16_t>() + slot_conv_offset + state_conv_offset),
        layout_.conv_history * layout_.conv_history_stride * sizeof(bf16_t),
        device_);

    auto conv_result = gpu::Tensor::create<bf16_t>(conv_view, conv_shape, 2, conv_strides);

    std::size_t slot_rec_offset =
        slot * layout_.num_gdn_states * layout_.num_v_heads * layout_.head_k * layout_.head_v;
    std::size_t state_rec_offset =
        state_index * layout_.num_v_heads * layout_.head_k * layout_.head_v;

    std::size_t rec_shape[3] = {layout_.num_v_heads, layout_.head_k, layout_.head_v};
    std::size_t rec_strides[3] = {
        layout_.head_k * layout_.head_v,
        layout_.head_v,
        1,
    };

    gpu::DeviceAllocationView rec_view = gpu::GpuArena::make_view(
        reinterpret_cast<std::byte*>(
            recurrent_memory_.data<fp16_t>() + slot_rec_offset + state_rec_offset),
        layout_.num_v_heads * layout_.head_k * layout_.head_v * sizeof(fp16_t),
        device_);

    auto rec_result = gpu::Tensor::create<fp16_t>(rec_view, rec_shape, 3, rec_strides);

    assert(conv_result.ok() && rec_result.ok());
    (void)conv_result;
    (void)rec_result;

    return GDNState{
        .conv_state = conv_result.value(),
        .recurrent_state = rec_result.value(),
    };
}

Status GdnStatePool::copy_slot_from(
    SequenceSlotId dst_slot,
    const GdnStatePool& src,
    SequenceSlotId src_slot,
    hipStream_t stream)
{
    if (dst_slot >= max_slots_) {
        return Status::out_of_range("copy_slot_from destination out of range", __FILE__, __LINE__);
    }
    if (src_slot >= src.max_slots_) {
        return Status::out_of_range("copy_slot_from source out of range", __FILE__, __LINE__);
    }
    const GdnStatePoolLayout& a = layout_;
    const GdnStatePoolLayout& b = src.layout_;
    if (a.num_gdn_states != b.num_gdn_states ||
        a.conv_dim != b.conv_dim ||
        a.conv_dim_padded != b.conv_dim_padded ||
        a.conv_history != b.conv_history ||
        a.conv_history_stride != b.conv_history_stride ||
        a.num_v_heads != b.num_v_heads ||
        a.head_k != b.head_k ||
        a.head_v != b.head_v) {
        return Status::invalid_argument("GDN slot copy layout mismatch", __FILE__, __LINE__);
    }

    if (layout_.num_gdn_states == 0) return Status::make_ok();

    const std::size_t conv_slot_elems =
        static_cast<std::size_t>(layout_.num_gdn_states)
        * layout_.conv_history * layout_.conv_history_stride;
    const std::size_t rec_slot_elems =
        static_cast<std::size_t>(layout_.num_gdn_states)
        * layout_.num_v_heads * layout_.head_k * layout_.head_v;

    bf16_t* dst_conv = conv_memory_.data<bf16_t>()
        + static_cast<std::size_t>(dst_slot) * conv_slot_elems;
    const bf16_t* src_conv = src.conv_memory_.data<bf16_t>()
        + static_cast<std::size_t>(src_slot) * conv_slot_elems;
    hipError_t err = hipMemcpyAsync(
        dst_conv, src_conv, conv_slot_elems * sizeof(bf16_t),
        hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error(
            "copy_slot_from conv", hipGetErrorString(err), __FILE__, __LINE__);
    }

    fp16_t* dst_rec = recurrent_memory_.data<fp16_t>()
        + static_cast<std::size_t>(dst_slot) * rec_slot_elems;
    const fp16_t* src_rec = src.recurrent_memory_.data<fp16_t>()
        + static_cast<std::size_t>(src_slot) * rec_slot_elems;
    err = hipMemcpyAsync(
        dst_rec, src_rec, rec_slot_elems * sizeof(fp16_t),
        hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error(
            "copy_slot_from recurrent", hipGetErrorString(err), __FILE__, __LINE__);
    }

    return Status::make_ok();
}

}
}
