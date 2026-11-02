#include <shelf/models/qwen35/runtime/gdn_spec_history.h>
#include <shelf/kernels/optimized/gdn/recurrence.h>

#include <vector>

namespace shelf::qwen35::runtime {

Result<GdnSpecHistory> create_gdn_spec_history(
    gpu::GpuArena& arena,
    const GdnStatePool& pool,
    uint32_t rows,
    bool with_recurrent) {
    if (rows == 0u) {
        return Status::invalid_argument("create_gdn_spec_history: rows is zero", __FILE__,
                                        __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    if (view.conv_slot_stride == 0u || view.recurrent_slot_stride == 0u) {
        return Status::invalid_state("create_gdn_spec_history: pool geometry is zero", __FILE__,
                                     __LINE__);
    }

    GdnSpecHistory history;
    history.rows = rows;
    history.conv_state_bytes = static_cast<std::size_t>(view.conv_slot_stride) * sizeof(bf16_t);
    history.recurrent_state_bytes =
        with_recurrent ? static_cast<std::size_t>(view.recurrent_slot_stride) * sizeof(fp16_t)
                       : 0u;

    {
        const std::size_t count = static_cast<std::size_t>(rows) * view.conv_slot_stride;
        auto conv_view = arena.allocate_aligned(count * sizeof(bf16_t), 256u);
        if (!conv_view.ok()) return conv_view.status();
        auto backing = conv_view.release();
        auto tensor = gpu::Tensor::create<bf16_t>(
            backing, {rows, view.conv_slot_stride}, {view.conv_slot_stride, 1u});
        if (!tensor.ok()) return tensor.status();
        history.conv = tensor.release();
    }
    if (with_recurrent) {
        const std::size_t count = static_cast<std::size_t>(rows) * view.recurrent_slot_stride;
        auto rec_view = arena.allocate_aligned(count * sizeof(fp16_t), 256u);
        if (!rec_view.ok()) return rec_view.status();
        auto backing = rec_view.release();
        auto tensor = gpu::Tensor::create<fp16_t>(
            backing, {rows, view.recurrent_slot_stride},
            {view.recurrent_slot_stride, 1u});
        if (!tensor.ok()) return tensor.status();
        history.recurrent = tensor.release();
    }

    history.initialized = true;
    return history;
}

GdnSpecHistoryDeviceView gdn_spec_history_view(
    const GdnSpecHistory& history,
    uint32_t capture_rows) noexcept {
    GdnSpecHistoryDeviceView view;
    if (!history.initialized || capture_rows == 0u || capture_rows > history.rows) {
        return view;
    }
    view.conv_history = history.conv.data<bf16_t>();
    view.recurrent_history = history.recurrent.data<fp16_t>();
    view.capture_rows = capture_rows;
    view.conv_row_stride = static_cast<uint64_t>(history.conv.dim(1));
    view.recurrent_row_stride =
        history.recurrent_state_bytes != 0u
            ? static_cast<uint64_t>(history.recurrent.dim(1))
            : 0u;
    return view;
}

Status shutdown_gdn_spec_history(GdnSpecHistory& history) noexcept {
    history.conv = gpu::Tensor{};
    history.recurrent = gpu::Tensor{};
    history.rows = 0u;
    history.conv_state_bytes = 0u;
    history.recurrent_state_bytes = 0u;
    history.initialized = false;
    return Status::make_ok();
}

Status restore_gdn_spec_history(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnSpecHistory& history,
    uint32_t history_row,
    hipStream_t stream) {
    if (!history.initialized) {
        return Status::invalid_state("restore_gdn_spec_history: not initialized", __FILE__,
                                     __LINE__);
    }
    if (history_row >= history.rows) {
        return Status::out_of_range("restore_gdn_spec_history: history row out of range",
                                    __FILE__, __LINE__);
    }
    if (slot >= pool.max_sequences()) {
        return Status::out_of_range("restore_gdn_spec_history: slot out of range", __FILE__,
                                    __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    bf16_t* conv_dst = view.conv_base + static_cast<std::size_t>(slot) * view.conv_slot_stride;
    fp16_t* rec_dst =
        view.recurrent_base + static_cast<std::size_t>(slot) * view.recurrent_slot_stride;
    const bf16_t* conv_src =
        history.conv.data<bf16_t>() +
        static_cast<std::size_t>(history_row) * history.conv.dim(1);

    hipError_t err = hipMemcpyAsync(conv_dst, conv_src, history.conv_state_bytes,
                                    hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("restore_gdn_spec_history conv copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (history.recurrent_state_bytes == 0u) return Status::make_ok();
    const fp16_t* rec_src =
        history.recurrent.data<fp16_t>() +
        static_cast<std::size_t>(history_row) * history.recurrent.dim(1);
    err = hipMemcpyAsync(rec_dst, rec_src, history.recurrent_state_bytes,
                         hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("restore_gdn_spec_history recurrent copy",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status restore_gdn_spec_conv(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnSpecHistory& history,
    uint32_t history_row,
    hipStream_t stream) {
    if (!history.initialized) {
        return Status::invalid_state("restore_gdn_spec_conv: not initialized", __FILE__,
                                     __LINE__);
    }
    if (history_row >= history.rows) {
        return Status::out_of_range("restore_gdn_spec_conv: history row out of range", __FILE__,
                                    __LINE__);
    }
    if (slot >= pool.max_sequences()) {
        return Status::out_of_range("restore_gdn_spec_conv: slot out of range", __FILE__,
                                    __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    bf16_t* conv_dst = view.conv_base + static_cast<std::size_t>(slot) * view.conv_slot_stride;
    const bf16_t* conv_src =
        history.conv.data<bf16_t>() +
        static_cast<std::size_t>(history_row) * history.conv.dim(1);
    const hipError_t err = hipMemcpyAsync(conv_dst, conv_src, history.conv_state_bytes,
                                          hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("restore_gdn_spec_conv copy", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status commit_gdn_compact_log(
    const GdnStatePool& pool,
    SequenceSlotId slot,
    const GdnCompactLogDeviceView& log,
    const ::shelf::runtime::DeviceRequestDescriptor* requests,
    uint32_t rows,
    uint32_t key_heads,
    hipStream_t stream,
    const int32_t* row_indices,
    uint32_t log_row_capacity) {
    if (log.delta == nullptr || log.k == nullptr || log.a == nullptr) {
        return Status::invalid_argument("commit_gdn_compact_log: log is null", __FILE__,
                                        __LINE__);
    }
    if (rows == 0u || requests == nullptr) {
        return Status::invalid_argument("commit_gdn_compact_log: args are null", __FILE__,
                                        __LINE__);
    }
    if (row_indices != nullptr && log_row_capacity == 0u) {
        return Status::invalid_argument("commit_gdn_compact_log: row capacity is zero",
                                        __FILE__, __LINE__);
    }
    if (slot >= pool.max_sequences()) {
        return Status::out_of_range("commit_gdn_compact_log: slot out of range", __FILE__,
                                    __LINE__);
    }
    const GdnStatePoolDeviceView view = pool.device_view();
    if (view.recurrent_base == nullptr || view.num_gdn_states == 0u) {
        return Status::invalid_state("commit_gdn_compact_log: pool not initialized", __FILE__,
                                     __LINE__);
    }
    ::shelf::kernel::GdnRecurrenceArgs a{};
    a.recurrent_state = view.recurrent_base;
    a.requests = requests;
    a.num_requests = 1u;
    a.actual_rows = rows;
    a.max_sequences = view.max_sequences;
    a.num_gdn_states = view.num_gdn_states;
    a.key_heads = key_heads;
    a.num_v_heads = view.num_v_heads;
    a.head_k = view.head_k;
    a.head_v = view.head_v;
    a.recurrent_slot_stride = view.recurrent_slot_stride;
    a.recurrent_layer_stride = view.recurrent_layer_stride;
    a.compact_delta = log.delta;
    a.compact_k = log.k;
    a.compact_a = log.a;
    a.compact_delta_layer_stride = log.delta_layer_stride;
    a.compact_k_layer_stride = log.k_layer_stride;
    a.compact_a_layer_stride = log.a_layer_stride;
    a.compact_rows = rows;
    a.compact_row_indices = row_indices;
    a.compact_row_capacity = log_row_capacity;
    const hipError_t err = ::shelf::kernel::launch_gdn_recurrence_f32_wmma_commit(a, stream);
    if (err != hipSuccess) {
        return Status::hip_error("commit_gdn_compact_log launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace shelf::qwen35::runtime
