#include <shelf/inference/state/spec_state_history.h>

#include <shelf/kernels/fused/ple_conv.h>

#include <cstddef>

namespace shelf::state {
namespace {

template <typename T>
Result<T*> allocate_history(gpu::GpuArena& arena, std::size_t count) {
    if (count == 0u) return static_cast<T*>(nullptr);
    auto alloc = arena.allocate_aligned(count * sizeof(T), 256u);
    if (!alloc.ok()) return alloc.status();
    return static_cast<T*>(alloc.release().data());
}

}  // namespace

Result<SpecStateHistory> create_spec_state_history(gpu::GpuArena& arena,
                                                   const SpecStatePools& pools,
                                                   uint32_t rows) {
    SpecStateHistory history;
    if (rows == 0u) return history;
    if (pools.qsa != nullptr) {
        const QsaKeyStatePoolDeviceView view = pools.qsa->device_view();
        if (view.num_qsa_states != 0u && view.compress_ratio != 0u && view.head_dim != 0u) {
            history.qsa.rows = rows;
            history.qsa.partial_state_stride =
                static_cast<std::uint64_t>(rows) * view.compress_ratio * view.head_dim;
            history.qsa.count_state_stride = static_cast<std::uint64_t>(rows) * 2u;
            history.qsa.journal_state_stride =
                static_cast<std::uint64_t>(rows) * view.head_dim;
            auto partial = allocate_history<bf16_t>(
                arena, static_cast<std::size_t>(view.num_qsa_states) *
                           history.qsa.partial_state_stride);
            if (!partial.ok()) return partial.status();
            auto counts = allocate_history<std::uint32_t>(
                arena, static_cast<std::size_t>(view.num_qsa_states) *
                           history.qsa.count_state_stride);
            if (!counts.ok()) return counts.status();
            auto journal = allocate_history<bf16_t>(
                arena, static_cast<std::size_t>(view.num_qsa_states) *
                           history.qsa.journal_state_stride);
            if (!journal.ok()) return journal.status();
            history.qsa.partial = partial.release();
            history.qsa.counts = counts.release();
            history.qsa.key_journal = journal.release();
        }
    }
    if (pools.ple != nullptr) {
        const PleConvStatePoolDeviceView view = pools.ple->device_view();
        if (view.channels != 0u && view.history_stride != 0u) {
            const std::size_t elems =
                static_cast<std::size_t>(view.history_stride) * ::shelf::kernel::kPleConvHistory;
            history.ple.rows = rows;
            history.ple.state_stride = static_cast<std::uint64_t>(rows) * elems;
            auto store = allocate_history<bf16_t>(
                arena, static_cast<std::size_t>(pools.ple->num_ple_states()) *
                           history.ple.state_stride);
            if (!store.ok()) return store.status();
            history.ple.store = store.release();
        }
    }
    return history;
}

Status restore_spec_state(const SpecStatePools& pools, SequenceSlotId slot,
                          const SpecStateHistory& history, uint32_t row,
                          hipStream_t stream) {
    if (pools.qsa != nullptr && history.qsa.partial != nullptr &&
        history.qsa.counts != nullptr) {
        Status st = restore_qsa_spec_state(*pools.qsa, slot, history.qsa, row, stream);
        if (!st.ok()) return st;
    }
    if (pools.ple != nullptr && history.ple.store != nullptr) {
        Status st = restore_ple_conv_spec_state(*pools.ple, slot, history.ple, row, stream);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

}  // namespace shelf::state
