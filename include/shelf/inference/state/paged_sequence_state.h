#pragma once

#include <shelf/inference/state/paged_types.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/runtime/request_handle.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace shelf {
namespace state {

struct PagedSequenceState {
    PagedSequenceState() noexcept
        : slot(kInvalidSlot),
          generation(0),
          position(0),
          max_seq_len(0),
          poisoned_(false) {}

    PagedSequenceState(const PagedSequenceState&) = delete;
    PagedSequenceState& operator=(const PagedSequenceState&) = delete;
    PagedSequenceState& operator=(PagedSequenceState&&) = delete;

    PagedSequenceState(PagedSequenceState&& other) noexcept
        : slot(std::exchange(other.slot, kInvalidSlot)),
          generation(std::exchange(other.generation, 0)),
          position(std::exchange(other.position, 0)),
          max_seq_len(std::exchange(other.max_seq_len, 0)),
          block_table(std::move(other.block_table)),
          kv_pool_(std::exchange(other.kv_pool_, nullptr)),
          seq_pool_(std::exchange(other.seq_pool_, nullptr)),
          recurrent_pool_(std::exchange(other.recurrent_pool_, nullptr)),
          max_blocks_(std::exchange(other.max_blocks_, 0)),
          poisoned_(std::exchange(other.poisoned_, false)) {}

    SequenceSlotId slot;
    uint32_t generation;
    uint32_t position;
    uint32_t max_seq_len;

    std::vector<PageId> block_table;

    ::shelf::runtime::RequestHandle request_handle() const noexcept {
        return ::shelf::runtime::RequestHandle{
            .slot = slot,
            .generation = generation,
        };
    }

    bool is_allocated() const noexcept { return slot != kInvalidSlot; }
    bool is_usable() const noexcept {
        return is_allocated() && generation != 0u && !poisoned_;
    }
    bool is_poisoned() const noexcept { return poisoned_; }

    void mark_poisoned() noexcept {
        poisoned_ = true;
    }

    PagedKVPool* kv_pool() const noexcept { return kv_pool_; }

 private:
    PagedKVPool* kv_pool_ = nullptr;
    SequenceSlotPool* seq_pool_ = nullptr;
    GdnStatePool* recurrent_pool_ = nullptr;
    uint32_t max_blocks_ = 0;
    bool poisoned_ = false;

    friend Result<PagedSequenceState> create_paged_sequence_state(
        SequenceSlotPool& seq_pool,
        GdnStatePool& recurrent_pool,
        PagedKVPool& kv_pool,
        uint32_t max_seq_len,
        hipStream_t stream);

    friend Status release_paged_sequence_state(PagedSequenceState& seq, hipStream_t stream);

    friend Status reserve_sequence_append(
        PagedSequenceState& seq,
        uint32_t append_tokens,
        hipStream_t stream);

    friend Status rollback_sequence_append(
        PagedSequenceState& seq,
        uint32_t old_block_count,
        hipStream_t stream);

    friend Status import_sequence_block_table(
        PagedSequenceState& dst,
        const PagedSequenceState& src,
        hipStream_t stream);

    friend Status commit_sequence_append(
        PagedSequenceState& seq,
        ::shelf::runtime::RequestHandle expected_handle,
        uint32_t committed_tokens);

    friend Status validate_sequence_append_commit(
        const PagedSequenceState& seq,
        ::shelf::runtime::RequestHandle expected_handle,
        uint32_t committed_tokens);

    friend void runtime_install_sequence_state(
        PagedSequenceState& dst,
        PagedSequenceState&& src);
};

Result<PagedSequenceState> create_paged_sequence_state(
    SequenceSlotPool& seq_pool,
    GdnStatePool& recurrent_pool,
    PagedKVPool& kv_pool,
    uint32_t max_seq_len,
    hipStream_t stream);

Status release_paged_sequence_state(PagedSequenceState& seq, hipStream_t stream);

Status reserve_sequence_append(
    PagedSequenceState& seq,
    uint32_t append_tokens,
    hipStream_t stream);

Status rollback_sequence_append(
    PagedSequenceState& seq,
    uint32_t old_block_count,
    hipStream_t stream);

Status import_sequence_block_table(
    PagedSequenceState& dst,
    const PagedSequenceState& src,
    hipStream_t stream);

Status commit_sequence_append(
    PagedSequenceState& seq,
    ::shelf::runtime::RequestHandle expected_handle,
    uint32_t committed_tokens);

Status validate_sequence_append_commit(
    const PagedSequenceState& seq,
    ::shelf::runtime::RequestHandle expected_handle,
    uint32_t committed_tokens);

void runtime_install_sequence_state(
    PagedSequenceState& dst,
    PagedSequenceState&& src);

}

}
