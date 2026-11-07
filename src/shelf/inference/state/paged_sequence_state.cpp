#include <shelf/inference/state/paged_sequence_state.h>
#include <hip/hip_runtime.h>

namespace shelf {
namespace state {

Result<PagedSequenceState> create_paged_sequence_state(
    SequenceSlotPool& seq_pool,
    GdnStatePool& recurrent_pool,
    PagedKVPool& kv_pool,
    uint32_t max_seq_len,
    hipStream_t stream) {

    uint32_t page_tokens = kv_pool.page_tokens();
    if (page_tokens == 0) {
        return Status::invalid_argument("page_tokens must be > 0", __FILE__, __LINE__);
    }

    uint32_t required_blocks = (max_seq_len + page_tokens - 1) / page_tokens;

    uint32_t max_blocks_per_seq = seq_pool.max_blocks_per_sequence();
    if (required_blocks > max_blocks_per_seq) {
        return Status::invalid_argument(
            "required_blocks exceeds max_blocks_per_sequence", __FILE__, __LINE__);
    }

    auto handle_result = seq_pool.allocate_handle();
    if (!handle_result.ok()) return handle_result.status();

    const ::shelf::runtime::RequestHandle handle = handle_result.value();

    auto clear_st = seq_pool.clear_slot(handle.slot, stream);
    if (!clear_st.ok()) {
        (void)seq_pool.release_handle(handle);
        return clear_st;
    }

    PagedSequenceState state;
    state.slot = handle.slot;
    state.generation = handle.generation;
    state.position = 0;
    state.max_seq_len = max_seq_len;
    state.max_blocks_ = max_blocks_per_seq;
    state.kv_pool_ = &kv_pool;
    state.seq_pool_ = &seq_pool;
    state.recurrent_pool_ = &recurrent_pool;
    state.poisoned_ = false;

    state.block_table.reserve(max_blocks_per_seq);

    auto init_result = recurrent_pool.initialize(state.slot, stream);
    if (!init_result.ok()) {
        (void)seq_pool.release_handle(handle);
        return init_result;
    }

    return state;
}

Status release_paged_sequence_state(PagedSequenceState& seq, hipStream_t stream) {
    if (!seq.is_allocated()) {
        return Status::invalid_argument("sequence state is not allocated", __FILE__, __LINE__);
    }

    if (seq.slot == kInvalidSlot) {
        return Status::invalid_argument("slot is invalid", __FILE__, __LINE__);
    }

    if (seq.kv_pool_ == nullptr || seq.seq_pool_ == nullptr || seq.recurrent_pool_ == nullptr) {
        return Status::invalid_state("sequence pool ownership is invalid", __FILE__, __LINE__);
    }

    for (PageId page : seq.block_table) {
        if (page == kInvalidPageId) {
            return Status::invalid_argument("invalid page id", __FILE__, __LINE__);
        }
        if (page >= seq.kv_pool_->num_pages()) {
            return Status::invalid_argument("page id out of range", __FILE__, __LINE__);
        }
    }

    hipError_t sync_error = hipStreamSynchronize(stream);
    if (sync_error != hipSuccess) {
        return Status::hip_error("hipStreamSynchronize(stream)", hipGetErrorString(sync_error), __FILE__, __LINE__);
    }

    const ::shelf::runtime::RequestHandle handle = seq.request_handle();

    auto release_pages = seq.kv_pool_->release_many(seq.block_table);
    if (!release_pages.ok()) return release_pages;

    auto release_recurrent = seq.recurrent_pool_->release(seq.slot);
    if (!release_recurrent.ok()) return release_recurrent;

    auto release_slot = seq.seq_pool_->release_handle(handle);
    if (!release_slot.ok()) return release_slot;

    seq.slot = kInvalidSlot;
    seq.generation = 0u;
    seq.position = 0;
    seq.max_seq_len = 0;
    seq.block_table.clear();
    seq.kv_pool_ = nullptr;
    seq.seq_pool_ = nullptr;
    seq.recurrent_pool_ = nullptr;
    seq.max_blocks_ = 0;

    return Status::make_ok();
}

Status reserve_sequence_append(
    PagedSequenceState& seq,
    uint32_t token_count,
    hipStream_t stream) {
    if (!seq.is_usable()) {
        return Status::invalid_argument("sequence state is not usable", __FILE__, __LINE__);
    }

    if (token_count == 0) {
        return Status::make_ok();
    }

    if (token_count > seq.max_seq_len - seq.position) {
        return Status::out_of_range("token_count exceeds max_seq_len", __FILE__, __LINE__);
    }

    const uint32_t page_tokens = seq.kv_pool_->page_tokens();
    if (page_tokens == 0) {
        return Status::invalid_state("page_tokens must be > 0", __FILE__, __LINE__);
    }

    const uint32_t end_position = seq.position + token_count;
    const uint32_t required_blocks =
        (end_position + page_tokens - 1) / page_tokens;

    const std::size_t current_blocks = seq.block_table.size();
    if (required_blocks < current_blocks) {
        return Status::make_ok();
    }

    const uint32_t new_blocks = required_blocks - static_cast<uint32_t>(current_blocks);
    if (new_blocks == 0) {
        return Status::make_ok();
    }

    if (required_blocks > seq.max_blocks_) {
        return Status::out_of_range(
            "required_blocks exceeds max_blocks_per_sequence", __FILE__, __LINE__);
    }

    if (new_blocks > seq.kv_pool_->num_free_pages()) {
        return Status::insufficient_memory(
            "not enough free pages for chunk", __FILE__, __LINE__);
    }

    const std::size_t old_size = seq.block_table.size();
    for (uint32_t i = 0; i < new_blocks; ++i) {
        auto page_result = seq.kv_pool_->allocate();
        if (!page_result.ok()) {
            for (std::size_t j = old_size; j < seq.block_table.size(); ++j) {
                seq.kv_pool_->release(seq.block_table[j]);
            }
            seq.block_table.resize(old_size);
            return page_result.status();
        }
        seq.block_table.push_back(page_result.release());
    }
    auto device_table = seq.seq_pool_->device_block_table(seq.slot);
    hipError_t err = hipMemcpyAsync(
        device_table.data<uint32_t>() + old_size,
        seq.block_table.data() + old_size,
        new_blocks * sizeof(uint32_t),
        hipMemcpyHostToDevice,
        stream);
    if (err != hipSuccess) {
        for (std::size_t j = old_size; j < seq.block_table.size(); ++j) {
            seq.kv_pool_->release(seq.block_table[j]);
        }
        seq.block_table.resize(old_size);
        return Status::hip_error(
            "reserve_sequence_append hipMemcpyAsync",
            hipGetErrorString(err),
            __FILE__, __LINE__);
    }

    return Status::make_ok();
}

Status rollback_sequence_append(
    PagedSequenceState& seq,
    uint32_t old_block_count,
    hipStream_t stream) {
    if (!seq.is_allocated()) {
        return Status::invalid_argument("sequence state is not allocated", __FILE__, __LINE__);
    }

    if (seq.seq_pool_ == nullptr || seq.kv_pool_ == nullptr) {
        return Status::invalid_state("sequence pool ownership is invalid", __FILE__, __LINE__);
    }

    if (old_block_count > seq.block_table.size()) {
        return Status::out_of_range("old_block_count exceeds block_table", __FILE__, __LINE__);
    }

    if (old_block_count == seq.block_table.size()) {
        return Status::make_ok();
    }

    const uint32_t new_block_count = static_cast<uint32_t>(seq.block_table.size());

    Status first_error = seq.seq_pool_->clear_blocks(
        seq.slot, old_block_count, new_block_count - old_block_count, stream);

    for (std::size_t i = old_block_count; i < seq.block_table.size(); ++i) {
        auto st = seq.kv_pool_->release(seq.block_table[i]);
        if (!st.ok() && first_error.ok()) {
            first_error = st;
        }
    }

    seq.block_table.resize(old_block_count);

    if (!first_error.ok()) {
        seq.mark_poisoned();
        return first_error;
    }

    return Status::make_ok();
}

Status import_sequence_block_table(
    PagedSequenceState& dst,
    const PagedSequenceState& src,
    hipStream_t stream) {
    if (!dst.is_allocated() || !src.is_allocated()) {
        return Status::invalid_argument("sequence state is not allocated", __FILE__, __LINE__);
    }
    if (dst.seq_pool_ == nullptr || dst.kv_pool_ == nullptr) {
        return Status::invalid_state("sequence pool ownership is invalid", __FILE__, __LINE__);
    }
    if (dst.block_table.size() > src.block_table.size()) {
        Status st = rollback_sequence_append(
            dst, static_cast<uint32_t>(src.block_table.size()), stream);
        if (!st.ok()) return st;
    }
    const std::size_t old_size = dst.block_table.size();
    for (std::size_t i = 0; i < old_size; ++i) {
        if (dst.block_table[i] != src.block_table[i]) {
            return Status::invalid_state(
                "pipeline mirror block table diverged from stage0", __FILE__, __LINE__);
        }
    }
    if (old_size == src.block_table.size()) {
        return Status::make_ok();
    }

    for (std::size_t i = old_size; i < src.block_table.size(); ++i) {
        Status st = dst.kv_pool_->adopt(src.block_table[i]);
        if (!st.ok()) {
            for (std::size_t j = old_size; j < dst.block_table.size(); ++j) {
                (void)dst.kv_pool_->release(dst.block_table[j]);
            }
            dst.block_table.resize(old_size);
            return st;
        }
        dst.block_table.push_back(src.block_table[i]);
    }

    auto device_table = dst.seq_pool_->device_block_table(dst.slot);
    hipError_t err = hipMemcpyAsync(
        device_table.data<uint32_t>() + old_size,
        dst.block_table.data() + old_size,
        (dst.block_table.size() - old_size) * sizeof(uint32_t),
        hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        for (std::size_t j = old_size; j < dst.block_table.size(); ++j) {
            (void)dst.kv_pool_->release(dst.block_table[j]);
        }
        dst.block_table.resize(old_size);
        return Status::hip_error(
            "import_sequence_block_table hipMemcpyAsync", hipGetErrorString(err),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status validate_sequence_append_commit(
    const PagedSequenceState& seq,
    ::shelf::runtime::RequestHandle expected_handle,
    uint32_t committed_tokens) {

    if (!seq.is_usable()) {
        return Status::invalid_state("sequence state is not usable", __FILE__, __LINE__);
    }

    if (!::shelf::runtime::request_handle_equal(seq.request_handle(), expected_handle)) {
        return Status::invalid_state(
            "stale request handle during sequence commit", __FILE__, __LINE__);
    }

    if (committed_tokens == 0u) {
        return Status::invalid_argument(
            "committed_tokens must be > 0", __FILE__, __LINE__);
    }

    if (committed_tokens > seq.max_seq_len - seq.position) {
        return Status::out_of_range(
            "sequence commit exceeds max_seq_len", __FILE__, __LINE__);
    }

    return Status::make_ok();
}

Status PagedSequenceState::append_pages(
    std::span<const PageId> pages,
    std::span<const uint8_t> owned,
    hipStream_t stream) {
    if (!is_usable()) {
        return Status::invalid_argument("sequence state is not usable", __FILE__, __LINE__);
    }
    if (seq_pool_ == nullptr || kv_pool_ == nullptr) {
        return Status::invalid_state("sequence pool ownership is invalid", __FILE__,
                                     __LINE__);
    }
    if (pages.size() != owned.size()) {
        return Status::invalid_argument("append_pages size mismatch", __FILE__, __LINE__);
    }
    if (pages.size() > max_blocks_ - block_table.size()) {
        return Status::out_of_range("append_pages exceeds max_blocks", __FILE__,
                                    __LINE__);
    }
    const std::size_t old_size = block_table.size();
    for (std::size_t i = 0; i < pages.size(); ++i) {
        const PageId page = pages[i];
        if (page == kInvalidPageId || page >= kv_pool_->num_pages()) {
            for (std::size_t j = old_size; j < block_table.size(); ++j) {
                (void)kv_pool_->release(block_table[j]);
            }
            block_table.resize(old_size);
            return Status::invalid_argument("append_pages page out of range", __FILE__,
                                            __LINE__);
        }
        if (owned[i] == 0u) {
            const Status attach_st = kv_pool_->attach(page);
            if (!attach_st.ok()) {
                for (std::size_t j = old_size; j < block_table.size(); ++j) {
                    (void)kv_pool_->release(block_table[j]);
                }
                block_table.resize(old_size);
                return attach_st;
            }
        }
        block_table.push_back(page);
    }
    auto device_table = seq_pool_->device_block_table(slot);
    const hipError_t err = hipMemcpyAsync(
        device_table.data<uint32_t>() + old_size,
        block_table.data() + old_size,
        pages.size() * sizeof(uint32_t),
        hipMemcpyHostToDevice,
        stream);
    if (err != hipSuccess) {
        for (std::size_t j = old_size; j < block_table.size(); ++j) {
            (void)kv_pool_->release(block_table[j]);
        }
        block_table.resize(old_size);
        return Status::hip_error("append_pages hipMemcpyAsync", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status commit_sequence_append(
    PagedSequenceState& seq,
    ::shelf::runtime::RequestHandle expected_handle,
    uint32_t committed_tokens) {
    auto st = validate_sequence_append_commit(seq, expected_handle, committed_tokens);
    if (!st.ok()) {
        return st;
    }

    seq.position += committed_tokens;

    return Status::make_ok();
}

void runtime_install_sequence_state(
    PagedSequenceState& dst,
    PagedSequenceState&& src) {
    if (dst.is_allocated()) {
        return;
    }
    dst.slot = std::exchange(src.slot, kInvalidSlot);
    dst.generation = std::exchange(src.generation, 0u);
    dst.position = std::exchange(src.position, 0);
    dst.max_seq_len = std::exchange(src.max_seq_len, 0);
    dst.block_table = std::move(src.block_table);
    dst.kv_pool_ = std::exchange(src.kv_pool_, nullptr);
    dst.seq_pool_ = std::exchange(src.seq_pool_, nullptr);
    dst.recurrent_pool_ = std::exchange(src.recurrent_pool_, nullptr);
    dst.max_blocks_ = std::exchange(src.max_blocks_, 0);
    dst.poisoned_ = std::exchange(src.poisoned_, false);
}

}
}
