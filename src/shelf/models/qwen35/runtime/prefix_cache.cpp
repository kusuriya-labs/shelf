#include <shelf/models/qwen35/runtime/prefix_cache.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <tuple>

namespace shelf {
namespace qwen35 {
namespace runtime {

using ::shelf::inference::prefix::PrefixKVConfig;
using ::shelf::inference::prefix::PrefixKVHit;
using ::shelf::inference::prefix::PrefixSequence;
using ::shelf::inference::prefix::PrefixSequenceSet;
using ::shelf::kernel::fp8e4m3_storage_t;

class Qwen35Sequence final : public PrefixSequence {
 public:
    explicit Qwen35Sequence(PagedSequenceState& state) : state_(state) {}

    PagedKVPool& kv_pool() override { return *state_.kv_pool(); }
    uint32_t position() const override { return state_.position; }
    std::span<const PageId> block_table() const override {
        return std::span<const PageId>(state_.block_table.data(),
                                       state_.block_table.size());
    }
    Status append_pages(std::span<const PageId> pages,
                        std::span<const uint8_t> owned,
                        hipStream_t stream) override {
        return state_.append_pages(pages, owned, stream);
    }
    Status commit_prefix(uint32_t tokens) override {
        return commit_sequence_append(state_, state_.request_handle(), tokens);
    }
    Status rollback_pages(uint32_t page_count, hipStream_t stream) override {
        return rollback_sequence_append(state_, page_count, stream);
    }

 private:
    PagedSequenceState& state_;
};

class Qwen35SequenceSet final : public PrefixSequenceSet {
 public:
    Qwen35SequenceSet(const PagedSequenceState& source, TpBatchHook* hook) {
        const uint32_t count = hook != nullptr ? hook->on_prefix_rank_count() : 1u;
        ranks_.reserve(count);
        for (uint32_t r = 0; r < count; ++r) {
            PagedSequenceState* seq = r == 0u
                ? const_cast<PagedSequenceState*>(&source)
                : hook->on_prefix_rank_sequence(source, r);
            if (seq == nullptr) {
                ranks_.clear();
                wrappers_.clear();
                return;
            }
            ranks_.push_back(seq);
            wrappers_.push_back(std::make_unique<Qwen35Sequence>(*seq));
        }
    }

    uint32_t rank_count() const override {
        return static_cast<uint32_t>(ranks_.size());
    }

    PrefixSequence& rank(uint32_t index) override { return *wrappers_.at(index); }

    bool valid() const noexcept { return !ranks_.empty(); }

    SequenceSlotId state_slot(uint32_t index) const {
        return ranks_.at(index)->slot;
    }

 private:
    std::vector<PagedSequenceState*> ranks_;
    std::vector<std::unique_ptr<Qwen35Sequence>> wrappers_;
};

class Qwen35PrefixState {
 public:
    static Result<std::unique_ptr<Qwen35PrefixState>> create(
        gpu::GpuArena& arena,
        GdnStatePool& active_state,
        const std::vector<PrefixCacheRankContext>& extra_ranks,
        const PrefixCacheConfig& config) {
        auto state = std::unique_ptr<Qwen35PrefixState>(new Qwen35PrefixState());
        state->active_state_ = &active_state;
        state->max_concurrent_ = config.max_concurrent;
        auto shadow = GdnStatePool::create(arena, 2u * config.max_concurrent,
                                           active_state.layout());
        if (!shadow.ok()) return shadow.status();
        state->shadow_state_.emplace(shadow.release());
        for (const PrefixCacheRankContext& ctx : extra_ranks) {
            if (ctx.active_state == nullptr || ctx.arena == nullptr ||
                ctx.stream == nullptr) {
                return Status::invalid_argument(
                    "prefix cache rank context is incomplete", __FILE__, __LINE__);
            }
            auto scope = shelf::gpu::ScopedDevice::create(ctx.device);
            if (!scope.ok()) return scope.status();
            auto shard = GdnStatePool::create(
                *ctx.arena, 2u * config.max_concurrent, ctx.active_state->layout());
            if (!shard.ok()) return shard.status();
            auto rank = std::make_unique<Rank>();
            rank->active = ctx.active_state;
            rank->shadow.emplace(shard.release());
            rank->stream = ctx.stream;
            rank->device = ctx.device;
            state->ranks_.push_back(std::move(rank));
        }
        const std::size_t ring_bytes = config.spec_ring_bytes;
        if (ring_bytes > 0u) {
            auto ring = arena.allocate_aligned(
                ring_bytes * static_cast<std::size_t>(config.max_concurrent), 128);
            if (!ring.ok()) return ring.status();
            state->ring_base_ = static_cast<std::byte*>(ring.release().data());
            state->ring_slot_bytes_ = ring_bytes;
            state->ring_slots_ = config.max_concurrent;
            state->ring_length_.assign(config.max_concurrent, 0u);
            state->ring_next_.assign(config.max_concurrent, 0u);
        }
        for (uint32_t i = config.max_concurrent; i-- > 0;) {
            state->capture_free_.push_back(i);
            state->hot_free_.push_back(i);
            if (i < state->ring_slots_) state->ring_free_.push_back(i);
        }
        return state;
    }

    Status shutdown() noexcept { return Status::make_ok(); }

    Result<uint32_t> acquire_capture_slot() {
        if (capture_free_.empty()) {
            return Status::insufficient_memory("prefix capture slots exhausted",
                                               __FILE__, __LINE__);
        }
        const uint32_t slot = capture_free_.back();
        capture_free_.pop_back();
        return slot;
    }

    void release_capture_slot(uint32_t slot) {
        if (slot < max_concurrent_) capture_free_.push_back(slot);
    }

    Result<uint32_t> acquire_hot_slot() {
        if (hot_free_.empty()) {
            return Status::insufficient_memory("prefix hot state slots exhausted",
                                               __FILE__, __LINE__);
        }
        const uint32_t slot = hot_free_.back();
        hot_free_.pop_back();
        return slot;
    }

    void release_hot_slot(uint32_t slot) {
        if (slot < max_concurrent_) hot_free_.push_back(slot);
    }

    Result<uint32_t> acquire_ring_slot() {
        if (ring_free_.empty()) {
            return Status::insufficient_memory("prefix ring slots exhausted",
                                               __FILE__, __LINE__);
        }
        const uint32_t slot = ring_free_.back();
        ring_free_.pop_back();
        return slot;
    }

    void release_ring_slot(uint32_t slot) {
        if (slot < ring_slots_) {
            ring_length_[slot] = 0u;
            ring_next_[slot] = 0u;
            ring_free_.push_back(slot);
        }
    }

    Status capture_gdn(uint32_t capture_slot, const Qwen35SequenceSet& seqs,
                       hipStream_t stream) {
        return copy_from_active(capture_slot, seqs, stream);
    }

    Status capture_gdn_to_hot(uint32_t hot_slot, const Qwen35SequenceSet& seqs,
                              hipStream_t stream) {
        return copy_from_active(max_concurrent_ + hot_slot, seqs, stream);
    }

    Status copy_capture_to_hot(uint32_t hot_slot, uint32_t capture_slot,
                               hipStream_t stream) {
        return copy_shadow(max_concurrent_ + hot_slot, capture_slot, stream);
    }

    Status restore_hot(uint32_t hot_slot, const Qwen35SequenceSet& seqs,
                       hipStream_t stream) {
        return copy_to_active(max_concurrent_ + hot_slot, seqs, stream);
    }

    Status save_ring(uint32_t ring_slot, const SpecRingRef& ring,
                     uint32_t position, hipStream_t stream) {
        if (ring_slot >= ring_slots_ || ring_slot_bytes_ == 0u || !ring.valid()) {
            return Status::invalid_argument("prefix ring slot unavailable", __FILE__,
                                            __LINE__);
        }
        const uint32_t next = std::min(ring.next_position, position);
        const uint32_t length = std::min(ring.length, next);
        const std::size_t half = ring_slot_bytes_ / 2u;
        std::byte* base =
            ring_base_ + static_cast<std::size_t>(ring_slot) * ring_slot_bytes_;
        hipError_t err = hipMemcpyAsync(base, ring.k, half, hipMemcpyDeviceToDevice,
                                        stream);
        if (err != hipSuccess) {
            return Status::hip_error("prefix ring save K", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        err = hipMemcpyAsync(base + half, ring.v, half, hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("prefix ring save V", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        ring_length_[ring_slot] = length;
        ring_next_[ring_slot] = next;
        return Status::make_ok();
    }

    Status load_ring(uint32_t ring_slot, SpecRingOut* ring, hipStream_t stream) {
        if (ring == nullptr || !ring->valid() || ring_slot >= ring_slots_ ||
            ring_slot_bytes_ == 0u) {
            return Status::make_ok();
        }
        const std::size_t half = ring_slot_bytes_ / 2u;
        const std::byte* base =
            ring_base_ + static_cast<std::size_t>(ring_slot) * ring_slot_bytes_;
        hipError_t err = hipMemcpyAsync(ring->k, base, half,
                                        hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("prefix ring load K", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        err = hipMemcpyAsync(ring->v, base + half, half, hipMemcpyDeviceToDevice,
                             stream);
        if (err != hipSuccess) {
            return Status::hip_error("prefix ring load V", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        ring->length = ring_length_[ring_slot];
        ring->next_position = ring_next_[ring_slot];
        return Status::make_ok();
    }

    uint32_t ring_length(uint32_t ring_slot) const {
        return ring_slot < ring_length_.size() ? ring_length_[ring_slot] : 0u;
    }

    uint32_t ring_next(uint32_t ring_slot) const {
        return ring_slot < ring_next_.size() ? ring_next_[ring_slot] : 0u;
    }

    bool ring_covers(uint32_t ring_slot, uint32_t position) const {
        return ring_length(ring_slot) != 0u && ring_next(ring_slot) == position;
    }

    uint32_t capture_slots() const noexcept { return max_concurrent_; }
    uint32_t hot_slots() const noexcept { return max_concurrent_; }

 private:
    struct Rank {
        GdnStatePool* active = nullptr;
        std::optional<GdnStatePool> shadow;
        hipStream_t stream = nullptr;
        int device = 0;
    };

    Qwen35PrefixState() = default;

    Status copy_from_active(uint32_t shadow_slot, const Qwen35SequenceSet& seqs,
                            hipStream_t stream) {
        Status st = shadow_state_->copy_slot_from(shadow_slot, *active_state_,
                                                  seqs.state_slot(0u), stream);
        if (!st.ok()) return st;
        for (uint32_t r = 0; r < ranks_.size(); ++r) {
            Rank& rank = *ranks_[r];
            auto scope = shelf::gpu::ScopedDevice::create(rank.device);
            if (!scope.ok()) return scope.status();
            st = rank.shadow->copy_slot_from(shadow_slot, *rank.active,
                                             seqs.state_slot(r + 1u), rank.stream);
            if (!st.ok()) return st;
        }
        return Status::make_ok();
    }

    Status copy_to_active(uint32_t shadow_slot, const Qwen35SequenceSet& seqs,
                          hipStream_t stream) {
        Status st = active_state_->copy_slot_from(seqs.state_slot(0u),
                                                  *shadow_state_, shadow_slot, stream);
        if (!st.ok()) return st;
        for (uint32_t r = 0; r < ranks_.size(); ++r) {
            Rank& rank = *ranks_[r];
            auto scope = shelf::gpu::ScopedDevice::create(rank.device);
            if (!scope.ok()) return scope.status();
            st = rank.active->copy_slot_from(seqs.state_slot(r + 1u), *rank.shadow,
                                             shadow_slot, rank.stream);
            if (!st.ok()) return st;
        }
        return Status::make_ok();
    }

    Status copy_shadow(uint32_t dst, uint32_t src, hipStream_t stream) {
        Status st = shadow_state_->copy_slot_from(dst, *shadow_state_, src, stream);
        if (!st.ok()) return st;
        for (auto& rank : ranks_) {
            auto scope = shelf::gpu::ScopedDevice::create(rank->device);
            if (!scope.ok()) return scope.status();
            st = rank->shadow->copy_slot_from(dst, *rank->shadow, src, rank->stream);
            if (!st.ok()) return st;
        }
        return Status::make_ok();
    }

    GdnStatePool* active_state_ = nullptr;
    std::optional<GdnStatePool> shadow_state_;
    std::vector<std::unique_ptr<Rank>> ranks_;
    uint32_t max_concurrent_ = 0;
    std::vector<uint32_t> capture_free_;
    std::vector<uint32_t> hot_free_;
    std::vector<uint32_t> ring_free_;

    std::byte* ring_base_ = nullptr;
    std::size_t ring_slot_bytes_ = 0;
    uint32_t ring_slots_ = 0;
    std::vector<uint32_t> ring_length_;
    std::vector<uint32_t> ring_next_;
};



PrefixCacheBytes prefix_cache_bytes(
    const PrefixCacheConfig& config,
    const GdnStatePoolLayout& layout) noexcept {
    PrefixCacheBytes bytes;
    if (config.max_concurrent == 0u) return bytes;
    bytes.state_slot =
        static_cast<std::size_t>(layout.num_gdn_states) * layout.conv_history *
            layout.conv_history_stride * sizeof(bf16_t) +
        static_cast<std::size_t>(layout.num_gdn_states) * layout.num_v_heads *
            layout.head_k * layout.head_v * sizeof(fp16_t);
    bytes.state =
        bytes.state_slot * 2u * static_cast<std::size_t>(config.max_concurrent);
    bytes.ring = static_cast<std::size_t>(config.spec_ring_bytes) *
                     static_cast<std::size_t>(
                         config.spec_ring_bytes > 0 ? config.max_concurrent : 0u);
    return bytes;
}

PrefixCache::PrefixCache(PrefixCache&&) noexcept = default;
PrefixCache& PrefixCache::operator=(PrefixCache&&) noexcept = default;
PrefixCache::~PrefixCache() = default;

Status PrefixCache::set_tp_hook(TpBatchHook* hook) noexcept {
    tp_hook_ = hook;
    return Status::make_ok();
}

Result<PrefixCache> PrefixCache::create(
    gpu::GpuArena& arena,
    PagedKVPool& active_kv_pool,
    GdnStatePool& active_state_pool,
    const PrefixCacheConfig& config,
    const std::vector<PrefixCacheRankContext>& extra_ranks) {
    if (config.max_concurrent == 0u) {
        return Status::invalid_argument("max_concurrent must be > 0", __FILE__,
                                        __LINE__);
    }
    auto state = Qwen35PrefixState::create(arena, active_state_pool, extra_ranks,
                                           config);
    if (!state.ok()) return state.status();

    std::vector<PagedKVPool*> extra_kv;
    extra_kv.reserve(extra_ranks.size());
    for (const PrefixCacheRankContext& ctx : extra_ranks) {
        if (ctx.active_kv == nullptr) {
            return Status::invalid_argument("prefix rank KV pool is null", __FILE__,
                                            __LINE__);
        }
        extra_kv.push_back(ctx.active_kv);
    }
    PrefixKVConfig kv_config;
    kv_config.min_prefix_tokens = config.min_prefix_tokens;
    auto kv = PrefixKVStore::create(arena, active_kv_pool, extra_kv, kv_config);
    if (!kv.ok()) return kv.status();

    PrefixCache cache;
    cache.config_ = config;
    cache.page_tokens_ = active_kv_pool.page_tokens();
    cache.state_ = state.release();
    cache.kv_ = std::make_unique<PrefixKVStore>(kv.release());
    cache.aux_ = std::make_shared<AuxTable>();
    cache.captures_.resize(config.max_concurrent);
    Qwen35PrefixState* state_ptr = cache.state_.get();
    std::shared_ptr<AuxTable> aux = cache.aux_;
    cache.kv_->set_evict_callback([state_ptr, aux](uint64_t payload) {
        auto it = aux->slots.find(payload);
        if (it == aux->slots.end()) return;
        const AuxSlot slot = it->second;
        state_ptr->release_hot_slot(slot.hot_slot);
        if (slot.ring_slot != 0xFFFFFFFFu) {
            state_ptr->release_ring_slot(slot.ring_slot);
        }
        aux->slots.erase(it);
    });
    return cache;
}

Status PrefixCache::shutdown() noexcept {
    if (kv_) {
        kv_->set_evict_callback({});
        (void)kv_->shutdown();
    }
    if (state_) (void)state_->shutdown();
    kv_.reset();
    state_.reset();
    aux_.reset();
    return Status::make_ok();
}

PrefixCache::CaptureState* PrefixCache::resolve_capture(
    CaptureId capture,
    const PagedSequenceState* seq) {
    const uint32_t index = capture & 0xFFFFu;
    const uint32_t generation = capture >> 16;
    if (index >= captures_.size()) return nullptr;
    CaptureState& state = captures_[index];
    if (!state.active || state.generation != generation) return nullptr;
    if (seq != nullptr &&
        (state.seq_slot != seq->slot || state.seq_generation != seq->generation)) {
        return nullptr;
    }
    return &state;
}

Result<uint32_t> PrefixCache::acquire_state_slot(hipStream_t) {
    auto hot = state_->acquire_hot_slot();
    if (hot.ok()) return hot;
    auto evicted = kv_->evict_lru();
    if (!evicted.ok()) return hot;
    return state_->acquire_hot_slot();
}

PrefixHit PrefixCache::lookup(const std::vector<uint32_t>& tokens) {
    PrefixHit out;
    if (!kv_ || !state_ || !aux_) return out;
    const PrefixKVHit hit = kv_->lookup(tokens);
    out.match_tokens = hit.match_tokens;
    if (!hit.hit()) return out;
    const auto it = aux_->slots.find(hit.payload);
    if (it == aux_->slots.end()) return out;
    out.entry_id = hit.entry_id;
    out.state_position = hit.position;
    out.match_tokens = hit.match_tokens;
    out.ring_length = state_->ring_length(it->second.ring_slot);
    out.ring_next_position = state_->ring_next(it->second.ring_slot);
    return out;
}

Status PrefixCache::save_input_boundary(
    const PagedSequenceState& seq,
    const std::vector<uint32_t>& tokens,
    uint32_t input_len,
    hipStream_t stream,
    const SpecRingRef* ring) {
    if (!kv_ || !state_ || !aux_) {
        return Status::invalid_state("prefix cache is not ready", __FILE__, __LINE__);
    }
    if (input_len != tokens.size()) {
        return Status::invalid_argument("save position must equal token count",
                                        __FILE__, __LINE__);
    }
    if ((input_len / page_tokens_) * page_tokens_ != seq.position) {
        return Status::invalid_argument(
            "save_input_boundary requires a page-aligned sequence position",
            __FILE__, __LINE__);
    }
    Qwen35SequenceSet seqs(seq, tp_hook_);
    if (!seqs.valid()) {
        return Status::invalid_state("prefix cache rank sequence missing", __FILE__,
                                     __LINE__);
    }
    auto hot = acquire_state_slot(stream);
    if (!hot.ok()) return hot.status();
    Status st = state_->capture_gdn_to_hot(hot.value(), seqs, stream);
    uint32_t ring_slot = 0xFFFFFFFFu;
    if (st.ok() && ring != nullptr && ring->valid()) {
        auto acquired = state_->acquire_ring_slot();
        if (!acquired.ok()) {
            st = acquired.status();
        } else {
            ring_slot = acquired.value();
            st = state_->save_ring(ring_slot, *ring, input_len, stream);
        }
    }
    if (!st.ok()) {
        state_->release_hot_slot(hot.value());
        if (ring_slot != 0xFFFFFFFFu) state_->release_ring_slot(ring_slot);
        return st;
    }
    const uint64_t payload = aux_->next++;
    auto saved = kv_->save(seqs, tokens, input_len, payload, stream);
    if (!saved.ok()) {
        state_->release_hot_slot(hot.value());
        if (ring_slot != 0xFFFFFFFFu) state_->release_ring_slot(ring_slot);
        if (saved.status().code() == Status::Code::out_of_range) {
            ++stats_.skipped_too_short;
            return Status::make_ok();
        }
        return saved.status();
    }
    aux_->slots.emplace(payload, AuxSlot{hot.value(), ring_slot});
    ++stats_.input_boundary_saves;
    ++stats_.saves;
    return Status::make_ok();
}

Result<CaptureId> PrefixCache::begin_capture(const PagedSequenceState& seq) {
    if (!state_) {
        return Status::invalid_state("prefix cache is not ready", __FILE__, __LINE__);
    }
    auto slot = state_->acquire_capture_slot();
    if (!slot.ok()) return slot.status();
    uint32_t index = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < static_cast<uint32_t>(captures_.size()); ++i) {
        if (!captures_[i].active) {
            index = i;
            break;
        }
    }
    if (index == 0xFFFFFFFFu) {
        state_->release_capture_slot(slot.value());
        return Status::insufficient_memory("no free capture slots", __FILE__,
                                           __LINE__);
    }
    CaptureState& state = captures_[index];
    state.generation = (state.generation % 0xFFFFu) + 1u;
    state.active = true;
    state.seq_slot = seq.slot;
    state.seq_generation = seq.generation;
    state.state_slot = slot.value();
    state.last_boundary = 0;
    ++stats_.captures_started;
    const CaptureId id = static_cast<CaptureId>((state.generation << 16) | index);
    if (seq.position != 0u && seq.position % page_tokens_ == 0u) {
        (void)capture_boundary(id, seq, 0);
    }
    return id;
}

Status PrefixCache::capture_boundary(
    CaptureId capture,
    const PagedSequenceState& seq,
    hipStream_t stream) {
    CaptureState* state = resolve_capture(capture, &seq);
    if (state == nullptr) {
        return Status::invalid_argument("capture handle mismatch", __FILE__,
                                        __LINE__);
    }
    const uint32_t boundary = (seq.position / page_tokens_) * page_tokens_;
    if (boundary <= state->last_boundary) return Status::make_ok();
    Qwen35SequenceSet seqs(seq, tp_hook_);
    if (!seqs.valid()) {
        return Status::invalid_state("prefix cache rank sequence missing", __FILE__,
                                     __LINE__);
    }
    const Status st = state_->capture_gdn(state->state_slot, seqs, stream);
    if (!st.ok()) return st;
    state->last_boundary = boundary;
    return Status::make_ok();
}

Status PrefixCache::cancel_capture(CaptureId capture) {
    CaptureState* state = resolve_capture(capture, nullptr);
    if (state == nullptr) {
        return Status::invalid_argument("invalid capture handle", __FILE__,
                                        __LINE__);
    }
    state_->release_capture_slot(state->state_slot);
    state->active = false;
    ++stats_.captures_cancelled;
    return Status::make_ok();
}

Status PrefixCache::save_output_tail(
    CaptureId capture,
    const PagedSequenceState& seq,
    const std::vector<uint32_t>& tokens,
    uint32_t position,
    hipStream_t stream,
    const SpecRingRef* ring) {
    CaptureState* state = resolve_capture(capture, &seq);
    if (state == nullptr) {
        return Status::invalid_argument("capture handle mismatch", __FILE__,
                                        __LINE__);
    }
    if (position != tokens.size()) {
        return Status::invalid_argument("save position must equal token count",
                                        __FILE__, __LINE__);
    }
    Qwen35SequenceSet seqs(seq, tp_hook_);
    if (!seqs.valid()) {
        return Status::invalid_state("prefix cache rank sequence missing", __FILE__,
                                     __LINE__);
    }
    const uint32_t boundary = position - (position % page_tokens_);
    if (state->last_boundary != boundary) {
        return Status::invalid_argument("save position does not match capture boundary",
                                        __FILE__, __LINE__);
    }
    const std::span<const uint32_t> saved_tokens(tokens.data(), boundary);
    auto hot = acquire_state_slot(stream);
    if (!hot.ok()) {
        return hot.status();
    }
    Status st = state_->copy_capture_to_hot(hot.value(), state->state_slot, stream);
    uint32_t ring_slot = 0xFFFFFFFFu;
    if (st.ok() && ring != nullptr && ring->valid()) {
        auto acquired = state_->acquire_ring_slot();
        if (!acquired.ok()) {
            st = acquired.status();
        } else {
            ring_slot = acquired.value();
            st = state_->save_ring(ring_slot, *ring, boundary, stream);
        }
    }
    if (!st.ok()) {
        state_->release_hot_slot(hot.value());
        if (ring_slot != 0xFFFFFFFFu) state_->release_ring_slot(ring_slot);
        return st;
    }
    const uint64_t payload = aux_->next++;
    auto saved = kv_->save(seqs, saved_tokens, boundary, payload, stream);
    if (!saved.ok()) {
        state_->release_hot_slot(hot.value());
        if (ring_slot != 0xFFFFFFFFu) state_->release_ring_slot(ring_slot);
        return saved.status();
    }
    aux_->slots.emplace(payload, AuxSlot{hot.value(), ring_slot});
    state_->release_capture_slot(state->state_slot);
    state->active = false;
    ++stats_.output_tail_saves;
    ++stats_.saves;
    return Status::make_ok();
}

Status PrefixCache::restore(
    uint64_t entry_id,
    PagedSequenceState& dst,
    hipStream_t stream,
    SpecRingOut* ring) {
    if (!kv_ || !state_ || !aux_) {
        return Status::invalid_state("prefix cache is not ready", __FILE__, __LINE__);
    }
    auto payload = kv_->payload_for(entry_id);
    if (!payload.ok()) return payload.status();
    auto position = kv_->position_for(entry_id);
    if (!position.ok()) return position.status();
    const auto it = aux_->slots.find(payload.value());
    if (it == aux_->slots.end()) {
        return Status::invalid_state("prefix aux slot missing", __FILE__, __LINE__);
    }
    Qwen35SequenceSet seqs(dst, tp_hook_);
    if (!seqs.valid()) {
        return Status::invalid_state("prefix cache rank sequence missing", __FILE__,
                                     __LINE__);
    }
    PrefixKVHit hit;
    hit.entry_id = entry_id;
    hit.position = position.value();
    hit.payload = payload.value();
    Status st = kv_->restore_pages(hit, seqs, stream);
    if (!st.ok()) return st;
    const AuxSlot slot = it->second;
    st = state_->restore_hot(slot.hot_slot, seqs, stream);
    if (st.ok()) st = state_->load_ring(slot.ring_slot, ring, stream);
    if (!st.ok()) {
        for (uint32_t r = 0; r < seqs.rank_count(); ++r) {
            (void)seqs.rank(r).rollback_pages(0u, stream);
        }
        return st;
    }
    for (uint32_t r = 0; r < seqs.rank_count(); ++r) {
        st = seqs.rank(r).commit_prefix(position.value());
        if (!st.ok()) {
            for (uint32_t j = 0; j < seqs.rank_count(); ++j) {
                (void)seqs.rank(j).rollback_pages(0u, stream);
            }
            return st;
        }
    }
    ++stats_.restores;
    return Status::make_ok();
}

PrefixCacheStats PrefixCache::stats() const noexcept {
    PrefixCacheStats out = stats_;
    if (kv_) {
        const auto kv_stats = kv_->stats();
        out.lookups = kv_stats.lookups;
        out.hits = kv_stats.hits;
        out.misses = kv_stats.misses;
        out.restores = kv_stats.restores;
        out.entries_evicted = kv_stats.entries_evicted;
        out.pages_evicted = kv_stats.pages_evicted;
        out.entries = kv_stats.entries;
        out.used_pages = kv_stats.hot_pages;
        out.radix_nodes = kv_stats.radix_nodes;
        out.skipped_too_short = kv_stats.skipped_too_short;
    }
    uint32_t active = 0;
    for (const CaptureState& capture : captures_) {
        if (capture.active) ++active;
    }
    out.captures_active = active;
    return out;
}

std::size_t PrefixCache::reserved_bytes() const noexcept {
    return 0;
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
