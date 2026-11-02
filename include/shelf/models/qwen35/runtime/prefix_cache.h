#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <memory>
#include <vector>

namespace ps {
namespace qwen35 {
namespace runtime {

struct PrefixCacheConfig {
    uint32_t hot_capacity_tokens = 16384;
    uint32_t cold_capacity_tokens = 65536;
    uint32_t max_concurrent = 4;
    uint32_t cold_state_slots = 16;
    uint32_t min_prefix_tokens = 64;
    uint32_t spec_ring_bytes = 0;
};

struct PrefixCacheGeometry {
    uint32_t page_tokens = 16;
    uint32_t num_attention_layers = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    KVCacheDType dtype = KVCacheDType::BF16;
};

struct PrefixCacheBytes {
    std::size_t hot_kv = 0;
    std::size_t hot_state = 0;
    std::size_t hot_ring = 0;
    std::size_t cold_kv = 0;
    std::size_t cold_state = 0;
    std::size_t cold_page = 0;
    std::size_t state_slot = 0;
    uint32_t hot_pages = 0;
    uint32_t cold_pages = 0;
    uint32_t cold_state_slots = 0;

    std::size_t hot_total() const noexcept { return hot_kv + hot_state + hot_ring; }
};

PrefixCacheBytes prefix_cache_bytes(
    const PrefixCacheConfig& config,
    const GdnStatePoolLayout& layout,
    const PrefixCacheGeometry& geometry) noexcept;

struct SpecRingRef {
    const bf16_t* k = nullptr;
    const bf16_t* v = nullptr;
    uint32_t length = 0;
    uint32_t next_position = 0;

    bool valid() const noexcept { return k != nullptr && v != nullptr; }
};

struct SpecRingOut {
    bf16_t* k = nullptr;
    bf16_t* v = nullptr;
    uint32_t length = 0;
    uint32_t next_position = 0;

    bool valid() const noexcept { return k != nullptr && v != nullptr; }
};

struct PrefixEntry {
    uint64_t id = 0;
    uint32_t position = 0;
    uint32_t node = 0;
    SequenceSlotId state_slot = kInvalidSlot;
    bool state_cold = false;
    uint32_t ring_slot = kInvalidSlot;
    uint32_t ring_length = 0;
    uint32_t ring_next_position = 0;
    uint64_t last_used = 0;
    std::vector<PageId> pages;
};

struct PrefixHit {
    uint64_t entry_id = 0;
    uint32_t state_position = 0;
    uint32_t match_tokens = 0;
    uint32_t ring_length = 0;
    uint32_t ring_next_position = 0;

    bool hit() const noexcept { return entry_id != 0; }

    bool ring_covers_state() const noexcept {
        return ring_length != 0 && ring_next_position == state_position;
    }
};

struct PrefixCacheStats {
    uint64_t saves = 0;
    uint64_t input_boundary_saves = 0;
    uint64_t output_tail_saves = 0;
    uint64_t skipped_too_short = 0;
    uint64_t skipped_too_large = 0;
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t restores = 0;
    uint64_t captures_started = 0;
    uint64_t captures_cancelled = 0;
    uint64_t entries_evicted = 0;
    uint64_t pages_evicted = 0;
    uint64_t pages_migrated_cold = 0;
    uint64_t pages_dropped_cold = 0;
    uint64_t pinned_demoted = 0;
    uint64_t states_migrated_cold = 0;
    uint32_t entries = 0;
    uint32_t used_pages = 0;
    uint32_t radix_nodes = 0;
    uint32_t captures_active = 0;
    uint32_t cold_pages_used = 0;
    uint32_t cold_states_used = 0;
};

using CaptureId = uint32_t;

inline constexpr CaptureId kInvalidCapture = 0;

class PinnedHostBuffer {
 public:
    PinnedHostBuffer() = default;
    ~PinnedHostBuffer() noexcept;

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    static Result<PinnedHostBuffer> allocate(std::size_t bytes, const char* name);

    Status shutdown() noexcept;

    void* data() const noexcept { return ptr_; }
    std::size_t size_bytes() const noexcept { return bytes_; }
    bool empty() const noexcept { return ptr_ == nullptr; }

 private:
    PinnedHostBuffer(void* ptr, std::size_t bytes, const char* name)
        : ptr_(ptr), bytes_(bytes), name_(name) {}

    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
    const char* name_ = "";
};

class TpBatchHook;

struct PrefixCacheRankContext {
    const PagedKVPool* active_kv = nullptr;
    GdnStatePool* active_state = nullptr;
    gpu::GpuArena* arena = nullptr;
    hipStream_t stream = nullptr;
    int device = 0;
};

class PrefixCache {
 public:
    static Result<PrefixCache> create(
        gpu::GpuArena& arena,
        const PagedKVPool& active_kv_pool,
        GdnStatePool& active_state_pool,
        const PrefixCacheConfig& config,
        const std::vector<PrefixCacheRankContext>& extra_ranks = {});

    Status set_tp_hook(TpBatchHook* hook) noexcept;

    PrefixCache(PrefixCache&&) = default;
    PrefixCache& operator=(PrefixCache&&) = default;
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    Status shutdown() noexcept;

    PrefixHit lookup(const std::vector<uint32_t>& tokens);

    Status save_input_boundary(
        const PagedSequenceState& seq,
        const std::vector<uint32_t>& tokens,
        uint32_t input_len,
        hipStream_t stream,
        const SpecRingRef* ring = nullptr);

    Result<CaptureId> begin_capture(const PagedSequenceState& seq);

    Status capture_boundary(
        CaptureId capture,
        const PagedSequenceState& seq,
        hipStream_t stream);

    Status cancel_capture(CaptureId capture);

    Status save_output_tail(
        CaptureId capture,
        const PagedSequenceState& seq,
        const std::vector<uint32_t>& tokens,
        uint32_t position,
        hipStream_t stream,
        const SpecRingRef* ring = nullptr);

    Status restore(
        uint64_t entry_id,
        PagedSequenceState& dst,
        hipStream_t stream,
        SpecRingOut* ring = nullptr);

    PrefixCacheStats stats() const noexcept;

    uint32_t page_tokens() const noexcept { return page_tokens_; }

 private:
    static constexpr uint32_t kInvalidNode = 0xFFFFFFFFu;

    struct PageLocation {
        bool cold = false;
        uint32_t index = 0;
    };

    struct RadixNode {
        uint32_t parent = 0;
        uint32_t depth_pages = 0;
        std::vector<uint32_t> edge_tokens;
        std::vector<PageId> edge_pages;
        std::vector<uint32_t> children;
        uint64_t last_used = 0;
        uint64_t entry_id = 0;
    };

    struct CaptureState {
        bool active = false;
        uint32_t generation = 0;
        SequenceSlotId seq_slot = kInvalidSlot;
        uint32_t seq_generation = 0;
        SequenceSlotId state_slot = kInvalidSlot;
        uint32_t last_boundary = 0;
    };

    struct WalkResult {
        uint32_t matched_pages = 0;
        uint64_t entry_id = 0;
        std::vector<uint32_t> visited;
        std::vector<PageId> path;
    };

    struct SavePlan {
        WalkResult walk;
        uint32_t page_count = 0;
        uint32_t shared_pages = 0;
        bool too_short = false;
        std::vector<PageId> pages;
    };

    PrefixCache(PagedKVPool&& shadow_kv, GdnStatePool&& shadow_state)
        : shadow_kv_(std::move(shadow_kv)), shadow_state_(std::move(shadow_state)) {}

    bool find_child(
        uint32_t node,
        const std::vector<uint32_t>& tokens,
        uint32_t query_pages,
        uint32_t consumed,
        uint32_t& child,
        uint32_t& matched_edge_pages) const;

    WalkResult walk(const std::vector<uint32_t>& tokens);

    uint32_t insert_path(
        const std::vector<uint32_t>& tokens,
        uint32_t page_count,
        const std::vector<PageId>& pages);

    void split_node(uint32_t child, uint32_t split_pages);

    uint32_t alloc_node();

    void remove_node(uint32_t node);

    PrefixEntry* find_entry(uint64_t id);

    PrefixEntry* find_lru_entry(bool cold_state);

    void remove_entry(uint64_t id);

    Result<PageId> alloc_page();

    void release_page(PageId page);

    Result<uint32_t> alloc_cold_page();

    Result<uint32_t> acquire_entry_state_slot(hipStream_t stream);

    Status demote_entry_state_to_cold(hipStream_t stream);

    Result<uint32_t> acquire_entry_ring_slot();

    Status save_entry_ring(
        uint32_t ring_slot,
        const SpecRingRef& ring,
        hipStream_t stream);

    Status load_entry_ring(
        uint32_t ring_slot,
        const SpecRingOut& ring,
        hipStream_t stream);

    Status make_room_for_pages(
        uint32_t needed,
        const std::vector<uint32_t>& visited,
        hipStream_t stream);

    Status migrate_one_leaf(
        const std::vector<uint32_t>& visited,
        const std::vector<char>& pinned_mark,
        bool allow_pinned,
        hipStream_t stream,
        bool& migrated);

    bool evict_one_leaf(const std::vector<uint32_t>& visited);

    void build_pinned_marks(std::vector<char>& marks) const;

    Status copy_page_to_cold(PageId page, uint32_t cold_index, hipStream_t stream) const;

    Status copy_page_from_cold(
        const PagedKVPool& dst_pool,
        PageId dst_page,
        uint32_t cold_index,
        hipStream_t stream) const;

    Status copy_state_to_cold(
        SequenceSlotId hot_slot,
        uint32_t cold_index,
        hipStream_t stream) const;

    Status copy_state_from_cold(
        const GdnStatePool& dst_pool,
        SequenceSlotId dst_slot,
        uint32_t cold_index,
        hipStream_t stream) const;

    CaptureState* resolve_capture(
        CaptureId capture,
        const PagedSequenceState* seq);

    Status prepare_save(
        const PagedSequenceState& seq,
        const std::vector<uint32_t>& tokens,
        uint32_t position,
        SavePlan& plan,
        hipStream_t stream);

    Status fill_pages(
        SavePlan& plan,
        const PagedSequenceState& seq,
        hipStream_t stream);

    void register_entry(
        SavePlan& plan,
        uint32_t state_slot,
        const std::vector<uint32_t>& tokens,
        uint32_t ring_slot = kInvalidSlot,
        uint32_t ring_length = 0,
        uint32_t ring_next_position = 0);

    struct RankShard {
        RankShard(PagedKVPool&& kv, GdnStatePool&& state)
            : shadow_kv(std::move(kv)), shadow_state(std::move(state)) {}
        PagedKVPool shadow_kv;
        GdnStatePool shadow_state;
        const PagedKVPool* active_kv = nullptr;
        GdnStatePool* active_state = nullptr;
        hipStream_t stream = nullptr;
        int device = 0;
    };

    PagedSequenceState* mirror_of(const PagedSequenceState& seq, uint32_t rank);
    Status capture_active_state(uint32_t dst_state_slot,
                                const PagedSequenceState& seq,
                                hipStream_t stream);
    Status roll_state(uint32_t dst_state_slot, uint32_t src_state_slot,
                      hipStream_t stream);
    void release_shadow(uint32_t index);

    PagedKVPool shadow_kv_;
    GdnStatePool shadow_state_;
    std::vector<std::unique_ptr<RankShard>> rank_shards_;
    TpBatchHook* tp_hook_ = nullptr;
    GdnStatePool* active_state_pool_ = nullptr;

    std::byte* ring_base_ = nullptr;
    std::size_t ring_slot_bytes_ = 0;
    uint32_t ring_slots_ = 0;

    PrefixCacheConfig config_;
    uint32_t page_tokens_ = 0;
    std::size_t cold_page_bytes_ = 0;
    std::size_t cold_state_bytes_ = 0;
    uint32_t cold_page_capacity_ = 0;
    uint32_t cold_state_capacity_ = 0;

    PinnedHostBuffer cold_kv_buffer_;
    PinnedHostBuffer cold_state_buffer_;

    std::vector<RadixNode> nodes_;
    std::vector<uint32_t> node_free_;
    std::vector<PrefixEntry> entries_;
    std::vector<PageLocation> page_location_;
    std::vector<uint32_t> page_refcount_;
    std::vector<PageId> page_free_;
    std::vector<uint32_t> cold_page_free_;
    std::vector<uint32_t> rolling_slots_free_;
    std::vector<uint32_t> hot_state_slots_free_;
    std::vector<uint32_t> cold_state_free_;
    std::vector<uint32_t> ring_slots_free_;
    std::vector<CaptureState> captures_;

    uint64_t next_entry_id_ = 1;
    uint64_t clock_ = 1;

    PrefixCacheStats stats_;
};

}
}
}
