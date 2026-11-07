#pragma once

#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace shelf {
namespace inference {
namespace prefix {

using ::shelf::state::PageId;
using ::shelf::state::PagedKVPool;

class PrefixSequence {
 public:
    virtual ~PrefixSequence() = default;

    virtual PagedKVPool& kv_pool() = 0;
    virtual uint32_t position() const = 0;
    virtual std::span<const PageId> block_table() const = 0;

    virtual Status append_pages(std::span<const PageId> pages,
                                std::span<const uint8_t> owned,
                                hipStream_t stream) = 0;
    virtual Status commit_prefix(uint32_t tokens) = 0;
    virtual Status rollback_pages(uint32_t page_count, hipStream_t stream) = 0;
};

class PrefixSequenceSet {
 public:
    virtual ~PrefixSequenceSet() = default;
    virtual uint32_t rank_count() const = 0;
    virtual PrefixSequence& rank(uint32_t rank) = 0;
};

struct PrefixKVConfig {
    uint32_t min_prefix_tokens = 64;
};

struct PrefixKVStats {
    uint64_t saves = 0;
    uint64_t skipped_too_short = 0;
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t restores = 0;
    uint64_t entries_evicted = 0;
    uint64_t pages_evicted = 0;
    uint32_t entries = 0;
    uint32_t hot_pages = 0;
    uint32_t radix_nodes = 0;
};

struct PrefixKVHit {
    uint64_t entry_id = 0;
    uint32_t position = 0;
    uint32_t match_tokens = 0;
    uint64_t payload = 0;

    bool hit() const noexcept { return entry_id != 0; }
};

class PrefixKVStore {
 public:
    using EvictCallback = std::function<void(uint64_t payload)>;

    static Result<PrefixKVStore> create(
        gpu::GpuArena& arena,
        PagedKVPool& kv0,
        const std::vector<PagedKVPool*>& extra_rank_kvs,
        const PrefixKVConfig& config);

    PrefixKVStore(PrefixKVStore&&) noexcept;
    PrefixKVStore& operator=(PrefixKVStore&&) noexcept;
    PrefixKVStore(const PrefixKVStore&) = delete;
    PrefixKVStore& operator=(const PrefixKVStore&) = delete;
    ~PrefixKVStore();

    PrefixKVHit lookup(std::span<const uint32_t> tokens);

    Result<uint64_t> save(
        PrefixSequenceSet& seqs,
        std::span<const uint32_t> tokens,
        uint32_t position,
        uint64_t payload,
        hipStream_t stream);

    Status restore_pages(
        const PrefixKVHit& hit,
        PrefixSequenceSet& seqs,
        hipStream_t stream);

    Result<uint64_t> payload_for(uint64_t entry_id) const;
    Result<uint32_t> position_for(uint64_t entry_id) const;
    Result<uint64_t> evict_lru();
    Result<uint64_t> lru_entry_id() const;
    std::vector<uint64_t> lru_entry_ids() const;

    Status erase(uint64_t entry_id);

    void set_evict_callback(EvictCallback callback) { evict_callback_ = std::move(callback); }

    Status reclaim_pages(uint32_t rank, uint32_t needed);

    PrefixKVStats stats() const noexcept;

    Status shutdown() noexcept;

 private:
    static constexpr uint32_t kInvalidNode = 0xFFFFFFFFu;

    struct RadixNode {
        uint32_t parent = 0;
        uint32_t depth_pages = 0;
        std::vector<uint32_t> edge_tokens;
        std::vector<PageId> edge_pages;
        std::vector<uint32_t> children;
        uint64_t last_used = 0;
        uint64_t entry_id = 0;
    };

    struct Entry {
        uint64_t id = 0;
        uint32_t position = 0;
        uint32_t node = 0;
        uint64_t payload = 0;
        uint64_t last_used = 0;
        std::vector<PageId> pages;
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

    PrefixKVStore() = default;

    PagedKVPool* pool_for_rank(uint32_t rank) const;
    void bind_pools();

    bool find_child(uint32_t node, std::span<const uint32_t> tokens,
                    uint32_t query_pages, uint32_t consumed,
                    uint32_t& child, uint32_t& matched_edge_pages) const;
    WalkResult walk(std::span<const uint32_t> tokens);
    uint32_t insert_path(std::span<const uint32_t> tokens, uint32_t page_count,
                         const std::vector<PageId>& pages);
    void split_node(uint32_t child, uint32_t split_pages);
    uint32_t alloc_node();
    void remove_node(uint32_t node);
    Entry* find_entry(uint64_t id);
    Entry* find_lru_entry();
    void remove_entry(uint64_t id);

    Result<PageId> alloc_hot_page(PageId pool_page,
                                  const std::vector<PageId>& shard_pages);
    void release_hot_refs(PageId page);
    void release_page(PageId page);

    Status prepare_save(PrefixSequenceSet& seqs, std::span<const uint32_t> tokens,
                        uint32_t position, SavePlan& plan, hipStream_t stream);
    Status fill_pages(SavePlan& plan, PrefixSequenceSet& seqs);
    void register_entry(SavePlan& plan, std::span<const uint32_t> tokens,
                        uint64_t payload);

    bool evict_one_leaf(const std::vector<uint32_t>& visited);

    std::vector<PagedKVPool*> pools_;
    std::vector<std::vector<uint32_t>> shard_hot_index_;

    PagedKVPool* kv0_ = nullptr;
    PrefixKVConfig config_;
    uint32_t page_tokens_ = 0;
    uint32_t hot_pages_used_ = 0;

    std::vector<RadixNode> nodes_;
    std::vector<uint32_t> node_free_;
    std::vector<Entry> entries_;
    std::vector<PageId> page_index_;
    std::vector<uint32_t> page_refcount_;
    std::vector<PageId> page_free_;

    uint64_t next_entry_id_ = 1;
    uint64_t clock_ = 1;
    bool pools_bound_ = false;

    EvictCallback evict_callback_;
    PrefixKVStats stats_;
};

}  // namespace prefix
}  // namespace inference
}  // namespace shelf
