#include <shelf/inference/prefix/prefix_kv_cache.h>

#include <algorithm>
#include <cstring>

namespace shelf {
namespace inference {
namespace prefix {

namespace {

using ::shelf::state::kInvalidPageId;

}  // namespace

Result<PrefixKVStore> PrefixKVStore::create(
    gpu::GpuArena& arena,
    PagedKVPool& kv0,
    const std::vector<PagedKVPool*>& extra_rank_kvs,
    const PrefixKVConfig& config) {
    (void)arena;
    const uint32_t page_tokens = kv0.page_tokens();
    if (page_tokens == 0u) {
        return Status::invalid_argument("page_tokens must be > 0", __FILE__, __LINE__);
    }
    if (config.min_prefix_tokens < 1u) {
        return Status::invalid_argument("min_prefix_tokens must be > 0", __FILE__,
                                        __LINE__);
    }
    PrefixKVStore store;
    store.kv0_ = &kv0;
    store.config_ = config;
    store.page_tokens_ = page_tokens;
    store.pools_.push_back(&kv0);
    for (PagedKVPool* pool : extra_rank_kvs) {
        if (pool == nullptr) {
            return Status::invalid_argument("rank KV pool is null", __FILE__, __LINE__);
        }
        store.pools_.push_back(pool);
    }
    store.shard_hot_index_.resize(store.pools_.size());
    store.nodes_.emplace_back();
    store.entries_.reserve(64);
    store.page_index_.reserve(kv0.num_pages());
    store.page_refcount_.reserve(kv0.num_pages());
    return store;
}

PrefixKVStore::PrefixKVStore(PrefixKVStore&& other) noexcept
    : pools_(std::move(other.pools_)),
      shard_hot_index_(std::move(other.shard_hot_index_)),
      kv0_(std::exchange(other.kv0_, nullptr)),
      config_(other.config_),
      page_tokens_(other.page_tokens_),
      hot_pages_used_(other.hot_pages_used_),
      nodes_(std::move(other.nodes_)),
      node_free_(std::move(other.node_free_)),
      entries_(std::move(other.entries_)),
      page_index_(std::move(other.page_index_)),
      page_refcount_(std::move(other.page_refcount_)),
      page_free_(std::move(other.page_free_)),
      next_entry_id_(other.next_entry_id_),
      clock_(other.clock_),
      pools_bound_(other.pools_bound_),
      evict_callback_(std::move(other.evict_callback_)),
      stats_(other.stats_) {}

PrefixKVStore& PrefixKVStore::operator=(PrefixKVStore&& other) noexcept {
    if (this == &other) return *this;
    shutdown();
    pools_ = std::move(other.pools_);
    shard_hot_index_ = std::move(other.shard_hot_index_);
    kv0_ = std::exchange(other.kv0_, nullptr);
    config_ = other.config_;
    page_tokens_ = other.page_tokens_;
    hot_pages_used_ = other.hot_pages_used_;
    nodes_ = std::move(other.nodes_);
    node_free_ = std::move(other.node_free_);
    entries_ = std::move(other.entries_);
    page_index_ = std::move(other.page_index_);
    page_refcount_ = std::move(other.page_refcount_);
    page_free_ = std::move(other.page_free_);
    next_entry_id_ = other.next_entry_id_;
    clock_ = other.clock_;
    pools_bound_ = other.pools_bound_;
    evict_callback_ = std::move(other.evict_callback_);
    stats_ = other.stats_;
    return *this;
}

PrefixKVStore::~PrefixKVStore() {
    (void)shutdown();
}

PagedKVPool* PrefixKVStore::pool_for_rank(uint32_t rank) const {
    if (rank >= pools_.size()) return nullptr;
    return pools_[rank];
}

void PrefixKVStore::bind_pools() {
    if (pools_bound_) return;
    pools_bound_ = true;
    for (uint32_t rank = 0; rank < pools_.size(); ++rank) {
        PagedKVPool* pool = pools_[rank];
        pool->set_reclaimer([this, rank](uint32_t needed) {
            return reclaim_pages(rank, needed).ok();
        });
    }
}

Status PrefixKVStore::shutdown() noexcept {
    while (!entries_.empty()) {
        remove_entry(entries_.back().id);
    }
    for (PageId page = 0; page < page_index_.size(); ++page) {
        if (page_index_[page] != kInvalidPageId) {
            release_hot_refs(page);
        }
    }
    for (PagedKVPool* pool : pools_) {
        if (pool != nullptr) pool->set_reclaimer({});
    }
    pools_.clear();
    shard_hot_index_.clear();
    nodes_.clear();
    node_free_.clear();
    entries_.clear();
    page_index_.clear();
    page_refcount_.clear();
    page_free_.clear();
    return Status::make_ok();
}

bool PrefixKVStore::find_child(
    uint32_t node,
    std::span<const uint32_t> tokens,
    uint32_t query_pages,
    uint32_t consumed,
    uint32_t& child,
    uint32_t& matched_edge_pages) const {
    for (uint32_t candidate : nodes_[node].children) {
        const RadixNode& n = nodes_[candidate];
        const uint32_t edge_pages = static_cast<uint32_t>(n.edge_pages.size());
        const uint32_t limit = std::min(edge_pages, query_pages - consumed);
        uint32_t matched = 0;
        while (matched < limit) {
            const uint32_t offset = (consumed + matched) * page_tokens_;
            bool equal = true;
            for (uint32_t t = 0; t < page_tokens_; ++t) {
                if (n.edge_tokens[matched * page_tokens_ + t] != tokens[offset + t]) {
                    equal = false;
                    break;
                }
            }
            if (!equal) break;
            ++matched;
        }
        if (matched > 0) {
            child = candidate;
            matched_edge_pages = matched;
            return true;
        }
    }
    return false;
}

PrefixKVStore::WalkResult PrefixKVStore::walk(std::span<const uint32_t> tokens) {
    WalkResult result;
    const uint32_t query_pages = static_cast<uint32_t>(tokens.size() / page_tokens_);
    nodes_[0].last_used = ++clock_;
    result.visited.push_back(0);
    uint32_t node = 0;
    uint32_t consumed = 0;
    while (consumed < query_pages) {
        uint32_t child = kInvalidNode;
        uint32_t matched_edge_pages = 0;
        if (!find_child(node, tokens, query_pages, consumed, child, matched_edge_pages)) {
            break;
        }
        const uint32_t edge_pages = static_cast<uint32_t>(nodes_[child].edge_pages.size());
        nodes_[child].last_used = ++clock_;
        result.visited.push_back(child);
        for (uint32_t i = 0; i < matched_edge_pages; ++i) {
            result.path.push_back(nodes_[child].edge_pages[i]);
        }
        consumed += matched_edge_pages;
        if (matched_edge_pages < edge_pages) break;
        node = child;
        if (nodes_[child].entry_id != 0) {
            result.entry_id = nodes_[child].entry_id;
        }
    }
    result.matched_pages = consumed;
    return result;
}

uint32_t PrefixKVStore::alloc_node() {
    if (!node_free_.empty()) {
        const uint32_t node = node_free_.back();
        node_free_.pop_back();
        nodes_[node] = RadixNode{};
        return node;
    }
    nodes_.emplace_back();
    return static_cast<uint32_t>(nodes_.size() - 1);
}

void PrefixKVStore::remove_node(uint32_t node) {
    if (node == 0) return;
    const uint32_t parent = nodes_[node].parent;
    std::vector<uint32_t>& siblings = nodes_[parent].children;
    siblings.erase(std::remove(siblings.begin(), siblings.end(), node), siblings.end());
    nodes_[node] = RadixNode{};
    node_free_.push_back(node);
}

void PrefixKVStore::split_node(uint32_t child, uint32_t split_pages) {
    const uint32_t split = alloc_node();
    RadixNode& tail = nodes_[split];
    RadixNode& head = nodes_[child];
    tail.parent = child;
    tail.depth_pages = head.depth_pages + split_pages;
    tail.edge_tokens.assign(
        head.edge_tokens.begin() + split_pages * page_tokens_, head.edge_tokens.end());
    tail.edge_pages.assign(head.edge_pages.begin() + split_pages, head.edge_pages.end());
    tail.children = std::move(head.children);
    head.children.clear();
    for (uint32_t moved : tail.children) {
        nodes_[moved].parent = split;
    }
    tail.entry_id = head.entry_id;
    head.entry_id = 0;
    if (tail.entry_id != 0) {
        Entry* entry = find_entry(tail.entry_id);
        if (entry != nullptr) entry->node = split;
    }
    head.edge_tokens.resize(split_pages * page_tokens_);
    head.edge_pages.resize(split_pages);
    tail.last_used = head.last_used;
    head.children.push_back(split);
}

uint32_t PrefixKVStore::insert_path(
    std::span<const uint32_t> tokens,
    uint32_t page_count,
    const std::vector<PageId>& pages) {
    uint32_t node = 0;
    uint32_t consumed = 0;
    nodes_[0].last_used = ++clock_;
    while (consumed < page_count) {
        uint32_t child = kInvalidNode;
        uint32_t matched_edge_pages = 0;
        if (!find_child(node, tokens, page_count, consumed, child, matched_edge_pages)) {
            const uint32_t fresh = alloc_node();
            RadixNode& n = nodes_[fresh];
            n.parent = node;
            n.depth_pages = consumed;
            n.edge_tokens.assign(tokens.begin() + consumed * page_tokens_,
                                 tokens.begin() + page_count * page_tokens_);
            n.edge_pages.assign(pages.begin() + consumed, pages.begin() + page_count);
            n.last_used = ++clock_;
            nodes_[node].children.push_back(fresh);
            return fresh;
        }
        const uint32_t edge_pages = static_cast<uint32_t>(nodes_[child].edge_pages.size());
        nodes_[child].last_used = ++clock_;
        if (matched_edge_pages == edge_pages) {
            node = child;
            consumed += matched_edge_pages;
            continue;
        }
        const uint32_t divergence = consumed + matched_edge_pages;
        if (matched_edge_pages > 0) {
            split_node(child, matched_edge_pages);
        }
        if (divergence == page_count) {
            return child;
        }
        const uint32_t parent = matched_edge_pages > 0 ? child : node;
        const uint32_t fresh = alloc_node();
        RadixNode& n = nodes_[fresh];
        n.parent = parent;
        n.depth_pages = divergence;
        n.edge_tokens.assign(tokens.begin() + divergence * page_tokens_,
                             tokens.begin() + page_count * page_tokens_);
        n.edge_pages.assign(pages.begin() + divergence, pages.begin() + page_count);
        n.last_used = ++clock_;
        nodes_[parent].children.push_back(fresh);
        return fresh;
    }
    return node;
}

PrefixKVStore::Entry* PrefixKVStore::find_entry(uint64_t id) {
    for (Entry& entry : entries_) {
        if (entry.id == id) return &entry;
    }
    return nullptr;
}

PrefixKVStore::Entry* PrefixKVStore::find_lru_entry() {
    Entry* best = nullptr;
    for (Entry& entry : entries_) {
        if (best == nullptr || entry.last_used < best->last_used) best = &entry;
    }
    return best;
}

void PrefixKVStore::remove_entry(uint64_t id) {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].id != id) continue;
        Entry removed = std::move(entries_[i]);
        if (i + 1 != entries_.size()) entries_[i] = std::move(entries_.back());
        entries_.pop_back();
        if (removed.node < nodes_.size() && nodes_[removed.node].entry_id == id) {
            nodes_[removed.node].entry_id = 0;
        }
        for (PageId page : removed.pages) {
            if (page < page_refcount_.size() && page_refcount_[page] > 0) {
                --page_refcount_[page];
            }
        }
        if (evict_callback_ && removed.payload != 0) evict_callback_(removed.payload);
        return;
    }
}

Result<PageId> PrefixKVStore::alloc_hot_page(
    PageId pool_page,
    const std::vector<PageId>& shard_pages) {
    PageId logical;
    if (!page_free_.empty()) {
        logical = page_free_.back();
        page_free_.pop_back();
    } else {
        logical = static_cast<PageId>(page_index_.size());
        page_index_.push_back(kInvalidPageId);
        page_refcount_.push_back(0);
        for (auto& index : shard_hot_index_) index.push_back(kInvalidPageId);
    }
    for (auto& index : shard_hot_index_) {
        if (index.size() <= logical) {
            index.resize(static_cast<std::size_t>(logical) + 1u, kInvalidPageId);
        }
    }
    const Status retain_st = pools_[0]->retain_cache(pool_page);
    if (!retain_st.ok()) return retain_st;
    for (std::size_t r = 1; r < pools_.size(); ++r) {
        if (r - 1 >= shard_pages.size() || shard_pages[r - 1] == kInvalidPageId) {
            (void)pools_[0]->release_cache(pool_page);
            return Status::invalid_state("prefix cache shard page missing", __FILE__,
                                         __LINE__);
        }
        const Status st = pools_[r]->retain_cache(shard_pages[r - 1]);
        if (!st.ok()) {
            (void)pools_[0]->release_cache(pool_page);
            for (std::size_t j = 1; j < r; ++j) {
                (void)pools_[j]->release_cache(shard_pages[j - 1]);
            }
            return st;
        }
        shard_hot_index_[r][logical] = shard_pages[r - 1];
    }
    page_index_[logical] = pool_page;
    page_refcount_[logical] = 0;
    ++hot_pages_used_;
    return logical;
}

void PrefixKVStore::release_hot_refs(PageId page) {
    if (page >= page_index_.size()) return;
    const PageId pool_page = page_index_[page];
    if (pool_page == kInvalidPageId) return;
    (void)pools_[0]->release_cache(pool_page);
    for (std::size_t r = 1; r < pools_.size(); ++r) {
        if (page < shard_hot_index_[r].size() &&
            shard_hot_index_[r][page] != kInvalidPageId) {
            (void)pools_[r]->release_cache(shard_hot_index_[r][page]);
            shard_hot_index_[r][page] = kInvalidPageId;
        }
    }
    if (hot_pages_used_ > 0) --hot_pages_used_;
}

void PrefixKVStore::release_page(PageId page) {
    if (page >= page_index_.size()) return;
    release_hot_refs(page);
    page_index_[page] = kInvalidPageId;
    page_refcount_[page] = 0;
    page_free_.push_back(page);
}

Status PrefixKVStore::prepare_save(
    PrefixSequenceSet& seqs,
    std::span<const uint32_t> tokens,
    uint32_t position,
    SavePlan& plan,
    hipStream_t stream) {
    if (static_cast<std::size_t>(position) != tokens.size()) {
        return Status::invalid_argument("save position must equal token count",
                                        __FILE__, __LINE__);
    }
    plan.page_count = position / page_tokens_;
    const PrefixSequence& rank0 = seqs.rank(0);
    if (rank0.block_table().size() < plan.page_count) {
        return Status::invalid_argument("sequence KV range shorter than save position",
                                        __FILE__, __LINE__);
    }
    if (position < config_.min_prefix_tokens || plan.page_count == 0u) {
        plan.too_short = true;
        return Status::make_ok();
    }
    plan.walk = walk(tokens);
    plan.shared_pages = std::min(plan.walk.matched_pages, plan.page_count);
    plan.pages.assign(plan.walk.path.begin(),
                      plan.walk.path.begin() + plan.shared_pages);
    return Status::make_ok();
}

Status PrefixKVStore::fill_pages(SavePlan& plan, PrefixSequenceSet& seqs) {
    const uint32_t start = plan.shared_pages;
    std::vector<PageId> shard_pages(pools_.size() > 1 ? pools_.size() - 1 : 0,
                                    kInvalidPageId);
    for (uint32_t r = 1; r < seqs.rank_count(); ++r) {
        if (seqs.rank(r).block_table().size() < plan.page_count) {
            return Status::invalid_state("rank mirror block table too short", __FILE__,
                                         __LINE__);
        }
    }
    for (uint32_t i = start; i < plan.page_count; ++i) {
        for (uint32_t r = 1; r < seqs.rank_count(); ++r) {
            shard_pages[r - 1] = seqs.rank(r).block_table()[i];
        }
        auto page = alloc_hot_page(seqs.rank(0).block_table()[i], shard_pages);
        if (!page.ok()) {
            for (uint32_t j = start; j < i; ++j) release_page(plan.pages[j]);
            plan.pages.resize(start);
            return page.status();
        }
        plan.pages.push_back(page.release());
    }
    return Status::make_ok();
}

void PrefixKVStore::register_entry(SavePlan& plan, std::span<const uint32_t> tokens,
                                   uint64_t payload) {
    const uint32_t endpoint = insert_path(tokens, plan.page_count, plan.pages);
    if (nodes_[endpoint].entry_id != 0) {
        remove_entry(nodes_[endpoint].entry_id);
    }
    Entry entry;
    entry.id = next_entry_id_++;
    entry.position = plan.page_count * page_tokens_;
    entry.node = endpoint;
    entry.payload = payload;
    entry.last_used = ++clock_;
    entry.pages = plan.pages;
    for (PageId page : entry.pages) {
        if (page < page_refcount_.size()) ++page_refcount_[page];
    }
    const uint64_t id = entry.id;
    entries_.push_back(std::move(entry));
    nodes_[endpoint].entry_id = id;
    ++stats_.saves;
}

Result<uint64_t> PrefixKVStore::save(
    PrefixSequenceSet& seqs,
    std::span<const uint32_t> tokens,
    uint32_t position,
    uint64_t payload,
    hipStream_t stream) {
    bind_pools();
    if (seqs.rank_count() != pools_.size()) {
        return Status::invalid_argument("prefix sequence rank count mismatch", __FILE__,
                                        __LINE__);
    }
    SavePlan plan;
    Status st = prepare_save(seqs, tokens, position, plan, stream);
    if (!st.ok()) return st;
    if (plan.too_short) {
        ++stats_.skipped_too_short;
        return Status::out_of_range("prefix is shorter than min_prefix_tokens",
                                    __FILE__, __LINE__);
    }
    st = fill_pages(plan, seqs);
    if (!st.ok()) return st;
    register_entry(plan, tokens, payload);
    return entries_.back().id;
}

PrefixKVHit PrefixKVStore::lookup(std::span<const uint32_t> tokens) {
    bind_pools();
    ++stats_.lookups;
    const WalkResult result = walk(tokens);
    PrefixKVHit hit;
    hit.match_tokens = result.matched_pages * page_tokens_;
    if (result.entry_id != 0) {
        Entry* entry = find_entry(result.entry_id);
        if (entry != nullptr) {
            hit.entry_id = entry->id;
            hit.position = entry->position;
            hit.payload = entry->payload;
            entry->last_used = ++clock_;
            ++stats_.hits;
            return hit;
        }
    }
    ++stats_.misses;
    return hit;
}

Status PrefixKVStore::restore_pages(
    const PrefixKVHit& hit,
    PrefixSequenceSet& seqs,
    hipStream_t stream) {
    bind_pools();
    Entry* entry = find_entry(hit.entry_id);
    if (entry == nullptr) {
        return Status::invalid_argument("prefix entry not found", __FILE__, __LINE__);
    }
    if (seqs.rank_count() != pools_.size()) {
        return Status::invalid_argument("prefix sequence rank count mismatch", __FILE__,
                                        __LINE__);
    }
    const uint32_t page_count = entry->position / page_tokens_;
    std::vector<std::vector<PageId>> pages(seqs.rank_count());
    std::vector<std::vector<uint8_t>> owned(seqs.rank_count());
    for (auto& list : pages) list.reserve(page_count);
    for (auto& list : owned) list.reserve(page_count);
    for (uint32_t i = 0; i < page_count; ++i) {
        const PageId cache_page = entry->pages[i];
        const PageId pool_page = page_index_[cache_page];
        if (pool_page == kInvalidPageId) {
            return Status::invalid_state("prefix page missing", __FILE__, __LINE__);
        }
        pages[0].push_back(pool_page);
        owned[0].push_back(0u);
        for (uint32_t r = 1; r < seqs.rank_count(); ++r) {
            if (cache_page >= shard_hot_index_[r].size() ||
                shard_hot_index_[r][cache_page] == kInvalidPageId) {
                return Status::invalid_state("prefix shard page missing", __FILE__,
                                             __LINE__);
            }
            pages[r].push_back(shard_hot_index_[r][cache_page]);
            owned[r].push_back(0u);
        }
    }
    for (uint32_t r = 0; r < seqs.rank_count(); ++r) {
        const Status st = seqs.rank(r).append_pages(pages[r], owned[r], stream);
        if (!st.ok()) {
            for (uint32_t j = 0; j < r; ++j) {
                (void)seqs.rank(j).rollback_pages(0u, stream);
            }
            return st;
        }
    }
    ++stats_.restores;
    return Status::make_ok();
}

Status PrefixKVStore::erase(uint64_t entry_id) {
    if (find_entry(entry_id) == nullptr) {
        return Status::invalid_argument("prefix entry not found", __FILE__, __LINE__);
    }
    remove_entry(entry_id);
    ++stats_.entries_evicted;
    return Status::make_ok();
}

Result<uint64_t> PrefixKVStore::payload_for(uint64_t entry_id) const {
    for (const Entry& entry : entries_) {
        if (entry.id == entry_id) return entry.payload;
    }
    return Status::invalid_argument("prefix entry not found", __FILE__, __LINE__);
}

Result<uint32_t> PrefixKVStore::position_for(uint64_t entry_id) const {
    for (const Entry& entry : entries_) {
        if (entry.id == entry_id) return entry.position;
    }
    return Status::invalid_argument("prefix entry not found", __FILE__, __LINE__);
}

Result<uint64_t> PrefixKVStore::evict_lru() {
    Entry* lru = find_lru_entry();
    if (lru == nullptr) {
        return Status::insufficient_memory("prefix cache has no entries to evict",
                                           __FILE__, __LINE__);
    }
    const uint64_t id = lru->id;
    remove_entry(id);
    ++stats_.entries_evicted;
    return id;
}

Result<uint64_t> PrefixKVStore::lru_entry_id() const {
    const Entry* lru = nullptr;
    for (const Entry& entry : entries_) {
        if (lru == nullptr || entry.last_used < lru->last_used) lru = &entry;
    }
    if (lru == nullptr) {
        return Status::insufficient_memory("prefix cache has no entries", __FILE__,
                                           __LINE__);
    }
    return lru->id;
}

std::vector<uint64_t> PrefixKVStore::lru_entry_ids() const {
    std::vector<const Entry*> ordered;
    ordered.reserve(entries_.size());
    for (const Entry& entry : entries_) ordered.push_back(&entry);
    std::sort(ordered.begin(), ordered.end(), [](const Entry* a, const Entry* b) {
        return a->last_used < b->last_used;
    });
    std::vector<uint64_t> ids;
    ids.reserve(ordered.size());
    for (const Entry* entry : ordered) ids.push_back(entry->id);
    return ids;
}

bool PrefixKVStore::evict_one_leaf(const std::vector<uint32_t>& visited) {
    uint32_t best = kInvalidNode;
    uint64_t best_used = 0;
    for (uint32_t node = 1; node < static_cast<uint32_t>(nodes_.size()); ++node) {
        const RadixNode& n = nodes_[node];
        if (n.edge_pages.empty()) continue;
        if (!n.children.empty()) continue;
        if (n.entry_id != 0) continue;
        if (std::find(visited.begin(), visited.end(), node) != visited.end()) continue;
        bool reclaimable = true;
        for (PageId page : n.edge_pages) {
            if (page < page_refcount_.size() && page_refcount_[page] != 0) {
                reclaimable = false;
                break;
            }
        }
        if (!reclaimable) continue;
        if (best == kInvalidNode || n.last_used < best_used) {
            best = node;
            best_used = n.last_used;
        }
    }
    if (best == kInvalidNode) return false;
    for (PageId page : nodes_[best].edge_pages) {
        release_page(page);
        ++stats_.pages_evicted;
    }
    remove_node(best);
    return true;
}

Status PrefixKVStore::reclaim_pages(uint32_t rank, uint32_t needed) {
    PagedKVPool* pool = pool_for_rank(rank);
    if (pool == nullptr) {
        return Status::invalid_argument("prefix rank out of range", __FILE__, __LINE__);
    }
    if (pool->num_truly_free_pages() >= needed) return Status::make_ok();
    const uint32_t limit = static_cast<uint32_t>(entries_.size() + nodes_.size()) + 16u;
    for (uint32_t guard = 0;
         guard < limit && pool->num_truly_free_pages() < needed; ++guard) {
        if (evict_one_leaf({})) continue;
        Entry* lru = find_lru_entry();
        if (lru == nullptr) break;
        const uint64_t id = lru->id;
        remove_entry(id);
        ++stats_.entries_evicted;
    }
    if (pool->num_truly_free_pages() < needed) {
        return Status::insufficient_memory("prefix cache cannot reclaim KV pages",
                                           __FILE__, __LINE__);
    }
    return Status::make_ok();
}

PrefixKVStats PrefixKVStore::stats() const noexcept {
    PrefixKVStats snapshot = stats_;
    snapshot.entries = static_cast<uint32_t>(entries_.size());
    snapshot.hot_pages = hot_pages_used_;
    uint32_t nodes = 0;
    for (const RadixNode& node : nodes_) {
        if (!node.edge_pages.empty()) ++nodes;
    }
    snapshot.radix_nodes = nodes;
    return snapshot;
}

}  // namespace prefix
}  // namespace inference
}  // namespace shelf
