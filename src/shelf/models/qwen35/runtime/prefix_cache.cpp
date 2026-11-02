#include <shelf/core/gpu/scoped_device.h>
#include <shelf/models/qwen35/runtime/tp_batch_hook.h>
#include <shelf/models/qwen35/runtime/prefix_cache.h>
#include <shelf/core/memory/size_format.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>
#include <unistd.h>

namespace shelf {
namespace qwen35 {
namespace runtime {

namespace {

std::size_t host_available_bytes() {
    std::FILE* fp = std::fopen("/proc/meminfo", "r");
    if (fp == nullptr) {
        return 0;
    }
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp) != nullptr) {
        if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) {
            break;
        }
    }
    std::fclose(fp);
    return kb * 1024u;
}

}

PinnedHostBuffer::~PinnedHostBuffer() noexcept {
    (void)shutdown();
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)),
      name_(other.name_) {}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
        (void)shutdown();
        ptr_ = std::exchange(other.ptr_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        name_ = other.name_;
    }
    return *this;
}

Result<PinnedHostBuffer> PinnedHostBuffer::allocate(std::size_t bytes, const char* name) {
    if (bytes == 0) {
        return PinnedHostBuffer(nullptr, 0, name);
    }
    void* ptr = nullptr;
    const hipError_t err = hipHostMalloc(&ptr, bytes);
    if (err != hipSuccess) {
        return Status::hip_error("hipHostMalloc", hipGetErrorString(err), __FILE__, __LINE__);
    }
    return PinnedHostBuffer(ptr, bytes, name);
}

Status PinnedHostBuffer::shutdown() noexcept {
    if (ptr_ == nullptr) {
        bytes_ = 0;
        return Status::make_ok();
    }
    void* ptr = std::exchange(ptr_, nullptr);
    bytes_ = 0;
    const hipError_t err = hipHostFree(ptr);
    if (err != hipSuccess) {
        return Status::hip_error(name_, hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

PrefixCacheBytes prefix_cache_bytes(
    const PrefixCacheConfig& config,
    const GdnStatePoolLayout& layout,
    const PrefixCacheGeometry& geometry) noexcept {
    PrefixCacheBytes bytes;
    const uint32_t page_tokens = geometry.page_tokens;
    if (page_tokens == 0u || config.max_concurrent == 0u) {
        return bytes;
    }
    bytes.hot_pages = config.hot_capacity_tokens / page_tokens;
    bytes.cold_pages = config.cold_capacity_tokens / page_tokens;
    bytes.cold_state_slots =
        config.cold_state_slots != 0 ? config.cold_state_slots : 4u * config.max_concurrent;

    const std::size_t elem_bytes = geometry.dtype == KVCacheDType::FP8_E4M3
        ? sizeof(fp8e4m3_storage_t)
        : sizeof(bf16_t);
    const std::size_t layers = geometry.num_attention_layers;
    const std::size_t kv_heads = geometry.kv_heads;
    const std::size_t head_dim = geometry.head_dim;
    const std::size_t scale_bytes =
        geometry.dtype == KVCacheDType::FP8_E4M3 ? sizeof(float) : 0u;

    bytes.hot_kv =
        2u * layers * static_cast<std::size_t>(bytes.hot_pages) * page_tokens * kv_heads
            * head_dim * elem_bytes
        + 2u * layers * static_cast<std::size_t>(bytes.hot_pages) * page_tokens * kv_heads
            * scale_bytes;
    bytes.state_slot =
        static_cast<std::size_t>(layout.num_gdn_states) * layout.conv_history
            * layout.conv_history_stride * sizeof(bf16_t)
        + static_cast<std::size_t>(layout.num_gdn_states) * layout.num_v_heads
            * layout.head_k * layout.head_v * sizeof(fp16_t);
    bytes.hot_state =
        bytes.state_slot * 2u * static_cast<std::size_t>(config.max_concurrent);
    bytes.hot_ring = static_cast<std::size_t>(config.spec_ring_bytes)
        * static_cast<std::size_t>(config.spec_ring_bytes > 0 ? config.max_concurrent : 0u);

    const std::size_t kv_layer_page_bytes = page_tokens * kv_heads * head_dim * elem_bytes;
    const std::size_t scale_layer_page_bytes = page_tokens * kv_heads * scale_bytes;
    bytes.cold_page =
        layers * (2u * kv_layer_page_bytes + 2u * scale_layer_page_bytes);
    bytes.cold_kv = bytes.cold_page * static_cast<std::size_t>(bytes.cold_pages);
    bytes.cold_state =
        bytes.state_slot * static_cast<std::size_t>(bytes.cold_state_slots);
    return bytes;
}

Result<PrefixCache> PrefixCache::create(
    gpu::GpuArena& arena,
    const PagedKVPool& active_kv_pool,
    GdnStatePool& active_state_pool,
    const PrefixCacheConfig& config,
    const std::vector<PrefixCacheRankContext>& extra_ranks) {
    if (config.max_concurrent == 0) {
        return Status::invalid_argument("max_concurrent must be > 0", __FILE__, __LINE__);
    }
    if (config.max_concurrent > 0xFFFFu) {
        return Status::invalid_argument(
            "max_concurrent exceeds capture handle space", __FILE__, __LINE__);
    }
    const uint32_t page_tokens = active_kv_pool.page_tokens();
    if (config.hot_capacity_tokens < page_tokens) {
        return Status::invalid_argument(
            "hot_capacity_tokens below one page", __FILE__, __LINE__);
    }

    const GdnStatePoolLayout& layout = active_state_pool.layout();
    const PrefixCacheGeometry geometry{
        .page_tokens = page_tokens,
        .num_attention_layers = active_kv_pool.num_attention_layers(),
        .kv_heads = active_kv_pool.kv_heads(),
        .head_dim = active_kv_pool.head_dim(),
        .dtype = active_kv_pool.dtype(),
    };
    const PrefixCacheBytes bytes = prefix_cache_bytes(config, layout, geometry);
    const uint32_t cold_state_slots = bytes.cold_state_slots;
    const uint32_t hot_pages = bytes.hot_pages;
    const uint32_t cold_pages = bytes.cold_pages;
    const std::size_t cold_page_bytes = bytes.cold_page;
    const std::size_t state_slot_bytes = bytes.state_slot;
    const std::size_t cold_kv_bytes = bytes.cold_kv;
    const std::size_t cold_state_bytes = bytes.cold_state;

    const std::size_t vram_needed = bytes.hot_total();
    const std::size_t vram_free = arena.capacity() - arena.used();
    if (vram_needed > vram_free) {
        return Status::insufficient_memory(
            "prefix cache hot tier needs " + gpu::format_gib(vram_needed)
            + " but only " + gpu::format_gib(vram_free) + " arena capacity free",
            __FILE__, __LINE__);
    }

    const std::size_t cold_needed = cold_kv_bytes + cold_state_bytes;
    if (cold_needed > 0) {
        std::size_t host_free = host_available_bytes();
        if (host_free == 0) {
            const long page_size = sysconf(_SC_PAGESIZE);
            const long avail_pages = sysconf(_SC_AVPHYS_PAGES);
            if (page_size > 0 && avail_pages > 0) {
                host_free = static_cast<std::size_t>(page_size)
                    * static_cast<std::size_t>(avail_pages);
            }
        }
        if (host_free > 0 && cold_needed > host_free) {
            return Status::insufficient_memory(
                "prefix cache cold tier needs " + gpu::format_gib(cold_needed)
                + " but only " + gpu::format_gib(host_free) + " host memory free",
                __FILE__, __LINE__);
        }
    }

    auto shadow_kv = PagedKVPool::create(
        arena,
        hot_pages,
        page_tokens,
        active_kv_pool.num_attention_layers(),
        active_kv_pool.kv_heads(),
        active_kv_pool.head_dim(),
        geometry.dtype);
    if (!shadow_kv.ok()) return shadow_kv.status();

    auto shadow_state = GdnStatePool::create(
        arena,
        2u * config.max_concurrent,
        layout);
    if (!shadow_state.ok()) return shadow_state.status();

    auto cold_kv = PinnedHostBuffer::allocate(cold_kv_bytes, "prefix cache cold KV");
    if (!cold_kv.ok()) return cold_kv.status();

    auto cold_state =
        PinnedHostBuffer::allocate(cold_state_bytes, "prefix cache cold state");
    if (!cold_state.ok()) return cold_state.status();

    const std::size_t ring_slot_bytes = config.spec_ring_bytes;
    const uint32_t ring_slots = ring_slot_bytes > 0 ? config.max_concurrent : 0u;
    auto ring_view = arena.allocate_aligned(
        ring_slot_bytes * static_cast<std::size_t>(ring_slots), 128);
    if (!ring_view.ok()) return ring_view.status();

    PrefixCache cache(shadow_kv.release(), shadow_state.release());
    cache.config_ = config;
    cache.page_tokens_ = page_tokens;
    cache.active_state_pool_ = &active_state_pool;
    cache.cold_page_bytes_ = cold_page_bytes;
    cache.cold_state_bytes_ = state_slot_bytes;
    cache.cold_page_capacity_ = cold_pages;
    cache.cold_state_capacity_ = cold_state_slots;
    cache.cold_kv_buffer_ = cold_kv.release();
    cache.cold_state_buffer_ = cold_state.release();
    cache.ring_base_ = static_cast<std::byte*>(ring_view.release().data());
    cache.ring_slot_bytes_ = ring_slot_bytes;
    cache.ring_slots_ = ring_slots;
    cache.captures_.resize(config.max_concurrent);
    for (const PrefixCacheRankContext& ctx : extra_ranks) {
        if (ctx.active_kv == nullptr || ctx.active_state == nullptr ||
            ctx.arena == nullptr || ctx.stream == nullptr) {
            return Status::invalid_argument(
                "prefix cache rank context is incomplete", __FILE__, __LINE__);
        }
        auto scope = shelf::gpu::ScopedDevice::create(ctx.device);
        if (!scope.ok()) return scope.status();
        auto shard_kv = PagedKVPool::create(
            *ctx.arena, hot_pages, page_tokens, ctx.active_kv->num_attention_layers(),
            ctx.active_kv->kv_heads(), ctx.active_kv->head_dim(), geometry.dtype);
        if (!shard_kv.ok()) return shard_kv.status();
        auto shard_state = GdnStatePool::create(
            *ctx.arena, 2u * config.max_concurrent, ctx.active_state->layout());
        if (!shard_state.ok()) return shard_state.status();
        auto shard = std::make_unique<RankShard>(shard_kv.release(),
                                                shard_state.release());
        shard->active_kv = ctx.active_kv;
        shard->active_state = ctx.active_state;
        shard->stream = ctx.stream;
        shard->device = ctx.device;
        cache.rank_shards_.push_back(std::move(shard));
    }
    cache.nodes_.emplace_back();
    cache.entries_.reserve(
        static_cast<std::size_t>(config.max_concurrent) + cold_state_slots);
    for (uint32_t i = config.max_concurrent; i-- > 0;) {
        cache.rolling_slots_free_.push_back(i);
    }
    for (uint32_t i = 2u * config.max_concurrent; i-- > config.max_concurrent;) {
        cache.hot_state_slots_free_.push_back(i);
    }
    for (uint32_t i = cold_state_slots; i-- > 0;) {
        cache.cold_state_free_.push_back(i);
    }
    for (uint32_t i = cold_pages; i-- > 0;) {
        cache.cold_page_free_.push_back(i);
    }
    for (uint32_t i = ring_slots; i-- > 0;) {
        cache.ring_slots_free_.push_back(i);
    }
    return cache;
}

Status PrefixCache::shutdown() noexcept {
    Status first_error = cold_kv_buffer_.shutdown();
    const Status state_st = cold_state_buffer_.shutdown();
    if (first_error.ok() && !state_st.ok()) first_error = state_st;
    return first_error;
}

bool PrefixCache::find_child(
    uint32_t node,
    const std::vector<uint32_t>& tokens,
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

PrefixCache::WalkResult PrefixCache::walk(const std::vector<uint32_t>& tokens) {
    WalkResult result;
    const uint32_t query_pages = static_cast<uint32_t>(tokens.size() / page_tokens_);
    const uint32_t root = 0;
    nodes_[root].last_used = ++clock_;
    result.visited.push_back(root);
    uint32_t node = root;
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

uint32_t PrefixCache::alloc_node() {
    if (!node_free_.empty()) {
        const uint32_t node = node_free_.back();
        node_free_.pop_back();
        nodes_[node] = RadixNode{};
        return node;
    }
    nodes_.emplace_back();
    return static_cast<uint32_t>(nodes_.size() - 1);
}

void PrefixCache::remove_node(uint32_t node) {
    const uint32_t parent = nodes_[node].parent;
    std::vector<uint32_t>& siblings = nodes_[parent].children;
    siblings.erase(
        std::remove(siblings.begin(), siblings.end(), node),
        siblings.end());
    nodes_[node] = RadixNode{};
    node_free_.push_back(node);
}

void PrefixCache::split_node(uint32_t child, uint32_t split_pages) {
    const uint32_t split = alloc_node();
    RadixNode& tail = nodes_[split];
    RadixNode& head = nodes_[child];
    tail.parent = child;
    tail.depth_pages = head.depth_pages + split_pages;
    tail.edge_tokens.assign(
        head.edge_tokens.begin() + split_pages * page_tokens_,
        head.edge_tokens.end());
    tail.edge_pages.assign(
        head.edge_pages.begin() + split_pages,
        head.edge_pages.end());
    tail.children = std::move(head.children);
    head.children.clear();
    for (uint32_t moved : tail.children) {
        nodes_[moved].parent = split;
    }
    tail.entry_id = head.entry_id;
    head.entry_id = 0;
    if (tail.entry_id != 0) {
        PrefixEntry* entry = find_entry(tail.entry_id);
        if (entry != nullptr) {
            entry->node = split;
        }
    }
    head.edge_tokens.resize(split_pages * page_tokens_);
    head.edge_pages.resize(split_pages);
    tail.last_used = head.last_used;
    head.children.push_back(split);
}

uint32_t PrefixCache::insert_path(
    const std::vector<uint32_t>& tokens,
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
            n.edge_tokens.assign(
                tokens.begin() + consumed * page_tokens_,
                tokens.begin() + page_count * page_tokens_);
            n.edge_pages.assign(
                pages.begin() + consumed,
                pages.begin() + page_count);
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
        n.edge_tokens.assign(
            tokens.begin() + divergence * page_tokens_,
            tokens.begin() + page_count * page_tokens_);
        n.edge_pages.assign(
            pages.begin() + divergence,
            pages.begin() + page_count);
        n.last_used = ++clock_;
        nodes_[parent].children.push_back(fresh);
        return fresh;
    }
    return node;
}

PrefixEntry* PrefixCache::find_entry(uint64_t id) {
    for (PrefixEntry& entry : entries_) {
        if (entry.id == id) return &entry;
    }
    return nullptr;
}

PrefixEntry* PrefixCache::find_lru_entry(bool cold_state) {
    PrefixEntry* best = nullptr;
    for (PrefixEntry& entry : entries_) {
        if (entry.state_cold != cold_state) continue;
        if (best == nullptr || entry.last_used < best->last_used) {
            best = &entry;
        }
    }
    return best;
}

void PrefixCache::remove_entry(uint64_t id) {
    for (std::size_t i = 0; i < entries_.size(); ++i) {
        if (entries_[i].id != id) continue;
        PrefixEntry removed = std::move(entries_[i]);
        if (i + 1 != entries_.size()) {
            entries_[i] = std::move(entries_.back());
        }
        entries_.pop_back();
        if (removed.node < nodes_.size() && nodes_[removed.node].entry_id == id) {
            nodes_[removed.node].entry_id = 0;
        }
        for (PageId page : removed.pages) {
            if (page < page_refcount_.size() && page_refcount_[page] > 0) {
                --page_refcount_[page];
            }
        }
        if (removed.state_cold) {
            cold_state_free_.push_back(removed.state_slot);
        } else {
            hot_state_slots_free_.push_back(removed.state_slot);
        }
        if (removed.ring_slot != kInvalidSlot) {
            ring_slots_free_.push_back(removed.ring_slot);
        }
        return;
    }
}

Result<PageId> PrefixCache::alloc_page() {
    auto physical = shadow_kv_.allocate();
    if (!physical.ok()) return physical.status();
    PageId logical;
    if (!page_free_.empty()) {
        logical = page_free_.back();
        page_free_.pop_back();
    } else {
        logical = static_cast<PageId>(page_location_.size());
        page_location_.emplace_back();
        page_refcount_.push_back(0);
    }
    const uint32_t index = physical.release();
    for (auto& shard : rank_shards_) {
        auto shard_physical = shard->shadow_kv.allocate();
        if (!shard_physical.ok()) {
            (void)shadow_kv_.release(index);
            return shard_physical.status();
        }
        if (shard_physical.value() != index) {
            (void)shadow_kv_.release(index);
            return Status::invalid_state(
                "prefix cache rank shard page allocation diverged", __FILE__, __LINE__);
        }
    }
    page_location_[logical] = PageLocation{.cold = false, .index = index};
    return logical;
}

void PrefixCache::release_page(PageId page) {
    if (page >= page_location_.size()) return;
    const PageLocation loc = page_location_[page];
    if (loc.cold) {
        cold_page_free_.push_back(loc.index);
    } else {
        (void)shadow_kv_.release(loc.index);
        release_shadow(loc.index);
    }
    page_location_[page] = PageLocation{.cold = false, .index = kInvalidNode};
    page_free_.push_back(page);
}

Result<uint32_t> PrefixCache::alloc_cold_page() {
    if (cold_page_free_.empty()) {
        return Status::insufficient_memory(
            "prefix cache cold KV tier exhausted", __FILE__, __LINE__);
    }
    const uint32_t index = cold_page_free_.back();
    cold_page_free_.pop_back();
    return index;
}

Result<uint32_t> PrefixCache::acquire_entry_state_slot(hipStream_t stream) {
    while (hot_state_slots_free_.empty()) {
        const Status st = demote_entry_state_to_cold(stream);
        if (!st.ok()) return st;
    }
    const uint32_t slot = hot_state_slots_free_.back();
    hot_state_slots_free_.pop_back();
    return slot;
}

Status PrefixCache::demote_entry_state_to_cold(hipStream_t stream) {
    if (!rank_shards_.empty()) {
        return Status::unsupported(
            "prefix cache state demotion to cold is not supported with tp rank "
            "shards; raise hot_capacity_tokens",
            __FILE__, __LINE__);
    }
    if (cold_state_free_.empty()) {
        PrefixEntry* cold_lru = find_lru_entry(true);
        if (cold_lru == nullptr) {
            return Status::invalid_state(
                "cold prefix state tier full without entries", __FILE__, __LINE__);
        }
        const uint64_t id = cold_lru->id;
        remove_entry(id);
        ++stats_.entries_evicted;
    }
    PrefixEntry* hot_lru = find_lru_entry(false);
    if (hot_lru == nullptr) {
        return Status::invalid_state(
            "no hot prefix state to demote", __FILE__, __LINE__);
    }
    const uint32_t cold_index = cold_state_free_.back();
    const Status st = copy_state_to_cold(hot_lru->state_slot, cold_index, stream);
    if (!st.ok()) {
        return st;
    }
    cold_state_free_.pop_back();
    const SequenceSlotId hot_slot = hot_lru->state_slot;
    hot_lru->state_cold = true;
    hot_lru->state_slot = cold_index;
    hot_state_slots_free_.push_back(hot_slot);
    ++stats_.states_migrated_cold;
    return Status::make_ok();
}

Result<uint32_t> PrefixCache::acquire_entry_ring_slot() {
    if (ring_slots_free_.empty()) {
        return Status::insufficient_memory(
            "prefix cache spec ring slots exhausted", __FILE__, __LINE__);
    }
    const uint32_t slot = ring_slots_free_.back();
    ring_slots_free_.pop_back();
    return slot;
}

Status PrefixCache::save_entry_ring(
    uint32_t ring_slot,
    const SpecRingRef& ring,
    hipStream_t stream) {
    const std::size_t half = ring_slot_bytes_ / 2u;
    std::byte* base = ring_base_ + static_cast<std::size_t>(ring_slot) * ring_slot_bytes_;
    hipError_t err = hipMemcpyAsync(base, ring.k, half, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("save_entry_ring K", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    err = hipMemcpyAsync(base + half, ring.v, half, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("save_entry_ring V", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status PrefixCache::load_entry_ring(
    uint32_t ring_slot,
    const SpecRingOut& ring,
    hipStream_t stream) {
    const std::size_t half = ring_slot_bytes_ / 2u;
    const std::byte* base =
        ring_base_ + static_cast<std::size_t>(ring_slot) * ring_slot_bytes_;
    hipError_t err = hipMemcpyAsync(ring.k, base, half, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("load_entry_ring K", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    err = hipMemcpyAsync(ring.v, base + half, half, hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("load_entry_ring V", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

void PrefixCache::build_pinned_marks(std::vector<char>& marks) const {
    marks.assign(page_location_.size(), 0);
    const uint32_t pin_count = std::min<uint32_t>(
        config_.max_concurrent, static_cast<uint32_t>(entries_.size()));
    if (pin_count == 0) return;
    std::vector<const PrefixEntry*> by_age;
    by_age.reserve(entries_.size());
    for (const PrefixEntry& entry : entries_) {
        by_age.push_back(&entry);
    }
    std::partial_sort(
        by_age.begin(), by_age.begin() + pin_count, by_age.end(),
        [](const PrefixEntry* a, const PrefixEntry* b) {
            return a->last_used > b->last_used;
        });
    for (uint32_t i = 0; i < pin_count; ++i) {
        for (PageId page : by_age[i]->pages) {
            if (page < marks.size()) marks[page] = 1;
        }
    }
}

Status PrefixCache::migrate_one_leaf(
    const std::vector<uint32_t>& visited,
    const std::vector<char>& pinned_mark,
    bool allow_pinned,
    hipStream_t stream,
    bool& migrated) {
    migrated = false;
    uint32_t best = kInvalidNode;
    uint64_t best_used = 0;
    bool best_pinned = false;
    for (uint32_t node = 1; node < static_cast<uint32_t>(nodes_.size()); ++node) {
        const RadixNode& n = nodes_[node];
        if (n.edge_pages.empty()) continue;
        if (!n.children.empty()) continue;
        if (std::find(visited.begin(), visited.end(), node) != visited.end()) continue;
        bool has_hot = false;
        bool pinned = false;
        for (PageId page : n.edge_pages) {
            if (page < page_location_.size() && !page_location_[page].cold) {
                has_hot = true;
            }
            if (page < pinned_mark.size() && pinned_mark[page] != 0) {
                pinned = true;
            }
        }
        if (!has_hot) continue;
        if (pinned && !allow_pinned) continue;
        if (best == kInvalidNode || n.last_used < best_used) {
            best = node;
            best_used = n.last_used;
            best_pinned = pinned;
        }
    }
    if (best == kInvalidNode) return Status::make_ok();
    for (PageId page : nodes_[best].edge_pages) {
        const PageLocation loc = page_location_[page];
        if (loc.cold) continue;
        auto cold = alloc_cold_page();
        if (!cold.ok()) break;
        const Status st = copy_page_to_cold(page, cold.value(), stream);
        if (!st.ok()) {
            cold_page_free_.push_back(cold.value());
            return st;
        }
        (void)shadow_kv_.release(loc.index);
        release_shadow(loc.index);
        page_location_[page] = PageLocation{.cold = true, .index = cold.release()};
        ++stats_.pages_migrated_cold;
        migrated = true;
    }
    if (migrated && best_pinned) {
        ++stats_.pinned_demoted;
    }
    return Status::make_ok();
}

bool PrefixCache::evict_one_leaf(const std::vector<uint32_t>& visited) {
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
        ++stats_.pages_dropped_cold;
    }
    remove_node(best);
    return true;
}

Status PrefixCache::make_room_for_pages(
    uint32_t needed,
    const std::vector<uint32_t>& visited,
    hipStream_t stream) {
    std::vector<char> pinned_mark;
    build_pinned_marks(pinned_mark);
    if (!rank_shards_.empty() && shadow_kv_.num_free_pages() < needed) {
        return Status::unsupported(
            "prefix cache hot overflow (cold migration) is not supported with "
            "tp rank shards; raise hot_capacity_tokens",
            __FILE__, __LINE__);
    }
    while (shadow_kv_.num_free_pages() < needed) {
        bool migrated = false;
        Status st = migrate_one_leaf(visited, pinned_mark, false, stream, migrated);
        if (!st.ok()) return st;
        if (migrated) continue;
        st = migrate_one_leaf(visited, pinned_mark, true, stream, migrated);
        if (!st.ok()) return st;
        if (migrated) continue;
        if (!evict_one_leaf(visited)) break;
    }
    return Status::make_ok();
}

Status PrefixCache::copy_page_to_cold(
    PageId page,
    uint32_t cold_index,
    hipStream_t stream) const {
    const PageLocation loc = page_location_[page];
    const bool fp8 = shadow_kv_.dtype() == KVCacheDType::FP8_E4M3;
    const std::size_t kv_layer_bytes =
        static_cast<std::size_t>(page_tokens_) * shadow_kv_.kv_heads()
        * shadow_kv_.head_dim()
        * (fp8 ? sizeof(fp8e4m3_storage_t) : sizeof(bf16_t));
    const std::size_t scale_layer_bytes =
        fp8 ? static_cast<std::size_t>(page_tokens_) * shadow_kv_.kv_heads()
                 * sizeof(float)
            : 0u;
    const std::size_t layer_region_bytes = 2u * kv_layer_bytes + 2u * scale_layer_bytes;
    uint8_t* base = static_cast<uint8_t*>(cold_kv_buffer_.data())
        + static_cast<std::size_t>(cold_index) * cold_page_bytes_;
    for (uint32_t layer = 0; layer < shadow_kv_.num_attention_layers(); ++layer) {
        uint8_t* region = base + static_cast<std::size_t>(layer) * layer_region_bytes;
        if (fp8) {
            hipError_t err = hipMemcpyAsync(
                region,
                shadow_kv_.k_view(layer, loc.index).data<fp8e4m3_storage_t>(),
                kv_layer_bytes, hipMemcpyDeviceToHost, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_to_cold K", hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                region + kv_layer_bytes,
                shadow_kv_.v_view(layer, loc.index).data<fp8e4m3_storage_t>(),
                kv_layer_bytes, hipMemcpyDeviceToHost, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_to_cold V", hipGetErrorString(err), __FILE__, __LINE__);
            }
            const std::size_t scale_offset =
                (static_cast<std::size_t>(layer) * shadow_kv_.num_pages() + loc.index)
                * page_tokens_ * shadow_kv_.kv_heads();
            err = hipMemcpyAsync(
                region + 2u * kv_layer_bytes,
                shadow_kv_.k_scale_pool().data<float>() + scale_offset,
                scale_layer_bytes, hipMemcpyDeviceToHost, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_to_cold K scale", hipGetErrorString(err),
                    __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                region + 2u * kv_layer_bytes + scale_layer_bytes,
                shadow_kv_.v_scale_pool().data<float>() + scale_offset,
                scale_layer_bytes, hipMemcpyDeviceToHost, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_to_cold V scale", hipGetErrorString(err),
                    __FILE__, __LINE__);
            }
        } else {
            hipError_t err = hipMemcpyAsync(
                region,
                shadow_kv_.k_view(layer, loc.index).data<bf16_t>(),
                kv_layer_bytes, hipMemcpyDeviceToHost, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_to_cold K", hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                region + kv_layer_bytes,
                shadow_kv_.v_view(layer, loc.index).data<bf16_t>(),
                kv_layer_bytes, hipMemcpyDeviceToHost, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_to_cold V", hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
    }
    return Status::make_ok();
}

Status PrefixCache::copy_page_from_cold(
    const PagedKVPool& dst_pool,
    PageId dst_page,
    uint32_t cold_index,
    hipStream_t stream) const {
    const bool fp8 = dst_pool.dtype() == KVCacheDType::FP8_E4M3;
    const std::size_t kv_layer_bytes =
        static_cast<std::size_t>(page_tokens_) * dst_pool.kv_heads()
        * dst_pool.head_dim()
        * (fp8 ? sizeof(fp8e4m3_storage_t) : sizeof(bf16_t));
    const std::size_t scale_layer_bytes =
        fp8 ? static_cast<std::size_t>(page_tokens_) * dst_pool.kv_heads()
                 * sizeof(float)
            : 0u;
    const std::size_t layer_region_bytes = 2u * kv_layer_bytes + 2u * scale_layer_bytes;
    const uint8_t* base = static_cast<const uint8_t*>(cold_kv_buffer_.data())
        + static_cast<std::size_t>(cold_index) * cold_page_bytes_;
    for (uint32_t layer = 0; layer < dst_pool.num_attention_layers(); ++layer) {
        const uint8_t* region =
            base + static_cast<std::size_t>(layer) * layer_region_bytes;
        if (fp8) {
            hipError_t err = hipMemcpyAsync(
                dst_pool.k_view(layer, dst_page).data<fp8e4m3_storage_t>(),
                region, kv_layer_bytes, hipMemcpyHostToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_from_cold K", hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                dst_pool.v_view(layer, dst_page).data<fp8e4m3_storage_t>(),
                region + kv_layer_bytes, kv_layer_bytes, hipMemcpyHostToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_from_cold V", hipGetErrorString(err), __FILE__, __LINE__);
            }
            const std::size_t dst_scale_offset =
                (static_cast<std::size_t>(layer) * dst_pool.num_pages() + dst_page)
                * page_tokens_ * dst_pool.kv_heads();
            err = hipMemcpyAsync(
                dst_pool.k_scale_pool().data<float>() + dst_scale_offset,
                region + 2u * kv_layer_bytes, scale_layer_bytes,
                hipMemcpyHostToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_from_cold K scale", hipGetErrorString(err),
                    __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                dst_pool.v_scale_pool().data<float>() + dst_scale_offset,
                region + 2u * kv_layer_bytes + scale_layer_bytes, scale_layer_bytes,
                hipMemcpyHostToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_from_cold V scale", hipGetErrorString(err),
                    __FILE__, __LINE__);
            }
        } else {
            hipError_t err = hipMemcpyAsync(
                dst_pool.k_view(layer, dst_page).data<bf16_t>(),
                region, kv_layer_bytes, hipMemcpyHostToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_from_cold K", hipGetErrorString(err), __FILE__, __LINE__);
            }
            err = hipMemcpyAsync(
                dst_pool.v_view(layer, dst_page).data<bf16_t>(),
                region + kv_layer_bytes, kv_layer_bytes, hipMemcpyHostToDevice, stream);
            if (err != hipSuccess) {
                return Status::hip_error(
                    "copy_page_from_cold V", hipGetErrorString(err), __FILE__, __LINE__);
            }
        }
    }
    return Status::make_ok();
}

Status PrefixCache::copy_state_to_cold(
    SequenceSlotId hot_slot,
    uint32_t cold_index,
    hipStream_t stream) const {
    const GdnStatePoolDeviceView view = shadow_state_.device_view();
    const std::size_t conv_bytes = view.conv_slot_stride * sizeof(bf16_t);
    const std::size_t rec_bytes = view.recurrent_slot_stride * sizeof(fp16_t);
    uint8_t* base = static_cast<uint8_t*>(cold_state_buffer_.data())
        + static_cast<std::size_t>(cold_index) * cold_state_bytes_;
    hipError_t err = hipMemcpyAsync(
        base,
        view.conv_base + static_cast<std::size_t>(hot_slot) * view.conv_slot_stride,
        conv_bytes, hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        return Status::hip_error(
            "copy_state_to_cold conv", hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipMemcpyAsync(
        base + conv_bytes,
        view.recurrent_base
            + static_cast<std::size_t>(hot_slot) * view.recurrent_slot_stride,
        rec_bytes, hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        return Status::hip_error(
            "copy_state_to_cold recurrent", hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status PrefixCache::copy_state_from_cold(
    const GdnStatePool& dst_pool,
    SequenceSlotId dst_slot,
    uint32_t cold_index,
    hipStream_t stream) const {
    const GdnStatePoolDeviceView view = dst_pool.device_view();
    const std::size_t conv_bytes = view.conv_slot_stride * sizeof(bf16_t);
    const std::size_t rec_bytes = view.recurrent_slot_stride * sizeof(fp16_t);
    const uint8_t* base = static_cast<const uint8_t*>(cold_state_buffer_.data())
        + static_cast<std::size_t>(cold_index) * cold_state_bytes_;
    hipError_t err = hipMemcpyAsync(
        view.conv_base + static_cast<std::size_t>(dst_slot) * view.conv_slot_stride,
        base, conv_bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error(
            "copy_state_from_cold conv", hipGetErrorString(err), __FILE__, __LINE__);
    }
    err = hipMemcpyAsync(
        view.recurrent_base
            + static_cast<std::size_t>(dst_slot) * view.recurrent_slot_stride,
        base + conv_bytes, rec_bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error(
            "copy_state_from_cold recurrent", hipGetErrorString(err), __FILE__, __LINE__);
    }
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

Status PrefixCache::prepare_save(
    const PagedSequenceState& seq,
    const std::vector<uint32_t>& tokens,
    uint32_t position,
    SavePlan& plan,
    hipStream_t stream) {
    if (static_cast<std::size_t>(position) != tokens.size()) {
        return Status::invalid_argument(
            "save position must equal tokens.size()", __FILE__, __LINE__);
    }
    plan.page_count = position / page_tokens_;
    if (seq.block_table.size() < plan.page_count) {
        return Status::invalid_argument(
            "sequence KV range shorter than save position", __FILE__, __LINE__);
    }
    if (position < config_.min_prefix_tokens || plan.page_count == 0) {
        plan.too_short = true;
        return Status::make_ok();
    }
    plan.walk = walk(tokens);
    plan.shared_pages = std::min(plan.walk.matched_pages, plan.page_count);
    plan.pages.assign(
        plan.walk.path.begin(),
        plan.walk.path.begin() + plan.shared_pages);
    const uint32_t needed = plan.page_count - plan.shared_pages;
    const Status room_st = make_room_for_pages(needed, plan.walk.visited, stream);
    if (!room_st.ok()) return room_st;
    if (shadow_kv_.num_free_pages() < needed) {
        ++stats_.skipped_too_large;
        return Status::insufficient_memory(
            "prefix cache capacity exceeded", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status PrefixCache::fill_pages(
    SavePlan& plan,
    const PagedSequenceState& seq,
    hipStream_t stream) {
    const uint32_t start = plan.shared_pages;
    for (uint32_t i = start; i < plan.page_count; ++i) {
        auto page = alloc_page();
        if (!page.ok()) {
            for (uint32_t j = start; j < i; ++j) {
                release_page(plan.pages[j]);
            }
            plan.pages.resize(start);
            return page.status();
        }
        plan.pages.push_back(page.release());
    }
    for (uint32_t i = start; i < plan.page_count; ++i) {
        const Status st = shadow_kv_.copy_page_from(
            page_location_[plan.pages[i]].index,
            *seq.kv_pool(),
            seq.block_table[i],
            stream);
        if (!st.ok()) {
            for (uint32_t j = start; j < plan.page_count; ++j) {
                release_page(plan.pages[j]);
            }
            plan.pages.resize(start);
            return st;
        }
    }
    if (!rank_shards_.empty()) {
        for (uint32_t r = 0; r < rank_shards_.size(); ++r) {
            const PagedSequenceState* mirror =
                mirror_of(seq, static_cast<uint32_t>(r) + 1u);
            if (mirror == nullptr) {
                return Status::invalid_state(
                    "prefix cache tp mirror sequence missing", __FILE__, __LINE__);
            }
            RankShard& shard = *rank_shards_[r];
            auto scope = shelf::gpu::ScopedDevice::create(shard.device);
            if (!scope.ok()) return scope.status();
            for (uint32_t i = start; i < plan.page_count; ++i) {
                const Status st = shard.shadow_kv.copy_page_from(
                    page_location_[plan.pages[i]].index,
                    *mirror->kv_pool(),
                    mirror->block_table[i],
                    shard.stream);
                if (!st.ok()) {
                    for (uint32_t j = start; j < plan.page_count; ++j) {
                        release_page(plan.pages[j]);
                    }
                    plan.pages.resize(start);
                    return st;
                }
            }
        }
        for (auto& shard : rank_shards_) {
            if (shard->shadow_kv.num_used_pages() != shadow_kv_.num_used_pages()) {
                return Status::invalid_state(
                    "prefix cache rank shard page usage diverged after fill",
                    __FILE__, __LINE__);
            }
        }
    }
    return Status::make_ok();
}

void PrefixCache::register_entry(
    SavePlan& plan,
    uint32_t state_slot,
    const std::vector<uint32_t>& tokens,
    uint32_t ring_slot,
    uint32_t ring_length,
    uint32_t ring_next_position) {
    const uint32_t endpoint = insert_path(tokens, plan.page_count, plan.pages);
    if (nodes_[endpoint].entry_id != 0) {
        remove_entry(nodes_[endpoint].entry_id);
    }
    PrefixEntry entry;
    entry.id = next_entry_id_++;
    entry.position = plan.page_count * page_tokens_;
    entry.node = endpoint;
    entry.state_slot = state_slot;
    entry.ring_slot = ring_slot;
    entry.ring_length = ring_length;
    entry.ring_next_position = ring_next_position;
    entry.last_used = ++clock_;
    entry.pages = plan.pages;
    const uint64_t id = entry.id;
    entries_.push_back(std::move(entry));
    nodes_[endpoint].entry_id = id;
    for (PageId page : plan.pages) {
        if (page < page_refcount_.size()) {
            ++page_refcount_[page];
        }
    }
    ++stats_.saves;
}

Status PrefixCache::set_tp_hook(TpBatchHook* hook) noexcept {
    tp_hook_ = hook;
    return Status::make_ok();
}

PagedSequenceState* PrefixCache::mirror_of(const PagedSequenceState& seq,
                                           uint32_t rank) {
    if (tp_hook_ == nullptr) return nullptr;
    return tp_hook_->on_prefix_rank_sequence(seq, rank);
}

void PrefixCache::release_shadow(uint32_t index) {
    for (auto& shard : rank_shards_) {
        (void)shard->shadow_kv.release(index);
    }
}

PrefixHit PrefixCache::lookup(const std::vector<uint32_t>& tokens) {
    ++stats_.lookups;
    const WalkResult result = walk(tokens);
    PrefixHit hit;
    hit.match_tokens = result.matched_pages * page_tokens_;
    if (result.entry_id != 0) {
        PrefixEntry* entry = find_entry(result.entry_id);
        if (entry != nullptr) {
            hit.entry_id = entry->id;
            hit.state_position = entry->position;
            if (entry->ring_slot != kInvalidSlot) {
                hit.ring_length = entry->ring_length;
                hit.ring_next_position = entry->ring_next_position;
            }
            entry->last_used = ++clock_;
            ++stats_.hits;
            return hit;
        }
    }
    ++stats_.misses;
    return hit;
}

Status PrefixCache::save_input_boundary(
    const PagedSequenceState& seq,
    const std::vector<uint32_t>& tokens,
    uint32_t input_len,
    hipStream_t stream,
    const SpecRingRef* ring) {
    if ((input_len / page_tokens_) * page_tokens_ != seq.position) {
        return Status::invalid_argument(
            "save_input_boundary requires the sequence position to equal the "
            "page aligned save position",
            __FILE__, __LINE__);
    }
    SavePlan plan;
    Status st = prepare_save(seq, tokens, input_len, plan, stream);
    if (!st.ok()) return st;
    if (plan.too_short) {
        ++stats_.skipped_too_short;
        return Status::make_ok();
    }
    auto slot = acquire_entry_state_slot(stream);
    if (!slot.ok()) return slot.status();
    uint32_t ring_slot = kInvalidSlot;
    if (ring != nullptr && ring->valid()) {
        auto acquired = acquire_entry_ring_slot();
        if (!acquired.ok()) {
            hot_state_slots_free_.push_back(slot.value());
            return acquired.status();
        }
        ring_slot = acquired.value();
        st = save_entry_ring(ring_slot, *ring, stream);
        if (!st.ok()) {
            ring_slots_free_.push_back(ring_slot);
            hot_state_slots_free_.push_back(slot.value());
            return st;
        }
    }
    st = fill_pages(plan, seq, stream);
    if (!st.ok()) {
        if (ring_slot != kInvalidSlot) ring_slots_free_.push_back(ring_slot);
        hot_state_slots_free_.push_back(slot.value());
        return st;
    }
    st = capture_active_state(slot.value(), seq, stream);
    if (!st.ok()) {
        for (uint32_t i = plan.shared_pages; i < plan.page_count; ++i) {
            release_page(plan.pages[i]);
        }
        plan.pages.resize(plan.shared_pages);
        if (ring_slot != kInvalidSlot) ring_slots_free_.push_back(ring_slot);
        hot_state_slots_free_.push_back(slot.value());
        return st;
    }
    register_entry(plan, slot.release(), tokens, ring_slot,
                   ring != nullptr ? ring->length : 0u,
                   ring != nullptr ? ring->next_position : 0u);
    ++stats_.input_boundary_saves;
    return Status::make_ok();
}

Result<CaptureId> PrefixCache::begin_capture(const PagedSequenceState& seq) {
    if (!seq.is_usable()) {
        return Status::invalid_argument("sequence is not usable", __FILE__, __LINE__);
    }
    uint32_t index = kInvalidNode;
    for (uint32_t i = 0; i < static_cast<uint32_t>(captures_.size()); ++i) {
        if (!captures_[i].active) {
            index = i;
            break;
        }
    }
    if (index == kInvalidNode) {
        return Status::insufficient_memory("no free capture slots", __FILE__, __LINE__);
    }
    if (rolling_slots_free_.empty()) {
        return Status::insufficient_memory(
            "no free rolling state slots", __FILE__, __LINE__);
    }
    CaptureState& state = captures_[index];
    state.generation = (state.generation % 0xFFFFu) + 1u;
    state.active = true;
    state.seq_slot = seq.slot;
    state.seq_generation = seq.generation;
    state.state_slot = rolling_slots_free_.back();
    rolling_slots_free_.pop_back();
    state.last_boundary = 0;
    ++stats_.captures_started;
    return static_cast<CaptureId>((state.generation << 16) | index);
}

Status PrefixCache::capture_active_state(uint32_t dst_state_slot,
                                         const PagedSequenceState& seq,
                                         hipStream_t stream) {
    Status st = shadow_state_.copy_slot_from(dst_state_slot, *active_state_pool_,
                                             seq.slot, stream);
    if (!st.ok()) return st;
    for (uint32_t r = 0; r < rank_shards_.size(); ++r) {
        const PagedSequenceState* mirror =
            mirror_of(seq, static_cast<uint32_t>(r) + 1u);
        if (mirror == nullptr) {
            return Status::invalid_state(
                "prefix cache tp mirror sequence missing (state capture)", __FILE__,
                __LINE__);
        }
        RankShard& shard = *rank_shards_[r];
        auto scope = shelf::gpu::ScopedDevice::create(shard.device);
        if (!scope.ok()) return scope.status();
        st = shard.shadow_state.copy_slot_from(dst_state_slot, *shard.active_state,
                                               mirror->slot, shard.stream);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Status PrefixCache::roll_state(uint32_t dst_state_slot, uint32_t src_state_slot,
                               hipStream_t stream) {
    Status st = shadow_state_.copy_slot_from(dst_state_slot, shadow_state_,
                                             src_state_slot, stream);
    if (!st.ok()) return st;
    for (auto& shard : rank_shards_) {
        auto scope = shelf::gpu::ScopedDevice::create(shard->device);
        if (!scope.ok()) return scope.status();
        st = shard->shadow_state.copy_slot_from(dst_state_slot, shard->shadow_state,
                                                src_state_slot, shard->stream);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Status PrefixCache::capture_boundary(
    CaptureId capture,
    const PagedSequenceState& seq,
    hipStream_t stream) {
    CaptureState* state = resolve_capture(capture, &seq);
    if (state == nullptr) {
        return Status::invalid_argument("capture handle mismatch", __FILE__, __LINE__);
    }
    const uint32_t boundary = (seq.position / page_tokens_) * page_tokens_;
    if (boundary <= state->last_boundary) {
        return Status::make_ok();
    }
    const Status st = capture_active_state(state->state_slot, seq, stream);
    if (!st.ok()) return st;
    state->last_boundary = boundary;
    return Status::make_ok();
}

Status PrefixCache::cancel_capture(CaptureId capture) {
    CaptureState* state = resolve_capture(capture, nullptr);
    if (state == nullptr) {
        return Status::invalid_argument("invalid capture handle", __FILE__, __LINE__);
    }
    rolling_slots_free_.push_back(state->state_slot);
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
        return Status::invalid_argument("capture handle mismatch", __FILE__, __LINE__);
    }
    SavePlan plan;
    Status st = prepare_save(seq, tokens, position, plan, stream);
    if (!st.ok()) return st;
    if (plan.too_short) {
        rolling_slots_free_.push_back(state->state_slot);
        state->active = false;
        ++stats_.skipped_too_short;
        return Status::make_ok();
    }
    const uint32_t target = plan.page_count * page_tokens_;
    if (state->last_boundary < target) {
        st = capture_active_state(state->state_slot, seq, stream);
        if (!st.ok()) return st;
        state->last_boundary = target;
    }
    st = fill_pages(plan, seq, stream);
    if (!st.ok()) return st;
    auto slot = acquire_entry_state_slot(stream);
    if (!slot.ok()) {
        for (uint32_t i = plan.shared_pages; i < plan.page_count; ++i) {
            release_page(plan.pages[i]);
        }
        plan.pages.resize(plan.shared_pages);
        return slot.status();
    }
    uint32_t ring_slot = kInvalidSlot;
    if (ring != nullptr && ring->valid()) {
        auto acquired = acquire_entry_ring_slot();
        if (!acquired.ok()) {
            for (uint32_t i = plan.shared_pages; i < plan.page_count; ++i) {
                release_page(plan.pages[i]);
            }
            plan.pages.resize(plan.shared_pages);
            hot_state_slots_free_.push_back(slot.value());
            return acquired.status();
        }
        ring_slot = acquired.value();
        st = save_entry_ring(ring_slot, *ring, stream);
        if (!st.ok()) {
            ring_slots_free_.push_back(ring_slot);
            for (uint32_t i = plan.shared_pages; i < plan.page_count; ++i) {
                release_page(plan.pages[i]);
            }
            plan.pages.resize(plan.shared_pages);
            hot_state_slots_free_.push_back(slot.value());
            return st;
        }
    }
    st = roll_state(slot.value(), state->state_slot, stream);
    if (!st.ok()) {
        for (uint32_t i = plan.shared_pages; i < plan.page_count; ++i) {
            release_page(plan.pages[i]);
        }
        plan.pages.resize(plan.shared_pages);
        if (ring_slot != kInvalidSlot) ring_slots_free_.push_back(ring_slot);
        hot_state_slots_free_.push_back(slot.value());
        return st;
    }
    rolling_slots_free_.push_back(state->state_slot);
    state->active = false;
    const uint32_t entry_position = plan.page_count * page_tokens_;
    uint32_t ring_length = 0u;
    uint32_t ring_next_position = 0u;
    if (ring != nullptr && ring->valid()) {
        ring_next_position = std::min(ring->next_position, entry_position);
        ring_length = std::min(ring->length, ring_next_position);
    }
    register_entry(plan, slot.release(), tokens, ring_slot, ring_length,
                   ring_next_position);
    ++stats_.output_tail_saves;
    return Status::make_ok();
}

Status PrefixCache::restore(
    uint64_t entry_id,
    PagedSequenceState& dst,
    hipStream_t stream,
    SpecRingOut* ring) {
    PrefixEntry* entry = find_entry(entry_id);
    if (entry == nullptr) {
        return Status::invalid_argument("prefix entry not found", __FILE__, __LINE__);
    }
    if (!dst.is_usable()) {
        return Status::invalid_argument(
            "destination sequence is not usable", __FILE__, __LINE__);
    }
    if (dst.position != 0 || !dst.block_table.empty()) {
        return Status::invalid_argument(
            "destination sequence must be fresh", __FILE__, __LINE__);
    }
    const uint32_t page_count = entry->position / page_tokens_;

    const Status reserve_st = reserve_sequence_append(dst, entry->position, stream);
    if (!reserve_st.ok()) return reserve_st;

    auto pin = [&]() {
        for (uint32_t i = 0; i < page_count; ++i) {
            if (entry->pages[i] < page_refcount_.size()) {
                ++page_refcount_[entry->pages[i]];
            }
        }
    };
    auto unpin = [&]() {
        for (uint32_t i = 0; i < page_count; ++i) {
            if (entry->pages[i] < page_refcount_.size() && page_refcount_[entry->pages[i]] > 0) {
                --page_refcount_[entry->pages[i]];
            }
        }
    };
    auto rollback = [&]() {
        unpin();
        (void)rollback_sequence_append(dst, 0, stream);
    };

    pin();
    Status st = Status::make_ok();
    for (uint32_t i = 0; i < page_count; ++i) {
        const PageLocation loc = page_location_[entry->pages[i]];
        if (loc.cold) {
            st = copy_page_from_cold(
                *dst.kv_pool(), dst.block_table[i], loc.index, stream);
        } else {
            st = dst.kv_pool()->copy_page_from(
                dst.block_table[i],
                shadow_kv_,
                loc.index,
                stream);
        }
        if (!st.ok()) {
            rollback();
            return st;
        }
    }
    if (entry->state_cold) {
        st = copy_state_from_cold(
            *active_state_pool_, dst.slot, entry->state_slot, stream);
    } else {
        st = active_state_pool_->copy_slot_from(
            dst.slot, shadow_state_, entry->state_slot, stream);
    }
    if (!st.ok()) {
        rollback();
        return st;
    }
    if (!rank_shards_.empty()) {
        for (uint32_t r = 0; r < rank_shards_.size(); ++r) {
            PagedSequenceState* mirror =
                mirror_of(dst, static_cast<uint32_t>(r) + 1u);
            if (mirror == nullptr) {
                rollback();
                return Status::invalid_state(
                    "prefix cache tp mirror sequence missing (restore)", __FILE__,
                    __LINE__);
            }
            RankShard& shard = *rank_shards_[r];
            auto scope = shelf::gpu::ScopedDevice::create(shard.device);
            if (!scope.ok()) {
                rollback();
                return scope.status();
            }
            const Status reserve_st =
                reserve_sequence_append(*mirror, entry->position, shard.stream);
            if (!reserve_st.ok()) {
                (void)rollback_sequence_append(*mirror, 0, shard.stream);
                rollback();
                return reserve_st;
            }
            Status mst = Status::make_ok();
            for (uint32_t i = 0; i < page_count; ++i) {
                const PageLocation loc = page_location_[entry->pages[i]];
                if (loc.cold) {
                    mst = Status::invalid_state(
                        "prefix cache tp restore encountered a cold page", __FILE__,
                        __LINE__);
                    break;
                }
                mst = mirror->kv_pool()->copy_page_from(
                    mirror->block_table[i], shard.shadow_kv, loc.index, shard.stream);
                if (!mst.ok()) break;
            }
            if (mst.ok()) {
                mst = shard.active_state->copy_slot_from(
                    mirror->slot, shard.shadow_state, entry->state_slot, shard.stream);
            }
            if (mst.ok()) {
                mst = commit_sequence_append(*mirror, mirror->request_handle(),
                                             entry->position);
            }
            if (!mst.ok()) {
                (void)rollback_sequence_append(*mirror, 0, shard.stream);
                rollback();
                return mst;
            }
        }
    }
    unpin();

    if (ring != nullptr && ring->valid() && entry->ring_slot != kInvalidSlot) {
        st = load_entry_ring(entry->ring_slot, *ring, stream);
        if (!st.ok()) return st;
        ring->length = entry->ring_length;
        ring->next_position = entry->ring_next_position;
    }

    st = commit_sequence_append(dst, dst.request_handle(), entry->position);
    if (!st.ok()) return st;

    entry->last_used = ++clock_;
    ++stats_.restores;
    return Status::make_ok();
}

PrefixCacheStats PrefixCache::stats() const noexcept {
    PrefixCacheStats snapshot = stats_;
    snapshot.entries = static_cast<uint32_t>(entries_.size());
    snapshot.used_pages = shadow_kv_.num_used_pages();
    uint32_t nodes = 0;
    for (const RadixNode& node : nodes_) {
        if (!node.edge_pages.empty()) ++nodes;
    }
    snapshot.radix_nodes = nodes;
    uint32_t active_captures = 0;
    for (const CaptureState& capture : captures_) {
        if (capture.active) ++active_captures;
    }
    snapshot.captures_active = active_captures;
    snapshot.cold_pages_used = cold_page_capacity_
        - static_cast<uint32_t>(cold_page_free_.size());
    snapshot.cold_states_used = cold_state_capacity_
        - static_cast<uint32_t>(cold_state_free_.size());
    return snapshot;
}

}
}
}
