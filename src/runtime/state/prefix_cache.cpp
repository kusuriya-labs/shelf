// shelf runtime/state — Prefix Cache（T16）の実装。
#include <shelf/runtime/state/prefix_cache.hpp>

#include <algorithm>

namespace shelf::runtime::state {

PrefixCache::PrefixCache(std::size_t num_blocks, std::size_t block_tokens)
    : block_tokens_(std::max<std::size_t>(block_tokens, 1u)) {
    blocks_.resize(std::max<std::size_t>(num_blocks, 1u));
}

void PrefixCache::touch(std::uint32_t block) {
    if (valid(block)) blocks_[block].last_used = ++clock_;
}

bool PrefixCache::pick_victim(std::uint32_t* out) const {
    bool found = false;
    std::uint64_t best = 0;
    for (std::uint32_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i].refcount != 0u) continue;
        if (!found || blocks_[i].last_used < best) {
            found = true;
            best = blocks_[i].last_used;
            *out = i;
        }
    }
    return found;
}

void PrefixCache::unregister(std::uint32_t block) {
    Block& b = blocks_[block];
    if (!b.in_map) return;
    by_key_.erase(b.key);
    b.in_map = false;
}

std::vector<PrefixCache::Found> PrefixCache::find(
    const std::vector<std::int32_t>& tokens) const {
    std::vector<Found> out;
    BlockKey parent = kRootBlockKey;
    const std::size_t full = tokens.size() / block_tokens_;
    for (std::size_t b = 0; b < full; ++b) {
        const std::int32_t* span = tokens.data() + b * block_tokens_;
        const BlockKey key = chain_block_key(parent, span, block_tokens_);
        const auto it = by_key_.find(key);
        if (it == by_key_.end()) break;
        const std::uint32_t id = it->second.block;
        if (!valid(id)) break;
        const Block& blk = blocks_[id];
        // 世代が違う・登録が外れた・トークンが違う（衝突）なら miss。
        if (it->second.generation != blk.generation || !blk.in_map) break;
        if (blk.tokens.size() != block_tokens_ ||
            !std::equal(span, span + block_tokens_, blk.tokens.begin()))
            break;
        out.push_back(Found{id, key});
        parent = key;
    }
    return out;
}

PrefixHit PrefixCache::lookup(const std::vector<std::int32_t>& tokens) const {
    const auto found = find(tokens);
    PrefixHit hit;
    for (const auto& f : found) hit.blocks.push_back(f.block);
    hit.matched_tokens = hit.blocks.size() * block_tokens_;
    if (hit.blocks.empty()) {
        ++misses_;
    } else {
        ++hits_;
    }
    return hit;
}

Result<PrefixHit> PrefixCache::acquire(const std::vector<std::int32_t>& tokens) {
    const auto found = find(tokens);
    PrefixHit hit;
    for (const auto& f : found) {
        hit.blocks.push_back(f.block);
        ++blocks_[f.block].refcount;
        touch(f.block);
    }
    hit.matched_tokens = hit.blocks.size() * block_tokens_;
    if (hit.blocks.empty()) {
        ++misses_;
    } else {
        ++hits_;
    }
    return hit;
}

Result<std::uint32_t> PrefixCache::allocate() {
    std::uint32_t victim = 0;
    if (!pick_victim(&victim))
        return Status::insufficient_memory("prefix cache: 空きブロックが無い", __FILE__,
                                           __LINE__);
    Block& b = blocks_[victim];
    unregister(victim);
    b.refcount = 1;
    b.generation += 1;   // 再利用で古い key を無効化
    b.key = BlockKey{};
    b.tokens.assign(block_tokens_, 0);
    touch(victim);
    return victim;
}

Status PrefixCache::insert(std::uint32_t block, const std::vector<std::int32_t>& tokens,
                           std::size_t block_index) {
    if (!valid(block))
        return Status::invalid_argument("prefix cache: block が範囲外", __FILE__, __LINE__);
    if (blocks_[block].in_map)
        return Status::invalid_state("prefix cache: block は登録済み", __FILE__, __LINE__);
    const std::size_t need = (block_index + 1u) * block_tokens_;
    if (tokens.size() < need)
        return Status::invalid_argument("prefix cache: tokens が短い", __FILE__, __LINE__);
    // 親から鎖状に key を作る（block_index まで）。
    BlockKey parent = kRootBlockKey;
    for (std::size_t b = 0; b < block_index; ++b)
        parent = chain_block_key(parent, tokens.data() + b * block_tokens_, block_tokens_);
    const std::int32_t* span = tokens.data() + block_index * block_tokens_;
    Block& blk = blocks_[block];
    blk.key = chain_block_key(parent, span, block_tokens_);
    blk.tokens.assign(span, span + block_tokens_);
    blk.in_map = true;
    by_key_[blk.key] = Entry{block, blk.generation};
    touch(block);
    return Status::make_ok();
}

void PrefixCache::release(std::uint32_t block) {
    if (!valid(block)) return;
    Block& b = blocks_[block];
    if (b.refcount != 0u) --b.refcount;
    touch(block);
}

void PrefixCache::release(const std::vector<std::uint32_t>& blocks) {
    for (const auto id : blocks) release(id);
}

Result<std::uint32_t> PrefixCache::fork(std::uint32_t block, bool* copied) {
    if (copied == nullptr)
        return Status::invalid_argument("prefix cache: copied が null", __FILE__, __LINE__);
    if (!valid(block) || blocks_[block].refcount == 0u)
        return Status::invalid_argument("prefix cache: block が参照されていない", __FILE__,
                                        __LINE__);
    if (blocks_[block].refcount == 1u) {
        *copied = false;
        return block;
    }
    auto fresh = allocate();
    if (!fresh.ok()) return fresh.status();
    const std::uint32_t nb = fresh.value();
    // トークン列は引き継ぐ（key は private なので登録しない）。
    blocks_[nb].tokens = blocks_[block].tokens;
    release(block);   // 旧ブロックの参照を 1 つ手放す
    *copied = true;
    return nb;
}

std::size_t PrefixCache::free_blocks() const noexcept {
    std::size_t n = 0;
    for (const auto& b : blocks_)
        if (b.refcount == 0u) ++n;
    return n;
}

std::size_t PrefixCache::referenced_blocks() const noexcept {
    std::size_t n = 0;
    for (const auto& b : blocks_)
        if (b.refcount != 0u) ++n;
    return n;
}

std::size_t PrefixCache::cached_blocks() const noexcept {
    std::size_t n = 0;
    for (const auto& b : blocks_)
        if (b.refcount == 0u && b.in_map) ++n;
    return n;
}

std::size_t PrefixCache::refcount(std::uint32_t block) const {
    return valid(block) ? blocks_[block].refcount : 0u;
}

std::uint64_t PrefixCache::generation(std::uint32_t block) const {
    return valid(block) ? blocks_[block].generation : 0u;
}

bool PrefixCache::registered(std::uint32_t block) const {
    return valid(block) && blocks_[block].in_map;
}

const std::int32_t* PrefixCache::block_tokens(std::uint32_t block) const {
    if (!valid(block) || !blocks_[block].in_map) return nullptr;
    return blocks_[block].tokens.data();
}

}  // namespace shelf::runtime::state
