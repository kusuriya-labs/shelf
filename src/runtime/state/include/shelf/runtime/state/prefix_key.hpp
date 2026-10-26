// shelf runtime/state — prefix-sharing (radix) block identity contract.
//
// The unit of paged-KV sharing is the KV BLOCK, exactly as attention kernels
// address it. A block's IDENTITY includes its whole token prefix: two blocks
// holding the same tokens at different prefixes are different blocks. This
// makes the shared-prefix structure a degenerate radix tree keyed by chained
// block hashes (functional RadixAttention without a trie):
//
//   key(block) = mix(key(parent block), tokens of block)
//
// Pool allocation, refcounts, LRU eviction and copy-on-write on fork are the
// later batching phase; THIS file is the ABI they all speak: the key, and the
// rule that a cache hit is only confirmed by comparing the token span
// (128-bit hash keeps collisions improbable; the token compare makes a wrong
// hit impossible).
#pragma once

#include <cstddef>
#include <compare>
#include <cstdint>

namespace shelf::runtime::state {

struct BlockKey {
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;
    bool valid() const noexcept { return hi != 0 || lo != 0; }
    auto operator<=>(const BlockKey&) const = default;
};

// Genesis key for the first block of a sequence (no parent).
inline constexpr BlockKey kRootBlockKey{0x9E3779B97F4A7C15ull, 0xBF58476D1CE4E5B9ull};

// Deterministic across processes and builds: pure mixing of the parent key
// with the block's token ids (length matters, token ids are int32 as in the
// block table / tokenizer).
BlockKey chain_block_key(const BlockKey& parent, const std::int32_t* tokens,
                         std::size_t count);

} // namespace shelf::runtime::state
