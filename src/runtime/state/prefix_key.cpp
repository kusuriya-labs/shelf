// shelf runtime/state — chained prefix block key (SplitMix64 finalizers).
#include <shelf/runtime/state/prefix_key.hpp>

namespace shelf::runtime::state {
namespace {

constexpr std::uint64_t kSplitMixA = 0xBF58476D1CE4E5B9ull;
constexpr std::uint64_t kSplitMixB = 0x94D049BB133111EBull;
constexpr std::uint64_t kGolden = 0x9E3779B97F4A7C15ull;

inline std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 30;
    x *= kSplitMixA;
    x ^= x >> 27;
    x *= kSplitMixB;
    x ^= x >> 31;
    return x;
}

} // namespace

BlockKey chain_block_key(const BlockKey& parent, const std::int32_t* tokens,
                         std::size_t count) {
    // Feed the token stream, then the LENGTH, into the parent key state.
    // Length finalization distinguishes prefix-extended token streams that
    // would otherwise fold identically at boundaries.
    std::uint64_t h0 = parent.hi;
    std::uint64_t h1 = parent.lo ^ (count * kGolden);
    for (std::size_t i = 0; i < count; ++i) {
        h0 = mix(h0 ^ static_cast<std::uint64_t>(static_cast<std::int32_t>(tokens[i]))
                     ^ ((i + 1) * kGolden));
        if ((i & 1u) == 1u) h1 = mix(h1 ^ h0);
    }
    h1 = mix(h1 ^ h0);
    h0 = mix(h0 ^ h1);
    return BlockKey{h0, h1};
}

} // namespace shelf::runtime::state
