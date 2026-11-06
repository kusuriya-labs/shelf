#include <shelf/core/memory/arena.h>
#include <shelf/inference/state/ple_conv_state_pool.h>
#include <shelf/inference/state/qsa_state_pool.h>

#include <cstdio>
#include <cstdint>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    } else {
        std::fprintf(stderr, "ok: %s\n", what);
    }
}

}  // namespace

int main() {
    constexpr std::uint32_t kSequences = 7u;
    constexpr std::uint32_t kQsaStates = 12u;
    constexpr std::uint32_t kQsaHeadDim = 128u;
    constexpr std::uint32_t kCompress = 4u;
    constexpr std::uint32_t kPleStates = 12u;
    constexpr std::uint32_t kPleChannels = 10240u;

    const auto qsa_layout = shelf::state::QsaKeyStatePoolLayout::make(
        kQsaStates, kQsaHeadDim, kCompress, 64u);
    const auto ple_layout = shelf::state::PleConvStatePoolLayout::make(kPleStates,
                                                                       kPleChannels);
    const auto qsa_bytes = shelf::state::qsa_key_state_pool_bytes(kSequences, qsa_layout);
    const std::size_t ple_bytes =
        shelf::state::ple_conv_state_pool_bytes(kSequences, ple_layout);

    check(qsa_bytes.pooled % shelf::state::kQsaKeyStatePoolAlignment == 0u &&
              qsa_bytes.partial % shelf::state::kQsaKeyStatePoolAlignment == 0u &&
              qsa_bytes.counts % shelf::state::kQsaKeyStatePoolAlignment == 0u,
          "qsa pool byte parts are alignment sized");
    check(ple_bytes % shelf::state::kPleConvStatePoolAlignment == 0u,
          "ple conv pool bytes are alignment sized");
    check(qsa_bytes.pooled >=
              static_cast<std::size_t>(kSequences) * kQsaStates * 64u * kQsaHeadDim *
                  sizeof(shelf::bf16_t),
          "qsa pooled bytes cover the elements");
    check(qsa_bytes.partial >=
              static_cast<std::size_t>(kSequences) * kQsaStates * kCompress * kQsaHeadDim *
                  sizeof(shelf::bf16_t),
          "qsa partial bytes cover the elements");
    check(qsa_bytes.counts >
              static_cast<std::size_t>(kSequences) * kQsaStates * 2u * sizeof(std::uint32_t),
          "qsa count bytes round the element bytes up to the pool alignment");

    auto arena = shelf::gpu::GpuArena::create(0, 128ull * 1024ull * 1024ull);
    check(arena.ok(), "arena create");
    if (!arena.ok()) return 1;

    const std::size_t used_before = arena.value().used();
    auto qsa_pool = shelf::state::QsaKeyStatePool::create(arena.value(), kSequences, qsa_layout);
    check(qsa_pool.ok(), "qsa state pool create");
    if (!qsa_pool.ok()) return 1;
    check(arena.value().used() - used_before == qsa_bytes.total(),
          "the qsa pool takes exactly the reserved bytes from the arena");
    check(qsa_pool.value().reserved_bytes() == qsa_bytes.total(),
          "the qsa pool reports the reserved bytes");

    const std::size_t used_with_qsa = arena.value().used();
    auto ple_pool = shelf::state::PleConvStatePool::create(arena.value(), kSequences, ple_layout);
    check(ple_pool.ok(), "ple conv state pool create");
    if (!ple_pool.ok()) return 1;
    check(arena.value().used() - used_with_qsa == ple_bytes,
          "the ple conv pool takes exactly the reserved bytes from the arena");
    check(ple_pool.value().reserved_bytes() == ple_bytes,
          "the ple conv pool reports the reserved bytes");

    check(arena.value().used() == used_before + qsa_bytes.total() + ple_bytes,
          "both pools together take exactly the reserved bytes");

    (void)ple_pool.release();
    (void)qsa_pool.release();
    auto shutdown = arena.value().shutdown();
    check(shutdown.ok(), "arena shutdown");

    std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
