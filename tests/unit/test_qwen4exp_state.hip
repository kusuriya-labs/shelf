#include <shelf/core/memory/arena.h>
#include <shelf/kernels/optimized/attention/qsa_indexer.h>
#include <shelf/models/qwen4exp/ple/ngram_hash.h>
#include <shelf/models/qwen4exp/ple/ple_conv.h>
#include <shelf/models/qwen4exp/runtime/qwen4exp_state.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace q4rt = ps::qwen4exp::runtime;
namespace ple = ps::qwen4exp::ple;

static_assert(q4rt::kIndexerKeyDim
                  == ps::kernel::kQsaIndexerHeadDim * ps::kernel::kQsaIndexerKvHeads,
              "indexer key dim must match the Q4c kernel contract");

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++passed;
        printf("PASS: %s\n", msg);
    } else {
        ++failed;
        printf("FAIL: %s\n", msg);
    }
}

namespace {

constexpr std::uint32_t kSlots = 2;
constexpr std::uint32_t kLayers = 3;
constexpr std::uint32_t kTokens = 16;
constexpr std::uint32_t kEos = 151645u;
constexpr std::uint32_t kConvDim = 10240;
constexpr std::size_t kArenaBytes = 64ull << 20;
constexpr std::size_t kIndexerSlotBytes =
    static_cast<std::size_t>(kLayers) * kTokens * q4rt::kIndexerKeyDim * sizeof(ps::bf16_t);

struct Rng {
    std::uint64_t state;

    explicit Rng(std::uint64_t seed) : state(seed) {}

    float next() {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        const float u = static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
        return -1.0f + 2.0f * u;
    }
};

ps::bf16_t key_pattern(std::uint32_t slot, std::uint32_t layer, std::uint32_t token,
                       std::uint32_t dim) {
    const std::uint32_t mixed =
        slot * 7919u + layer * 131u + token * 257u + dim * 17u + 1u;
    ps::bf16_t v;
    v.data = static_cast<std::uint16_t>(1u + (mixed % 60000u));
    return v;
}

bool region_is_zero(const void* device_ptr, std::size_t bytes, std::vector<std::uint8_t>& host) {
    host.resize(bytes);
    if (hipMemcpy(host.data(), device_ptr, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }
    for (std::size_t i = 0; i < bytes; ++i) {
        if (host[i] != 0u) return false;
    }
    return true;
}

void fill_keys(q4rt::IndexerKeyPool& pool, std::uint32_t slot, std::uint32_t layer,
               std::uint32_t token) {
    ps::bf16_t row[q4rt::kIndexerKeyDim];
    for (std::uint32_t d = 0; d < q4rt::kIndexerKeyDim; ++d) {
        row[d] = key_pattern(slot, layer, token, d);
    }
    (void)hipMemcpy(pool.key_row(slot, layer, token), row, sizeof(row), hipMemcpyHostToDevice);
}

void test_indexer(ps::gpu::GpuArena& arena, hipStream_t stream) {
    auto created = q4rt::IndexerKeyPool::create(arena, kSlots, kLayers, kTokens);
    if (!created.ok()) {
        check(false, "idx: create");
        return;
    }
    q4rt::IndexerKeyPool& pool = created.value();

    check(pool.reserved_bytes() == kIndexerSlotBytes * kSlots,
          "idx: capacity is slots x layers x tokens x 128 bf16");
    check(pool.bytes_per_slot() == kIndexerSlotBytes, "idx: per-slot byte contract");
    check(pool.key_dim() == 128, "idx: key dim 128");
    check(pool.device() == 0, "idx: physical device recorded from arena");
    check(pool.max_sequences() == kSlots && pool.num_layers() == kLayers &&
              pool.max_tokens() == kTokens,
          "idx: geometry exposed");

    bool offset_ok = true;
    const ps::bf16_t* base = pool.key_memory().data<ps::bf16_t>();
    for (std::uint32_t s = 0; s < kSlots; ++s) {
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            const std::size_t want =
                (static_cast<std::size_t>(s) * kLayers + l) * kTokens * q4rt::kIndexerKeyDim;
            if (pool.key_elements_offset(s, l) != want) offset_ok = false;
        }
    }
    check(offset_ok, "idx: offset equals slot-major then layer-major layout");

    check(!pool.is_initialized(0), "idx: slot starts uninitialized");
    check(pool.key_cache(0, 0) == nullptr, "idx: no kernel pointer before initialize");
    check(pool.initialize(0, stream).ok(), "idx: initialize slot0");
    check(!pool.initialize(0, stream).ok(), "idx: double initialize rejected");
    check(pool.is_initialized(0) && pool.num_used_slots() == 1,
          "idx: initialize marks the slot used");
    (void)hipStreamSynchronize(stream);
    std::vector<std::uint8_t> host;
    check(region_is_zero(base, kIndexerSlotBytes, host),
          "idx: initialize zero-fills the whole slot");

    ps::kernel::QsaIndexerArgs args;
    args.key_cache = pool.key_cache(0, 1);
    check(args.key_cache == base + pool.key_elements_offset(0, 1),
          "idx: pointer handed to launch_qsa_indexer matches offset");

    std::vector<ps::bf16_t> expected(static_cast<std::size_t>(kTokens) * q4rt::kIndexerKeyDim);

    for (std::uint32_t t = 0; t < 8; ++t) {
        fill_keys(pool, 0, 1, t);
        for (std::uint32_t d = 0; d < q4rt::kIndexerKeyDim; ++d) {
            expected[static_cast<std::size_t>(t) * q4rt::kIndexerKeyDim + d] =
                key_pattern(0, 1, t, d);
        }
    }
    (void)hipStreamSynchronize(stream);

    for (std::uint32_t t = 8; t < 12; ++t) {
        fill_keys(pool, 0, 1, t);
        for (std::uint32_t d = 0; d < q4rt::kIndexerKeyDim; ++d) {
            expected[static_cast<std::size_t>(t) * q4rt::kIndexerKeyDim + d] =
                key_pattern(0, 1, t, d);
        }
    }
    (void)hipStreamSynchronize(stream);

    for (std::uint32_t t = 12; t < kTokens; ++t) {
        fill_keys(pool, 0, 1, t);
        for (std::uint32_t d = 0; d < q4rt::kIndexerKeyDim; ++d) {
            expected[static_cast<std::size_t>(t) * q4rt::kIndexerKeyDim + d] =
                key_pattern(0, 1, t, d);
        }
    }
    (void)hipStreamSynchronize(stream);

    std::vector<ps::bf16_t> readback(expected.size());
    if (hipMemcpy(readback.data(), pool.key_cache(0, 1), expected.size() * sizeof(ps::bf16_t),
                  hipMemcpyDeviceToHost)
        != hipSuccess) {
        check(false, "idx: readback of layer1 key cache");
        return;
    }
    check(std::memcmp(readback.data(), expected.data(),
                      expected.size() * sizeof(ps::bf16_t))
              == 0,
          "idx: prefill chunks and decode rows survive in readback");

    bool isolation_ok = true;
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        if (l == 1) continue;
        if (!region_is_zero(base + pool.key_elements_offset(0, l),
                            static_cast<std::size_t>(kTokens) * q4rt::kIndexerKeyDim
                                * sizeof(ps::bf16_t),
                            host)) {
            isolation_ok = false;
        }
    }
    if (!region_is_zero(base + pool.key_elements_offset(1, 1),
                        static_cast<std::size_t>(kTokens) * q4rt::kIndexerKeyDim
                            * sizeof(ps::bf16_t),
                        host)) {
        isolation_ok = false;
    }
    check(isolation_ok, "idx: other layers and slots stay untouched");

    check(pool.reset(0, stream).ok(), "idx: reset on cancel clears the request");
    (void)hipStreamSynchronize(stream);
    check(region_is_zero(base, kIndexerSlotBytes, host),
          "idx: cancel reset zero-fills every layer of the slot");
    check(pool.is_initialized(0) && pool.num_used_slots() == 1,
          "idx: reset keeps slot ownership");

    check(pool.release(0).ok(), "idx: release at EOS disposal");
    check(!pool.is_initialized(0) && pool.num_used_slots() == 0,
          "idx: release frees the slot");
    check(pool.key_cache(0, 1) == nullptr && pool.key_row(0, 1, 0) == nullptr,
          "idx: pointers unavailable after release");

    check(pool.initialize(0, stream).ok(), "idx: slot reuse re-initializes");
    (void)hipStreamSynchronize(stream);
    for (std::uint32_t t = 0; t < 4; ++t) fill_keys(pool, 0, 1, t);
    (void)hipStreamSynchronize(stream);
    check(region_is_zero(base + pool.key_elements_offset(0, 1)
                             + static_cast<std::size_t>(4) * q4rt::kIndexerKeyDim,
                         static_cast<std::size_t>(kTokens - 4) * q4rt::kIndexerKeyDim
                             * sizeof(ps::bf16_t),
                         host),
          "idx: reused slot carries no residue beyond written rows");

    check(pool.shutdown().ok(), "idx: shutdown");
    check(pool.shutdown().ok(), "idx: double shutdown is a no-op");
    check(pool.key_cache(0, 1) == nullptr && pool.key_row(0, 1, 0) == nullptr,
          "idx: shutdown clears pointers");
    check(!pool.initialize(0, stream).ok() && !pool.reset(0, stream).ok() &&
              !pool.release(0).ok(),
          "idx: operations after shutdown are rejected");
}

void test_conv() {
    auto created = q4rt::PleConvStatePool::create(kSlots, kConvDim);
    if (!created.ok()) {
        check(false, "conv: create");
        return;
    }
    q4rt::PleConvStatePool& pool = created.value();

    check(pool.conv_history() == ple::kPleConvStateLen && pool.conv_history() == 9,
          "conv: conv_history is 9 from G4a");
    check(pool.conv_dim() == kConvDim, "conv: conv_dim is the PLE hidden dim");
    check(pool.state(0) == nullptr, "conv: no state before initialize");

    Rng rng(1234u);
    std::vector<float> weight(static_cast<std::size_t>(kConvDim) * ple::kPleConvKernel);
    for (float& v : weight) v = rng.next() * 0.02f;
    constexpr std::uint32_t kTokensTotal = 12;
    std::vector<float> input(static_cast<std::size_t>(kTokensTotal) * kConvDim);
    for (float& v : input) v = rng.next();

    std::vector<float> full(static_cast<std::size_t>(kTokensTotal) * kConvDim, 0.0f);
    std::vector<float> chunked(static_cast<std::size_t>(kTokensTotal) * kConvDim, 0.0f);

    check(!pool.convolve(0, weight.data(), input.data(), 1, full.data()).ok(),
          "conv: uninitialized slot rejected");
    check(pool.initialize(0).ok(), "conv: initialize slot0");
    check(!pool.initialize(0).ok(), "conv: double initialize rejected");
    check(pool.is_initialized(0) && pool.num_used_slots() == 1,
          "conv: initialize marks the slot used");

    check(pool.initialize(1).ok(), "conv: initialize slot1");
    const ple::PleConvState* st = pool.state(0);
    check(st != nullptr && st->buffer.size()
                              == static_cast<std::size_t>(kConvDim) * ple::kPleConvStateLen,
          "conv: owned state allocates dim x 9 history");

    check(pool.convolve(0, weight.data(), input.data(), kTokensTotal, full.data()).ok(),
          "conv: full prefill run");

    const std::uint32_t cuts[2] = {5, 3};
    std::uint32_t begin = 0;
    bool chunks_ok = true;
    for (std::uint32_t cut : cuts) {
        chunks_ok = chunks_ok
            && pool
                   .convolve(1, weight.data(), input.data() + static_cast<std::size_t>(begin) * kConvDim,
                             cut, chunked.data() + static_cast<std::size_t>(begin) * kConvDim)
                   .ok();
        begin += cut;
    }
    check(chunks_ok, "conv: prefill chunk boundary runs through the owner");

    bool decode_ok = true;
    for (; begin < kTokensTotal; ++begin) {
        decode_ok = decode_ok
            && pool
                   .convolve(1, weight.data(), input.data() + static_cast<std::size_t>(begin) * kConvDim,
                             1, chunked.data() + static_cast<std::size_t>(begin) * kConvDim)
                   .ok();
    }
    check(decode_ok, "conv: one-token decode continuation runs through the owner");
    check(full == chunked, "conv: chunked and decode path equals full prefill bitwise");

    const ple::PleConvState* st0 = pool.state(0);
    const ple::PleConvState* st1 = pool.state(1);
    check(st0 != nullptr && st1 != nullptr && st0->buffer == st1->buffer,
          "conv: history continues across chunks");

    check(pool.reset(1).ok(), "conv: reset on cancel");
    const ple::PleConvState* st1_after = pool.state(1);
    bool cleared = st1_after != nullptr;
    if (cleared) {
        for (float v : st1_after->buffer) {
            if (v != 0.0f) {
                cleared = false;
                break;
            }
        }
    }
    check(cleared, "conv: cancel reset clears every history tap");

    Rng rng_b(555u);
    std::vector<float> input_b(static_cast<std::size_t>(7) * kConvDim);
    for (float& v : input_b) v = rng_b.next();

    std::vector<float> out_b(7ull * kConvDim, 0.0f);
    check(pool.convolve(1, weight.data(), input_b.data(), 7, out_b.data()).ok(),
          "conv: request B runs after cancel");

    ple::PleConvState fresh;
    fresh.reset(kConvDim);
    std::vector<float> out_b_fresh(7ull * kConvDim, 0.0f);
    check(ple::ple_dilated_conv1d(weight.data(), input_b.data(), 7, kConvDim, fresh,
                                  out_b_fresh.data())
              .ok(),
          "conv: reference fresh run");
    check(out_b == out_b_fresh, "conv: canceled slot leaves no residue");

    check(pool.release(0).ok(), "conv: release at EOS disposal");
    check(pool.state(0) == nullptr && pool.num_used_slots() == 1,
          "conv: release drops ownership");
    check(pool.initialize(0).ok(), "conv: slot reuse re-initializes");
    std::vector<float> out_b2(7ull * kConvDim, 0.0f);
    check(pool.convolve(0, weight.data(), input_b.data(), 7, out_b2.data()).ok(),
          "conv: reused slot runs request B");
    check(out_b2 == out_b_fresh, "conv: reused slot carries no residue");

    check(pool.shutdown().ok(), "conv: shutdown");
    check(pool.shutdown().ok(), "conv: double shutdown is a no-op");
    check(pool.state(1) == nullptr && !pool.convolve(1, weight.data(), input_b.data(), 1,
                                                     out_b2.data())
                                           .ok(),
          "conv: operations after shutdown are rejected");
}

void test_ngram(ps::gpu::GpuArena& arena, hipStream_t stream) {
    ple::NgramHashParams params;
    ple::NgramIdTables tables;
    if (!ple::build_ngram_id_tables(params, tables).ok()) {
        check(false, "ng: build id tables");
        return;
    }

    auto created = q4rt::PleNgramContext::create(arena, kSlots, kEos);
    if (!created.ok()) {
        check(false, "ng: create");
        return;
    }
    q4rt::PleNgramContext& ctx = created.value();

    check(ctx.max_sequences() == kSlots && ctx.eos_token() == kEos, "ng: geometry exposed");
    check(ctx.context(0) == nullptr && ctx.device_context(0) == nullptr,
          "ng: no context before initialize");
    check(!ctx.upload(0, stream).ok(), "ng: upload before initialize rejected");
    check(!ctx.advance(1, nullptr, 1).ok(), "ng: advance before initialize rejected");

    check(ctx.initialize(0).ok(), "ng: initialize slot0");
    check(!ctx.initialize(0).ok(), "ng: double initialize rejected");
    const std::uint32_t* c0 = ctx.context(0);
    check(c0 != nullptr && c0[0] == kEos && c0[1] == kEos,
          "ng: initial context is [eos, eos]");

    check(ctx.advance(0, nullptr, 0).ok(), "ng: zero-length advance is a no-op");

    const std::uint32_t input[5] = {101, 102, 103, 104, 105};
    std::uint64_t full[5 * ple::kNgramHeads];
    check(ple::compute_ngram_ids(tables, c0, 2, input, 5, kEos, full).ok(),
          "ng: reference full prefill");

    std::uint64_t part1[2 * ple::kNgramHeads];
    check(ple::compute_ngram_ids(tables, ctx.context(0), 2, input, 2, kEos, part1).ok(),
          "ng: first prefill chunk runs");
    check(ctx.advance(0, input, 2).ok(), "ng: advance after first chunk");

    std::uint64_t part2[2 * ple::kNgramHeads];
    check(ple::compute_ngram_ids(tables, ctx.context(0), 2, input + 2, 2, kEos, part2).ok(),
          "ng: second prefill chunk runs");
    check(ctx.advance(0, input + 2, 2).ok(), "ng: advance after second chunk");

    std::uint64_t step[ple::kNgramHeads];
    check(ple::compute_ngram_ids(tables, ctx.context(0), 2, input + 4, 1, kEos, step).ok(),
          "ng: decode token runs");
    check(ctx.advance(0, input + 4, 1).ok(), "ng: advance after decode token");

    bool ids_ok = std::memcmp(part1, full, sizeof(part1)) == 0
        && std::memcmp(part2, full + 2 * ple::kNgramHeads, sizeof(part2)) == 0
        && std::memcmp(step, full + 4 * ple::kNgramHeads, sizeof(step)) == 0;
    check(ids_ok, "ng: chunked and decode ids equal the full prefill");

    const std::uint32_t* after = ctx.context(0);
    check(after[0] == input[3] && after[1] == input[4],
          "ng: context holds the last two tokens");

    check(ctx.upload(0, stream).ok(), "ng: upload to the transfer buffer");
    (void)hipStreamSynchronize(stream);
    const std::uint32_t* dev0 = ctx.device_context(0);
    check(dev0 == ctx.device_memory().data<std::uint32_t>(),
          "ng: device context row offset for slot0");
    std::uint32_t dev_host[ple::kNgramContextTokens];
    check(hipMemcpy(dev_host, dev0, sizeof(dev_host), hipMemcpyDeviceToHost) == hipSuccess,
          "ng: readback of the device transfer buffer");
    check(dev_host[0] == after[0] && dev_host[1] == after[1],
          "ng: device buffer mirrors the host context");

    check(ctx.initialize(1).ok(), "ng: initialize slot1");
    check(ctx.device_context(1)
              == ctx.device_memory().data<std::uint32_t>() + ple::kNgramContextTokens,
          "ng: device context row offset for slot1");
    const std::uint32_t* c1 = ctx.context(1);
    check(c1[0] == kEos && c1[1] == kEos, "ng: fresh slot starts at [eos, eos]");

    check(ctx.reset(0).ok(), "ng: reset on cancel");
    const std::uint32_t* c0_after_reset = ctx.context(0);
    check(c0_after_reset[0] == kEos && c0_after_reset[1] == kEos,
          "ng: cancel reset restores the initial context");

    check(ctx.advance(0, input, 2).ok(), "ng: advance before the EOS token");
    const std::uint32_t eos_token[1] = {kEos};
    check(ctx.advance(0, eos_token, 1).ok(), "ng: EOS token advances the context");
    const std::uint32_t* c0_eos = ctx.context(0);
    check(c0_eos[1] == kEos, "ng: EOS token lands in the context tail");

    check(ctx.release(0).ok(), "ng: release at EOS disposal");
    check(ctx.context(0) == nullptr && ctx.device_context(0) == nullptr &&
              ctx.num_used_slots() == 1,
          "ng: release drops ownership");

    check(ctx.initialize(0).ok(), "ng: slot reuse re-initializes");
    const std::uint32_t* c0_reused = ctx.context(0);
    check(c0_reused[0] == kEos && c0_reused[1] == kEos,
          "ng: reused slot carries no residue");

    std::uint64_t reused_ids[2 * ple::kNgramHeads];
    std::uint64_t fresh_ids[2 * ple::kNgramHeads];
    check(ple::compute_ngram_ids(tables, c0_reused, 2, input, 2, kEos, reused_ids).ok() &&
              ple::compute_ngram_ids(tables, c1, 2, input, 2, kEos, fresh_ids).ok(),
          "ng: reused and fresh contexts compute ids");
    check(std::memcmp(reused_ids, fresh_ids, sizeof(reused_ids)) == 0,
          "ng: reused slot matches a fresh request bitwise");

    check(!ctx.advance(0, nullptr, 1).ok(), "ng: null input with count rejected");

    check(ctx.shutdown().ok(), "ng: shutdown");
    check(ctx.shutdown().ok(), "ng: double shutdown is a no-op");
    check(ctx.context(0) == nullptr && ctx.device_context(0) == nullptr,
          "ng: shutdown clears host and device views");
    check(!ctx.initialize(0).ok() && !ctx.advance(0, input, 1).ok() &&
              !ctx.upload(0, stream).ok(),
          "ng: operations after shutdown are rejected");
}

void run_round(hipStream_t stream) {
    auto arena_result = ps::gpu::GpuArena::create(0, kArenaBytes);
    if (!arena_result.ok()) return;
    ps::gpu::GpuArena& arena = arena_result.value();

    auto keys_result = q4rt::IndexerKeyPool::create(arena, kSlots, kLayers, kTokens);
    auto conv_result = q4rt::PleConvStatePool::create(kSlots, kConvDim);
    auto ngram_result = q4rt::PleNgramContext::create(arena, kSlots, kEos);
    if (!keys_result.ok() || !conv_result.ok() || !ngram_result.ok()) {
        check(false, "leak: round pool create");
        return;
    }
    q4rt::IndexerKeyPool& keys = keys_result.value();
    q4rt::PleConvStatePool& conv = conv_result.value();
    q4rt::PleNgramContext& ngram = ngram_result.value();

    if (!keys.initialize(0, stream).ok() || !conv.initialize(0).ok() ||
        !ngram.initialize(0).ok()) {
        check(false, "leak: round initialize");
        return;
    }

    fill_keys(keys, 0, 0, 0);
    fill_keys(keys, 0, 1, 7);

    Rng rng(7u);
    std::vector<float> weight(static_cast<std::size_t>(kConvDim) * ple::kPleConvKernel);
    for (float& v : weight) v = rng.next() * 0.02f;
    std::vector<float> input(3ull * kConvDim);
    for (float& v : input) v = rng.next();
    std::vector<float> output(3ull * kConvDim, 0.0f);
    (void)conv.convolve(0, weight.data(), input.data(), 3, output.data());

    const std::uint32_t ids[3] = {11, 12, 13};
    (void)ngram.advance(0, ids, 3);
    (void)ngram.upload(0, stream);

    (void)keys.reset(0, stream);
    (void)conv.reset(0);
    (void)ngram.reset(0);
    (void)keys.release(0);
    (void)conv.release(0);
    (void)ngram.release(0);
    (void)hipStreamSynchronize(stream);

    (void)keys.shutdown();
    (void)conv.shutdown();
    (void)ngram.shutdown();
    (void)arena.shutdown();
}

void test_leak() {
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        check(false, "leak: stream create");
        return;
    }

    run_round(stream);

    std::size_t free_before = 0;
    std::size_t total_before = 0;
    std::size_t free_after = 0;
    std::size_t total_after = 0;
    (void)hipMemGetInfo(&free_before, &total_before);

    for (int round = 0; round < 8; ++round) {
        run_round(stream);
    }

    (void)hipMemGetInfo(&free_after, &total_after);
    check(total_before == total_after, "leak: device total unchanged");
    const std::ptrdiff_t drift =
        static_cast<std::ptrdiff_t>(free_after) - static_cast<std::ptrdiff_t>(free_before);
    printf("  free memory drift after 8 pool rounds: %td bytes\n", drift);
    check(drift >= -static_cast<std::ptrdiff_t>(1u << 20),
          "leak: no device memory leak beyond 1 MiB");
    (void)hipStreamDestroy(stream);
}

}

int main() {
    printf("=== qwen4exp state owners: IndexerKeyPool / PleConvStatePool / PleNgramContext ===\n");
    if (hipSetDevice(0) != hipSuccess) {
        printf("FAIL: hipSetDevice(0)\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        printf("FAIL: hipStreamCreate\n");
        return 1;
    }

    {
        auto arena_result = ps::gpu::GpuArena::create(0, kArenaBytes);
        if (!arena_result.ok()) {
            printf("FAIL: arena create: %s\n", arena_result.status().message().c_str());
            (void)hipStreamDestroy(stream);
            return 1;
        }
        ps::gpu::GpuArena& arena = arena_result.value();
        test_indexer(arena, stream);
        test_ngram(arena, stream);
        check(arena.shutdown().ok(), "arena: shutdown after pool shutdown");
    }

    test_conv();
    test_leak();

    (void)hipStreamDestroy(stream);

    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
