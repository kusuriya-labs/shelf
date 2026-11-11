#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/runtime/prefix_cache.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/kernels/kv_types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

namespace {

using shelf::bf16_t;
using shelf::state::GdnStatePool;
using shelf::state::GdnStatePoolLayout;
using shelf::kernel::KVCacheDType;
using shelf::state::PageId;
using shelf::state::PagedKVPool;
using shelf::state::PagedSequenceState;
using shelf::state::SequenceSlotPool;
using shelf::state::commit_sequence_append;
using shelf::state::create_paged_sequence_state;
using shelf::state::release_paged_sequence_state;
using shelf::state::reserve_sequence_append;
using shelf::qwen35::runtime::kInvalidReuseSlot;
using shelf::qwen35::runtime::PrefixCache;
using shelf::qwen35::runtime::PrefixCacheConfig;
using shelf::qwen35::runtime::PrefixCacheStats;
using shelf::qwen35::runtime::PrefixCacheStrategy;
using shelf::qwen35::runtime::PrefixHit;

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const char* name) {
    if (cond) {
        ++g_pass;
        std::printf("PASS: %s\n", name);
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", name);
    }
}

constexpr uint32_t kPageTokens = 4;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 4;
constexpr uint32_t kAttnLayers = 2;
constexpr uint32_t kMaxSeqLen = 64;

struct Env {
    shelf::gpu::GpuArena* arena = nullptr;
    PagedKVPool* kv_bf16 = nullptr;
    PagedKVPool* kv_fp8 = nullptr;
    PagedKVPool* kv_octave = nullptr;
    GdnStatePool* gdn = nullptr;
    SequenceSlotPool* slots = nullptr;
    hipStream_t stream = nullptr;
};

std::optional<PagedSequenceState> make_seq(Env& env, PagedKVPool& kv) {
    auto result = create_paged_sequence_state(
        *env.slots, *env.gdn, kv, kMaxSeqLen, env.stream);
    if (!result.ok()) {
        std::printf("create_paged_sequence_state failed: %s\n",
                    result.status().message().c_str());
        return std::nullopt;
    }
    return result.release();
}

std::optional<PrefixCache> make_cache(
    Env& env,
    PagedKVPool& kv,
    const PrefixCacheConfig& config) {
    auto result = PrefixCache::create(*env.arena, kv, *env.gdn, config);
    if (!result.ok()) {
        std::printf("PrefixCache::create failed: %s\n",
                    result.status().message().c_str());
        return std::nullopt;
    }
    return result.release();
}

bool append_tokens(PagedSequenceState& seq, uint32_t count, hipStream_t stream) {
    const shelf::Status reserve_st = reserve_sequence_append(seq, count, stream);
    if (!reserve_st.ok()) return false;
    return commit_sequence_append(seq, seq.request_handle(), count).ok();
}

std::vector<uint32_t> make_tokens(uint32_t count, uint32_t base) {
    std::vector<uint32_t> tokens(count);
    for (uint32_t i = 0; i < count; ++i) {
        tokens[i] = base + i;
    }
    return tokens;
}

std::size_t kv_page_bytes(const PagedKVPool& pool) {
    if (pool.dtype() == KVCacheDType::OCTAVE) {
        return static_cast<std::size_t>(pool.page_tokens()) * pool.kv_heads() *
               shelf::kernel::octave_k_slot_bytes(pool.octave_format());
    }
    const std::size_t elems = static_cast<std::size_t>(pool.page_tokens())
        * pool.kv_heads() * pool.head_dim();
    return pool.dtype() == KVCacheDType::FP8_E4M3
               ? elems
               : elems * sizeof(bf16_t);
}

std::size_t scale_page_bytes(const PagedKVPool& pool) {
    if (pool.dtype() != KVCacheDType::FP8_E4M3) return 0;
    return static_cast<std::size_t>(pool.page_tokens()) * pool.kv_heads()
        * sizeof(float);
}

uint8_t kv_byte(uint32_t tag, uint32_t layer, uint32_t page, std::size_t i, bool is_v) {
    return static_cast<uint8_t>(
        (tag * 31u + layer * 97u + page * 13u
         + static_cast<uint32_t>(i) * (is_v ? 3u : 1u)) & 0xFFu);
}

uint8_t state_byte(uint32_t tag, uint32_t state_index, uint32_t part, std::size_t i) {
    return static_cast<uint8_t>(
        (tag * 131u + state_index * 41u + part * 7u
         + static_cast<uint32_t>(i)) & 0xFFu);
}

float scale_value(uint32_t tag, uint32_t layer, uint32_t page, std::size_t i) {
    return static_cast<float>(tag)
        + static_cast<float>(layer) * 0.25f
        + static_cast<float>(page) * 0.5f
        + static_cast<float>(i) * 0.125f;
}

bool seed_kv(
    PagedKVPool& pool,
    const PagedSequenceState& seq,
    uint32_t tag,
    hipStream_t stream) {
    if (hipStreamSynchronize(stream) != hipSuccess) return false;
    const std::size_t kbytes = kv_page_bytes(pool);
    const std::size_t sbytes = scale_page_bytes(pool);
    for (uint32_t layer = 0; layer < pool.num_attention_layers(); ++layer) {
        for (std::size_t p = 0; p < seq.block_table.size(); ++p) {
            const PageId page = seq.block_table[p];
            std::vector<uint8_t> kbuf(kbytes);
            std::vector<uint8_t> vbuf(kbytes);
            for (std::size_t i = 0; i < kbytes; ++i) {
                kbuf[i] = kv_byte(tag, layer, static_cast<uint32_t>(p), i, false);
                vbuf[i] = kv_byte(tag, layer, static_cast<uint32_t>(p), i, true);
            }
            if (hipMemcpy(pool.k_view(layer, page).data<bf16_t>(), kbuf.data(),
                          kbytes, hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
            if (hipMemcpy(pool.v_view(layer, page).data<bf16_t>(), vbuf.data(),
                          kbytes, hipMemcpyHostToDevice) != hipSuccess) {
                return false;
            }
            if (sbytes > 0) {
                const std::size_t base =
                    (static_cast<std::size_t>(layer) * pool.num_pages() + page)
                    * pool.page_tokens() * pool.kv_heads();
                const std::size_t count = pool.page_tokens() * pool.kv_heads();
                std::vector<float> ks(count);
                std::vector<float> vs(count);
                for (std::size_t i = 0; i < count; ++i) {
                    ks[i] = scale_value(tag, layer, static_cast<uint32_t>(p), i);
                    vs[i] = ks[i] + 1.0f;
                }
                if (hipMemcpy(pool.k_scale_pool().data<float>() + base, ks.data(),
                              sbytes, hipMemcpyHostToDevice) != hipSuccess) {
                    return false;
                }
                if (hipMemcpy(pool.v_scale_pool().data<float>() + base, vs.data(),
                              sbytes, hipMemcpyHostToDevice) != hipSuccess) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool verify_kv(
    PagedKVPool& pool,
    const PagedSequenceState& seq,
    uint32_t tag,
    hipStream_t stream) {
    if (hipStreamSynchronize(stream) != hipSuccess) return false;
    const std::size_t kbytes = kv_page_bytes(pool);
    const std::size_t sbytes = scale_page_bytes(pool);
    for (uint32_t layer = 0; layer < pool.num_attention_layers(); ++layer) {
        for (std::size_t p = 0; p < seq.block_table.size(); ++p) {
            const PageId page = seq.block_table[p];
            std::vector<uint8_t> kbuf(kbytes);
            std::vector<uint8_t> vbuf(kbytes);
            if (hipMemcpy(kbuf.data(), pool.k_view(layer, page).data<bf16_t>(),
                          kbytes, hipMemcpyDeviceToHost) != hipSuccess) {
                return false;
            }
            if (hipMemcpy(vbuf.data(), pool.v_view(layer, page).data<bf16_t>(),
                          kbytes, hipMemcpyDeviceToHost) != hipSuccess) {
                return false;
            }
            for (std::size_t i = 0; i < kbytes; ++i) {
                if (kbuf[i] != kv_byte(tag, layer, static_cast<uint32_t>(p), i, false)) {
                    return false;
                }
                if (vbuf[i] != kv_byte(tag, layer, static_cast<uint32_t>(p), i, true)) {
                    return false;
                }
            }
            if (sbytes > 0) {
                const std::size_t base =
                    (static_cast<std::size_t>(layer) * pool.num_pages() + page)
                    * pool.page_tokens() * pool.kv_heads();
                const std::size_t count = pool.page_tokens() * pool.kv_heads();
                std::vector<float> ks(count);
                std::vector<float> vs(count);
                if (hipMemcpy(ks.data(), pool.k_scale_pool().data<float>() + base,
                              sbytes, hipMemcpyDeviceToHost) != hipSuccess) {
                    return false;
                }
                if (hipMemcpy(vs.data(), pool.v_scale_pool().data<float>() + base,
                              sbytes, hipMemcpyDeviceToHost) != hipSuccess) {
                    return false;
                }
                for (std::size_t i = 0; i < count; ++i) {
                    const float expect_k =
                        scale_value(tag, layer, static_cast<uint32_t>(p), i);
                    if (ks[i] != expect_k) return false;
                    if (vs[i] != expect_k + 1.0f) return false;
                }
            }
        }
    }
    return true;
}

bool seed_state(
    GdnStatePool& pool,
    shelf::state::SequenceSlotId slot,
    uint32_t tag,
    hipStream_t stream) {
    if (hipStreamSynchronize(stream) != hipSuccess) return false;
    for (uint32_t s = 0; s < pool.num_gdn_states(); ++s) {
        const auto state = pool.gdn_state(slot, s);
        const std::size_t conv_count =
            static_cast<std::size_t>(pool.conv_history()) * pool.conv_history_stride();
        std::vector<uint8_t> conv_bytes(conv_count * sizeof(bf16_t));
        for (std::size_t i = 0; i < conv_bytes.size(); ++i) {
            conv_bytes[i] = state_byte(tag, s, 0, i);
        }
        if (hipMemcpy(state.conv_state.data<bf16_t>(), conv_bytes.data(),
                      conv_bytes.size(), hipMemcpyHostToDevice) != hipSuccess) {
            return false;
        }
        const std::size_t rec_count = static_cast<std::size_t>(pool.num_v_heads())
            * pool.head_k() * pool.head_v();
        std::vector<uint8_t> rec_bytes(rec_count * sizeof(shelf::fp16_t));
        for (std::size_t i = 0; i < rec_bytes.size(); ++i) {
            rec_bytes[i] = state_byte(tag, s, 1, i);
        }
        if (hipMemcpy(state.recurrent_state.data<shelf::fp16_t>(), rec_bytes.data(),
                      rec_bytes.size(), hipMemcpyHostToDevice) != hipSuccess) {
            return false;
        }
    }
    return true;
}

bool verify_state(
    GdnStatePool& pool,
    shelf::state::SequenceSlotId slot,
    uint32_t tag,
    hipStream_t stream) {
    if (hipStreamSynchronize(stream) != hipSuccess) return false;
    for (uint32_t s = 0; s < pool.num_gdn_states(); ++s) {
        const auto state = pool.gdn_state(slot, s);
        const std::size_t conv_count =
            static_cast<std::size_t>(pool.conv_history()) * pool.conv_history_stride();
        std::vector<uint8_t> conv_bytes(conv_count * sizeof(bf16_t));
        if (hipMemcpy(conv_bytes.data(), state.conv_state.data<bf16_t>(),
                      conv_bytes.size(), hipMemcpyDeviceToHost) != hipSuccess) {
            return false;
        }
        for (std::size_t i = 0; i < conv_bytes.size(); ++i) {
            if (conv_bytes[i] != state_byte(tag, s, 0, i)) return false;
        }
        const std::size_t rec_count = static_cast<std::size_t>(pool.num_v_heads())
            * pool.head_k() * pool.head_v();
        std::vector<uint8_t> rec_bytes(rec_count * sizeof(shelf::fp16_t));
        if (hipMemcpy(rec_bytes.data(), state.recurrent_state.data<shelf::fp16_t>(),
                      rec_bytes.size(), hipMemcpyDeviceToHost) != hipSuccess) {
            return false;
        }
        for (std::size_t i = 0; i < rec_bytes.size(); ++i) {
            if (rec_bytes[i] != state_byte(tag, s, 1, i)) return false;
        }
    }
    return true;
}

void free_seq(std::optional<PagedSequenceState>& seq, hipStream_t stream) {
    if (seq.has_value()) {
        const shelf::Status st = release_paged_sequence_state(*seq, stream);
        if (!st.ok()) {
            std::printf("release_paged_sequence_state failed: %s\n",
                        st.message().c_str());
            ++g_fail;
        }
        seq.reset();
    }
}

void test_radix(Env& env) {
    PrefixCacheConfig config;
    config.max_concurrent = 3;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "radix cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    auto seq_a = make_seq(env, *env.kv_bf16);
    check(seq_a.has_value(), "radix seq a");
    if (!seq_a.has_value()) return;
    check(append_tokens(*seq_a, 8, env.stream), "radix append a");
    const std::vector<uint32_t> tokens_a = make_tokens(8, 1000);
    check(seed_kv(*env.kv_bf16, *seq_a, 1, env.stream), "radix seed kv a");
    check(seed_state(*env.gdn, seq_a->slot, 1, env.stream), "radix seed state a");
    check(cache.save_input_boundary(*seq_a, tokens_a, 8, env.stream).ok(),
          "radix save a");

    const PrefixHit hit_a = cache.lookup(tokens_a);
    check(hit_a.hit() && hit_a.state_position == 8 && hit_a.match_tokens == 8,
          "radix page aligned longest match");

    auto seq_e = make_seq(env, *env.kv_bf16);
    check(seq_e.has_value(), "radix seq e");
    if (!seq_e.has_value()) return;
    check(append_tokens(*seq_e, 4, env.stream), "radix append e");
    const std::vector<uint32_t> tokens_e = {
        tokens_a[0], tokens_a[1], tokens_a[2], tokens_a[3], 5000, 5001};
    check(seed_kv(*env.kv_bf16, *seq_e, 2, env.stream), "radix seed kv e");
    check(seed_state(*env.gdn, seq_e->slot, 2, env.stream), "radix seed state e");
    check(cache.save_input_boundary(*seq_e, tokens_e, 6, env.stream).ok(),
          "radix save unaligned tail");

    const PrefixHit hit_e = cache.lookup(tokens_e);
    check(hit_e.hit() && hit_e.match_tokens == 4,
          "radix tail page stays out of tree");
    check(cache.stats().used_pages == 2,
          "radix unaligned save allocates no tail page");

    auto seq_g = make_seq(env, *env.kv_bf16);
    check(seq_g.has_value(), "radix seq g");
    if (!seq_g.has_value()) return;
    check(append_tokens(*seq_g, 8, env.stream), "radix append g");
    const std::vector<uint32_t> tokens_g = {
        tokens_a[0], tokens_a[1], tokens_a[2], tokens_a[3],
        6000, 6001, 6002, 6003};
    check(seed_kv(*env.kv_bf16, *seq_g, 3, env.stream), "radix seed kv g");
    check(seed_state(*env.gdn, seq_g->slot, 3, env.stream), "radix seed state g");
    check(cache.save_input_boundary(*seq_g, tokens_g, 8, env.stream).ok(),
          "radix save divergent sibling");

    const PrefixHit hit_g = cache.lookup(tokens_g);
    check(hit_g.hit() && hit_g.match_tokens == 8 && hit_g.state_position == 8,
          "radix divergent sibling match");
    const PrefixHit hit_a2 = cache.lookup(tokens_a);
    check(hit_a2.hit() && hit_a2.match_tokens == 8 && hit_a2.state_position == 8,
          "radix entry survives node split");
    const PrefixCacheStats stats = cache.stats();
    check(stats.radix_nodes == 3, "radix node count after split");
    check(stats.entries == 3, "radix entry count");
    check(stats.used_pages == 3, "radix used page count");

    const PrefixHit hit_miss = cache.lookup(make_tokens(8, 9999));
    check(!hit_miss.hit() && hit_miss.match_tokens == 0, "radix miss on new prefix");

    free_seq(seq_g, env.stream);
    free_seq(seq_e, env.stream);
    free_seq(seq_a, env.stream);
}

void test_refcount_share(Env& env) {
    PrefixCacheConfig config;
    config.max_concurrent = 3;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "share cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    auto seq_a = make_seq(env, *env.kv_bf16);
    auto seq_b = make_seq(env, *env.kv_bf16);
    auto seq_c = make_seq(env, *env.kv_bf16);
    check(seq_a.has_value() && seq_b.has_value() && seq_c.has_value(),
          "share seqs");
    if (!seq_a.has_value() || !seq_b.has_value() || !seq_c.has_value()) return;

    check(append_tokens(*seq_a, 8, env.stream), "share append a");
    const std::vector<uint32_t> tokens_a = make_tokens(8, 1000);
    check(seed_kv(*env.kv_bf16, *seq_a, 10, env.stream), "share seed kv a");
    check(seed_state(*env.gdn, seq_a->slot, 10, env.stream), "share seed state a");
    check(cache.save_input_boundary(*seq_a, tokens_a, 8, env.stream).ok(),
          "share save a");

    check(append_tokens(*seq_b, 16, env.stream), "share append b");
    std::vector<uint32_t> tokens_b = tokens_a;
    for (uint32_t i = 0; i < 8; ++i) {
        tokens_b.push_back(2000 + i);
    }
    check(seed_kv(*env.kv_bf16, *seq_b, 10, env.stream), "share seed kv b");
    check(seed_state(*env.gdn, seq_b->slot, 11, env.stream), "share seed state b");
    check(cache.save_input_boundary(*seq_b, tokens_b, 16, env.stream).ok(),
          "share save b");

    const PrefixHit hit_b = cache.lookup(tokens_b);
    check(hit_b.hit() && hit_b.state_position == 16, "share lookup b");
    const uint64_t entry_b = hit_b.entry_id;
    check(cache.stats().used_pages == 4, "share pages after child save");
    check(cache.stats().entries == 2, "share entries after child save");

    check(append_tokens(*seq_c, 8, env.stream), "share append c");
    const std::vector<uint32_t> tokens_c = {
        tokens_a[0], tokens_a[1], tokens_a[2], tokens_a[3],
        3000, 3001, 3002, 3003};
    check(seed_kv(*env.kv_bf16, *seq_c, 12, env.stream), "share seed kv c");
    check(seed_state(*env.gdn, seq_c->slot, 12, env.stream), "share seed state c");
    check(cache.save_input_boundary(*seq_c, tokens_c, 8, env.stream).ok(),
          "share save c");

    const PrefixCacheStats stats_after = cache.stats();
    check(stats_after.entries == 3, "share keeps all entries");
    check(stats_after.entries_evicted == 0, "share does not evict");
    check(stats_after.used_pages == 5, "share pages after third save");

    const PrefixHit hit_a = cache.lookup(tokens_a);
    check(hit_a.hit() && hit_a.state_position == 8, "share parent still hits");

    auto dst = make_seq(env, *env.kv_bf16);
    check(dst.has_value(), "share restore dst");
    if (!dst.has_value()) return;
    check(cache.restore(entry_b, *dst, env.stream).ok(), "share restore child");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "share restore sync");
        return;
    }
    check(dst->position == 16, "share restore position");
    check(verify_kv(*env.kv_bf16, *dst, 10, env.stream),
          "share restore kv bytes");
    check(verify_state(*env.gdn, dst->slot, 11, env.stream),
          "share restore state bytes");

    free_seq(dst, env.stream);
    free_seq(seq_c, env.stream);
    free_seq(seq_b, env.stream);
    free_seq(seq_a, env.stream);
}

void test_hit_rule(Env& env) {
    PrefixCacheConfig config;
    config.max_concurrent = 2;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "rule cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    auto seq_a = make_seq(env, *env.kv_bf16);
    auto seq_b = make_seq(env, *env.kv_bf16);
    check(seq_a.has_value() && seq_b.has_value(), "rule seqs");
    if (!seq_a.has_value() || !seq_b.has_value()) return;

    check(append_tokens(*seq_a, 8, env.stream), "rule append a");
    const std::vector<uint32_t> tokens_a = make_tokens(8, 1000);
    check(seed_kv(*env.kv_bf16, *seq_a, 21, env.stream), "rule seed kv a");
    check(seed_state(*env.gdn, seq_a->slot, 21, env.stream), "rule seed state a");
    check(cache.save_input_boundary(*seq_a, tokens_a, 8, env.stream).ok(),
          "rule save a");

    check(append_tokens(*seq_b, 16, env.stream), "rule append b");
    std::vector<uint32_t> tokens_b = tokens_a;
    for (uint32_t i = 0; i < 8; ++i) {
        tokens_b.push_back(2000 + i);
    }
    check(seed_kv(*env.kv_bf16, *seq_b, 21, env.stream), "rule seed kv b");
    check(seed_state(*env.gdn, seq_b->slot, 22, env.stream), "rule seed state b");
    check(cache.save_input_boundary(*seq_b, tokens_b, 16, env.stream).ok(),
          "rule save b");

    const PrefixHit hit_b = cache.lookup(tokens_b);
    check(hit_b.hit() && hit_b.state_position == 16,
          "rule deepest state position wins");
    const PrefixHit hit_a = cache.lookup(tokens_a);
    check(hit_a.hit() && hit_a.state_position == 8,
          "rule shorter query selects shorter entry");

    std::vector<uint32_t> diverge_page2 = tokens_a;
    for (uint32_t i = 0; i < 8; ++i) {
        diverge_page2.push_back(5000 + i);
    }
    const PrefixHit hit_diverge = cache.lookup(diverge_page2);
    check(hit_diverge.hit() && hit_diverge.match_tokens == 8
              && hit_diverge.state_position == 8,
          "rule entry beyond match is not selected");

    std::vector<uint32_t> diverge_page1 = {
        tokens_a[0], tokens_a[1], tokens_a[2], tokens_a[3],
        6000, 6001, 6002, 6003};
    const PrefixHit hit_early = cache.lookup(diverge_page1);
    check(!hit_early.hit() && hit_early.match_tokens == 4,
          "rule state position beyond match misses");

    PrefixCacheConfig small_config;
    small_config.max_concurrent = 1;
    small_config.min_prefix_tokens = 4;
    auto small_cache_opt = make_cache(env, *env.kv_bf16, small_config);
    check(small_cache_opt.has_value(), "rule small cache create");
    if (!small_cache_opt.has_value()) return;
    PrefixCache& small_cache = *small_cache_opt;

    auto seq_x = make_seq(env, *env.kv_bf16);
    auto seq_y = make_seq(env, *env.kv_bf16);
    auto seq_z = make_seq(env, *env.kv_bf16);
    check(seq_x.has_value() && seq_y.has_value() && seq_z.has_value(),
          "rule small seqs");
    if (!seq_x.has_value() || !seq_y.has_value() || !seq_z.has_value()) return;

    check(append_tokens(*seq_x, 8, env.stream), "rule append x");
    const std::vector<uint32_t> tokens_x = make_tokens(8, 11000);
    check(seed_kv(*env.kv_bf16, *seq_x, 23, env.stream), "rule seed kv x");
    check(seed_state(*env.gdn, seq_x->slot, 23, env.stream), "rule seed state x");
    check(small_cache.save_input_boundary(*seq_x, tokens_x, 8, env.stream).ok(),
          "rule small save x");

    check(append_tokens(*seq_y, 8, env.stream), "rule append y");
    const std::vector<uint32_t> tokens_y = make_tokens(8, 12000);
    check(seed_kv(*env.kv_bf16, *seq_y, 24, env.stream), "rule seed kv y");
    check(seed_state(*env.gdn, seq_y->slot, 24, env.stream), "rule seed state y");
    check(small_cache.save_input_boundary(*seq_y, tokens_y, 8, env.stream).ok(),
          "rule small save y");
    check(small_cache.stats().entries == 1, "rule small keeps one entry");

    check(append_tokens(*seq_z, 8, env.stream), "rule append z");
    const std::vector<uint32_t> tokens_z = make_tokens(8, 13000);
    check(seed_kv(*env.kv_bf16, *seq_z, 25, env.stream), "rule seed kv z");
    check(seed_state(*env.gdn, seq_z->slot, 25, env.stream), "rule seed state z");
    check(small_cache.save_input_boundary(*seq_z, tokens_z, 8, env.stream).ok(),
          "rule small save z");
    check(small_cache.stats().entries == 1, "rule small entry evicted");
    check(small_cache.stats().entries_evicted == 2, "rule small eviction count");

    const PrefixHit hit_stateless = small_cache.lookup(tokens_x);
    check(!hit_stateless.hit() && hit_stateless.match_tokens == 8,
          "rule stateless path misses");

    free_seq(seq_z, env.stream);
    free_seq(seq_y, env.stream);
    free_seq(seq_x, env.stream);
    free_seq(seq_b, env.stream);
    free_seq(seq_a, env.stream);
}

void test_roundtrip(Env& env, PagedKVPool& kv, uint32_t tag) {
    PrefixCacheConfig config;
    config.max_concurrent = 2;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, kv, config);
    check(cache_opt.has_value(), "roundtrip cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    auto seq = make_seq(env, kv);
    check(seq.has_value(), "roundtrip seq");
    if (!seq.has_value()) return;
    check(append_tokens(*seq, 16, env.stream), "roundtrip append");
    const std::vector<uint32_t> tokens = make_tokens(16, 7000);
    check(seed_kv(kv, *seq, tag, env.stream), "roundtrip seed kv");
    check(seed_state(*env.gdn, seq->slot, tag, env.stream), "roundtrip seed state");
    check(cache.save_input_boundary(*seq, tokens, 16, env.stream).ok(),
          "roundtrip save");

    const PrefixHit hit = cache.lookup(tokens);
    check(hit.hit() && hit.state_position == 16, "roundtrip lookup");

    auto dst = make_seq(env, kv);
    check(dst.has_value(), "roundtrip dst");
    if (!dst.has_value()) return;
    check(cache.restore(hit.entry_id, *dst, env.stream).ok(), "roundtrip restore");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "roundtrip sync");
        return;
    }
    check(dst->position == 16, "roundtrip position page aligned");
    check(verify_kv(kv, *dst, tag, env.stream), "roundtrip kv bytes");
    check(verify_state(*env.gdn, dst->slot, tag, env.stream), "roundtrip state bytes");

    const bool reserved = reserve_sequence_append(*dst, 8, env.stream).ok();
    check(reserved, "post restore reserve append");
    const bool committed = reserved
        && commit_sequence_append(*dst, dst->request_handle(), 8).ok();
    check(committed, "post restore commit append");
    check(committed && dst->position == 24, "post restore position link");
    check(committed && dst->block_table.size() == 6, "post restore block table");

    free_seq(dst, env.stream);
    free_seq(seq, env.stream);
}

void test_capture(Env& env) {
    PrefixCacheConfig config;
    config.max_concurrent = 2;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "capture cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    auto seq = make_seq(env, *env.kv_bf16);
    check(seq.has_value(), "capture seq");
    if (!seq.has_value()) return;

    const auto capture = cache.begin_capture(*seq);
    check(capture.ok() && capture.value() != shelf::qwen35::runtime::kInvalidCapture,
          "capture begin");
    if (!capture.ok()) return;
    const shelf::qwen35::runtime::CaptureId capture_id = capture.value();

    const struct {
        uint32_t delta;
        uint32_t state_tag;
    } steps[] = {
        {4, 11},
        {4, 12},
        {4, 13},
        {4, 14},
    };
    for (const auto& step : steps) {
        if (!append_tokens(*seq, step.delta, env.stream)) {
            check(false, "capture append");
            return;
        }
        if (!seed_state(*env.gdn, seq->slot, step.state_tag, env.stream)) {
            check(false, "capture seed state");
            return;
        }
        const shelf::Status st = cache.capture_boundary(capture_id, *seq, env.stream);
        if (!st.ok()) {
            check(false, "capture boundary");
            return;
        }
    }
    check(seq->position == 16, "capture position advance");

    const std::vector<uint32_t> tokens = make_tokens(16, 8000);
    check(seed_kv(*env.kv_bf16, *seq, 50, env.stream), "capture seed kv");
    check(cache.save_output_tail(capture_id, *seq, tokens, 16, env.stream).ok(),
          "capture save output tail");
    const PrefixCacheStats stats = cache.stats();
    check(stats.output_tail_saves == 1, "capture save counted");
    check(stats.captures_active == 0, "capture slot consumed by save");

    const PrefixHit hit = cache.lookup(tokens);
    check(hit.hit() && hit.state_position == 16, "capture lookup");

    auto dst = make_seq(env, *env.kv_bf16);
    check(dst.has_value(), "capture dst");
    if (!dst.has_value()) return;
    check(cache.restore(hit.entry_id, *dst, env.stream).ok(), "capture restore");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "capture restore sync");
        return;
    }
    check(dst->position == 16, "capture restore position");
    check(verify_state(*env.gdn, dst->slot, 14, env.stream),
          "capture rolling keeps latest boundary");
    check(verify_kv(*env.kv_bf16, *dst, 50, env.stream), "capture restore kv bytes");
    free_seq(dst, env.stream);

    const shelf::Status stale_st = cache.cancel_capture(capture_id);
    check(stale_st.code() == shelf::Status::Code::invalid_argument,
          "capture cancel after save rejected");

    const auto capture_h1 = cache.begin_capture(*seq);
    const auto capture_h2 = cache.begin_capture(*seq);
    check(capture_h1.ok() && capture_h2.ok(), "capture concurrent begins");
    const auto capture_h3 = cache.begin_capture(*seq);
    check(!capture_h3.ok(), "capture budget exhausted");
    check(cache.stats().captures_active == 2, "capture active gauge");
    if (capture_h1.ok()) {
        check(cache.cancel_capture(capture_h1.value()).ok(), "capture cancel one");
    }
    const auto capture_h4 = cache.begin_capture(*seq);
    check(capture_h4.ok(), "capture begin after cancel");
    if (capture_h1.ok()) {
        const shelf::Status again_st = cache.cancel_capture(capture_h1.value());
        check(again_st.code() == shelf::Status::Code::invalid_argument,
              "capture cancel stale rejected");
    }
    if (capture_h2.ok()) {
        check(cache.cancel_capture(capture_h2.value()).ok(), "capture cancel h2");
    }
    if (capture_h4.ok()) {
        check(cache.cancel_capture(capture_h4.value()).ok(), "capture cancel h4");
    }
    check(cache.stats().captures_active == 0, "capture active after cancels");

    auto seq_floor = make_seq(env, *env.kv_bf16);
    check(seq_floor.has_value(), "capture floor seq");
    if (!seq_floor.has_value()) return;
    check(append_tokens(*seq_floor, 18, env.stream), "capture floor append");
    const auto capture_floor = cache.begin_capture(*seq_floor);
    check(capture_floor.ok(), "capture floor begin");
    if (!capture_floor.ok()) return;
    check(seed_state(*env.gdn, seq_floor->slot, 21, env.stream),
          "capture floor seed state");
    check(cache.capture_boundary(capture_floor.value(), *seq_floor, env.stream).ok(),
          "capture floor boundary");
    const std::vector<uint32_t> tokens_floor = make_tokens(18, 9000);
    check(seed_kv(*env.kv_bf16, *seq_floor, 51, env.stream), "capture floor seed kv");
    check(cache.save_output_tail(
              capture_floor.value(), *seq_floor, tokens_floor, 18, env.stream).ok(),
          "capture floor save unaligned position");

    const PrefixHit hit_floor = cache.lookup(tokens_floor);
    check(hit_floor.hit() && hit_floor.state_position == 16,
          "capture floor position aligned down");

    auto dst_floor = make_seq(env, *env.kv_bf16);
    check(dst_floor.has_value(), "capture floor dst");
    if (!dst_floor.has_value()) return;
    check(cache.restore(hit_floor.entry_id, *dst_floor, env.stream).ok(),
          "capture floor restore");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "capture floor sync");
        return;
    }
    check(dst_floor->position == 16, "capture floor restore position");
    check(verify_state(*env.gdn, dst_floor->slot, 21, env.stream),
          "capture floor state bytes");
    check(verify_kv(*env.kv_bf16, *dst_floor, 51, env.stream),
          "capture floor kv bytes");
    free_seq(dst_floor, env.stream);

    const shelf::Status bad_input = cache.save_input_boundary(
        *seq, tokens_floor, 17, env.stream);
    check(bad_input.code() == shelf::Status::Code::invalid_argument,
          "save position mismatch rejected");

    const auto capture_bad = cache.begin_capture(*seq_floor);
    check(capture_bad.ok(), "capture bad handle begin");
    if (capture_bad.ok()) {
        const shelf::Status bad_tail = cache.save_output_tail(
            capture_bad.value(), *seq_floor, tokens_floor, 17, env.stream);
        check(bad_tail.code() == shelf::Status::Code::invalid_argument,
              "output tail position mismatch rejected");
        check(cache.cancel_capture(capture_bad.value()).ok(),
              "capture cancel after rejected save");
    }

    PrefixCacheConfig strict_config;
    strict_config.max_concurrent = 1;
    strict_config.min_prefix_tokens = 64;
    auto strict_cache_opt = make_cache(env, *env.kv_bf16, strict_config);
    check(strict_cache_opt.has_value(), "capture strict cache create");
    if (!strict_cache_opt.has_value()) return;
    PrefixCache& strict_cache = *strict_cache_opt;

    auto seq_short = make_seq(env, *env.kv_bf16);
    check(seq_short.has_value(), "capture short seq");
    if (!seq_short.has_value()) return;
    check(append_tokens(*seq_short, 16, env.stream), "capture short append");
    const std::vector<uint32_t> tokens_short = make_tokens(16, 15000);
    check(seed_kv(*env.kv_bf16, *seq_short, 60, env.stream), "capture short seed kv");
    check(seed_state(*env.gdn, seq_short->slot, 60, env.stream),
          "capture short seed state");
    check(strict_cache.save_input_boundary(*seq_short, tokens_short, 16, env.stream).ok(),
          "capture short save skipped softly");
    const PrefixCacheStats strict_stats = strict_cache.stats();
    check(strict_stats.skipped_too_short == 1, "capture short skip counted");
    check(strict_stats.saves == 0 && strict_stats.entries == 0,
          "capture short stored nothing");
    check(!strict_cache.lookup(tokens_short).hit(), "capture short miss");

    free_seq(seq_short, env.stream);
    free_seq(seq_floor, env.stream);
    free_seq(seq, env.stream);
}

void test_eviction(Env& env) {
    PrefixCacheConfig config;
    config.max_concurrent = 1;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "eviction cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    auto seq_a = make_seq(env, *env.kv_bf16);
    auto seq_b = make_seq(env, *env.kv_bf16);
    check(seq_a.has_value() && seq_b.has_value(), "eviction seqs");
    if (!seq_a.has_value() || !seq_b.has_value()) return;

    check(append_tokens(*seq_a, 8, env.stream), "eviction append a");
    const std::vector<uint32_t> tokens_a = make_tokens(8, 1000);
    check(seed_kv(*env.kv_bf16, *seq_a, 31, env.stream), "eviction seed kv a");
    check(seed_state(*env.gdn, seq_a->slot, 31, env.stream), "eviction seed state a");
    check(cache.save_input_boundary(*seq_a, tokens_a, 8, env.stream).ok(),
          "eviction save a");
    check(cache.lookup(tokens_a).hit(), "eviction a hits");

    check(append_tokens(*seq_b, 8, env.stream), "eviction append b");
    const std::vector<uint32_t> tokens_b = make_tokens(8, 4000);
    check(seed_kv(*env.kv_bf16, *seq_b, 32, env.stream), "eviction seed kv b");
    check(seed_state(*env.gdn, seq_b->slot, 32, env.stream), "eviction seed state b");
    check(cache.save_input_boundary(*seq_b, tokens_b, 8, env.stream).ok(),
          "eviction save b");

    const PrefixCacheStats stats = cache.stats();
    check(stats.entries == 1, "eviction keeps one entry");
    check(stats.entries_evicted == 1, "eviction drops the old entry");
    check(!cache.lookup(tokens_a).hit(), "eviction old entry misses");
    const PrefixHit hit_b = cache.lookup(tokens_b);
    check(hit_b.hit() && hit_b.state_position == 8, "eviction new entry hits");

    auto dst = make_seq(env, *env.kv_bf16);
    check(dst.has_value(), "eviction dst");
    if (!dst.has_value()) return;
    check(cache.restore(hit_b.entry_id, *dst, env.stream).ok(), "eviction restore");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "eviction restore sync");
        return;
    }
    check(verify_kv(*env.kv_bf16, *dst, 32, env.stream), "eviction restore kv bytes");
    check(verify_state(*env.gdn, dst->slot, 32, env.stream),
          "eviction restore state bytes");

    free_seq(dst, env.stream);
    free_seq(seq_b, env.stream);
    free_seq(seq_a, env.stream);
}

uint32_t save_slot(PrefixCache& cache, PagedSequenceState& seq,
                   const std::vector<uint32_t>& tokens, uint64_t owner,
                   hipStream_t stream) {
    const PrefixHit hit = cache.lookup(tokens);
    const uint32_t slot = cache.reserve_reuse_slot(owner, hit);
    if (slot == kInvalidReuseSlot) return slot;
    const uint32_t input_len = static_cast<uint32_t>(tokens.size());
    if (input_len == 0u || input_len % kPageTokens != 0u ||
        seq.position != input_len) {
        return kInvalidReuseSlot;
    }
    if (!cache.save_slot_boundary(owner, seq, tokens, input_len, stream).ok()) {
        return kInvalidReuseSlot;
    }
    return slot;
}

void test_slot(Env& env) {
    PrefixCacheConfig config;
    config.max_concurrent = 3;
    config.min_prefix_tokens = 4;
    config.strategy = PrefixCacheStrategy::Slot;
    config.slot_count = 2;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "slot cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;
    check(cache.reuse_slot_count() == 2, "slot count");

    auto seq_a = make_seq(env, *env.kv_bf16);
    auto seq_b = make_seq(env, *env.kv_bf16);
    auto seq_c = make_seq(env, *env.kv_bf16);
    check(seq_a.has_value() && seq_b.has_value() && seq_c.has_value(), "slot seqs");
    if (!seq_a.has_value() || !seq_b.has_value() || !seq_c.has_value()) return;

    check(append_tokens(*seq_a, 8, env.stream), "slot append a");
    const std::vector<uint32_t> tokens_a = make_tokens(8, 1000);
    check(seed_kv(*env.kv_bf16, *seq_a, 71, env.stream), "slot seed kv a");
    check(seed_state(*env.gdn, seq_a->slot, 71, env.stream), "slot seed state a");
    check(save_slot(cache, *seq_a, tokens_a, 1, env.stream) != kInvalidReuseSlot,
          "slot save a");
    check(cache.stats().entries == 1, "slot one entry after a");
    check(cache.lookup(tokens_a).hit(), "slot a hits");

    check(append_tokens(*seq_b, 8, env.stream), "slot append b");
    const std::vector<uint32_t> tokens_b = make_tokens(8, 2000);
    check(seed_kv(*env.kv_bf16, *seq_b, 72, env.stream), "slot seed kv b");
    check(seed_state(*env.gdn, seq_b->slot, 72, env.stream), "slot seed state b");
    check(save_slot(cache, *seq_b, tokens_b, 2, env.stream) != kInvalidReuseSlot,
          "slot save b");
    check(cache.stats().entries == 2, "slot two entries");

    const PrefixHit hit_b = cache.lookup(tokens_b);
    check(hit_b.hit() && hit_b.slot != kInvalidReuseSlot, "slot hit carries slot");

    auto seq_a2 = make_seq(env, *env.kv_bf16);
    check(seq_a2.has_value(), "slot extend seq");
    if (!seq_a2.has_value()) return;
    check(append_tokens(*seq_a2, 12, env.stream), "slot append extended a");
    std::vector<uint32_t> tokens_a2 = tokens_a;
    for (uint32_t i = 0; i < 4; ++i) tokens_a2.push_back(3000 + i);
    check(seed_kv(*env.kv_bf16, *seq_a2, 73, env.stream), "slot seed kv extended");
    check(seed_state(*env.gdn, seq_a2->slot, 73, env.stream), "slot seed state extended");
    check(save_slot(cache, *seq_a2, tokens_a2, 3, env.stream) != kInvalidReuseSlot,
          "slot update a");
    check(cache.stats().entries == 2, "slot update keeps entry count");
    const PrefixHit hit_a2 = cache.lookup(tokens_a2);
    check(hit_a2.hit() && hit_a2.state_position == 12, "slot updated entry hits");
    check(!cache.lookup(tokens_a).hit(), "slot old entry replaced");

    check(append_tokens(*seq_c, 8, env.stream), "slot append c");
    const std::vector<uint32_t> tokens_c = make_tokens(8, 4000);
    check(seed_kv(*env.kv_bf16, *seq_c, 74, env.stream), "slot seed kv c");
    check(seed_state(*env.gdn, seq_c->slot, 74, env.stream), "slot seed state c");
    check(save_slot(cache, *seq_c, tokens_c, 4, env.stream) != kInvalidReuseSlot,
          "slot save c evicts lru");
    check(cache.stats().entries == 2, "slot lru keeps entry count");

    const uint32_t slot_owner = cache.reserve_reuse_slot(5, cache.lookup(tokens_c));
    check(slot_owner != kInvalidReuseSlot, "slot reserve for writer");
    const PrefixHit same_hit = cache.lookup(tokens_c);
    const uint32_t other_slot = cache.reserve_reuse_slot(6, same_hit);
    check(other_slot != kInvalidReuseSlot && other_slot != slot_owner,
          "slot parallel writer does not share reservation");
    cache.release_reuse_slot(5);
    cache.release_reuse_slot(6);

    auto dst = make_seq(env, *env.kv_bf16);
    check(dst.has_value(), "slot restore dst");
    if (!dst.has_value()) return;
    const PrefixHit hit_restore = cache.lookup(tokens_c);
    check(hit_restore.hit(), "slot restore hit");
    check(cache.restore(hit_restore.entry_id, *dst, env.stream).ok(), "slot restore");
    if (hipStreamSynchronize(env.stream) == hipSuccess) {
        check(dst->position == 8, "slot restore position");
        check(verify_kv(*env.kv_bf16, *dst, 74, env.stream), "slot restore kv bytes");
        check(verify_state(*env.gdn, dst->slot, 74, env.stream),
              "slot restore state bytes");
    } else {
        check(false, "slot restore sync");
    }
    free_seq(dst, env.stream);

    free_seq(seq_a2, env.stream);
    free_seq(seq_c, env.stream);
    free_seq(seq_b, env.stream);
    free_seq(seq_a, env.stream);
}

void test_spec_ring(Env& env) {
    constexpr uint32_t kRingBytes = 64;
    PrefixCacheConfig config;
    config.max_concurrent = 1;
    config.min_prefix_tokens = 4;
    config.spec_ring_bytes = 2 * kRingBytes;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "spec ring cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    bf16_t* ring_k = nullptr;
    bf16_t* ring_v = nullptr;
    if (hipMalloc(&ring_k, kRingBytes) != hipSuccess ||
        hipMalloc(&ring_v, kRingBytes) != hipSuccess) {
        check(false, "spec ring hipMalloc");
        return;
    }
    std::vector<uint8_t> host_k(kRingBytes);
    std::vector<uint8_t> host_v(kRingBytes);
    for (uint32_t i = 0; i < kRingBytes; ++i) {
        host_k[i] = static_cast<uint8_t>(i * 3u + 7u);
        host_v[i] = static_cast<uint8_t>(i * 5u + 11u);
    }
    if (hipMemcpy(ring_k, host_k.data(), kRingBytes, hipMemcpyHostToDevice) !=
            hipSuccess ||
        hipMemcpy(ring_v, host_v.data(), kRingBytes, hipMemcpyHostToDevice) !=
            hipSuccess) {
        check(false, "spec ring seed memcpy");
        hipFree(ring_k);
        hipFree(ring_v);
        return;
    }

    auto seq = make_seq(env, *env.kv_bf16);
    check(seq.has_value(), "spec ring seq");
    if (!seq.has_value()) {
        hipFree(ring_k);
        hipFree(ring_v);
        return;
    }
    check(append_tokens(*seq, 8, env.stream), "spec ring append");
    const std::vector<uint32_t> tokens = make_tokens(8, 40000);
    check(seed_kv(*env.kv_bf16, *seq, 90, env.stream), "spec ring seed kv");
    check(seed_state(*env.gdn, seq->slot, 90, env.stream), "spec ring seed state");

    shelf::qwen35::runtime::SpecRingRef ring_ref;
    ring_ref.k = ring_k;
    ring_ref.v = ring_v;
    ring_ref.length = 8;
    ring_ref.next_position = 8;
    check(cache.save_input_boundary(*seq, tokens, 8, env.stream, &ring_ref).ok(),
          "spec ring save with ring");

    const PrefixHit hit = cache.lookup(tokens);
    check(hit.hit() && hit.state_position == 8, "spec ring hit");

    bf16_t* out_k = nullptr;
    bf16_t* out_v = nullptr;
    if (hipMalloc(&out_k, kRingBytes) != hipSuccess ||
        hipMalloc(&out_v, kRingBytes) != hipSuccess) {
        check(false, "spec ring out hipMalloc");
        hipFree(ring_k);
        hipFree(ring_v);
        return;
    }
    shelf::qwen35::runtime::SpecRingOut ring_out;
    ring_out.k = out_k;
    ring_out.v = out_v;

    auto dst = make_seq(env, *env.kv_bf16);
    check(dst.has_value(), "spec ring dst");
    if (!dst.has_value()) {
        hipFree(ring_k);
        hipFree(ring_v);
        hipFree(out_k);
        hipFree(out_v);
        return;
    }
    check(cache.restore(hit.entry_id, *dst, env.stream, &ring_out).ok(),
          "spec ring restore");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "spec ring restore sync");
        hipFree(ring_k);
        hipFree(ring_v);
        hipFree(out_k);
        hipFree(out_v);
        return;
    }
    check(dst->position == 8, "spec ring restore position");
    check(verify_kv(*env.kv_bf16, *dst, 90, env.stream), "spec ring restore kv");
    check(verify_state(*env.gdn, dst->slot, 90, env.stream), "spec ring restore state");
    check(ring_out.length == 8 && ring_out.next_position == 8, "spec ring meta");

    std::vector<uint8_t> back_k(kRingBytes);
    std::vector<uint8_t> back_v(kRingBytes);
    check(hipMemcpy(back_k.data(), out_k, kRingBytes, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "spec ring copy back k");
    check(hipMemcpy(back_v.data(), out_v, kRingBytes, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "spec ring copy back v");
    check(back_k == host_k, "spec ring k bytes roundtrip");
    check(back_v == host_v, "spec ring v bytes roundtrip");

    free_seq(dst, env.stream);
    free_seq(seq, env.stream);
    hipFree(ring_k);
    hipFree(ring_v);
    hipFree(out_k);
    hipFree(out_v);
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }

    auto arena_result = shelf::gpu::GpuArena::create(0, 128ull * 1024ull * 1024ull);
    if (!arena_result.ok()) {
        std::printf("SKIP: arena (%s)\n", arena_result.status().message().c_str());
        return 77;
    }
    shelf::gpu::GpuArena arena = arena_result.release();

    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) {
        std::printf("FAIL: hipStreamCreate\n");
        return 1;
    }

    auto kv_bf16_result = PagedKVPool::create(
        arena, 64, kPageTokens, kAttnLayers, kKvHeads, kHeadDim,
        KVCacheDType::BF16);
    auto kv_fp8_result = PagedKVPool::create(
        arena, 64, kPageTokens, kAttnLayers, kKvHeads, kHeadDim,
        KVCacheDType::FP8_E4M3);
    auto kv_octave_result = PagedKVPool::create(
        arena, 64, kPageTokens, kAttnLayers, kKvHeads, 256,
        KVCacheDType::OCTAVE,
        {shelf::kernel::OctaveKMode::Compact, 4, 4, 4});
    auto gdn_result = GdnStatePool::create(
        arena, 12, GdnStatePoolLayout::make(2, 8, 3, 2, 4, 4));
    auto slots_result = SequenceSlotPool::create(arena, 12, 64);

    if (!kv_bf16_result.ok() || !kv_fp8_result.ok() || !kv_octave_result.ok() || !gdn_result.ok() ||
        !slots_result.ok()) {
        std::printf("FAIL: pool create\n");
        (void)hipStreamDestroy(stream);
        return 1;
    }

    PagedKVPool kv_bf16 = kv_bf16_result.release();
    PagedKVPool kv_fp8 = kv_fp8_result.release();
    PagedKVPool kv_octave = kv_octave_result.release();
    GdnStatePool gdn = gdn_result.release();
    SequenceSlotPool slots = slots_result.release();

    Env env;
    env.arena = &arena;
    env.kv_bf16 = &kv_bf16;
    env.kv_fp8 = &kv_fp8;
    env.kv_octave = &kv_octave;
    env.gdn = &gdn;
    env.slots = &slots;
    env.stream = stream;

    auto empty_result = GdnStatePool::create(arena, 12, GdnStatePoolLayout{});
    check(empty_result.ok(), "empty GDN pool create");
    if (empty_result.ok()) {
        GdnStatePool empty = empty_result.release();
        check(empty.reserved_bytes() == 0, "empty GDN pool allocates no state");
        check(empty.initialize(0, stream).ok(), "empty GDN slot initialize");
        check(!empty.initialize(0, stream).ok(), "empty GDN rejects duplicate initialize");
        check(empty.reset(0, stream).ok(), "empty GDN slot reset");
        check(empty.copy_slot_from(1, empty, 0, stream).ok(), "empty GDN slot copy");
        check(empty.release(0).ok() && empty.num_used_slots() == 0,
              "empty GDN slot release");
        Env empty_env = env;
        empty_env.gdn = &empty;
        test_roundtrip(empty_env, *env.kv_bf16, 80);
    }

    const char* only = std::getenv("SHELF_PFX_TEST");
    auto run = [&](const char* name, void (*fn)(Env&)) {
        if (only == nullptr || std::strcmp(only, name) == 0) fn(env);
    };
    run("radix", test_radix);
    run("share", test_refcount_share);
    run("rule", test_hit_rule);
    run("roundtrip-bf16", +[](Env& e) { test_roundtrip(e, *e.kv_bf16, 77); });
    run("roundtrip-fp8", +[](Env& e) { test_roundtrip(e, *e.kv_fp8, 78); });
    run("roundtrip-octave", +[](Env& e) { test_roundtrip(e, *e.kv_octave, 79); });
    run("capture", test_capture);
    run("eviction", test_eviction);
    run("slot", test_slot);
    run("ring", test_spec_ring);

    const shelf::Status arena_release = arena.shutdown();
    check(arena_release.ok(), "arena shutdown");

    if (hipStreamDestroy(stream) != hipSuccess) {
        std::printf("FAIL: hipStreamDestroy\n");
        ++g_fail;
    }

    std::printf("passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
