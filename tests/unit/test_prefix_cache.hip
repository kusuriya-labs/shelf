#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/runtime/prefix_cache.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/kernels/kv_types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
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
using shelf::qwen35::runtime::PrefixCache;
using shelf::qwen35::runtime::PrefixCacheConfig;
using shelf::qwen35::runtime::PrefixCacheStats;
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
    config.hot_capacity_tokens = 1024;
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
    config.hot_capacity_tokens = 1024;
    config.max_concurrent = 1;
    config.cold_state_slots = 1;
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
          "share save c evicts parent");

    const PrefixCacheStats stats_after = cache.stats();
    check(stats_after.entries == 2, "share parent entry evicted");
    check(stats_after.entries_evicted == 1, "share parent eviction count");
    check(stats_after.used_pages == 5, "share pages survive entry eviction");

    const PrefixHit hit_a = cache.lookup(tokens_a);
    check(!hit_a.hit() && hit_a.match_tokens == 8,
          "share stateless node yields miss");

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
    config.hot_capacity_tokens = 1024;
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
    small_config.hot_capacity_tokens = 1024;
    small_config.max_concurrent = 1;
    small_config.cold_state_slots = 1;
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
    check(small_cache.stats().entries == 2, "rule small demote keeps entries");

    check(append_tokens(*seq_z, 8, env.stream), "rule append z");
    const std::vector<uint32_t> tokens_z = make_tokens(8, 13000);
    check(seed_kv(*env.kv_bf16, *seq_z, 25, env.stream), "rule seed kv z");
    check(seed_state(*env.gdn, seq_z->slot, 25, env.stream), "rule seed state z");
    check(small_cache.save_input_boundary(*seq_z, tokens_z, 8, env.stream).ok(),
          "rule small save z");
    check(small_cache.stats().entries == 2, "rule small entry evicted");
    check(small_cache.stats().entries_evicted == 1, "rule small eviction count");

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
    config.hot_capacity_tokens = 256;
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
    config.hot_capacity_tokens = 1024;
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
    strict_config.hot_capacity_tokens = 1024;
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
    config.hot_capacity_tokens = 16;
    config.cold_capacity_tokens = 8;
    config.max_concurrent = 1;
    config.cold_state_slots = 1;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "eviction cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    auto seq1 = make_seq(env, *env.kv_bf16);
    auto seq2 = make_seq(env, *env.kv_bf16);
    auto seq3 = make_seq(env, *env.kv_bf16);
    auto seq4 = make_seq(env, *env.kv_bf16);
    check(seq1.has_value() && seq2.has_value() && seq3.has_value() && seq4.has_value(),
          "eviction seqs");
    if (!seq1.has_value() || !seq2.has_value() || !seq3.has_value() ||
        !seq4.has_value()) {
        return;
    }

    check(append_tokens(*seq1, 8, env.stream), "eviction append 1");
    const std::vector<uint32_t> tokens1 = make_tokens(8, 1000);
    check(seed_kv(*env.kv_bf16, *seq1, 31, env.stream), "eviction seed kv 1");
    check(seed_state(*env.gdn, seq1->slot, 31, env.stream), "eviction seed state 1");
    check(cache.save_input_boundary(*seq1, tokens1, 8, env.stream).ok(),
          "eviction save 1");
    check(cache.stats().used_pages == 2, "eviction first save pages");

    check(append_tokens(*seq2, 8, env.stream), "eviction append 2");
    const std::vector<uint32_t> tokens2 = make_tokens(8, 4000);
    check(seed_kv(*env.kv_bf16, *seq2, 32, env.stream), "eviction seed kv 2");
    check(seed_state(*env.gdn, seq2->slot, 32, env.stream), "eviction seed state 2");
    check(cache.save_input_boundary(*seq2, tokens2, 8, env.stream).ok(),
          "eviction save 2");
    check(cache.stats().entries == 2, "eviction demote keeps both entries");
    check(cache.stats().states_migrated_cold == 1, "eviction state demoted to cold");
    check(cache.stats().used_pages == 4, "eviction pages after demote");

    check(append_tokens(*seq3, 8, env.stream), "eviction append 3");
    const std::vector<uint32_t> tokens3 = make_tokens(8, 6000);
    check(seed_kv(*env.kv_bf16, *seq3, 33, env.stream), "eviction seed kv 3");
    check(seed_state(*env.gdn, seq3->slot, 33, env.stream), "eviction seed state 3");
    check(cache.save_input_boundary(*seq3, tokens3, 8, env.stream).ok(),
          "eviction save 3 cold migration");
    check(cache.stats().pages_migrated_cold == 2, "eviction cold migration page count");
    check(cache.stats().entries_evicted == 1, "eviction cold overflow drops entry");
    check(cache.stats().entries == 2, "eviction entries after overflow");
    check(cache.stats().used_pages == 4, "eviction hot page budget after migration");
    check(cache.stats().cold_pages_used == 2, "eviction cold page gauge");

    check(append_tokens(*seq4, 8, env.stream), "eviction append 4");
    const std::vector<uint32_t> tokens4 = make_tokens(8, 8000);
    check(seed_kv(*env.kv_bf16, *seq4, 34, env.stream), "eviction seed kv 4");
    check(seed_state(*env.gdn, seq4->slot, 34, env.stream), "eviction seed state 4");
    check(cache.save_input_boundary(*seq4, tokens4, 8, env.stream).ok(),
          "eviction save 4 leaf lru");
    check(cache.stats().pages_evicted == 2, "eviction leaf lru page count");
    check(cache.stats().pages_dropped_cold == 2, "eviction dropped instead of cold");
    check(cache.stats().used_pages == 4, "eviction leaf lru page budget");
    check(cache.stats().entries == 2, "eviction entry after leaf eviction");

    free_seq(seq4, env.stream);
    free_seq(seq3, env.stream);
    free_seq(seq2, env.stream);
    free_seq(seq1, env.stream);

    {
    PrefixCacheConfig blocked_config;
    blocked_config.hot_capacity_tokens = 16;
    blocked_config.cold_capacity_tokens = 0;
    blocked_config.max_concurrent = 2;
    blocked_config.min_prefix_tokens = 4;
    auto blocked_cache_opt = make_cache(env, *env.kv_bf16, blocked_config);
    check(blocked_cache_opt.has_value(), "eviction blocked cache create");
    if (!blocked_cache_opt.has_value()) return;
    PrefixCache& blocked_cache = *blocked_cache_opt;

    auto seq4 = make_seq(env, *env.kv_bf16);
    auto seq5 = make_seq(env, *env.kv_bf16);
    auto seq6 = make_seq(env, *env.kv_bf16);
    check(seq4.has_value() && seq5.has_value() && seq6.has_value(),
          "eviction blocked seqs");
    if (!seq4.has_value() || !seq5.has_value() || !seq6.has_value()) return;

    check(append_tokens(*seq4, 8, env.stream), "eviction blocked append 4");
    const std::vector<uint32_t> tokens4 = make_tokens(8, 1000);
    check(seed_kv(*env.kv_bf16, *seq4, 41, env.stream), "eviction blocked seed kv 4");
    check(seed_state(*env.gdn, seq4->slot, 41, env.stream),
          "eviction blocked seed state 4");
    check(blocked_cache.save_input_boundary(*seq4, tokens4, 8, env.stream).ok(),
          "eviction blocked save 4");

    check(append_tokens(*seq5, 8, env.stream), "eviction blocked append 5");
    const std::vector<uint32_t> tokens5 = make_tokens(8, 4000);
    check(seed_kv(*env.kv_bf16, *seq5, 42, env.stream), "eviction blocked seed kv 5");
    check(seed_state(*env.gdn, seq5->slot, 42, env.stream),
          "eviction blocked seed state 5");
    check(blocked_cache.save_input_boundary(*seq5, tokens5, 8, env.stream).ok(),
          "eviction blocked save 5");

    check(append_tokens(*seq6, 8, env.stream), "eviction blocked append 6");
    const std::vector<uint32_t> tokens6 = make_tokens(8, 7000);
    check(seed_kv(*env.kv_bf16, *seq6, 43, env.stream), "eviction blocked seed kv 6");
    check(seed_state(*env.gdn, seq6->slot, 43, env.stream),
          "eviction blocked seed state 6");

    const uint32_t active_before = env.kv_bf16->num_used_pages();
    const shelf::Status blocked_st =
        blocked_cache.save_input_boundary(*seq6, tokens6, 8, env.stream);
    const uint32_t active_after = env.kv_bf16->num_used_pages();

    check(blocked_st.code() == shelf::Status::Code::insufficient_memory,
          "eviction reports skipped too large");
    const PrefixCacheStats stats = blocked_cache.stats();
    check(stats.skipped_too_large == 1, "eviction too large counted");
    check(stats.pages_evicted == 0, "eviction protected pages untouched");
    check(stats.entries == 2, "eviction entries untouched on failure");
    check(stats.used_pages == 4, "eviction shadow pages untouched on failure");
    check(active_before == active_after, "eviction active pool untouched");

    free_seq(seq6, env.stream);
    free_seq(seq5, env.stream);
    free_seq(seq4, env.stream);
    }
}

bool save_prefix(
    Env& env,
    PrefixCache& cache,
    PagedKVPool& kv,
    uint32_t base,
    uint32_t tag,
    std::optional<PagedSequenceState>& seq_out) {
    std::optional<PagedSequenceState> seq = make_seq(env, kv);
    if (!seq.has_value()) return false;
    seq_out.emplace(std::move(seq.value()));
    if (!append_tokens(*seq_out, 8, env.stream)) return false;
    const std::vector<uint32_t> tokens = make_tokens(8, base);
    if (!seed_kv(kv, *seq_out, tag, env.stream)) return false;
    if (!seed_state(*env.gdn, seq_out->slot, tag, env.stream)) return false;
    return cache.save_input_boundary(*seq_out, tokens, 8, env.stream).ok();
}

void test_tier_migration(Env& env) {
    PrefixCacheConfig config;
    config.hot_capacity_tokens = 16;
    config.cold_capacity_tokens = 32;
    config.max_concurrent = 1;
    config.cold_state_slots = 4;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "tier cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    std::optional<PagedSequenceState> seq_a;
    std::optional<PagedSequenceState> seq_b;
    std::optional<PagedSequenceState> seq_c;
    check(save_prefix(env, cache, *env.kv_bf16, 20000, 70, seq_a), "tier save a");
    check(save_prefix(env, cache, *env.kv_bf16, 21000, 71, seq_b), "tier save b");
    check(save_prefix(env, cache, *env.kv_bf16, 22000, 72, seq_c), "tier save c");

    const PrefixCacheStats stats = cache.stats();
    check(stats.pages_migrated_cold == 2, "tier oldest pages migrated to cold");
    check(stats.cold_pages_used == 2, "tier cold page gauge");
    check(stats.used_pages == 4, "tier hot page budget constant");
    check(stats.states_migrated_cold == 2, "tier entry states demoted to cold");
    check(stats.cold_states_used == 2, "tier cold state gauge");
    check(stats.pinned_demoted == 0, "tier pinned pages not demoted");

    const PrefixHit hit_a = cache.lookup(make_tokens(8, 20000));
    check(hit_a.hit() && hit_a.state_position == 8, "tier cold entry still hits");

    auto dst_a = make_seq(env, *env.kv_bf16);
    check(dst_a.has_value(), "tier dst a");
    if (!dst_a.has_value()) return;
    check(cache.restore(hit_a.entry_id, *dst_a, env.stream).ok(),
          "tier restore cold entry");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "tier restore sync");
        return;
    }
    check(dst_a->position == 8, "tier cold restore position");
    check(verify_kv(*env.kv_bf16, *dst_a, 70, env.stream), "tier cold restore kv bytes");
    check(verify_state(*env.gdn, dst_a->slot, 70, env.stream),
          "tier cold restore state bytes");

    const PrefixHit hit_b = cache.lookup(make_tokens(8, 21000));
    check(hit_b.hit() && hit_b.state_position == 8, "tier pinned entry still hits");
    auto dst_b = make_seq(env, *env.kv_bf16);
    check(dst_b.has_value(), "tier dst b");
    if (!dst_b.has_value()) return;
    check(cache.restore(hit_b.entry_id, *dst_b, env.stream).ok(),
          "tier restore pinned entry");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "tier restore pinned sync");
        return;
    }
    check(verify_kv(*env.kv_bf16, *dst_b, 71, env.stream),
          "tier pinned restore kv bytes");
    check(verify_state(*env.gdn, dst_b->slot, 71, env.stream),
          "tier pinned restore state bytes");

    free_seq(dst_b, env.stream);
    free_seq(dst_a, env.stream);
    free_seq(seq_c, env.stream);
    free_seq(seq_b, env.stream);
    free_seq(seq_a, env.stream);
}

void test_pin_priority(Env& env) {
    PrefixCacheConfig config;
    config.hot_capacity_tokens = 24;
    config.cold_capacity_tokens = 64;
    config.max_concurrent = 2;
    config.cold_state_slots = 8;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "pin cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    std::optional<PagedSequenceState> seq_a;
    std::optional<PagedSequenceState> seq_b;
    std::optional<PagedSequenceState> seq_c;
    std::optional<PagedSequenceState> seq_d;
    check(save_prefix(env, cache, *env.kv_bf16, 24000, 75, seq_a), "pin save a");
    check(save_prefix(env, cache, *env.kv_bf16, 25000, 76, seq_b), "pin save b");
    check(save_prefix(env, cache, *env.kv_bf16, 26000, 77, seq_c), "pin save c");
    check(cache.stats().pages_migrated_cold == 0, "pin no migration before pressure");
    check(save_prefix(env, cache, *env.kv_bf16, 27000, 78, seq_d), "pin save d");

    const PrefixCacheStats stats = cache.stats();
    check(stats.pages_migrated_cold == 2, "pin oldest unpinned pages migrated first");
    check(stats.pinned_demoted == 0, "pin recent pages stay in hot tier");
    check(stats.cold_pages_used == 2, "pin cold page gauge");
    check(stats.states_migrated_cold == 2, "pin entry states demoted");
    check(stats.cold_states_used == 2, "pin cold state gauge");
    check(stats.used_pages == 6, "pin hot page budget constant");

    const PrefixHit hit_b = cache.lookup(make_tokens(8, 25000));
    check(hit_b.hit() && hit_b.state_position == 8, "pin second entry hits");
    auto dst_b = make_seq(env, *env.kv_bf16);
    check(dst_b.has_value(), "pin dst b");
    if (!dst_b.has_value()) return;
    check(cache.restore(hit_b.entry_id, *dst_b, env.stream).ok(), "pin restore b");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "pin restore sync");
        return;
    }
    check(verify_kv(*env.kv_bf16, *dst_b, 76, env.stream), "pin restore b kv bytes");
    check(verify_state(*env.gdn, dst_b->slot, 76, env.stream),
          "pin restore b state bytes");

    const PrefixHit hit_a = cache.lookup(make_tokens(8, 24000));
    check(hit_a.hit() && hit_a.state_position == 8, "pin oldest entry hits");
    auto dst_a = make_seq(env, *env.kv_bf16);
    check(dst_a.has_value(), "pin dst a");
    if (!dst_a.has_value()) return;
    check(cache.restore(hit_a.entry_id, *dst_a, env.stream).ok(), "pin restore a");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "pin restore a sync");
        return;
    }
    check(verify_kv(*env.kv_bf16, *dst_a, 75, env.stream), "pin restore a kv bytes");
    check(verify_state(*env.gdn, dst_a->slot, 75, env.stream),
          "pin restore a state bytes");

    free_seq(dst_a, env.stream);
    free_seq(dst_b, env.stream);
    free_seq(seq_d, env.stream);
    free_seq(seq_c, env.stream);
    free_seq(seq_b, env.stream);
    free_seq(seq_a, env.stream);
}

void test_state_tier(Env& env) {
    PrefixCacheConfig config;
    config.hot_capacity_tokens = 16;
    config.cold_capacity_tokens = 32;
    config.max_concurrent = 1;
    config.cold_state_slots = 1;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "state tier cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    std::optional<PagedSequenceState> seq_a;
    std::optional<PagedSequenceState> seq_b;
    std::optional<PagedSequenceState> seq_c;
    check(save_prefix(env, cache, *env.kv_bf16, 28000, 80, seq_a), "state tier save a");
    check(save_prefix(env, cache, *env.kv_bf16, 29000, 81, seq_b), "state tier save b");
    check(cache.stats().states_migrated_cold == 1, "state tier first demote");
    check(cache.stats().entries == 2, "state tier entries after demote");

    check(save_prefix(env, cache, *env.kv_bf16, 30000, 82, seq_c), "state tier save c");
    const PrefixCacheStats stats = cache.stats();
    check(stats.entries == 2, "state tier entries after cold overflow");
    check(stats.entries_evicted == 1, "state tier cold lru removed");
    check(stats.states_migrated_cold == 2, "state tier second demote");
    check(stats.cold_states_used == 1, "state tier cold state gauge");
    check(stats.pages_migrated_cold == 2, "state tier page migration");

    const PrefixHit hit_a = cache.lookup(make_tokens(8, 28000));
    check(!hit_a.hit() && hit_a.match_tokens == 8, "state tier removed entry misses");

    const PrefixHit hit_b = cache.lookup(make_tokens(8, 29000));
    check(hit_b.hit() && hit_b.state_position == 8, "state tier demoted entry hits");
    auto dst_b = make_seq(env, *env.kv_bf16);
    check(dst_b.has_value(), "state tier dst b");
    if (!dst_b.has_value()) return;
    check(cache.restore(hit_b.entry_id, *dst_b, env.stream).ok(),
          "state tier restore cold state");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "state tier restore sync");
        return;
    }
    check(dst_b->position == 8, "state tier restore position");
    check(verify_kv(*env.kv_bf16, *dst_b, 81, env.stream),
          "state tier restore kv bytes");
    check(verify_state(*env.gdn, dst_b->slot, 81, env.stream),
          "state tier restore gdn bytes");

    const PrefixHit hit_c = cache.lookup(make_tokens(8, 30000));
    check(hit_c.hit() && hit_c.state_position == 8, "state tier hot entry hits");
    auto dst_c = make_seq(env, *env.kv_bf16);
    check(dst_c.has_value(), "state tier dst c");
    if (!dst_c.has_value()) return;
    check(cache.restore(hit_c.entry_id, *dst_c, env.stream).ok(),
          "state tier restore hot state");
    if (hipStreamSynchronize(env.stream) != hipSuccess) {
        check(false, "state tier restore c sync");
        return;
    }
    check(verify_state(*env.gdn, dst_c->slot, 82, env.stream),
          "state tier restore hot gdn bytes");

    free_seq(dst_c, env.stream);
    free_seq(dst_b, env.stream);
    free_seq(seq_c, env.stream);
    free_seq(seq_b, env.stream);
    free_seq(seq_a, env.stream);
}

void test_spec_ring(Env& env) {
    constexpr uint32_t kRingBytes = 64;
    PrefixCacheConfig config;
    config.hot_capacity_tokens = 16;
    config.cold_capacity_tokens = 32;
    config.max_concurrent = 1;
    config.cold_state_slots = 1;
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

void test_tier_fail_closed(Env& env) {
    PrefixCacheConfig config;
    config.hot_capacity_tokens = 16;
    config.cold_capacity_tokens = 8;
    config.max_concurrent = 1;
    config.cold_state_slots = 2;
    config.min_prefix_tokens = 4;
    auto cache_opt = make_cache(env, *env.kv_bf16, config);
    check(cache_opt.has_value(), "fail closed cache create");
    if (!cache_opt.has_value()) return;
    PrefixCache& cache = *cache_opt;

    std::optional<PagedSequenceState> seq_a;
    std::optional<PagedSequenceState> seq_b;
    std::optional<PagedSequenceState> seq_c;
    check(save_prefix(env, cache, *env.kv_bf16, 31000, 85, seq_a), "fail closed save a");
    check(save_prefix(env, cache, *env.kv_bf16, 32000, 86, seq_b), "fail closed save b");
    check(save_prefix(env, cache, *env.kv_bf16, 33000, 87, seq_c), "fail closed save c");

    auto seq_d = make_seq(env, *env.kv_bf16);
    check(seq_d.has_value(), "fail closed seq d");
    if (!seq_d.has_value()) return;
    check(append_tokens(*seq_d, 8, env.stream), "fail closed append d");
    const std::vector<uint32_t> tokens_d = make_tokens(8, 34000);
    check(seed_kv(*env.kv_bf16, *seq_d, 88, env.stream), "fail closed seed kv d");
    check(seed_state(*env.gdn, seq_d->slot, 88, env.stream), "fail closed seed state d");

    const uint32_t active_before = env.kv_bf16->num_used_pages();
    const shelf::Status st =
        cache.save_input_boundary(*seq_d, tokens_d, 8, env.stream);
    const uint32_t active_after = env.kv_bf16->num_used_pages();

    check(st.code() == shelf::Status::Code::insufficient_memory,
          "fail closed reports insufficient memory");
    const PrefixCacheStats stats = cache.stats();
    check(stats.skipped_too_large == 1, "fail closed too large counted");
    check(stats.pages_migrated_cold == 2, "fail closed migration limited by cold tier");
    check(stats.pages_dropped_cold == 0, "fail closed no pages dropped");
    check(stats.pinned_demoted == 0, "fail closed no pin demotion");
    check(stats.entries == 3, "fail closed entries untouched");
    check(stats.used_pages == 4, "fail closed shadow pages untouched");
    check(active_before == active_after, "fail closed active pool untouched");

    const PrefixHit hit_b = cache.lookup(make_tokens(8, 32000));
    check(hit_b.hit() && hit_b.state_position == 8, "fail closed entries still hit");

    free_seq(seq_d, env.stream);
    free_seq(seq_c, env.stream);
    free_seq(seq_b, env.stream);
    free_seq(seq_a, env.stream);
}

}

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
    auto gdn_result = GdnStatePool::create(
        arena, 12, GdnStatePoolLayout::make(2, 8, 3, 2, 4, 4));
    auto slots_result = SequenceSlotPool::create(arena, 12, 64);

    if (!kv_bf16_result.ok() || !kv_fp8_result.ok() || !gdn_result.ok() ||
        !slots_result.ok()) {
        std::printf("FAIL: pool create\n");
        (void)hipStreamDestroy(stream);
        return 1;
    }

    PagedKVPool kv_bf16 = kv_bf16_result.release();
    PagedKVPool kv_fp8 = kv_fp8_result.release();
    GdnStatePool gdn = gdn_result.release();
    SequenceSlotPool slots = slots_result.release();

    Env env;
    env.arena = &arena;
    env.kv_bf16 = &kv_bf16;
    env.kv_fp8 = &kv_fp8;
    env.gdn = &gdn;
    env.slots = &slots;
    env.stream = stream;

    test_radix(env);
    test_refcount_share(env);
    test_hit_rule(env);
    test_roundtrip(env, kv_bf16, 77);
    test_roundtrip(env, kv_fp8, 78);
    test_capture(env);
    test_eviction(env);
    test_tier_migration(env);
    test_pin_priority(env);
    test_state_tier(env);
    test_spec_ring(env);
    test_tier_fail_closed(env);

    const shelf::Status arena_release = arena.shutdown();
    check(arena_release.ok(), "arena shutdown");

    if (hipStreamDestroy(stream) != hipSuccess) {
        std::printf("FAIL: hipStreamDestroy\n");
        ++g_fail;
    }

    std::printf("passed=%d failed=%d\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
