#include "../support/hy4_logical_ranks.h"
#include <shelf/models/hy4_flash/target.h>
#include <shelf/models/hy4_flash/generate.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace shelf::hy4 {
void benchmark_hy4_decode(Residency &gpu);
void benchmark_hy4_index(Residency &gpu);
void check_hy4_prefill_capacity(Residency &gpu, int rows);
void benchmark_hy4_prefill(Residency &gpu, int rows);
void benchmark_hy4_moe(Residency &gpu);
void benchmark_hy4_router(Residency &gpu);
void benchmark_hy4_steps(Residency &gpu);
void benchmark_hy4_ctx(Residency &gpu, int ctx);
void benchmark_hy4_bits3(Residency &gpu);
}

namespace {
std::atomic<int> remaining_instantiations{0};
using namespace shelf;
using namespace shelf::hy4;

void share_expert_weights(Checkpoint &model, bool allow_mixed = false) {
    if (model.mixed && !allow_mixed)
        throw std::invalid_argument("expert 領域を共有する試験には 4 bpw checkpoint が必要です");
    model.bytes[0] = 0;
    model.upload_order[0].clear();
    for (auto &[key, weight] : model.local[0]) {
        auto pos = key.find(".mlp.experts.");
        if (pos != std::string::npos && !key.starts_with("model.layers.1."))
            continue;
        weight.offset = (model.bytes[0] + 15) & ~size_t(15);
        model.bytes[0] = weight.offset + weight.bytes;
        model.upload_order[0].push_back(&weight);
    }
    for (auto &[key, weight] : model.local[0]) {
        auto pos = key.find(".mlp.experts.");
        if (pos != std::string::npos) {
            const auto &src = model.local[0].at("model.layers.1" + key.substr(pos));
            weight.offset = src.offset;
            weight.bytes = src.bytes;
            weight.spec.shape = src.spec.shape;
        }
    }
}

void logical_lm_head_tp(Residency &gpu, Checkpoint &sharded) {
    if (!sharded.split_lm_head || sharded.ranks != gpu.ranks)
        throw std::runtime_error("logical_lm_head_tp の前提が不正です");
    const int ranks = gpu.ranks;
    gpu.lm_head_tp = true;
    gpu.model.split_lm_head = true;
    const auto transport = std::getenv("SHELF_HY4_EP_TRANSPORT");
    const bool host = transport && std::string_view(transport) == "host";
    gpu.lm_host_fallback = host;
    for (int r = 0; r < ranks; ++r) {
        size_t weight_bytes = 0;
        for (const char *suffix : {"lm_head.suh", "lm_head.svh", "lm_head.trellis"})
            weight_bytes += ((sharded.at(suffix, r).bytes + 255) & ~size_t(255));
        const size_t buf_bytes = 2 * lm_hidden_chunk_bytes + 2 * 4 * lm_argmax_stride +
                                 (r == 0 ? 2 * lm_logits_chunk_bytes : 0) + (r > 0 ? weight_bytes : 0) + 4096;
        auto result = gpu::GpuArena::create(0, buf_bytes);
        if (!result.ok())
            throw std::runtime_error(result.status().message());
        gpu.arenas.push_back(result.release());
        auto alloc = [&](size_t sz) {
            auto mem = gpu.arenas.back().allocate_aligned(sz, 256);
            if (!mem.ok())
                throw std::runtime_error(mem.status().message());
            return mem.value().data();
        };
        gpu.lm_hidden_buf[r] = alloc(2 * lm_hidden_chunk_bytes);
        gpu.lm_head_recv[r] = alloc(2 * 4 * lm_argmax_stride);
        if (r == 0)
            gpu.lm_logits_recv[0] = alloc(2 * lm_logits_chunk_bytes);
        if (host && !gpu.lm_staging[r])
            checked(hipHostMalloc(&gpu.lm_staging[r], Residency::lm_staging_bytes(), hipHostMallocPortable));
        for (int p = 0; p < 2; ++p)
            for (auto events : {&gpu.lm_ready, &gpu.lm_done, &gpu.lm_arrived, &gpu.lm_consumed})
                gpu.create_event((*events)[r][p], r);
        if (r == 0) {
            for (const char *suffix : {"lm_head.suh", "lm_head.svh", "lm_head.trellis"}) {
                const auto &src_w = sharded.at(suffix, 0);
                auto &dst_w = gpu.model.local[0].at(suffix);
                dst_w.spec = src_w.spec;
                dst_w.bytes = src_w.bytes;
                checked(hipMemcpy(gpu.weight_base[0] + dst_w.offset, src_w.host, src_w.bytes, hipMemcpyHostToDevice));
            }
        } else {
            uint8_t *wbase = static_cast<uint8_t *>(alloc(weight_bytes));
            gpu.weight_base[r] = wbase;
            size_t off = 0;
            for (const char *suffix : {"lm_head.suh", "lm_head.svh", "lm_head.trellis"}) {
                Weight w = sharded.at(suffix, r);
                w.offset = off;
                checked(hipMemcpy(wbase + off, w.host, w.bytes, hipMemcpyHostToDevice));
                off += (w.bytes + 255) & ~size_t(255);
                gpu.model.local[r].emplace(suffix, std::move(w));
            }
        }
    }
}

void share_tp_weights(Checkpoint &model) {
    if (model.mixed)
        throw std::invalid_argument("expert 領域を共有する試験には 4 bpw checkpoint が必要です");
    for (int r = 0; r < model.ranks; ++r) {
        model.bytes[r] = 0;
        model.upload_order[r].clear();
        for (auto &[key, weight] : model.local[r]) {
            auto pos = key.find(".mlp.experts.");
            if (pos != std::string::npos && !key.starts_with("model.layers.1."))
                continue;
            weight.offset = (model.bytes[r] + 15) & ~size_t(15);
            model.bytes[r] = weight.offset + weight.bytes;
            model.upload_order[r].push_back(&weight);
        }
        for (auto &[key, weight] : model.local[r]) {
            auto pos = key.find(".mlp.experts.");
            if (pos != std::string::npos)
                weight.offset = model.local[r].at("model.layers.1" + key.substr(pos)).offset;
        }
    }
}

Owners make_tp_owners(int mg) {
    Owners layout{};
    for (auto &row : layout)
        for (int e = 0; e < 512; ++e)
            row[e] = e % mg;
    return layout;
}

Path resolve_engram(const Path &body) {
    const char *env = std::getenv("SHELF_HY4_ENGRAM_DIR");
    return env && *env ? Path(env) : body;
}

void reserve_logical(Residency &gpu, size_t work = size_t(1) << 30) {
    auto align = [](size_t n) { return (n + 255) & ~size_t(255); };
    gpu.ranks = gpu.model.ranks;
    gpu.lm_head_tp = gpu.model.split_lm_head;
    gpu.lm_host_fallback = false;
    gpu.lm_parity = 0;
    gpu.lm_ready_rec = {};
    gpu.lm_done_rec = {};
    gpu.lm_arrived_rec = {};
    gpu.lm_consumed_rec = {};
    gpu.kv_page_stride = gpu.pages_for_tokens(gpu.tokens);
    gpu.kv_pages_total = size_t(gpu.max_slots) * gpu.kv_page_stride;
    gpu.physical.fill(0);
    std::array<std::array<bool, 4>, 4> peer{};
    for (auto &row : peer)
        row.fill(true);
    gpu.graph_room = graph_topology(gpu.ep_size, gpu.tp_size, true, true, peer);
    for (int r = 0; r < gpu.ranks; ++r) {
        const bool attention_rank = gpu.tp_size > 1 || r == 0;
        size_t overhead = work + Residency::matrix_bytes;
        if (attention_rank) {
            overhead += align(41 * 512 * sizeof(int)) + align(41 * 512 * sizeof(float)) + 256;
            overhead += align(size_t(gpu.max_slots) * gpu.kv_page_stride * sizeof(int32_t));
            for (int i = 0; i < 42; ++i) {
                if (!Residency::is_sparse_layer(i)) {
                    overhead += align(gpu.max_slots * 294 * std::min(gpu.tokens, size_t(518)));
                } else {
                    overhead += align(gpu.kv_pages_total * kv_page_tokens * 294);
                    overhead += align(gpu.kv_pages_total * kv_page_groups * 128 * sizeof(uint16_t));
                    overhead += align(size_t(gpu.max_slots) * 12 * 128 * sizeof(uint16_t));
                }
            }
            for (const auto &[name, ignored] : gpu.folded[r])
                overhead += size_t(gpu.model.shape_of(name, r)->at(1)) * 512 * 2;
        }
        if (gpu.tp_size > 1)
            overhead += tp_prefill_rows * 3072u * 2u;
        if (gpu.tp_size > 1 && gpu.ep_size > 1)
            overhead += tp_prefill_rows * 1536u * 2u;
        if (gpu.lm_head_tp) {
            overhead += 2 * lm_hidden_chunk_bytes + 2 * 4 * lm_argmax_stride;
            if (r == 0 || gpu.tp_size > 1)
                overhead += 2 * lm_logits_chunk_bytes;
        }
        size_t bytes = align(gpu.model.bytes[r]) + overhead;
        auto result = gpu::GpuArena::create(0, bytes);
        if (!result.ok())
            throw std::runtime_error(result.status().message());
        gpu.arenas.push_back(result.release());
        auto allocation = gpu.arenas.back().allocate_aligned(bytes, 256);
        if (!allocation.ok())
            throw std::runtime_error(allocation.status().message());
        checked(hipMemset(allocation.value().data(), 0, bytes));
        Region buffer{static_cast<uint8_t *>(allocation.value().data()), bytes};
        checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
        gpu.weight_base[r] = static_cast<uint8_t *>(buffer.next(gpu.model.bytes[r]));
        if (attention_rank) {
            gpu.page_table[r] = static_cast<int32_t *>(buffer.next(size_t(gpu.max_slots) * gpu.kv_page_stride * sizeof(int32_t)));
            checked(hipMemset(gpu.page_table[r], 0xff, size_t(gpu.max_slots) * gpu.kv_page_stride * sizeof(int32_t)));
            for (int i = 0; i < 42; ++i) {
                if (!Residency::is_sparse_layer(i)) {
                    gpu.cache[r][i] = static_cast<uint8_t *>(
                        buffer.next(gpu.max_slots * 294 * std::min(gpu.tokens, size_t(518))));
                } else {
                    gpu.cache[r][i] = static_cast<uint8_t *>(buffer.next(gpu.kv_pages_total * kv_page_tokens * 294));
                    gpu.index_keys[r][i] = static_cast<uint16_t *>(
                        buffer.next(gpu.kv_pages_total * kv_page_groups * 128 * sizeof(uint16_t)));
                    gpu.index_pending[r][i] = static_cast<uint16_t *>(
                        buffer.next(size_t(gpu.max_slots) * 12 * 128 * sizeof(uint16_t)));
                }
            }
            for (auto &[name, ptr] : gpu.folded[r])
                ptr = buffer.next(size_t(gpu.model.shape_of(name, r)->at(1)) * 512 * 2);
        }
        if (gpu.tp_size > 1)
            gpu.tp_recv[r] = buffer.next(tp_prefill_rows * 3072u * 2u);
        if (gpu.tp_size > 1 && gpu.ep_size > 1)
            gpu.ep_recv[r] = buffer.next(tp_prefill_rows * 1536u * 2u);
        if (gpu.lm_head_tp) {
            gpu.lm_hidden_buf[r] = buffer.next(2 * lm_hidden_chunk_bytes);
            gpu.lm_head_recv[r] = buffer.next(2 * 4 * lm_argmax_stride);
            if (r == 0 || gpu.tp_size > 1)
                gpu.lm_logits_recv[r] = buffer.next(2 * lm_logits_chunk_bytes);
        }
        gpu.reserve_experts(r, buffer);
        gpu.workspace[r] = {static_cast<uint8_t *>(buffer.next(work)), work};
    }
    if (gpu.tp_size > 1) {
        for (int g = 0; g < gpu.ep_size; ++g)
            gpu.tp_barrier[g] = std::make_unique<SyncBarrier>(gpu.tp_size);
        if (gpu.ep_size > 1)
            for (int t = 0; t < gpu.tp_size; ++t)
                gpu.ep_barrier[t] = std::make_unique<SyncBarrier>(gpu.ep_size);
        gpu.all_barrier = std::make_unique<SyncBarrier>(gpu.ranks);
        gpu.capture_barrier = std::make_unique<SyncBarrier>(gpu.ranks);
    }
    gpu.kv_page_refs.assign(gpu.kv_pages_total, 0);
    gpu.kv_free_pages.resize(gpu.kv_pages_total);
    for (size_t i = 0; i < gpu.kv_pages_total; ++i)
        gpu.kv_free_pages[i] = int32_t(gpu.kv_pages_total - 1 - i);
    gpu.slot_page_tables.assign(gpu.max_slots, {});
    gpu.upload();
}
void reset_history(Residency &gpu) {
    gpu.release_slot_pages(0);
    if (!gpu.allocate_slot_pages(0, gpu.tokens))
        throw std::runtime_error("KV ページ割当に失敗しました");
    std::vector<uint8_t> kv(gpu.tokens * 294);
    for (size_t row = 0; row < gpu.tokens; ++row) {
        auto rec = kv.data() + row * 294;
        for (int c = 0; c < 294; ++c)
            rec[c] = uint8_t(11 * row + 7 * c);
        for (int off : {128, 258, 292}) {
            rec[off] = 0;
            rec[off + 1] = 0x30;
        }
    }
    std::vector<uint16_t> groups((gpu.tokens / 4 + 12) * 128);
    for (size_t i = 0; i < groups.size(); ++i)
        groups[i] = uint16_t(std::bit_cast<uint32_t>(float(int(i * 37 % 101) - 50) / 64) >> 16);
    for (int r = 0; r < gpu.ranks; ++r) {
        if (!gpu.cache[r][0])
            continue;
        for (int l = 0; l < 42; ++l) {
            if (!Residency::is_sparse_layer(l)) {
                auto slots = std::min(gpu.tokens, size_t(518));
                checked(hipMemcpy(gpu.cache[r][l], kv.data(), slots * 294, hipMemcpyHostToDevice));
                continue;
            }
            const auto &pages = gpu.slot_page_tables[0];
            const size_t total_groups = gpu.tokens / 4;
            for (size_t pi = 0; pi < pages.size(); ++pi) {
                const int32_t phys = pages[pi];
                const size_t tok_off = pi * kv_page_tokens;
                const size_t tok_cnt = std::min(kv_page_tokens, gpu.tokens - tok_off);
                checked(hipMemcpy(gpu.cache[r][l] + size_t(phys) * kv_page_tokens * 294,
                                  kv.data() + tok_off * 294, tok_cnt * 294, hipMemcpyHostToDevice));
                const size_t grp_off = pi * kv_page_groups;
                if (grp_off < total_groups) {
                    const size_t grp_cnt = std::min(kv_page_groups, total_groups - grp_off);
                    checked(hipMemcpy(gpu.index_keys[r][l] + size_t(phys) * kv_page_groups * 128,
                                      groups.data() + grp_off * 128, grp_cnt * 256, hipMemcpyHostToDevice));
                }
            }
            checked(hipMemcpy(gpu.index_pending[r][l], groups.data() + total_groups * 128,
                              12 * 256, hipMemcpyHostToDevice));
        }
    }
}

void check_sliding_chunks(Residency &gpu) {
    Ops op(gpu);
    std::vector<int> ids(20);
    std::iota(ids.begin(), ids.end(), 100);
    std::vector<uint16_t> reference;
    for (int chunk : {1, 8, 15, 20}) {
        reset_history(gpu);
        checked(hipMemsetAsync(op.arena.base, 0xff, op.arena.size, op.stream));
        std::vector<uint16_t> output(ids.size() * 3072);
        for (int i = 0; i < int(ids.size()); i += chunk) {
            int rows = std::min(chunk, int(ids.size()) - i);
            op.arena.cursor = 0;
            Mat x = op.embed({ids.begin() + i, ids.begin() + i + rows}, false);
            Selection selection;
            Mat result = attention(op, "model.layers.2.", 2, 511 + i, x, selection, false);
            op.copy(output.data() + size_t(i) * 3072, result.p, result.bytes(), hipMemcpyDeviceToHost);
        }
        if (reference.empty())
            reference = output;
        if (output != reference)
            throw std::runtime_error("sliding window の境界でチャンク間の出力が異なります");
    }
}

struct Output {
    std::vector<int> ids, draft;
    std::vector<uint16_t> state;
    std::vector<float> target_probs, draft_probs;
};

std::vector<Output> run(Residency &gpu, bool graphs) {
    setenv("SHELF_HY4_GRAPH", graphs ? "1" : "0", 1);
    setenv("SHELF_HY4_DRAFT_GRAPH", graphs ? "1" : "0", 1);
    reset_history(gpu);
    Target target(gpu, 0.8f, 42, 0, 1);
    target.history.assign(7, 120000);
    const int drafts = gpu.draft_tokens;
    const int block_rows = drafts + 1;
    const std::vector<int> full_verify{802, 15063, 2576, 12286, 859, 444};
    std::vector<Output> results;
    for (int raw_keep : {0, 3, 0, 0, 4, 2, 5}) {
        int keep = raw_keep ? std::min(raw_keep, block_rows) : 0;
        if (results.size() == 2) {
            reset_history(gpu);
            target.history.assign(2044, 120000);
        }
        int start = target.history.size();
        std::vector<int> ids = keep ? std::vector<int>(full_verify.begin(), full_verify.begin() + block_rows)
                                    : std::vector<int>{start};
        Output out;
        out.ids = target.forward(ids, keep != 0);
        out.state.resize(ids.size() * 3072);
        target.op.copy(out.state.data(), target.state.p, out.state.size() * 2, hipMemcpyDeviceToHost);
        if (keep) {
            out.target_probs.resize(block_rows * 120832);
            target.op.copy(out.target_probs.data(), target.target_probs, out.target_probs.size() * 4, hipMemcpyDeviceToHost);
            target.accept(keep, block_rows);
        }
        out.draft = target.propose(500 + start);
        if (int(out.draft.size()) != drafts)
            throw std::runtime_error("DFly2 の候補数が設定と一致しません");
        out.draft_probs.resize(drafts * 120832);
        target.op.copy(out.draft_probs.data(), target.draft_probs, out.draft_probs.size() * 4, hipMemcpyDeviceToHost);
        results.push_back(std::move(out));
    }
    return results;
}

bool same_bits(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void check_capture_failure(Residency &gpu) {
    const auto before_target = gpu.target_replays, before_draft = gpu.draft_replays;
    const int block_rows = gpu.draft_tokens + 1;
    const std::vector<int> full_verify{850, 401, 82, 44, 276, 903};
    const std::vector<int> verify_ids(full_verify.begin(), full_verify.begin() + block_rows);
    const int keep = std::min(3, block_rows);
    std::vector<int> expected_ids, expected_draft;
    for (bool inject : {false, true}) {
        setenv("SHELF_HY4_GRAPH", inject ? "1" : "0", 1);
        setenv("SHELF_HY4_DRAFT_GRAPH", inject ? "1" : "0", 1);
        reset_history(gpu);
        Target target(gpu, 0.f, 177, 0, 1.f);
        target.history.assign(2047, 120000);
        remaining_instantiations = inject ? 3 : 0;
        auto ids = target.forward(verify_ids, true);
        target.accept(keep, block_rows);
        remaining_instantiations = inject ? 1 : 0;
        auto draft = target.propose(ids[keep - 1]);
        if (!inject) {
            expected_ids = ids;
            expected_draft = draft;
        } else if (ids != expected_ids || draft != expected_draft || !gpu.target_capture_failed ||
                   !gpu.draft_capture_failed || gpu.target_replays != before_target || gpu.draft_replays != before_draft)
            throw std::runtime_error("capture 失敗後の eager 実行が基準と一致しません");
    }
}

void compare(const Path &body, const Path &draft, int ranks) {
    Checkpoint model(body, resolve_engram(body), draft, Owners{}, 1, 1);
    share_expert_weights(model);
    Residency gpu(model, 2080);
    auto status = gpu.reserve();
    if (!status.ok())
        throw std::runtime_error(status.message());
    gpu.upload();
    logical_ranks(gpu, ranks);
    if (ranks == 1)
        check_sliding_chunks(gpu);
    auto verify_run = [&](int k) {
        gpu.draft_tokens = k;
        auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
        auto eager = run(gpu, false);
        if (gpu.target_replays != target_before || gpu.draft_replays != draft_before)
            throw std::runtime_error("無効化した Graph が起動しました");
        auto graph = run(gpu, true);
        for (size_t i = 0; i < eager.size(); ++i) {
            if (eager[i].ids != graph[i].ids || eager[i].state != graph[i].state || !same_bits(eager[i].target_probs, graph[i].target_probs))
                throw std::runtime_error("Target の照合失敗: k=" + std::to_string(k) + " step=" + std::to_string(i));
            if (eager[i].draft != graph[i].draft || !same_bits(eager[i].draft_probs, graph[i].draft_probs))
                throw std::runtime_error("DFly2 の照合失敗: k=" + std::to_string(k) + " step=" + std::to_string(i));
        }
        if (gpu.target_replays - target_before != 7 || gpu.draft_replays - draft_before != 7)
            throw std::runtime_error("Graph の replay 回数が異なります");
    };
    verify_run(gpu.draft_tokens);
    if (gpu.target_victim != 4 || gpu.draft_victim != 2)
        throw std::runtime_error("Graph の capture 回数が異なります");
    if (ranks == 1) {
        for (int k : {2, 3, 5})
            verify_run(k);
        gpu.draft_tokens = default_draft_tokens;
    }
    check_capture_failure(gpu);
    printf("ep=%d k=%d target=7/7 draft=7/7 target_captures=4 draft_captures=2 capture_fallback=pass\n", ranks, gpu.draft_tokens);
    gpu.shutdown();
}
void compare_draft_counts(const Path &body, const Path &draft) {
    Checkpoint model(body, resolve_engram(body), draft, Owners{}, 1, 1);
    share_expert_weights(model);
    for (auto &owners : model.owners)
        owners.fill(0);
    std::vector<int> prompt_short{802, 15063, 2576, 12286};
    std::vector<int> prompt_512(510);
    std::vector<int> prompt_2048(2046);
    for (size_t i = 0; i < prompt_512.size(); ++i)
        prompt_512[i] = 100 + int(i * 13 % 97);
    for (size_t i = 0; i < prompt_2048.size(); ++i)
        prompt_2048[i] = 100 + int(i * 7 % 101);
    const std::vector<std::vector<int>> cases{prompt_short, prompt_512, prompt_2048};
    const std::vector<int> counts{12, 10, 10};
    std::vector<std::vector<int>> baselines;
    {
        Residency single_gpu(model, 2100, 1);
        auto status = single_gpu.reserve();
        if (!status.ok())
            throw std::runtime_error(status.message());
        single_gpu.upload();
        model.has_draft = false;
        setenv("SHELF_HY4_GRAPH", "0", 1);
        setenv("SHELF_HY4_DRAFT_GRAPH", "0", 1);
        for (size_t c = 0; c < cases.size(); ++c)
            baselines.push_back(generate(single_gpu, cases[c], counts[c], 1024, 0.f, 0, {}));
        model.has_draft = true;
        for (int k : {2, 3, 4, 5}) {
            single_gpu.draft_tokens = k;
            for (bool graph : {false, true}) {
                setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
                setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
                for (size_t c = 0; c < cases.size(); ++c) {
                    auto target_before = single_gpu.target_replays, draft_before = single_gpu.draft_replays;
                    auto got = generate(single_gpu, cases[c], counts[c], 1024, 0.f, 0, {}, 0, 1.f, {}, false, nullptr,
                                        nullptr, nullptr, k);
                    if (got != baselines[c])
                        throw std::runtime_error("単一スロット greedy 不一致: k=" + std::to_string(k) +
                                                 " case=" + std::to_string(c) + " graph=" + std::to_string(graph));
                    if (graph && (single_gpu.target_replays == target_before || single_gpu.draft_replays == draft_before))
                        throw std::runtime_error("単一スロット Graph が起動しませんでした: k=" + std::to_string(k));
                }
            }
            printf("k_parity_single k=%d=pass\n", k);
            fflush(stdout);
        }
        single_gpu.shutdown();
    }
    setenv("SHELF_HY4_BATCH_DFLY", "1", 1);
    Residency gpu(model, 2100, 16);
    auto status = gpu.reserve();
    if (!status.ok())
        throw std::runtime_error(status.message());
    gpu.upload();
    model.has_draft = true;
    for (int k : {2, 3, 4, 5}) {
        gpu.draft_tokens = k;
        for (bool graph : {false, true}) {
            setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
            setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
            for (int batch : {1, 2, 4, 8, 16}) {
                Hy4BatchOptions opts{1024, 0.f, 0, 0, 1.f, false, k};
                std::vector<std::vector<int>> inputs(batch, prompt_short);
                auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
                auto results = generate_batch(gpu, inputs, std::vector<int>(batch, counts[0]), opts);
                for (int i = 0; i < batch; ++i)
                    if (results[i] != baselines[0])
                        throw std::runtime_error("バッチ greedy 不一致: k=" + std::to_string(k) +
                                                 " B=" + std::to_string(batch) + " graph=" + std::to_string(graph));
                if (graph && batch == 1) {
                    std::vector<int> one_out;
                    auto sess_target_before = gpu.target_replays, sess_draft_before = gpu.draft_replays;
                    Hy4BatchSession one_session(
                        gpu, [&](int, int token) { one_out.push_back(token); }, [](int, const auto &, const char *) {}, k);
                    one_session.admit({0, prompt_short, counts[0], opts});
                    while (one_session.active() || one_session.queued())
                        one_session.step();
                    if (one_out != baselines[0])
                        throw std::runtime_error("単一セッション greedy 不一致: k=" + std::to_string(k));
                    if (gpu.target_replays == sess_target_before || gpu.draft_replays == sess_draft_before)
                        throw std::runtime_error("B=1 バッチセッション Graph が起動しませんでした: k=" + std::to_string(k));
                }
                if (graph && batch >= 2 && (gpu.target_replays == target_before || gpu.draft_replays == draft_before))
                    throw std::runtime_error("バッチ Graph が起動しませんでした: k=" + std::to_string(k));
                if (batch == 2 || batch == 4 || batch == 8 || batch == 16) {
                    std::vector<std::vector<int>> mixed_inputs;
                    std::vector<int> mixed_counts;
                    for (int i = 0; i < batch; ++i) {
                        size_t idx = (batch <= 4) ? ((i % 2) + 1) : (i % 3);
                        mixed_inputs.push_back(cases[idx]);
                        mixed_counts.push_back(counts[idx]);
                    }
                    auto mixed = generate_batch(gpu, mixed_inputs, mixed_counts, opts);
                    for (int i = 0; i < batch; ++i) {
                        size_t idx = (batch <= 4) ? ((i % 2) + 1) : (i % 3);
                        if (mixed[i] != baselines[idx])
                            throw std::runtime_error("境界混在バッチ greedy 不一致: k=" + std::to_string(k) +
                                                     " B=" + std::to_string(batch) + " graph=" + std::to_string(graph));
                    }
                }
            }
        }
        printf("k_parity k=%d single_and_batch(1,2,4,8,16)_512_2048=pass\n", k);
        fflush(stdout);
    }
    setenv("SHELF_HY4_GRAPH", "1", 1);
    setenv("SHELF_HY4_DRAFT_GRAPH", "1", 1);
    std::vector<std::vector<int>> session_out(4);
    Hy4BatchSession session(gpu, [&](int id, int token) { session_out[id].push_back(token); },
                            [](int, const auto &, const char *) {}, 2);
    for (int i = 0; i < 4; ++i)
        session.admit({i, cases[i % 3], counts[i % 3], {1024}});
    int step_idx = 0;
    while (session.active() || session.queued()) {
        session.set_draft_tokens(2 + (step_idx++ % 4));
        session.step();
    }
    for (int i = 0; i < 4; ++i)
        if (session_out[i] != baselines[i % 3])
            throw std::runtime_error("動的 K 切替 Hy4BatchSession が非投機と一致しません");
    printf("k_dynamic_session=pass\n");
    gpu.shutdown();
}
void compare_batches(const Path &body, const Path &draft, int ranks) {
    Checkpoint model(body, resolve_engram(body), draft, Owners{}, 1, 1);
    share_expert_weights(model);
    const std::vector<std::vector<int>> prompts{{802, 15063}, {12286, 859, 444}, {401, 82, 44, 276}, {120000}};
    constexpr int count = 4;
    for (auto [speculate, slots] : {std::pair{false, 32}, std::pair{true, 32}, std::pair{true, 4}}) {
        for (auto &owners : model.owners)
            owners.fill(0);
        setenv("SHELF_HY4_BATCH_DFLY", speculate ? "1" : "0", 1);
        Residency gpu(model, 32, slots);
        if (gpu.batch_draft != speculate) throw std::runtime_error("バッチ DFly2 の設定が異なります");
        auto status = gpu.reserve();
        if (!status.ok()) throw std::runtime_error(status.message());
        gpu.upload();
        logical_ranks(gpu, ranks);
        std::vector<std::vector<int>> baseline;
        model.has_draft = speculate;
        for (const auto &prompt : prompts)
            baseline.push_back(generate_batch(gpu, {prompt}, {count}, {}).front());
        model.has_draft = true;
        for (int batch = 2; slots == 32 && batch <= 32; ++batch) {
            if (batch > 16 && batch != 24 && batch != 32) continue;
            if (ranks != 1 && batch != 2 && batch != 4 && batch != 8 && batch != 16 && batch != 32) continue;
            std::vector<std::vector<int>> inputs;
            for (int i = 0; i < batch; ++i)
                inputs.push_back(prompts[i % prompts.size()]);
            for (bool graph : {false, true}) {
                setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
                setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
                auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
                auto results = generate_batch(gpu, inputs, std::vector<int>(batch, count), {});
                for (int i = 0; i < batch; ++i)
                    if (results.at(i) != baseline[i % prompts.size()])
                        throw std::runtime_error("バッチ生成が単独実行と異なります: B=" + std::to_string(batch) +
                                                 " slot=" + std::to_string(i));
                bool active = graph && graph_topology(ranks, 1, gpu.push, gpu.graph_room, gpu.peer);
                if ((gpu.target_replays > target_before) != active ||
                    (gpu.draft_replays > draft_before) != (active && speculate))
                    throw std::runtime_error("バッチ Graph の起動条件が異なります");
                printf("ep=%d draft=%d graph=%d batch=%d greedy=%d/%d\n", ranks, speculate, graph, batch, batch, batch);
                fflush(stdout);
            }
        }
        if (slots == 4) {
            std::vector<int> near_end(29, 802);
            auto tail = generate_batch(gpu, {near_end}, {3}, {}).front();
            std::array<std::vector<int>, 7> streamed;
            std::array<std::string, 7> reason;
            std::vector<int> started;
            Hy4BatchSession session(gpu, [&](int id, int token) {
                if (streamed[id].empty()) started.push_back(id);
                streamed[id].push_back(token);
            }, [&](int id, const std::vector<int> &ids, const char *finish) {
                if (ids != streamed[id]) throw std::runtime_error("配信した ID と終了通知が異なります");
                reason[id] = finish;
            });
            for (int id = 0; id < 6; ++id)
                session.admit({id, id == 5 ? near_end : prompts[id % 4], id == 5 ? 3 : count, {}});
            session.admit({6, prompts[2], count, {}});
            session.cancel(6);
            if (session.active() != 4 || session.queued() != 2)
                throw std::runtime_error("待機列の件数が異なります");
            for (int i = 0; i < 5; ++i) session.step();
            session.cancel(1);
            for (int i = 0; session.active() || session.queued(); ++i) {
                if (i > 64) throw std::runtime_error("バッチ処理が完了しません");
                session.step();
            }
            if (started != std::vector<int>{0, 1, 2, 3, 4, 5} || !session.draft_rounds() ||
                reason[1] != "cancelled" || reason[6] != "cancelled" || streamed[5] != tail)
                throw std::runtime_error("FIFO、キャンセルまたは KV 残量の検証に失敗しました");
            for (int id : {0, 2, 3, 4})
                if (streamed[id] != baseline[id % 4])
                    throw std::runtime_error("スロット再利用後の生成 ID が異なります");
            printf("ep=%d session=pass draft_rounds=%zu\n", ranks, session.draft_rounds());
        }
        if (slots == 4 && gpu.push) {
            std::vector<std::vector<int>> inputs{prompts[0], prompts[1]};
            std::vector<std::vector<int>> expected{baseline[0], baseline[1]};
            for (bool draft_only : {true, false}) {
                setenv("SHELF_HY4_GRAPH", draft_only ? "0" : "1", 1);
                setenv("SHELF_HY4_DRAFT_GRAPH", draft_only ? "1" : "0", 1);
                auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
                if (generate_batch(gpu, inputs, {count, count}, {}) != expected ||
                    (gpu.target_replays > target_before) == draft_only ||
                    (gpu.draft_replays > draft_before) != draft_only)
                    throw std::runtime_error("バッチ Graph の個別切り替えが異なります");
            }
            if (ranks == 1) {
                Hy4BatchOptions options{16, 0.7f, 0, 20, 0.95f};
                for (uint64_t seed : {117, 277}) {
                    options.seed = seed;
                    std::vector<std::vector<int>> sampled;
                    for (bool graph : {false, true}) {
                        setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
                        setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
                        auto ids = generate_batch(gpu, inputs, {6, 6}, options);
                        if (!graph) sampled = ids;
                        else if (ids != sampled)
                            throw std::runtime_error("バッチ Graph のサンプリング結果が異なります");
                    }
                }
            }
            for (bool fail_draft : {true, false}) {
                setenv("SHELF_HY4_GRAPH", fail_draft ? "0" : "1", 1);
                setenv("SHELF_HY4_DRAFT_GRAPH", fail_draft ? "1" : "0", 1);
                std::string errors;
                for (auto &plan : gpu.target_graphs) plan.release(gpu, errors);
                for (auto &plan : gpu.draft_graphs) plan.release(gpu, errors);
                if (!errors.empty()) throw std::runtime_error(errors);
                remaining_instantiations = fail_draft ? 1 : (ranks == 1 ? 2 : 3);
                auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
                if (generate_batch(gpu, inputs, {count, count}, {}) != expected ||
                    !(fail_draft ? gpu.draft_capture_failed : gpu.target_capture_failed) ||
                    gpu.target_replays != target_before || gpu.draft_replays != draft_before)
                    throw std::runtime_error("バッチ capture 失敗後の eager 復帰が異なります");
            }
        }
        gpu.shutdown();
    }
    setenv("SHELF_HY4_BATCH_DFLY", "1", 1);
    for (auto &owners : model.owners) owners.fill(0);
    Residency gpu(model, 2070, 2);
    auto status = gpu.reserve();
    if (!status.ok()) throw std::runtime_error(status.message());
    gpu.upload();
    logical_ranks(gpu, ranks);
    Hy4BatchOptions options{128};
    std::vector<std::vector<int>> inputs{std::vector<int>(2047), std::vector<int>(2053)};
    for (auto &input : inputs)
        for (size_t i = 0; i < input.size(); ++i) input[i] = 100 + int(i % 7);
    std::vector<std::vector<int>> expected;
    for (bool graph : {false, true}) {
        setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
        setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
        if (!graph)
            for (auto &input : inputs)
                expected.push_back(generate_batch(gpu, {input}, {6}, options).front());
        auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
        std::vector<std::vector<int>> generated(2);
        Hy4BatchSession session(gpu, [&](int id, int token) { generated[id].push_back(token); },
                                 [](int, const auto &, const char *) {});
        for (int i = 0; i < 2; ++i) session.admit({i, inputs[i], 6, options});
        while (session.active() || session.queued()) session.step();
        if (generated != expected)
            throw std::runtime_error("位置 2047・2053 混在時の生成 ID が異なります");
        if (graph && gpu.push && (gpu.target_replays == target_before || gpu.draft_replays == draft_before))
            throw std::runtime_error("疎な attention でバッチ Graph が起動しませんでした");
    }
    gpu.shutdown();
    if (ranks == 1) {
        auto make_prompt = [](int len, int salt) {
            std::vector<int> p(len);
            for (int i = 0; i < len; ++i) p[i] = 100 + (i + salt) % 13;
            return p;
        };
        std::vector<std::vector<int>> mix_prompts{
            make_prompt(128, 0), make_prompt(2045, 3), make_prompt(2099, 5), make_prompt(50, 7)};
        std::vector<std::vector<int>> sparse32(32, make_prompt(2099, 11));
        for (bool speculate : {false, true}) {
            setenv("SHELF_HY4_BATCH_DFLY", speculate ? "1" : "0", 1);
            for (auto &owners : model.owners) owners.fill(0);
            Residency mix_gpu(model, 2120, 32);
            auto mix_status = mix_gpu.reserve();
            if (!mix_status.ok()) throw std::runtime_error(mix_status.message());
            mix_gpu.upload();
            model.has_draft = speculate;
            std::vector<std::vector<int>> mix_ref;
            for (const auto &p : mix_prompts)
                mix_ref.push_back(generate_batch(mix_gpu, {p}, {16}, {1024, 0, 0, 0, 1}).front());
            std::vector<int> s32_ref = generate_batch(mix_gpu, {sparse32[0]}, {8}, {1024, 0, 0, 0, 1}).front();
            model.has_draft = true;
            for (bool graph : {false, true}) {
                setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
                setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
                auto got_mix = generate_batch(mix_gpu, mix_prompts, std::vector<int>(4, 16), {1024, 0, 0, 0, 1});
                for (size_t i = 0; i < mix_prompts.size(); ++i)
                    if (got_mix[i] != mix_ref[i])
                        throw std::runtime_error("長短混在バッチ生成が単独実行と異なります: draft=" +
                                                 std::to_string(speculate) + " graph=" + std::to_string(graph) +
                                                 " slot=" + std::to_string(i));
                auto got_32 = generate_batch(mix_gpu, sparse32, std::vector<int>(32, 8), {1024, 0, 0, 0, 1});
                for (int i = 0; i < 32; ++i)
                    if (got_32[i] != s32_ref)
                        throw std::runtime_error("c=32 長文バッチ生成が単独実行と異なります: draft=" +
                                                 std::to_string(speculate) + " graph=" + std::to_string(graph) +
                                                 " slot=" + std::to_string(i));
            }
            printf("mixed_sparse=pass c32_sparse=pass draft=%d\n", speculate);
            fflush(stdout);
            mix_gpu.shutdown();
        }
        setenv("SHELF_HY4_BATCH_DFLY", "1", 1);
    }
}

std::vector<float> prefill_logits(Residency &gpu, const std::vector<int> &prompt, int chunk) {
    Target target(gpu, 0.f, 0, 0, 1.f);
    for (size_t i = 0; i < prompt.size(); i += chunk) {
        size_t end = std::min(i + chunk, prompt.size());
        target.forward({prompt.begin() + i, prompt.begin() + end}, false, end == prompt.size());
    }
    auto device = target.head_logits(target.state.row(0));
    std::vector<float> logits(120832);
    target.op.copy(logits.data(), device, logits.size() * sizeof(float), hipMemcpyDeviceToHost);
    return logits;
}

void check_prefill(Residency &gpu, const std::vector<int> &prompt) {
    std::vector<float> reference;
    for (int pattern : {0x7f, 0xff}) {
        for (int chunk : {1, 6, 8, 15, 16, 20, 32}) {
            for (int r = 0; r < gpu.ranks; ++r) {
                checked(hipSetDevice(gpu.physical[r]));
                checked(hipMemsetAsync(gpu.workspace[r].base, pattern, gpu.workspace[r].size, gpu.compute[r]));
            }
            checked(hipSetDevice(gpu.physical[0]));
            gpu.release_slot_pages(0);
            if (!gpu.allocate_slot_pages(0, gpu.tokens))
                throw std::runtime_error("KV ページ割当に失敗しました");
            for (int layer = 0; layer < 42; ++layer) {
                if (!Residency::is_sparse_layer(layer)) {
                    auto slots = std::min(gpu.tokens, size_t(518));
                    checked(hipMemsetAsync(gpu.cache[0][layer], pattern, slots * 294, gpu.compute[0]));
                } else {
                    checked(hipMemsetAsync(gpu.cache[0][layer], pattern,
                                           gpu.kv_pages_total * kv_page_tokens * 294, gpu.compute[0]));
                    checked(hipMemsetAsync(gpu.index_keys[0][layer], pattern,
                                           gpu.kv_pages_total * kv_page_groups * 256, gpu.compute[0]));
                    checked(hipMemsetAsync(gpu.index_pending[0][layer], pattern, 12 * 256, gpu.compute[0]));
                }
            }
            auto logits = prefill_logits(gpu, prompt, chunk);
            if (reference.empty()) {
                if (!std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); }))
                    throw std::runtime_error("prefill logits に非有限値があります");
                reference = logits;
            }
            if (!same_bits(reference, logits))
                throw std::runtime_error("初期値またはチャンク長によって prefill logits が変化しました: chunk=" +
                                         std::to_string(chunk));
        }
    }
}

void compare_transports(const Path &body, const Path &draft, int ranks, const Owners *placement = nullptr) {
    Checkpoint model(body, resolve_engram(body), draft, placement ? *placement : Owners{}, placement ? ranks : 1, 1);
    if (!placement)
        share_expert_weights(model);
    const std::vector<int> prompt32{802, 8778, 299, 9480, 341, 39144, 13, 360, 46544, 52315, 533,
                                    10044, 565, 1823, 269, 7370, 587, 269, 10575, 2417, 18221, 302,
                                    269, 18212, 13, 44635, 269, 1419, 4782, 269, 46544, 2929};
    const std::vector<int> prompt(prompt32.begin(), prompt32.begin() + 20);
    std::vector<int> reference;
    std::vector<float> reference_logits;
    for (bool speculate : {false, true}) {
        model.has_draft = speculate;
        for (const char *transport : {"push", "peer", "host"}) {
            setenv("SHELF_HY4_EP_TRANSPORT", transport, 1);
            setenv("SHELF_HY4_BATCH_DFLY", speculate ? "1" : "0", 1);
            for (int slots : {1, 16}) {
                if (!placement)
                    for (auto &owners : model.owners)
                        owners.fill(0);
                Residency gpu(model, 64, slots);
                auto status = gpu.reserve();
                if (!status.ok())
                    throw std::runtime_error(status.message());
                gpu.upload();
                if (!placement)
                    logical_ranks(gpu, ranks);
                if (slots == 16 && !speculate && std::string_view(transport) == "host")
                    check_prefill(gpu, prompt32);
                if (slots == 1) {
                    auto logits = prefill_logits(gpu, prompt, 20);
                    if (reference_logits.empty())
                        reference_logits = logits;
                    if (!same_bits(logits, reference_logits))
                        throw std::runtime_error("転送方式によって prefill logits が変化しました");
                    for (bool graphs : {false, true}) {
                        setenv("SHELF_HY4_GRAPH", graphs ? "1" : "0", 1);
                        setenv("SHELF_HY4_DRAFT_GRAPH", graphs ? "1" : "0", 1);
                        auto ids = generate(gpu, prompt, 4, 20, 0, 0, {});
                        if (reference.empty())
                            reference = ids;
                        if (ids != reference)
                            throw std::runtime_error("転送方式、投機または Graph 設定によって生成 ID が変化しました");
                    }
                } else {
                    for (int batch : {1, 2, 4, 8, 16}) {
                        auto ids = generate_batch(gpu, std::vector<std::vector<int>>(batch, prompt),
                                                  std::vector<int>(batch, 4), {});
                        for (const auto &lane : ids)
                            if (lane != reference)
                                throw std::runtime_error("転送方式またはバッチ件数によって生成 ID が変化しました");
                        printf("ep=%d transport=%s draft=%d batch=%d greedy=%d/%d\n",
                               ranks, transport, speculate, batch, batch, batch);
                        fflush(stdout);
                    }
                }
                gpu.shutdown();
            }
        }
    }
}

void compare_lm_head_tp(const Path &body, const Path &draft) {
    const Path engram = resolve_engram(body);
    setenv("SHELF_HY4_LM_HEAD_TP", "0", 1);
    Checkpoint model_off(body, engram, draft, Owners{}, 4, 1);
    if (model_off.split_lm_head)
        throw std::runtime_error("SHELF_HY4_LM_HEAD_TP=0 でも split_lm_head が有効になりました");
    setenv("SHELF_HY4_LM_HEAD_TP", "1", 1);
    Checkpoint model_on(body, engram, draft, Owners{}, 4, 1);
    if (!model_on.split_lm_head)
        throw std::runtime_error("4 rank 時に split_lm_head が有効になりませんでした");
    Checkpoint model_ep3(body, engram, draft, Owners{}, 3, 1);
    if (model_ep3.split_lm_head)
        throw std::runtime_error("割り切れない 3 rank で split_lm_head が有効になりました");
    const size_t saved = model_off.bytes[0] - model_on.bytes[0];
    if (saved < 200000000)
        throw std::runtime_error("rank 0 の lm_head 削減量が期待未満です: " + std::to_string(saved));
    printf("lm_head_tp_rank0_saved_bytes=%zu lm_head_rank0_bytes=%zu\n",
           saved, model_on.at("lm_head.trellis", 0).bytes + model_on.at("lm_head.suh", 0).bytes +
                      model_on.at("lm_head.svh", 0).bytes);
    fflush(stdout);

    const std::vector<int> prompt{802, 8778, 299, 9480, 341, 39144, 13, 360, 46544, 52315, 533,
                                  10044, 565, 1823, 269, 7370, 587, 269, 10575, 2417};
    std::vector<float> ref_logits;
    std::vector<int> ref_greedy, ref_sample;
    std::vector<std::vector<int>> ref_batch;
    {
        Checkpoint base_model(body, engram, draft, Owners{}, 1, 1);
        share_expert_weights(base_model);
        Residency gpu(base_model, 64, 1);
        auto st = gpu.reserve();
        if (!st.ok())
            throw std::runtime_error(st.message());
        gpu.upload();
        logical_ranks(gpu, 4);
        setenv("SHELF_HY4_GRAPH", "0", 1);
        setenv("SHELF_HY4_DRAFT_GRAPH", "0", 1);
        ref_logits = prefill_logits(gpu, prompt, 20);
        ref_greedy = generate(gpu, prompt, 8, 20, 0.f, 0, {}, 0, 1.f, {}, true, nullptr, nullptr, nullptr, 4);
        ref_sample = generate(gpu, prompt, 6, 20, 0.8f, 42, {}, 0, 1.f, {}, true, nullptr, nullptr, nullptr, 4);
        gpu.shutdown();
    }
    {
        Checkpoint base_model(body, engram, draft, Owners{}, 1, 1);
        share_expert_weights(base_model);
        Residency gpu(base_model, 64, 4);
        auto st = gpu.reserve();
        if (!st.ok())
            throw std::runtime_error(st.message());
        gpu.upload();
        logical_ranks(gpu, 4);
        setenv("SHELF_HY4_GRAPH", "0", 1);
        setenv("SHELF_HY4_DRAFT_GRAPH", "0", 1);
        ref_batch = generate_batch(gpu, std::vector<std::vector<int>>(4, prompt), std::vector<int>(4, 6),
                                   {1024, 0.f, 0, 0, 1.f, true, 4});
        gpu.shutdown();
    }
    for (const char *transport : {"push", "peer", "host"}) {
        setenv("SHELF_HY4_EP_TRANSPORT", transport, 1);
        for (int slots : {1, 4}) {
            Checkpoint base_model(body, engram, draft, Owners{}, 1, 1);
            share_expert_weights(base_model);
            Residency gpu(base_model, 64, slots);
            auto st = gpu.reserve();
            if (!st.ok())
                throw std::runtime_error(st.message());
            gpu.upload();
            logical_ranks(gpu, 4);
            logical_lm_head_tp(gpu, model_on);
            if (slots == 1) {
                auto got_logits = prefill_logits(gpu, prompt, 20);
                for (size_t i = 0; i < ref_logits.size(); ++i)
                    if (std::fabs(got_logits[i] - ref_logits[i]) > 1e-4f)
                        throw std::runtime_error("lm_head TP prefill logits が基準と一致しません");
                for (bool graphs : {false, true}) {
                    if (graphs && std::string_view(transport) != "push")
                        continue;
                    setenv("SHELF_HY4_GRAPH", graphs ? "1" : "0", 1);
                    setenv("SHELF_HY4_DRAFT_GRAPH", graphs ? "1" : "0", 1);
                    for (int k : {2, 4, 5}) {
                        auto got_g =
                            generate(gpu, prompt, 8, 20, 0.f, 0, {}, 0, 1.f, {}, true, nullptr, nullptr, nullptr, k);
                        if (got_g != ref_greedy)
                            throw std::runtime_error("lm_head TP greedy 生成が基準と一致しません: k=" +
                                                     std::to_string(k) + " graph=" + std::to_string(graphs));
                    }
                    auto got_s =
                        generate(gpu, prompt, 6, 20, 0.8f, 42, {}, 0, 1.f, {}, true, nullptr, nullptr, nullptr, 4);
                    if (got_s != ref_sample)
                        throw std::runtime_error("lm_head TP sample 生成が基準と一致しません");
                }
            } else {
                for (bool graphs : {false, true}) {
                    if (graphs && std::string_view(transport) != "push")
                        continue;
                    setenv("SHELF_HY4_GRAPH", graphs ? "1" : "0", 1);
                    setenv("SHELF_HY4_DRAFT_GRAPH", graphs ? "1" : "0", 1);
                    auto got_b = generate_batch(gpu, std::vector<std::vector<int>>(4, prompt), std::vector<int>(4, 6),
                                                {1024, 0.f, 0, 0, 1.f, true, 4});
                    if (got_b != ref_batch)
                        throw std::runtime_error("lm_head TP バッチ生成が基準と一致しません");
                }
            }
            printf("lm_head_tp ep=4 transport=%s slots=%d pass\n", transport, slots);
            fflush(stdout);
            gpu.shutdown();
        }
    }
    unsetenv("SHELF_HY4_EP_TRANSPORT");
    setenv("SHELF_HY4_GRAPH", "1", 1);
    setenv("SHELF_HY4_DRAFT_GRAPH", "1", 1);
}

template <class Fn> void run_ranks(Residency &gpu, Fn &&fn) {
    gpu.reset_collectives();
    std::array<std::exception_ptr, 4> failures{};
    std::array<std::jthread, 4> workers;
    for (int r = 0; r < gpu.ranks; ++r)
        workers[r] = std::jthread([&, r] {
            try {
                checked(hipSetDevice(gpu.physical[r]));
                fn(r);
            } catch (...) {
                failures[r] = std::current_exception();
                gpu.abort_collectives();
            }
        });
    for (auto &w : workers)
        w.join();
    for (auto &f : failures)
        if (f)
            std::rethrow_exception(f);
}

std::vector<Output> run_tp(Residency &gpu, bool graphs) {
    setenv("SHELF_HY4_GRAPH", graphs ? "1" : "0", 1);
    setenv("SHELF_HY4_DRAFT_GRAPH", graphs ? "1" : "0", 1);
    reset_history(gpu);
    std::vector<Output> rank0_results;
    run_ranks(gpu, [&](int rank) {
        Target target(gpu, 0.8f, 42, 0, 1, rank, 5);
        target.history.assign(7, 120000);
        std::vector<Output> results;
        for (int keep : {0, 3, 0, 0, 4, 2, 5}) {
            if (results.size() == 2) {
                gpu.all_barrier->arrive_and_wait();
                if (rank == 0)
                    reset_history(gpu);
                gpu.all_barrier->arrive_and_wait();
                target.history.assign(2044, 120000);
            }
            int start = target.history.size();
            std::vector<int> ids = keep ? std::vector<int>{802, 15063, 2576, 12286, 859, 444} : std::vector<int>{start};
            Output out;
            out.ids = target.forward(ids, keep != 0);
            if (rank == 0) {
                out.state.resize(ids.size() * 3072);
                target.op.copy(out.state.data(), target.state.p, out.state.size() * 2, hipMemcpyDeviceToHost);
                if (keep) {
                    out.target_probs.resize(6 * 120832);
                    target.op.copy(out.target_probs.data(), target.target_probs, out.target_probs.size() * 4,
                                   hipMemcpyDeviceToHost);
                }
            }
            if (keep)
                target.accept(keep, 6);
            out.draft = target.propose(500 + start);
            if (rank == 0) {
                out.draft_probs.resize(5 * 120832);
                target.op.copy(out.draft_probs.data(), target.draft_probs, out.draft_probs.size() * 4,
                               hipMemcpyDeviceToHost);
                results.push_back(std::move(out));
            } else {
                results.emplace_back();
            }
        }
        if (rank == 0)
            rank0_results = std::move(results);
    });
    return rank0_results;
}

void check_tp_capture_failure(Residency &gpu) {
    const auto before_target = gpu.target_replays, before_draft = gpu.draft_replays;
    std::vector<int> expected_ids, expected_draft;
    for (bool inject : {false, true}) {
        setenv("SHELF_HY4_GRAPH", inject ? "1" : "0", 1);
        setenv("SHELF_HY4_DRAFT_GRAPH", inject ? "1" : "0", 1);
        reset_history(gpu);
        std::vector<int> rank0_ids, rank0_draft;
        run_ranks(gpu, [&](int rank) {
            Target target(gpu, 0.f, 177, 0, 1.f, rank, 5);
            target.history.assign(2047, 120000);
            if (rank == 0)
                remaining_instantiations = inject ? 3 : 0;
            gpu.all_barrier->arrive_and_wait();
            auto ids = target.forward({850, 401, 82, 44, 276, 903}, true);
            target.accept(3, 6);
            gpu.all_barrier->arrive_and_wait();
            if (rank == 0)
                remaining_instantiations = inject ? 1 : 0;
            gpu.all_barrier->arrive_and_wait();
            auto draft = target.propose(ids[2]);
            if (rank == 0) {
                rank0_ids = std::move(ids);
                rank0_draft = std::move(draft);
            }
        });
        if (!inject) {
            expected_ids = rank0_ids;
            expected_draft = rank0_draft;
        } else if (rank0_ids != expected_ids || rank0_draft != expected_draft || !gpu.target_capture_failed ||
                   !gpu.draft_capture_failed || gpu.target_replays != before_target || gpu.draft_replays != before_draft)
            throw std::runtime_error("TP capture 失敗後の eager 実行が基準と一致しません");
    }
}

std::vector<float> tp_prefill_logits(Residency &gpu, const std::vector<int> &prompt, int chunk, int *head_arg = nullptr) {
    std::vector<float> logits(120832);
    int argmax_id = -1;
    run_ranks(gpu, [&](int rank) {
        Target target(gpu, 0.f, 0, 0, 1.f, rank);
        for (size_t i = 0; i < prompt.size(); i += chunk) {
            size_t end = std::min(i + chunk, prompt.size());
            target.forward({prompt.begin() + i, prompt.begin() + end}, false, end == prompt.size());
        }
        auto mark = target.op.arena.cursor;
        int *arg_dev = static_cast<int *>(target.op.arena.next(sizeof(int)));
        target.head_argmax(target.state.row(0), arg_dev);
        if (rank == 0)
            target.op.copy(&argmax_id, arg_dev, sizeof(int), hipMemcpyDeviceToHost);
        target.op.arena.cursor = mark;
        auto device = target.head_logits(target.state.row(0));
        if (rank == 0)
            target.op.copy(logits.data(), device, logits.size() * sizeof(float), hipMemcpyDeviceToHost);
    });
    if (head_arg)
        *head_arg = argmax_id;
    if (argmax_id != int(std::max_element(logits.begin(), logits.end()) - logits.begin()))
        throw std::runtime_error("TP head_argmax が head_logits の argmax と一致しません");
    return logits;
}

double logits_l2(const std::vector<float> &a, const std::vector<float> &b, double *rel = nullptr) {
    double sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i]) - double(b[i]);
        sq += d * d;
        ref_sq += double(b[i]) * double(b[i]);
    }
    double l2 = std::sqrt(sq);
    if (rel)
        *rel = ref_sq > 0.0 ? l2 / std::sqrt(ref_sq) : 0.0;
    return l2;
}

int argmax_of(const std::vector<float> &logits) {
    return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

void compare_tp(const Path &body, const Path &engram, const Path &draft) {
    const std::vector<int> prompt20{802, 8778, 299, 9480, 341, 39144, 13, 360, 46544, 52315,
                                    533, 10044, 565, 1823, 269, 7370, 587, 269, 10575, 2417};
    const std::vector<std::vector<int>> batch_prompts{{802, 15063}, {12286, 859, 444}, {401, 82, 44, 276}, {120000}};
    std::vector<float> ref_logits20;
    std::vector<int> ref_gen_nodraft, ref_gen_draft;
    std::vector<std::vector<int>> ref_batch_nodraft, ref_batch_draft;
    {
        Checkpoint ref_model(body, engram, draft, Owners{}, 1, 1);
        share_tp_weights(ref_model);
        Residency ref_gpu(ref_model, 2080);
        auto st = ref_gpu.reserve();
        if (!st.ok())
            throw std::runtime_error(st.message());
        ref_gpu.upload();
        ref_logits20 = prefill_logits(ref_gpu, prompt20, 20);
        ref_model.has_draft = false;
        ref_gen_nodraft = generate(ref_gpu, prompt20, 4, 20, 0.f, 0, {});
        for (const auto &bp : batch_prompts)
            ref_batch_nodraft.push_back(generate(ref_gpu, bp, 4, 20, 0.f, 0, {}));
        ref_model.has_draft = true;
        ref_gen_draft = generate(ref_gpu, prompt20, 4, 20, 0.f, 0, {});
        for (const auto &bp : batch_prompts)
            ref_batch_draft.push_back(generate(ref_gpu, bp, 4, 20, 0.f, 0, {}));
        ref_gpu.shutdown();
    }
    const char *filter = std::getenv("SHELF_HY4_TP_FILTER");
    for (auto [ep, tp] : {std::pair{4, 1}, std::pair{2, 2}, std::pair{1, 4}}) {
        if (filter && std::string_view(filter) != (std::to_string(ep) + "," + std::to_string(tp)))
            continue;
        Owners owners = make_tp_owners(tp == 4 ? 4 : ep);
        Checkpoint model(body, engram, draft, owners, ep, tp);
        share_tp_weights(model);
        if (tp == 1) {
            setenv("SHELF_HY4_EP_TRANSPORT", "push", 1);
            model.has_draft = true;
            Residency gpu(model, 2080);
            reserve_logical(gpu);
            auto logits = prefill_logits(gpu, prompt20, 20);
            double rel = 0.0;
            double l2 = logits_l2(logits, ref_logits20, &rel);
            int r_arg = argmax_of(ref_logits20), t_arg = argmax_of(logits);
            printf("ep=%d tp=%d transport=push prefill20_l2=%.6f rel=%.6e ref_arg=%d(%.4f) tp_arg=%d(%.4f)\n",
                   ep, tp, l2, rel, r_arg, ref_logits20[r_arg], t_arg, logits[t_arg]);
            fflush(stdout);
            gpu.shutdown();
        }
        if (tp > 1) {
            std::vector<float> push_logits;
            std::vector<int> tp_gen_nodraft, tp_gen_draft;
            for (const char *transport : {"push", "peer", "host"}) {
                setenv("SHELF_HY4_EP_TRANSPORT", transport, 1);
                model.has_draft = true;
                Residency gpu(model, 2080);
                reserve_logical(gpu);
                auto logits = tp_prefill_logits(gpu, prompt20, 20);
                double rel = 0.0;
                double l2 = logits_l2(logits, ref_logits20, &rel);
                int r_arg = argmax_of(ref_logits20), t_arg = argmax_of(logits);
                printf("ep=%d tp=%d transport=%s prefill20_l2=%.6f rel=%.6e ref_arg=%d(%.4f) tp_arg=%d(%.4f) s0=%.3f/%.3f\n",
                       ep, tp, transport, l2, rel, r_arg, ref_logits20[r_arg], t_arg, logits[t_arg],
                       ref_logits20[0], logits[0]);
                fflush(stdout);
                if (!std::all_of(logits.begin(), logits.end(), [](float v) { return std::isfinite(v); }) || rel > 0.35)
                    throw std::runtime_error("TP prefill logits が許容誤差を超えました");
                if (push_logits.empty())
                    push_logits = logits;
                else if (logits_l2(logits, push_logits) > 1e-4)
                    throw std::runtime_error("TP transport 間で logits が異なります");
                for (bool speculate : {false, true}) {
                    model.has_draft = speculate;
                    for (bool graphs : {false, true}) {
                        setenv("SHELF_HY4_GRAPH", graphs ? "1" : "0", 1);
                        setenv("SHELF_HY4_DRAFT_GRAPH", graphs ? "1" : "0", 1);
                        auto &expected = speculate ? tp_gen_draft : tp_gen_nodraft;
                        const bool repeat = !graphs && (std::string_view(transport) == "push" ||
                                                       (tp == 4 && std::string_view(transport) == "peer"));
                        const int runs = repeat ? 5 : 1;
                        for (int i = 0; i < runs; ++i) {
                            auto ids = generate(gpu, prompt20, 4, 20, 0.f, 0, {});
                            if (ids.empty())
                                throw std::runtime_error("TP generate の出力が空です");
                            if (expected.empty())
                                expected = ids;
                            else if (ids != expected)
                                throw std::runtime_error("TP generate の生成 ID が一致しません");
                        }
                        if (repeat) {
                            printf("ep=%d tp=%d transport=%s draft=%d deterministic=%d/%d ids=",
                                   ep, tp, transport, speculate, runs, runs);
                            for (size_t i = 0; i < expected.size(); ++i)
                                printf("%s%d", i ? "," : "", expected[i]);
                            printf("\n");
                        }
                        if (!graphs && std::string_view(transport) == "push") {
                            const auto &ref_bp = speculate ? ref_batch_draft : ref_batch_nodraft;
                            int bp_matches = 0;
                            for (size_t bi = 0; bi < batch_prompts.size(); ++bi) {
                                auto bids = generate(gpu, batch_prompts[bi], 4, 20, 0.f, 0, {});
                                if (bids == ref_bp[bi])
                                    ++bp_matches;
                            }
                            printf("ep=%d tp=%d draft=%d short_prompt_ref_match=%d/%zu\n", ep, tp, speculate, bp_matches,
                                   batch_prompts.size());
                            fflush(stdout);
                        }
                    }
                }
                if (std::string_view(transport) == "push") {
                    model.has_draft = true;
                    const auto before_target = gpu.target_replays, before_draft = gpu.draft_replays;
                    auto eager = run_tp(gpu, false);
                    auto graph = run_tp(gpu, true);
                    for (size_t i = 0; i < eager.size(); ++i) {
                        if (eager[i].ids != graph[i].ids || eager[i].state != graph[i].state ||
                            !same_bits(eager[i].target_probs, graph[i].target_probs))
                            throw std::runtime_error("TP Target eager/graph が一致しません");
                        if (eager[i].draft != graph[i].draft || !same_bits(eager[i].draft_probs, graph[i].draft_probs))
                            throw std::runtime_error("TP DFly2 eager/graph が一致しません");
                    }
                    if (gpu.target_replays - before_target != 7 || gpu.draft_replays - before_draft != 7)
                        throw std::runtime_error("TP Graph の replay 回数が不正です");
                    check_tp_capture_failure(gpu);
                    printf("ep=%d tp=%d graph_parity=pass target=7/7 draft=7/7 capture_fallback=pass\n", ep, tp);
                }
                gpu.shutdown();
            }
            if (tp == 4) {
                setenv("SHELF_HY4_EP_TRANSPORT", "push", 1);
                model.has_draft = true;
                Residency gpu(model, 2080);
                reserve_logical(gpu);
                std::vector<int> prompt1536(1536);
                for (size_t i = 0; i < prompt1536.size(); ++i)
                    prompt1536[i] = 100 + int((i * 37) % 1024);
                auto l_big = tp_prefill_logits(gpu, prompt1536, 1536);
                auto l_chunk = tp_prefill_logits(gpu, prompt1536, 512);
                if (!std::all_of(l_big.begin(), l_big.end(), [](float v) { return std::isfinite(v); }))
                    throw std::runtime_error("TP=4 1536 token prefill に非有限値があります");
                double rel = 0.0;
                double l2 = logits_l2(l_big, l_chunk, &rel);
                if (argmax_of(l_big) != argmax_of(l_chunk))
                    throw std::runtime_error("TP=4 1536 token prefill の argmax が分割時と一致しません");
                printf("ep=%d tp=%d chunk1536_vs_512_l2=%.6f rel=%.6e argmax=match\n", ep, tp, l2, rel);
                gpu.shutdown();
            }
        }
        for (auto [speculate, slots] : {std::pair{false, 16}, std::pair{true, 16}, std::pair{true, 4}}) {
            setenv("SHELF_HY4_EP_TRANSPORT", "push", 1);
            setenv("SHELF_HY4_BATCH_DFLY", speculate ? "1" : "0", 1);
            model.has_draft = true;
            Residency gpu(model, 64, slots);
            reserve_logical(gpu);
            std::vector<std::vector<int>> baseline;
            model.has_draft = speculate;
            for (const auto &prompt : batch_prompts)
                baseline.push_back(generate_batch(gpu, {prompt}, {4}, {}).front());
            model.has_draft = true;
            for (int batch : {2, 4, 8, 16}) {
                if (slots < batch)
                    continue;
                std::vector<std::vector<int>> inputs;
                for (int i = 0; i < batch; ++i)
                    inputs.push_back(batch_prompts[i % batch_prompts.size()]);
                for (bool graph : {false, true}) {
                    setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
                    setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
                    auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
                    auto results = generate_batch(gpu, inputs, std::vector<int>(batch, 4), {});
                    for (int i = 0; i < batch; ++i)
                        if (results.at(i) != baseline[i % batch_prompts.size()])
                            throw std::runtime_error("TP バッチ生成が単独実行と異なります");
                    bool active = graph && graph_topology(ep, tp, gpu.push, gpu.graph_room, gpu.peer);
                    if ((gpu.target_replays > target_before) != active ||
                        (gpu.draft_replays > draft_before) != (active && speculate))
                        throw std::runtime_error("TP バッチ Graph の起動条件が異なります");
                    printf("ep=%d tp=%d draft=%d graph=%d batch=%d greedy=%d/%d\n", ep, tp, speculate, graph, batch,
                           batch, batch);
                    fflush(stdout);
                }
            }
            if (slots == 4 && tp > 1) {
                std::vector<int> near_end(61, 802);
                auto tail = generate_batch(gpu, {near_end}, {3}, {}).front();
                std::array<std::vector<int>, 7> streamed;
                std::array<std::string, 7> reason;
                std::vector<int> started;
                Hy4BatchSession session(gpu, [&](int id, int token) {
                    if (streamed[id].empty())
                        started.push_back(id);
                    streamed[id].push_back(token);
                }, [&](int id, const std::vector<int> &ids, const char *finish) {
                    if (ids != streamed[id])
                        throw std::runtime_error("TP セッション配信 ID と終了通知が異なります");
                    reason[id] = finish;
                });
                for (int id = 0; id < 6; ++id)
                    session.admit({id, id == 5 ? near_end : batch_prompts[id % 4], id == 5 ? 3 : 4, {}});
                session.admit({6, batch_prompts[2], 4, {}});
                session.cancel(6);
                for (int i = 0; i < 5; ++i)
                    session.step();
                session.cancel(1);
                for (int i = 0; session.active() || session.queued(); ++i) {
                    if (i > 64)
                        throw std::runtime_error("TP セッションが完了しません");
                    session.step();
                }
                if (started != std::vector<int>{0, 1, 2, 3, 4, 5} || !session.draft_rounds() ||
                    reason[1] != "cancelled" || reason[6] != "cancelled" || streamed[5] != tail)
                    throw std::runtime_error("TP セッションの FIFO/キャンセル検証に失敗しました");
                for (int id : {0, 2, 3, 4})
                    if (streamed[id] != baseline[id % 4])
                        throw std::runtime_error("TP スロット再利用後の生成 ID が異なります");
                printf("ep=%d tp=%d session=pass draft_rounds=%zu\n", ep, tp, session.draft_rounds());
            }
            gpu.shutdown();
        }
        if (tp > 1) {
            setenv("SHELF_HY4_BATCH_DFLY", "1", 1);
            model.has_draft = true;
            Residency gpu(model, 2070, 2);
            reserve_logical(gpu);
            Hy4BatchOptions options{128};
            std::vector<std::vector<int>> inputs{std::vector<int>(2047), std::vector<int>(2053)};
            for (auto &input : inputs)
                for (size_t i = 0; i < input.size(); ++i)
                    input[i] = 100 + int(i % 7);
            std::vector<std::vector<int>> expected;
            for (bool graph : {false, true}) {
                setenv("SHELF_HY4_GRAPH", graph ? "1" : "0", 1);
                setenv("SHELF_HY4_DRAFT_GRAPH", graph ? "1" : "0", 1);
                if (!graph)
                    for (auto &input : inputs)
                        expected.push_back(generate_batch(gpu, {input}, {6}, options).front());
                auto target_before = gpu.target_replays, draft_before = gpu.draft_replays;
                std::vector<std::vector<int>> generated(2);
                Hy4BatchSession session(gpu, [&](int id, int token) { generated[id].push_back(token); },
                                         [](int, const auto &, const char *) {});
                for (int i = 0; i < 2; ++i)
                    session.admit({i, inputs[i], 6, options});
                while (session.active() || session.queued())
                    session.step();
                if (generated != expected) {
                    for (size_t i = 0; i < 2; ++i) {
                        fprintf(stderr, "graph=%d slot=%zu expected:", int(graph), i);
                        for (int id : expected[i]) fprintf(stderr, " %d", id);
                        fprintf(stderr, "\ngraph=%d slot=%zu generated:", int(graph), i);
                        for (int id : generated[i]) fprintf(stderr, " %d", id);
                        fprintf(stderr, "\n");
                    }
                    throw std::runtime_error("TP 位置 2047・2053 混在時の生成 ID が異なります");
                }
                if (graph && (gpu.target_replays == target_before || gpu.draft_replays == draft_before))
                    throw std::runtime_error("TP 疎な attention でバッチ Graph が起動しませんでした");
            }
            auto free_pages = gpu.kv_free_pages;
            std::sort(free_pages.begin(), free_pages.end());
            if (free_pages.size() != gpu.kv_pages_total)
                throw std::runtime_error("TP セッションの終了後に未解放の KV ページがあります");
            for (size_t i = 0; i < free_pages.size(); ++i)
                if (free_pages[i] != int32_t(i) || gpu.kv_page_refs[i] != 0)
                    throw std::runtime_error("TP セッションの終了後に KV ページの所有状態が一致しません");
            gpu.shutdown();
            printf("ep=%d tp=%d sparse_2047_2053=pass ids0=", ep, tp);
            for (size_t i = 0; i < expected[0].size(); ++i)
                printf("%s%d", i ? "," : "", expected[0][i]);
            printf(" ids1=");
            for (size_t i = 0; i < expected[1].size(); ++i)
                printf("%s%d", i ? "," : "", expected[1][i]);
            printf("\n");
        }
        fflush(stdout);
    }
}

}
extern "C" hipError_t __real_hipEventRecord(hipEvent_t, hipStream_t);
extern "C" hipError_t __wrap_hipEventRecord(hipEvent_t event, hipStream_t stream) {
    if (std::getenv("SHELF_HY4_CHECK_EVENT_REUSE")) {
        auto status = hipEventQuery(event);
        if (status == hipErrorNotReady) {
            std::fprintf(stderr, "完了前の event が再使用されました\n");
            return hipErrorIllegalState;
        }
        if (status != hipSuccess)
            return status;
    }
    return __real_hipEventRecord(event, stream);
}
extern "C" hipError_t __real_hipGraphInstantiate(hipGraphExec_t *, hipGraph_t, hipGraphNode_t *, char *, size_t);
extern "C" hipError_t __wrap_hipGraphInstantiate(hipGraphExec_t *exec, hipGraph_t graph, hipGraphNode_t *node,
                                                 char *log, size_t length) {
    if (remaining_instantiations > 0 && --remaining_instantiations == 0)
        return hipErrorOutOfMemory;
    return __real_hipGraphInstantiate(exec, graph, node, log, length);
}
int main(int argc, char **argv) {
    try {
        bool micro = argc == 3 && std::string_view(argv[2]) == "microbench";
        bool moe = argc == 3 && std::string_view(argv[2]) == "moebench";
        bool bits3 = argc == 3 && std::string_view(argv[2]) == "bits3bench";
        bool router_b = argc == 3 && std::string_view(argv[2]) == "routerbench";
        bool steps = (argc == 5 || argc == 6) && std::string_view(argv[3]) == "stepbench";
        bool ctx_b = (argc == 6 || argc == 7) && std::string_view(argv[3]) == "ctxbench";
        if (argc == 7 && std::string_view(argv[2]) == "prefillbench") {
            int length = std::stoi(argv[3]), chunk = std::stoi(argv[4]), ranks = std::stoi(argv[5]);
            if (length < 1 || length > 131072 || chunk < 1 || chunk > max_prefill_chunk || ranks < 1 || ranks > 4)
                throw std::invalid_argument("prefill 計測の引数が範囲外です");
            Checkpoint model(argv[1], resolve_engram(argv[1]), {}, Owners{}, 1, 1);
            share_expert_weights(model);
            Residency gpu(model, length + 1);
            auto status = gpu.reserve();
            if (!status.ok()) throw std::runtime_error(status.message());
            gpu.upload();
            logical_ranks(gpu, ranks);
            if (std::string_view(argv[6]) == "index") {
                benchmark_hy4_index(gpu);
            } else if (std::string_view(argv[6]) == "capacity") {
                check_hy4_prefill_capacity(gpu, chunk);
            } else if (std::string_view(argv[6]) == "moe") {
                benchmark_hy4_prefill(gpu, chunk);
            } else {
                std::vector<int> ids(length);
                for (int i = 0; i < length; ++i) ids[i] = 100 + i * 37 % 128;
                generate(gpu, ids, 1, chunk, 0.f, 0, argv[6]);
            }
            gpu.shutdown();
            return 0;
        }
        if (micro || moe || bits3 || router_b || steps || ctx_b) {
            Path draft = (steps || ctx_b) && std::string_view(argv[2]) != "-" ? Path(argv[2]) : Path{};
            Checkpoint model(argv[1], resolve_engram(argv[1]), draft, Owners{}, 1, 1);
            share_expert_weights(model, bits3 || moe || steps || ctx_b);
            int slots = (steps || ctx_b) ? std::stoi(argv[4]) : 1;
            int ctx = ctx_b ? std::stoi(argv[5]) : 128;
            size_t reserve_tokens = ctx_b ? size_t(ctx + 128) : size_t(128);
            Residency gpu(model, reserve_tokens, slots);
            auto status = gpu.reserve();
            if (!status.ok()) throw std::runtime_error(status.message());
            gpu.upload();
            if (ctx_b) {
                int ranks = argc == 7 ? std::stoi(argv[6]) : 1;
                if (ranks < 1 || ranks > 4) throw std::invalid_argument("論理 EP は 1〜4 です");
                logical_ranks(gpu, ranks);
                benchmark_hy4_ctx(gpu, ctx);
            } else if (steps) {
                int ranks = argc == 6 ? std::stoi(argv[5]) : 3;
                if (ranks < 1 || ranks > 4) throw std::invalid_argument("論理 EP は 1〜4 です");
                logical_ranks(gpu, ranks);
                benchmark_hy4_steps(gpu);
            } else if (bits3) {
                benchmark_hy4_bits3(gpu);
            } else if (moe) {
                benchmark_hy4_moe(gpu);
            } else if (router_b) {
                benchmark_hy4_router(gpu);
            } else {
                benchmark_hy4_decode(gpu);
            }
            gpu.shutdown();
            return 0;
        }

        if ((argc == 4 || argc == 5) && std::string_view(argv[3]) == "tp") {
            compare_tp(argv[1], argc == 5 ? Path(argv[4]) : resolve_engram(argv[1]), argv[2]);
            return 0;
        }
        if ((argc == 4 || argc == 5) && std::string_view(argv[3]) == "transport") {
            if (argc == 5) {
                int count = 0;
                checked(hipGetDeviceCount(&count));
                auto placement = read_owners(argv[4]);
                compare_transports(argv[1], argv[2], count, &placement);
            } else {
                for (int ranks : {3, 4})
                    compare_transports(argv[1], argv[2], ranks);
            }
            return 0;
        }
        if (argc == 4 && std::string_view(argv[3]) == "kparity") {
            compare_draft_counts(argv[1], argv[2]);
            return 0;
        }
        if (argc == 4 && std::string_view(argv[3]) == "lmhead") {
            compare_lm_head_tp(argv[1], argv[2]);
            return 0;
        }
        if (argc != 3 && (argc != 4 || std::string_view(argv[3]) != "batch"))
            return 2;
        if (argc == 4) {
            for (int ranks : {1, 2, 3, 4}) compare_batches(argv[1], argv[2], ranks);
        } else {
            for (int ranks : {1, 3, 4}) compare(argv[1], argv[2], ranks);
            compare_draft_counts(argv[1], argv[2]);
            compare_lm_head_tp(argv[1], argv[2]);
        }
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
