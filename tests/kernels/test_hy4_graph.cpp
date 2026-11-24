#include <shelf/models/hy4_flash/target.h>
#include <shelf/models/hy4_flash/generate.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace shelf::hy4 {
void verify_hy4_upload(Residency &gpu);
void benchmark_hy4_decode(Residency &gpu);
void benchmark_hy4_index(Residency &gpu);
void check_hy4_prefill_capacity(Residency &gpu, int rows);
void benchmark_hy4_prefill(Residency &gpu, int rows);
void benchmark_hy4_moe(Residency &gpu);
void benchmark_hy4_router(Residency &gpu);
void benchmark_hy4_steps(Residency &gpu);
void verify_hy4_layers(Residency &gpu, const Path &input, const Path &dump);
bool trace_hy4_prefill(Residency &gpu, const Path &input, int chunk, const Path &dump);
}

namespace {
int remaining_instantiations = 0;
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

void logical_ranks(Residency &gpu, int ranks) {
    gpu.ranks = gpu.ep_size = ranks;
    for (int r = 1; r < ranks; ++r) {
        gpu.physical[r] = 0;
        gpu.matrices[r] = gpu.matrices[0];
        auto result = gpu::GpuArena::create(0, 512 << 20);
        if (!result.ok())
            throw std::runtime_error(result.status().message());
        gpu.arenas.push_back(result.release());
        auto memory = gpu.arenas.back().allocate_aligned(512 << 20, 256);
        if (!memory.ok())
            throw std::runtime_error(memory.status().message());
        gpu.workspace[r] = {static_cast<uint8_t *>(memory.value().data()), 512 << 20};
        checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
    }
    const auto transport = std::getenv("SHELF_HY4_EP_TRANSPORT");
    const bool host = transport && std::string_view(transport) == "host";
    for (int r = 0; r < ranks; ++r)
        for (int other = 0; other < ranks; ++other)
            if (other != r) {
                gpu.peer[r][other] = !host;
                if (host)
                    checked(hipHostMalloc(&gpu.small_staging[r][other], gpu.transfer_packet_bytes(), hipHostMallocPortable));
                for (int p = 0; p < 2; ++p) {
                    constexpr unsigned flags = hipEventDisableTiming | hipEventReleaseToSystem;
                    checked(hipEventCreateWithFlags(&gpu.ready[r][other][p], flags));
                    checked(hipEventCreateWithFlags(&gpu.ack[r][other][p], flags));
                }
            }
    for (auto &layer : gpu.model.owners)
        for (int e = 0; e < 512; ++e)
            layer[e] = e % ranks;
    checked(hipMemcpy(gpu.moe_owners[0], gpu.model.owners.data(), sizeof(Owners), hipMemcpyHostToDevice));
    if (ranks > 1) {
        std::vector<Hy4ExpertProjection> all(Residency::matrix_bytes / sizeof(Hy4ExpertProjection));
        checked(hipMemcpy(all.data(), gpu.matrices[0], Residency::matrix_bytes, hipMemcpyDeviceToHost));
        for (int r = 0; r < ranks; ++r) {
            auto table = all;
            for (size_t layer = 0; layer < gpu.model.owners.size(); ++layer)
                for (int e = 0; e < 512; ++e)
                    if (gpu.model.owners[layer][e] != r)
                        std::fill_n(table.data() + (layer * 512 + e) * 3, 3, Hy4ExpertProjection{});
            checked(hipMemcpy(gpu.matrices[r], table.data(), Residency::matrix_bytes, hipMemcpyHostToDevice));
        }
    }
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
    for (int l = 0; l < 42; ++l) {
        if (!Residency::is_sparse_layer(l)) {
            auto slots = std::min(gpu.tokens, size_t(518));
            checked(hipMemcpy(gpu.cache[0][l], kv.data(), slots * 294, hipMemcpyHostToDevice));
            continue;
        }
        const auto &pages = gpu.slot_page_tables[0];
        const size_t total_groups = gpu.tokens / 4;
        for (size_t pi = 0; pi < pages.size(); ++pi) {
            const int32_t phys = pages[pi];
            const size_t tok_off = pi * kv_page_tokens;
            const size_t tok_cnt = std::min(kv_page_tokens, gpu.tokens - tok_off);
            checked(hipMemcpy(gpu.cache[0][l] + size_t(phys) * kv_page_tokens * 294,
                              kv.data() + tok_off * 294, tok_cnt * 294, hipMemcpyHostToDevice));
            const size_t grp_off = pi * kv_page_groups;
            if (grp_off < total_groups) {
                const size_t grp_cnt = std::min(kv_page_groups, total_groups - grp_off);
                checked(hipMemcpy(gpu.index_keys[0][l] + size_t(phys) * kv_page_groups * 128,
                                  groups.data() + grp_off * 128, grp_cnt * 256, hipMemcpyHostToDevice));
            }
        }
        checked(hipMemcpy(gpu.index_pending[0][l], groups.data() + total_groups * 128,
                          12 * 256, hipMemcpyHostToDevice));
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
    Checkpoint model(body, body, draft, Owners{}, 1, 1);
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
    Checkpoint model(body, body, draft, Owners{}, 1, 1);
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
                Hy4BatchOptions opts{1024, 0.f, 0, 0, 1.f, k};
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
    Checkpoint model(body, body, draft, Owners{}, 1, 1);
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
    Checkpoint model(body, body, draft, placement ? *placement : Owners{}, placement ? ranks : 1, 1);
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
            Checkpoint model(argv[1], argv[1], {}, Owners{}, 1, 1);
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
            Checkpoint model(argv[1], argv[1], draft, Owners{}, 1, 1);
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
        if (argc == 7 && std::string_view(argv[3]) == "layers") {
            auto owners = read_owners(argv[4]);
            int ranks = 1;
            for (const auto &layer : owners)
                ranks = std::max(ranks, 1 + *std::max_element(layer.begin(), layer.end()));
            Checkpoint model(argv[1], argv[1], argv[2], Owners{}, 1, 1);
            share_expert_weights(model);
            Residency gpu(model, 64);
            auto status = gpu.reserve();
            if (!status.ok())
                throw std::runtime_error(status.message());
            gpu.upload();
            logical_ranks(gpu, ranks);
            model.owners = owners;
            checked(hipMemcpy(gpu.moe_owners[0], owners.data(), sizeof(owners), hipMemcpyHostToDevice));
            verify_hy4_layers(gpu, argv[5], argv[6]);
            gpu.shutdown();
            return 0;
        }
        if ((argc == 4 || argc == 5 || argc == 8) && std::string_view(argv[3]) == "upload") {
            int count = 1;
            checked(hipGetDeviceCount(&count));
            bool shared = argc == 4 || std::string_view(argv[4]) == "-";
            if (shared && count != 1)
                throw std::runtime_error("複数 GPU の検証には配置表を指定してください");
            Owners owners = shared ? Owners{} : read_owners(argv[4]);
            Checkpoint model(argv[1], argv[1], argv[2], owners, count, 1);
            if (shared)
                share_expert_weights(model);
            Residency gpu(model, argc == 8 ? 4096 : 64);
            auto status = gpu.reserve();
            if (!status.ok())
                throw std::runtime_error(status.message());
            gpu.upload();
            bool match = argc != 8 || trace_hy4_prefill(gpu, argv[5], std::stoi(argv[6]), argv[7]);
            verify_hy4_upload(gpu);
            gpu.shutdown();
            if (!match)
                throw std::runtime_error("初回の logits と層別照合時の logits が異なります");
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
        if (argc != 3 && (argc != 4 || std::string_view(argv[3]) != "batch"))
            return 2;
        if (argc == 4) {
            for (int ranks : {1, 2, 3, 4}) compare_batches(argv[1], argv[2], ranks);
        } else {
            for (int ranks : {1, 3, 4}) compare(argv[1], argv[2], ranks);
            compare_draft_counts(argv[1], argv[2]);
        }
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
