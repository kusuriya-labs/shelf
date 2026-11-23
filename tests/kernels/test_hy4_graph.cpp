#include <shelf/models/hy4_flash/target.h>
#include <shelf/models/hy4_flash/generate.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace {
int remaining_instantiations = 0;
using namespace shelf;
using namespace shelf::hy4;

void share_expert_weights(Checkpoint &model) {
    if (model.mixed)
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
        if (pos != std::string::npos)
            weight.offset = model.local[0].at("model.layers.1" + key.substr(pos)).offset;
    }
}

void logical_ranks(Residency &gpu, int ranks) {
    gpu.ranks = gpu.ep_size = ranks;
    for (int r = 1; r < ranks; ++r) {
        gpu.physical[r] = 0;
        constexpr size_t work = 64 << 20, bytes = work + Residency::matrix_bytes;
        auto result = gpu::GpuArena::create(0, bytes);
        if (!result.ok())
            throw std::runtime_error(result.status().message());
        gpu.arenas.push_back(result.release());
        auto memory = gpu.arenas.back().allocate_aligned(bytes, 256);
        if (!memory.ok())
            throw std::runtime_error(memory.status().message());
        Region buffer{static_cast<uint8_t *>(memory.value().data()), bytes};
        gpu.reserve_experts(r, buffer);
        gpu.workspace[r] = {static_cast<uint8_t *>(buffer.next(work)), work};
        checked(hipStreamCreateWithFlags(&gpu.compute[r], hipStreamNonBlocking));
    }
    const auto transport = std::getenv("SHELF_HY4_EP_TRANSPORT");
    const bool host = transport && std::string_view(transport) == "host";
    if (host && ranks > 1)
        checked(hipHostMalloc(&gpu.staging, max_prefill_chunk * 3072u * 2u, hipHostMallocPortable));
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
        auto slots = gpu.index_keys[0][l] ? gpu.tokens : std::min(gpu.tokens, size_t(518));
        checked(hipMemcpy(gpu.cache[0][l], kv.data(), slots * 294, hipMemcpyHostToDevice));
        if (gpu.index_keys[0][l])
            checked(hipMemcpy(gpu.index_keys[0][l], groups.data(), groups.size() * 2, hipMemcpyHostToDevice));
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
    std::vector<Output> results;
    for (int keep : {0, 3, 0, 0, 4, 2, 5}) {
        if (results.size() == 2) {
            reset_history(gpu);
            target.history.assign(2044, 120000);
        }
        int start = target.history.size();
        std::vector<int> ids = keep ? std::vector<int>{802, 15063, 2576, 12286, 859, 444} : std::vector<int>{start};
        Output out;
        out.ids = target.forward(ids, keep != 0);
        out.state.resize(ids.size() * 3072);
        target.op.copy(out.state.data(), target.state.p, out.state.size() * 2, hipMemcpyDeviceToHost);
        if (keep) {
            out.target_probs.resize(6 * 120832);
            target.op.copy(out.target_probs.data(), target.target_probs, out.target_probs.size() * 4, hipMemcpyDeviceToHost);
            target.accept(keep, 6);
        }
        out.draft = target.propose(500 + start);
        out.draft_probs.resize(5 * 120832);
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
    std::vector<int> expected_ids, expected_draft;
    for (bool inject : {false, true}) {
        setenv("SHELF_HY4_GRAPH", inject ? "1" : "0", 1);
        setenv("SHELF_HY4_DRAFT_GRAPH", inject ? "1" : "0", 1);
        reset_history(gpu);
        Target target(gpu, 0.f, 177, 0, 1.f);
        target.history.assign(2047, 120000);
        remaining_instantiations = inject ? 3 : 0;
        auto ids = target.forward({850, 401, 82, 44, 276, 903}, true);
        target.accept(3, 6);
        remaining_instantiations = inject ? 1 : 0;
        auto draft = target.propose(ids[2]);
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
    auto eager = run(gpu, false);
    if (gpu.target_replays || gpu.draft_replays)
        throw std::runtime_error("無効化した Graph が起動しました");
    auto graph = run(gpu, true);
    for (size_t i = 0; i < eager.size(); ++i) {
        if (eager[i].ids != graph[i].ids || eager[i].state != graph[i].state || !same_bits(eager[i].target_probs, graph[i].target_probs))
            throw std::runtime_error("Target の照合失敗: step=" + std::to_string(i));
        if (eager[i].draft != graph[i].draft || !same_bits(eager[i].draft_probs, graph[i].draft_probs))
            throw std::runtime_error("DFly2 の照合失敗: step=" + std::to_string(i));
    }
    if (gpu.target_replays != 7 || gpu.draft_replays != 7 || gpu.target_victim != 4 || gpu.draft_victim != 2)
        throw std::runtime_error("Graph の capture 回数または replay 回数が異なります");
    check_capture_failure(gpu);
    printf("ep=%d target=7/7 draft=7/7 target_captures=4 draft_captures=2 capture_fallback=pass\n", ranks);
    gpu.shutdown();
}
void compare_batches(const Path &body, const Path &draft, int ranks) {
    Checkpoint model(body, body, draft, Owners{}, 1, 1);
    share_expert_weights(model);
    const std::vector<std::vector<int>> prompts{{802, 15063}, {12286, 859, 444}, {401, 82, 44, 276}, {120000}};
    constexpr int count = 4;
    for (auto [speculate, slots] : {std::pair{false, 16}, std::pair{true, 16}, std::pair{true, 4}}) {
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
        for (int batch : {2, 4, 8, 16}) {
            if (slots == 4) break;
            std::vector<std::vector<int>> inputs;
            for (int i = 0; i < batch; ++i)
                inputs.push_back(prompts[i % prompts.size()]);
            auto results = generate_batch(gpu, inputs, std::vector<int>(batch, count), {});
            for (int i = 0; i < batch; ++i)
                if (results.at(i) != baseline[i % prompts.size()])
                    throw std::runtime_error("バッチ生成が単独実行と異なります: B=" + std::to_string(batch) +
                                             " slot=" + std::to_string(i));
            printf("ep=%d draft=%d batch=%d greedy=%d/%d\n", ranks, speculate, batch, batch, batch);
            fflush(stdout);
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
        gpu.shutdown();
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
            for (int layer = 0; layer < 42; ++layer) {
                auto slots = gpu.index_keys[0][layer] ? gpu.tokens : std::min(gpu.tokens, size_t(518));
                checked(hipMemsetAsync(gpu.cache[0][layer], pattern, slots * 294, gpu.compute[0]));
                if (gpu.index_keys[0][layer])
                    checked(hipMemsetAsync(gpu.index_keys[0][layer], pattern, (gpu.tokens / 4 + 12) * 256,
                                           gpu.compute[0]));
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
extern "C" hipError_t __real_hipGraphInstantiate(hipGraphExec_t *, hipGraph_t, hipGraphNode_t *, char *, size_t);
extern "C" hipError_t __wrap_hipGraphInstantiate(hipGraphExec_t *exec, hipGraph_t graph, hipGraphNode_t *node,
                                                 char *log, size_t length) {
    if (remaining_instantiations > 0 && --remaining_instantiations == 0)
        return hipErrorOutOfMemory;
    return __real_hipGraphInstantiate(exec, graph, node, log, length);
}
int main(int argc, char **argv) {
    try {
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
        if (argc != 3 && (argc != 4 || std::string_view(argv[3]) != "batch"))
            return 2;
        for (int ranks : {1, 3, 4})
            if (argc == 4) compare_batches(argv[1], argv[2], ranks);
            else compare(argv[1], argv[2], ranks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
