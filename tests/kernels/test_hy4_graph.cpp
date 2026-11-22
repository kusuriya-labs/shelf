#include <shelf/models/hy4_flash/target.h>
#include <shelf/models/hy4_flash/generate.h>
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
int remaining_instantiations = 0;
using namespace shelf;
using namespace shelf::hy4;

void share_expert_weights(Checkpoint &model) {
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
        gpu.matrices[r] = gpu.matrices[0];
        auto result = gpu::GpuArena::create(0, 64 << 20);
        if (!result.ok())
            throw std::runtime_error(result.status().message());
        gpu.arenas.push_back(result.release());
        auto memory = gpu.arenas.back().allocate_aligned(64 << 20, 256);
        if (!memory.ok())
            throw std::runtime_error(memory.status().message());
        gpu.workspace[r] = {static_cast<uint8_t *>(memory.value().data()), 64 << 20};
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
                    checked(hipEventCreateWithFlags(&gpu.ready[r][other][p], hipEventDisableTiming));
                    checked(hipEventCreateWithFlags(&gpu.ack[r][other][p], hipEventDisableTiming));
                }
            }
    for (auto &layer : gpu.model.owners)
        for (int e = 0; e < 512; ++e)
            layer[e] = e % ranks;
    checked(hipMemcpy(gpu.moe_owners[0], gpu.model.owners.data(), sizeof(Owners), hipMemcpyHostToDevice));
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

}
extern "C" hipError_t __real_hipGraphInstantiate(hipGraphExec_t *, hipGraph_t, hipGraphNode_t *, char *, size_t);
extern "C" hipError_t __wrap_hipGraphInstantiate(hipGraphExec_t *exec, hipGraph_t graph, hipGraphNode_t *node,
                                                 char *log, size_t length) {
    if (remaining_instantiations > 0 && --remaining_instantiations == 0)
        return hipErrorOutOfMemory;
    return __real_hipGraphInstantiate(exec, graph, node, log, length);
}
int main(int argc, char **argv) {
    if (argc != 3 && (argc != 4 || std::string_view(argv[3]) != "batch"))
        return 2;
    try {
        for (int ranks : {1, 4})
            if (argc == 4) compare_batches(argv[1], argv[2], ranks);
            else compare(argv[1], argv[2], ranks);
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
