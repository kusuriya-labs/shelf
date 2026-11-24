#include <shelf/models/hy4_flash/generate.h>
#include <shelf/models/hy4_flash/residency.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace shelf::hy4;

void share_expert_weights(Checkpoint &model) {
    if (model.mixed)
        throw std::invalid_argument("4 bpw checkpoint required");
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

std::vector<int> make_tokens(size_t len, int seed) {
    std::vector<int> ids(len);
    for (size_t i = 0; i < len; ++i)
        ids[i] = 100 + int((i * 37 + size_t(seed) * 97 + (i / 17) * 13) % 256);
    return ids;
}

struct TurnOutcome {
    std::vector<int> generated;
    std::string reason;
    size_t restored = 0;
    double prefill_ms = 0.0;
    double elapsed_ms = 0.0;
};

std::vector<TurnOutcome> run_session_sequence(
    Residency &gpu,
    bool prefix_cache,
    const std::vector<std::vector<int>> &prompts,
    const std::vector<int> &counts,
    bool ignore_eos = false) {
    std::vector<TurnOutcome> out(prompts.size());
    Hy4BatchSession session(
        gpu,
        [&](int id, int) {
            auto &entry = out[size_t(id - 1)];
            if (entry.prefill_ms == 0.0)
                entry.prefill_ms = -1.0;
        },
        [&](int id, const std::vector<int> &ids, const char *reason) {
            auto &entry = out[size_t(id - 1)];
            entry.generated = ids;
            entry.reason = reason ? reason : "";
        });
    session.set_prefix_cache(prefix_cache);
    for (size_t i = 0; i < prompts.size(); ++i) {
        auto t0 = std::chrono::steady_clock::now();
        double first_tok_ms = 0.0;
        session.admit({int(i + 1), prompts[i], counts[i], {1024, 0.f, 0, 0, 1.f, ignore_eos}});
        while (session.active() || session.queued()) {
            session.step();
            if (first_tok_ms == 0.0 && out[i].prefill_ms < 0.0) {
                first_tok_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
                out[i].prefill_ms = first_tok_ms;
            }
        }
        auto t1 = std::chrono::steady_clock::now();
        out[i].restored = session.last_restored_tokens();
        out[i].elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return out;
}

void require_equal(const char *label, const std::vector<int> &a, const std::vector<int> &b) {
    if (a != b) {
        fprintf(stderr, "MISMATCH %s: len_a=%zu len_b=%zu\n", label, a.size(), b.size());
        for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
            if (a[i] != b[i]) {
                fprintf(stderr, "  first diff at %zu: %d vs %d\n", i, a[i], b[i]);
                break;
            }
        }
        throw std::runtime_error(std::string("token mismatch: ") + label);
    }
}

void test_session_suite(Checkpoint &model, bool has_draft) {
    Residency gpu(model, 4096, 2);
    auto st = gpu.reserve();
    if (!st.ok()) throw std::runtime_error(st.message());
    gpu.upload();

    auto p1 = make_tokens(525, 1);
    std::vector<int> cold_t1, cold_t2, cold_t3;
    std::vector<int> p2, p3;
    double cold_t2_ms = 0.0, cached_t2_ms = 0.0;

    {
        Hy4BatchSession cached(gpu, {}, [&](int id, const std::vector<int> &ids, const char *) {
            if (id == 1) cold_t1 = ids;
            else if (id == 2) cold_t2 = ids;
            else if (id == 3) cold_t3 = ids;
        });
        cached.set_prefix_cache(true);
        cached.admit({1, p1, 20, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        if (cached.last_restored_tokens() != 0)
            throw std::runtime_error("turn 1 expected restored=0");

        p2 = p1;
        p2.insert(p2.end(), cold_t1.begin(), cold_t1.end());
        auto d2 = make_tokens(19, 2);
        p2.insert(p2.end(), d2.begin(), d2.end());

        auto t0 = std::chrono::steady_clock::now();
        cached.admit({2, p2, 20, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        cached_t2_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        size_t r2 = cached.last_restored_tokens();
        if (r2 == 0)
            throw std::runtime_error("turn 2 expected restored > 0");

        p3 = p2;
        p3.insert(p3.end(), cold_t2.begin(), cold_t2.end());
        auto d3 = make_tokens(21, 3);
        p3.insert(p3.end(), d3.begin(), d3.end());
        cached.admit({3, p3, 16, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        size_t r3 = cached.last_restored_tokens();
        if (r3 <= r2)
            throw std::runtime_error("turn 3 expected restored > turn 2");

        auto stats = cached.prefix_stats();
        fprintf(stderr, "[draft=%d] swa_525_3turn: r2=%zu/%zu r3=%zu/%zu hits=%zu saves=%zu\n",
                int(has_draft), r2, p2.size(), r3, p3.size(), stats.hits, stats.saves);
    }

    {
        auto t0 = std::chrono::steady_clock::now();
        auto off_t2 = run_session_sequence(gpu, false, {p2}, {20});
        cold_t2_ms = off_t2[0].elapsed_ms;
        auto off_t3 = run_session_sequence(gpu, false, {p3}, {16});
        require_equal("swa_525_turn2", cold_t2, off_t2[0].generated);
        require_equal("swa_525_turn3", cold_t3, off_t3[0].generated);
        fprintf(stderr, "[draft=%d] timing_525_t2: cold=%.1fms cached=%.1fms\n",
                int(has_draft), cold_t2_ms, cached_t2_ms);
    }

    auto sp1 = make_tokens(2051, 10);
    std::vector<int> sp1_gen, sp2_gen;
    std::vector<int> sp2;
    double sp_cold_pp = 0.0, sp_cached_pp = 0.0;
    {
        double t2_pp = 0.0;
        std::chrono::steady_clock::time_point t2_start;
        Hy4BatchSession cached(gpu,
            [&](int id, int) {
                if (id == 12 && t2_pp == 0.0)
                    t2_pp = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2_start).count();
            },
            [&](int id, const std::vector<int> &ids, const char *) {
                if (id == 11) sp1_gen = ids;
                else if (id == 12) sp2_gen = ids;
            });
        cached.set_prefix_cache(true);
        cached.admit({11, sp1, 12, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        sp2 = sp1;
        sp2.insert(sp2.end(), sp1_gen.begin(), sp1_gen.end());
        auto sd2 = make_tokens(31, 11);
        sp2.insert(sp2.end(), sd2.begin(), sd2.end());

        t2_start = std::chrono::steady_clock::now();
        cached.admit({12, sp2, 12, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        sp_cached_pp = t2_pp;
        size_t r = cached.last_restored_tokens();
        if (r < 2048 || r % 4 != 0)
            throw std::runtime_error("2048 boundary expected restored >= 2048 multiple of 4");
        fprintf(stderr, "[draft=%d] sparse_2051: restored=%zu/%zu cached_pp=%.1fms\n",
                int(has_draft), r, sp2.size(), sp_cached_pp);
    }
    auto off_sp2 = run_session_sequence(gpu, false, {sp2}, {12});
    sp_cold_pp = off_sp2[0].prefill_ms;
    require_equal("sparse_2051_turn2", sp2_gen, off_sp2[0].generated);
    fprintf(stderr, "[draft=%d] sparse_2051_pp_speed: cold_pp=%.1fms cached_pp=%.1fms\n",
            int(has_draft), sp_cold_pp, sp_cached_pp);
    if (sp_cached_pp >= sp_cold_pp)
        throw std::runtime_error("expected cached prefill faster than cold at 2051+ len");

    {
        auto div_p1 = make_tokens(640, 20);
        std::vector<int> div1_gen, div2_gen, div_early_gen;
        std::vector<int> div_p2 = div_p1;
        auto tail = make_tokens(25, 21);
        div_p2.insert(div_p2.end(), tail.begin(), tail.end());
        std::vector<int> div_early(div_p1.begin(), div_p1.begin() + 603);
        div_early.insert(div_early.end(), tail.begin(), tail.end());

        Hy4BatchSession cached(gpu, {}, [&](int id, const std::vector<int> &ids, const char *) {
            if (id == 21) div1_gen = ids;
            else if (id == 22) div2_gen = ids;
            else if (id == 23) div_early_gen = ids;
        });
        cached.set_prefix_cache(true);
        cached.admit({21, div_p1, 96, {1024, 0.f, 0, 0, 1.f, true}});
        while (cached.active() || cached.queued()) cached.step();
        if (div1_gen.size() != 96)
            throw std::runtime_error("ignore_eos expected full 96 tokens");
        cached.admit({22, div_p2, 16, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        size_t r_div = cached.last_restored_tokens();
        if (r_div != 640)
            throw std::runtime_error("prompt-boundary diverge expected swa_snap restored=640, got " + std::to_string(r_div));
        auto off_div2 = run_session_sequence(gpu, false, {div_p2}, {16});
        require_equal("swa_snap_diverge", div2_gen, off_div2[0].generated);

        cached.admit({23, div_early, 12, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        size_t r_early = cached.last_restored_tokens();
        if (r_early != 0)
            throw std::runtime_error("early SWA-wrapped diverge must safely miss");
        auto off_early = run_session_sequence(gpu, false, {div_early}, {12});
        require_equal("swa_wrap_safe_miss", div_early_gen, off_early[0].generated);
        fprintf(stderr, "[draft=%d] swa_snap_diverge: restored=%zu/%zu early_miss=%zu PASS\n",
                int(has_draft), r_div, div_p2.size(), r_early);
    }

    {
        auto a1 = make_tokens(531, 30);
        auto b1 = make_tokens(1117, 31);
        std::map<int, std::vector<int>> res_on;
        std::map<int, size_t> rest_on;
        Hy4BatchSession *ptr = nullptr;
        Hy4BatchSession cached(gpu, {}, [&](int id, const std::vector<int> &ids, const char *) {
            res_on[id] = ids;
            rest_on[id] = ptr->last_restored_tokens();
        });
        ptr = &cached;
        cached.set_prefix_cache(true);
        cached.admit({101, a1, 12, {1024, 0.f, 0, 0, 1.f, false}});
        cached.admit({102, b1, 12, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();

        auto a2 = a1;
        a2.insert(a2.end(), res_on[101].begin(), res_on[101].end());
        auto da = make_tokens(15, 32);
        a2.insert(a2.end(), da.begin(), da.end());

        auto b2 = b1;
        b2.insert(b2.end(), res_on[102].begin(), res_on[102].end());
        auto db = make_tokens(17, 33);
        b2.insert(b2.end(), db.begin(), db.end());

        cached.admit({201, a2, 12, {1024, 0.f, 0, 0, 1.f, false}});
        cached.admit({202, b2, 12, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        if (rest_on[201] == 0 || rest_on[202] == 0)
            throw std::runtime_error("concurrent B=2 expected both slots restored > 0");

        auto mismatch = make_tokens(400, 99);
        cached.admit({203, mismatch, 10, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        if (rest_on[203] != 0)
            throw std::runtime_error("mismatch expected restored=0");

        auto off_a2 = run_session_sequence(gpu, false, {a2}, {12});
        auto off_b2 = run_session_sequence(gpu, false, {b2}, {12});
        auto off_mis = run_session_sequence(gpu, false, {mismatch}, {10});
        require_equal("b2_a2", res_on[201], off_a2[0].generated);
        require_equal("b2_b2", res_on[202], off_b2[0].generated);
        require_equal("mismatch", res_on[203], off_mis[0].generated);

        cached.admit({301, mismatch, 12, {1024, 0.f, 0, 0, 1.f, false}});
        cached.step();
        cached.cancel(301);
        while (cached.active() || cached.queued()) cached.step();
        auto mis_ext = mismatch;
        mis_ext.push_back(12345);
        cached.admit({302, mis_ext, 10, {1024, 0.f, 0, 0, 1.f, false}});
        while (cached.active() || cached.queued()) cached.step();
        if (rest_on[302] != 0)
            throw std::runtime_error("cancelled request must invalidate slot reuse");
        require_equal("post_cancel", res_on[302], run_session_sequence(gpu, false, {mis_ext}, {10})[0].generated);
        fprintf(stderr, "[draft=%d] concurrent_b2_mismatch_cancel: r201=%zu r202=%zu PASS\n",
                int(has_draft), rest_on[201], rest_on[202]);
    }

    gpu.shutdown();
}

void test_single_slot_generate(Checkpoint &model) {
    Residency gpu(model, 4096, 1);
    auto st = gpu.reserve();
    if (!st.ok()) throw std::runtime_error(st.message());
    gpu.upload();

    Hy4PrefixSlot slot;
    Hy4PrefixStats stats;
    stats.enabled = true;
    auto p1 = make_tokens(611, 50);
    size_t r1 = 999, r2 = 0;
    auto g1 = generate(gpu, p1, 16, 1024, 0.f, 0, {}, 0, 1.f, {}, true, &slot, &stats, &r1);
    if (r1 != 0 || g1.size() != 16)
        throw std::runtime_error("single-slot turn 1 failed");
    auto p2 = p1;
    p2.insert(p2.end(), g1.begin(), g1.end());
    auto d2 = make_tokens(11, 51);
    p2.insert(p2.end(), d2.begin(), d2.end());
    auto g2_cached = generate(gpu, p2, 16, 1024, 0.f, 0, {}, 0, 1.f, {}, true, &slot, &stats, &r2);
    if (r2 == 0)
        throw std::runtime_error("single-slot turn 2 expected restored > 0");
    auto g2_cold = generate(gpu, p2, 16, 1024, 0.f, 0, {}, 0, 1.f, {}, true);
    require_equal("single_slot_generate", g2_cached, g2_cold);
    fprintf(stderr, "single_slot_generate: restored=%zu/%zu hits=%zu saves=%zu PASS\n",
            r2, p2.size(), stats.hits, stats.saves);
    gpu.shutdown();
}
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s MODEL_DIR DRAFT_DIR [ENGRAM_DIR]\n", argv[0]);
        return 2;
    }
    const char *engram_dir = argc == 4 ? argv[3] : argv[1];
    try {
        Checkpoint model_draft(argv[1], engram_dir, argv[2], Owners{}, 1, 1);
        share_expert_weights(model_draft);
        test_session_suite(model_draft, true);
        test_single_slot_generate(model_draft);

        Checkpoint model_nodraft(argv[1], engram_dir, {}, Owners{}, 1, 1);
        share_expert_weights(model_nodraft);
        test_session_suite(model_nodraft, false);
        fprintf(stderr, "ALL_HY4_PREFIX_CACHE_TESTS_PASSED\n");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
}
