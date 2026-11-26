#include <shelf/models/qwen35/runtime/dflash2_adaptive_batch.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace rt = shelf::qwen35::runtime;

static int g_failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::printf("FAIL %s\n", name);
        ++g_failures;
    }
}

static int controller_tests() {
    rt::DFlash2AdaptiveController ctrl;
    check(ctrl.current_k == 5u && ctrl.switches == 0 && ctrl.rounds == 0, "initial state");
    bool rejected = false;
    try {
        rt::DFlash2AdaptiveController bad(4u, 0u, 0.25);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "reject zero interval");
    rejected = false;
    try {
        rt::DFlash2AdaptiveController bad(4u, 8u, 0.0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "reject zero alpha");
    ctrl.observe(1u, 5u, false, 0u);
    ctrl.observe(6u, 5u, false, 1u);
    check(ctrl.rounds == 0 && ctrl.current_k == 5u, "reject bad observations");

    for (uint32_t configured : {1u, 4u, 5u, 7u, 15u}) {
        for (uint32_t batch : {1u, 8u, 11u, 12u, 16u, 32u}) {
            rt::DFlash2AdaptiveController c(configured);
            const auto ceiling = batch >= 12u ? std::min(configured, 4u) : configured;
            c.apply_batch_limit(batch);
            for (int r = 0; r < 32; ++r) c.observe(0u, c.current_k, false, batch);
            check(c.current_k == std::min(2u, configured), "low acceptance reaches floor");
            const auto before = c.switches;
            for (int r = 0; r < 128; ++r) c.observe(c.current_k, c.current_k, false, batch);
            check(c.current_k == ceiling, "full acceptance reaches configured ceiling");
            check(c.switches - before == ceiling - std::min(2u, configured), "one step per switch");
            const auto at_ceiling = c.switches;
            for (uint32_t r = 0; r < 64; ++r)
                c.observe((ceiling + r % 2u) / 2u, ceiling, false, batch);
            check(c.switches == at_ceiling, "steady half acceptance holds the ceiling");
            for (int r = 0; r < 128; ++r) c.observe(0u, c.current_k, false, batch);
            check(c.current_k == std::min(2u, configured), "returns to floor");
        }
    }
    for (uint32_t batch : {11u, 12u}) {
        rt::DFlash2AdaptiveController c(7u, 2u);
        check(c.apply_batch_limit(batch) == (batch == 11u ? 5u : 4u), "cap before first verify");
        c.apply_batch_limit(12u);
        for (int r = 0; r < 2; ++r) c.observe(1u, 4u, false, batch);
        check(c.current_k == (batch == 11u ? 4u : 3u), "demotion at the batch boundary");
        rt::DFlash2AdaptiveController promote(4u, 2u);
        for (int r = 0; r < 2; ++r) promote.observe(0u, 4u, false, batch);
        promote.observe(2u, 3u, false, batch);
        promote.observe(1u, 3u, false, batch);
        check(promote.current_k == (batch == 11u ? 4u : 3u), "promotion at the batch boundary");
    }
    rt::DFlash2AdaptiveController zero(0u);
    zero.observe(0u, 0u, false, 1u);
    check(zero.current_k == 0u && zero.rounds == 0u, "zero drafts guard");

    rt::DFlash2AdaptiveController half(7u);
    for (int r = 0; r < 24; ++r) half.observe(0u, half.current_k, false, 1u);
    for (int r = 0; r < 7; ++r) half.observe(1u, 2u, false, 32u);
    check(half.current_k == 2u, "minimum switch interval");
    half.observe(1u, 2u, false, 32u);
    check(half.current_k == 3u, "one accepted out of two promotes K");
    half.observe(1u, 3u, false, 32u);
    check(half.acc_ema == 1.0 / 3.0, "new width seeds a fresh EMA");
    for (int r = 0; r < 64; ++r) half.observe(1u, 3u, false, 32u);
    check(half.current_k == 3u && half.switches == 4u, "steady acceptance avoids oscillation");
    const auto before = half;
    for (int r = 0; r < 32; ++r) {
        half.observe(3u, 3u, true, 1u);
        half.observe(1u, 1u, false, 1u);
    }
    check(half.acc_ema == before.acc_ema && half.switches == before.switches &&
          half.rounds_since_switch == before.rounds_since_switch && half.rounds == before.rounds,
          "ngram hits and clipped rounds do not train or advance dwell time");
    return g_failures;
}

static int batch_tests() {
    rt::DFlash2BatcherStats stats;
    rt::DFlash2BatcherBinding binding;
    binding.num_drafts = 15u;
    binding.stats = &stats;
    rt::ContinuousBatcherConfig config;
    config.page_tokens = 16u;
    std::vector<rt::RuntimeRequest> live(32);
    std::vector<rt::RuntimeRequest*> active;
    for (auto& request : live) {
        request.state = rt::RequestState::Active;
        request.max_new_tokens = 100u;
        request.sequence.position = 20u;
        request.sequence.max_seq_len = 512u;
        request.dflash = std::make_unique<rt::DFlash2RequestSession>();
        request.dflash->ready = true;
        request.dflash->adaptive = rt::DFlash2AdaptiveController(binding.num_drafts);
        active.push_back(&request);
    }
    uint64_t round = 0;
    rt::select_dflash2_adaptive(active, binding, config, round);
    uint32_t total_rows = 0;
    for (const auto& request : live) {
        check(request.dflash->adaptive_rows >= 2u && request.dflash->adaptive_rows <= 5u,
              "c=32 sessions all speculate");
        total_rows += request.dflash->adaptive_rows;
    }
    check(round == 1u && stats.adaptive_rounds == 1u, "c=32 round recorded");
    check(total_rows / live.size() == 5u, "c=32 caps K at four drafts");
    check(total_rows <= rt::kDFlash2SpecMaxVerifyRows, "c=32 verify rows fit the cap");
    check(rt::dflash2_session_rows(16u, 5u) * 32u <= rt::kDFlash2SpecMaxVerifyRows,
          "c=32 reserved rows fit the cap");

    active.resize(2);
    rt::DFlash2AdaptiveController single(binding.num_drafts);
    for (int r = 0; r < 100; ++r) {
        const uint32_t batch = rt::select_dflash2_adaptive(active, binding, config, round);
        for (auto* request : active) {
            auto& session = *request->dflash;
            const uint32_t k = session.adaptive_rows - 1u;
            session.adaptive.observe(request == active.front() ? k : 0u, k, false, batch);
        }
        single.observe(single.current_k, single.current_k, false, 1u);
    }
    rt::select_dflash2_adaptive(active, binding, config, round);
    check(active[0]->dflash->adaptive_rows == 16u && active[1]->dflash->adaptive_rows == 3u,
          "mixed sessions verify independently at K=15 and K=2");
    check(single.current_k == 15u && single.switches == 10u && single.rounds == 100u,
          "single request has independent history");
    check(stats.adaptive_k_counts.size() == 16u && stats.adaptive_k_counts[15] > 0u,
          "histogram includes configured ceiling");
    const auto switches = active.front()->dflash->adaptive.switches;
    for (std::size_t i = active.size(); i < live.size(); ++i) active.push_back(&live[i]);
    rt::select_dflash2_adaptive(active, binding, config, round);
    check(active.front()->dflash->adaptive_rows == 5u &&
          active.front()->dflash->adaptive.switches == switches + 1u,
          "new high concurrency cap applies before verify");
    active.resize(2);
    binding.num_drafts = 1u;
    for (auto* request : active)
        request->dflash->adaptive = rt::DFlash2AdaptiveController(binding.num_drafts);
    rt::select_dflash2_adaptive(active, binding, config, round);
    for (const auto* request : active)
        check(request->dflash->adaptive_rows == 2u, "one draft stays within the budget");
    auto& tail = live.front();
    tail.generated_tokens = tail.max_new_tokens - 1u;
    rt::select_dflash2_adaptive(active, binding, config, round);
    check(tail.dflash->adaptive_rows == 1u, "an admitted tail runs one target row in its own session");
    tail.generated_tokens = 0u;
    tail.prefix_capture = 1u;
    tail.sequence.position = config.page_tokens - 1u;
    rt::select_dflash2_adaptive(active, binding, config, round);
    check(tail.dflash->adaptive_rows == 1u, "a boundary without replay support keeps the anchor");
    return g_failures;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--check-cap-override") == 0)
        return shelf::runtime::dflash2_policy_max_sessions() == 4u ? 0 : 1;
    unsetenv("SHELF_DFLASH2_POLICY_MAX_SESSIONS");
    check(shelf::runtime::dflash2_policy_max_sessions() == 32u, "default capacity is c=32");
    controller_tests();
    batch_tests();
    return g_failures ? 1 : 0;
}
