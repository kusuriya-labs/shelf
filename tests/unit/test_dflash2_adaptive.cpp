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
    check(ctrl.current_k == 4u && ctrl.switches == 0 && ctrl.rounds == 0, "initial state");
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
    ctrl.observe(1.0, 0u);
    ctrl.observe(-0.1, 1u);
    check(ctrl.rounds == 0 && ctrl.current_k == 4u, "reject bad observations");

    for (uint32_t batch : {1u, 16u}) {
        rt::DFlash2AdaptiveController c(4u, 8u, 0.25);
        for (int r = 0; r < 7; ++r)
            check(c.observe(0.5, batch) == 4u && c.switches == 0, "before first switch");
        check(c.observe(0.5, batch) == 2u && c.switches == 1, "drop to K=2");
        for (int r = 0; r < 7; ++r)
            check(c.observe(2.5, batch) == 2u && c.switches == 1, "hysteresis holds K=2");
        int k = c.current_k;
        for (int r = 0; r < 16 && k != 4; ++r) k = c.observe(2.5, batch);
        check(k == 4, "recover to K=4");
        const uint64_t switches_before = c.switches;
        k = c.current_k;
        for (int r = 0; r < 24 && k != 5; ++r) k = c.observe(4.0, batch);
        check(k == 5 && c.switches == switches_before + 1, "raise to K=5");
        k = c.current_k;
        for (int r = 0; r < 24 && k != 3; ++r) k = c.observe(1.4, batch);
        check(k == 3, "drop to K=3");
        k = c.current_k;
        for (int r = 0; r < 24 && k != 2; ++r) k = c.observe(0.6, batch);
        check(k == 2, "drop to K=2 again");
        check(c.acc_ema > 0.0 && c.acc_ema < 5.0, "EMA stays bounded");
    }

    {
        rt::DFlash2AdaptiveController c(4u, 8u, 0.25);
        for (int r = 0; r < 8; ++r) c.observe(0.8, 32u);
        check(c.current_k == 2u, "c32 drops to K=2");
        int k = c.current_k;
        for (int r = 0; r < 24 && k != 3; ++r) k = c.observe(2.4, 32u);
        check(k == 3, "c32 raises to K=3");
        k = c.current_k;
        for (int r = 0; r < 24 && k != 4; ++r) k = c.observe(4.0, 32u);
        check(k == 4, "c32 raises to K=4");
        for (int r = 0; r < 24; ++r)
            check(c.observe(5.0, 32u) == 4u, "c32 never selects K=5");
    }

    struct BandCase {
        uint32_t start_k;
        uint32_t batch;
        double low;
        double high;
    };
    const std::vector<BandCase> bands{
        {2u, 1u, 1.10, 1.20}, {3u, 1u, 1.10, 1.20},
        {3u, 1u, 1.70, 1.90}, {4u, 1u, 1.70, 1.90},
        {4u, 16u, 3.10, 3.30}, {5u, 16u, 3.10, 3.30},
        {2u, 32u, 1.70, 1.90}, {3u, 32u, 1.70, 1.90},
        {3u, 32u, 3.10, 3.30}, {4u, 32u, 3.10, 3.30},
    };
    for (const auto& band : bands) {
        rt::DFlash2AdaptiveController c(band.start_k, 8u, 0.25);
        for (int r = 0; r < 64; ++r) {
            const uint32_t got = c.observe((r % 2) ? band.low : band.high, band.batch);
            if (got != band.start_k || c.switches != 0) {
                check(false, "hysteresis band");
                break;
            }
        }
    }
    return g_failures;
}

static int batch_tests() {
    rt::DFlash2BatcherStats stats;
    rt::DFlash2BatcherBinding binding;
    binding.num_drafts = 5u;
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

    for (auto* request : active) {
        request->dflash->adaptive.current_k = 2u;
        request->dflash->adaptive.has_sample = true;
        request->dflash->adaptive.acc_ema = 0.2;
        request->dflash->adaptive_rows = 0u;
    }
    rt::select_dflash2_adaptive(active, binding, config, round);
    for (const auto& request : live)
        check(request.dflash->adaptive_rows == 3u, "low acceptance degrades to K=2");

    rt::DFlash2AdaptiveController observed(4u);
    const uint32_t next_k = rt::observe_dflash2_adaptive(
        *live[0].dflash, 5u, static_cast<uint32_t>(live.size()), binding.num_drafts);
    check(next_k <= 4u, "observer respects the c32 cap");
    check(observed.current_k == 4u, "independent controller unchanged");

    binding.num_drafts = 1u;
    rt::select_dflash2_adaptive(active, binding, config, round);
    for (const auto& request : live)
        check(request.dflash->adaptive_rows == 2u, "one draft stays within the budget");
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
