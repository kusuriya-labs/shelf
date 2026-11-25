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
    ctrl.observe(1.0, 0u);
    ctrl.observe(-0.1, 1u);
    check(ctrl.rounds == 0 && ctrl.current_k == 5u, "reject bad observations");

    for (uint32_t batch : {1u, 16u, 32u}) {
        rt::DFlash2AdaptiveController c(4u);
        for (int r = 0; r < 16; ++r) c.observe(0.0, batch);
        check(c.current_k == 2u, "low acceptance reaches K=2");
        const auto switches = c.switches;
        for (int r = 0; r < 7; ++r) c.observe(2.0, batch);
        check(c.switches == switches, "minimum switch interval");
        rt::DFlash2AdaptiveController half(2u);
        for (int r = 0; r < 8; ++r) half.observe(1.0, batch);
        check(half.current_k == 3u, "one accepted draft out of two promotes K");
        for (int r = 0; r < 40; ++r) c.observe(c.current_k, batch);
        check(c.current_k == (batch > 16u ? 4u : 5u), "full acceptance recovers from K=2");
        for (int r = 0; r < 64; ++r) c.observe(0.48 * c.current_k, batch);
        check(c.current_k == (batch > 16u ? 4u : 5u), "hysteresis holds K");
        check(c.acc_ema >= 0.0 && c.acc_ema <= 1.0, "normalized EMA");
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
