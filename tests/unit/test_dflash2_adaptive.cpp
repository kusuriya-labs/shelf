#include <shelf/models/qwen35/runtime/dflash2_adaptive.h>
#include <shelf/models/qwen35/runtime/dflash2_adaptive_batch.h>
#include <shelf/models/qwen35/runtime/dflash2_adaptive_costs.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace adaptive = shelf::qwen35::runtime::adaptive;

int main() {
    int failures = 0;
    auto check = [&](bool condition, const char* name) {
        if (!condition) {
            std::printf("FAIL %s\n", name);
            ++failures;
        }
    };
    adaptive::Acceptance acceptance(3, 0.5);
    check(std::abs(acceptance.gain(3) - 0.875) < 1e-9, "expected prefix length");
    acceptance.observe(3, 1, 1.0);
    check(acceptance.conditional[0] == 1.0 && acceptance.conditional[1] == 0.0 &&
              acceptance.conditional[2] == 0.5,
          "censored observations");
    adaptive::Acceptance changing(2, 0.5);
    for (int i = 0; i < 32; ++i) changing.observe(2, 2);
    check(changing.gain(2) > 1.9, "high acceptance adaptation");
    for (int i = 0; i < 32; ++i) changing.observe(2, 0);
    check(changing.gain(2) < 0.02, "low acceptance adaptation");
    for (int i = 0; i < 64; ++i) changing.observe(2, 2);
    check(changing.gain(2) > 1.9, "acceptance recovery");
    std::vector<uint32_t> disabled{100, 0, 120};
    check(adaptive::pick_probe(disabled, 63, 64) == UINT32_MAX, "probe cadence");
    check(adaptive::pick_probe(disabled, 64, 64) == 2, "oldest probe");
    check(adaptive::pick_probe(disabled, 64, 0) == UINT32_MAX, "disabled probes");
    adaptive::CostModel model;
    model.baseline_ms = {0.0, 10.0, 15.0};
    model.speculation_ms = {{}, {{2, 12.0}, {4, 18.0}}};
    check(std::abs(model.milliseconds(2, 1, 3) - 25.0) < 1e-9,
          "interpolated mixed batch cost");
    check(model.milliseconds(2, 0, 0) == 15.0, "normal batch cost");
    check(std::isinf(model.milliseconds(2, 1, 5)), "unmeasured cost");
    model.baseline_ms[1] = 0.0;
    check(std::isinf(model.milliseconds(2, 1, 3)), "missing normal decode cost");
    std::vector<adaptive::Acceptance> requests{
        adaptive::Acceptance(2, 0.9), adaptive::Acceptance(2, 0.01)};
    auto cost = [](uint32_t sessions, uint32_t rows) {
        return 1.0 + 0.4 * sessions + 0.01 * rows;
    };
    auto selected = adaptive::choose(requests, 2, 2, cost, {}, 0.03);
    check(selected.rows == std::vector<uint32_t>({3, 0}), "mixed requests");
    auto expensive = [](uint32_t sessions, uint32_t) {
        return sessions ? 100.0 : 1.0;
    };
    selected = adaptive::choose(requests, 2, 2, expensive, {}, 0.03);
    check(selected.rows == std::vector<uint32_t>({0, 0}), "disable expensive speculation");
    const auto probed = adaptive::choose(requests, 2, 2, expensive, {}, 0.03, 1, 1, 3);
    check(probed.rows == std::vector<uint32_t>({0, 3}), "forced probe");
    std::vector<uint32_t> previous{2, 0};
    selected = adaptive::choose(requests, 2, 2, cost, previous, 0.5);
    check(selected.rows == previous, "hysteresis");
    selected = adaptive::choose(requests, 2, 1, cost, std::vector<uint32_t>{3, 0}, 0.99);
    check(selected.rows == std::vector<uint32_t>({2, 0}), "draft cap");
    {
        const auto measured = adaptive::qwen38_27b_exl3_fp8_cost_model();
        auto model_cost = [&](uint32_t sessions, uint32_t rows) {
            return measured.milliseconds(8u, sessions, rows);
        };
        std::vector<adaptive::Acceptance> cold(8, adaptive::Acceptance(7));
        const auto initial = adaptive::choose(cold, 8u, 7u, model_cost, {}, 0.03, 8u);
        check(std::any_of(initial.rows.begin(), initial.rows.end(),
                          [](uint32_t rows) { return rows != 0u; }),
              "cold start explores speculation");
        for (auto& request : cold) {
            for (int i = 0; i < 8; ++i) request.observe(7, 0);
        }
        const auto rejected = adaptive::choose(cold, 8u, 7u, model_cost, {}, 0.03, 8u);
        check(std::all_of(rejected.rows.begin(), rejected.rows.end(),
                          [](uint32_t rows) { return rows == 0u; }),
              "persistent rejection disables speculation");
    }
    std::vector<std::vector<double>> row_gains{{0.0, 0.0, 0.9, 0.95, 1.8}};
    const auto arbitrary = adaptive::choose_rows(row_gains, 1, 4,
        [](uint32_t sessions, uint32_t rows) { return 2.0 + sessions + 0.1 * rows; }, {}, 0.0);
    check(arbitrary.rows == std::vector<uint32_t>({4}), "row budget interface");
    std::vector<adaptive::Acceptance> limited(3, adaptive::Acceptance(2, 0.9));
    limited[1] = adaptive::Acceptance(2, 0.99);
    selected = adaptive::choose(limited, 3, 2, cost, {}, 0.0, 1);
    check(selected.rows == std::vector<uint32_t>({0, 3, 0}), "session capacity");
    {
        namespace rt = shelf::qwen35::runtime;
        adaptive::CostModel costly;
        costly.baseline_ms = {0.0, 9.0, 10.0};
        costly.speculation_ms = {{}, {{2, 100.0}, {3, 101.0}}};
        rt::DFlash2BatcherStats stats;
        rt::DFlash2BatcherBinding binding;
        binding.num_drafts = 2u;
        binding.adaptive_cost = &costly;
        binding.stats = &stats;
        rt::ContinuousBatcherConfig config;
        std::vector<rt::RuntimeRequest> live(2);
        std::vector<rt::RuntimeRequest*> active;
        for (auto& request : live) {
            request.state = rt::RequestState::Active;
            request.max_new_tokens = 100u;
            request.generated_tokens = 1u;
            request.sequence.position = 20u;
            request.sequence.max_seq_len = 512u;
            request.dflash = std::make_unique<rt::DFlash2RequestSession>();
            request.dflash->ready = true;
            request.dflash->acceptance = adaptive::Acceptance(2, 0.5);
            request.dflash->adaptive_probe_age = 64u;
            active.push_back(&request);
        }
        uint64_t round = 63u;
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[0].dflash->adaptive_rows == 3u && live[1].dflash->adaptive_rows == 0u,
              "batch probe is limited to one request");
        check(stats.adaptive_k_counts[2] == 1u && stats.adaptive_k_counts[0] == 1u,
              "batch choice metrics");
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(live[0].dflash->adaptive_rows == 0u && live[1].dflash->adaptive_rows == 0u,
              "probe returns to previous choice");
        live[0].dflash.reset();
        live[0].pending_decode_token = 12;
        live[1].state = rt::RequestState::Finished;
        stats = {};
        rt::select_dflash2_adaptive(active, binding, config, round);
        check(stats.adaptive_rounds == 1u && stats.adaptive_k_counts[0] == 1u &&
                  stats.adaptive_predicted_tokens == 1.0,
              "normal-only batch metrics");
    }
    for (uint32_t seed = 1; seed <= 16; ++seed) {
        std::vector<adaptive::Acceptance> values(3, adaptive::Acceptance(3));
        for (uint32_t i = 0; i < values.size(); ++i) {
            for (uint32_t j = 0; j < 3; ++j) {
                values[i].conditional[j] = ((seed * (i + 3) + j * 7) % 9 + 1) / 10.0;
            }
        }
        auto trial_cost = [seed](uint32_t sessions, uint32_t rows) {
            return 2.0 + 0.2 * sessions + 0.03 * rows + 0.01 * seed;
        };
        const auto decision = adaptive::choose(values, 3, 3, trial_cost, {}, 0.0);
        double best = 0.0;
        for (uint32_t a = 0; a <= 3; ++a) {
            for (uint32_t b = 0; b <= 3; ++b) {
                for (uint32_t c = 0; c <= 3; ++c) {
                    const uint32_t sessions = (a > 0) + (b > 0) + (c > 0);
                    const uint32_t rows = a + b + c + sessions;
                    const double score = (3.0 + values[0].gain(a) + values[1].gain(b) +
                                          values[2].gain(c)) / trial_cost(sessions, rows);
                    best = std::max(best, score);
                }
            }
        }
        check(std::abs(decision.predicted - best) < 1e-10, "exhaustive optimum");
        for (uint32_t cap = 1; cap <= 2; ++cap) {
            const auto capped = adaptive::choose(values, 3, 3, trial_cost, {}, 0.0, cap);
            double capped_best = 0.0;
            for (uint32_t a = 0; a <= 3; ++a) {
                for (uint32_t b = 0; b <= 3; ++b) {
                    for (uint32_t c = 0; c <= 3; ++c) {
                        const uint32_t sessions = (a > 0) + (b > 0) + (c > 0);
                        if (sessions > cap) continue;
                        const uint32_t rows = a + b + c + sessions;
                        const double score = (3.0 + values[0].gain(a) + values[1].gain(b) +
                                              values[2].gain(c)) / trial_cost(sessions, rows);
                        capped_best = std::max(capped_best, score);
                    }
                }
            }
            check(std::abs(capped.predicted - capped_best) < 1e-10,
                  "exhaustive optimum with session cap");
        }
    }
    return failures ? 1 : 0;
}
