#include <shelf/models/hy4_flash/generate.h>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace shelf::hy4;

static void test_adaptive_recovery() {
    Hy4DflyAdaptiveController controller;
    for (int i = 0; i < 16; ++i)
        controller.observe(0, 1);
    if (controller.current_k != 2)
        throw std::runtime_error("低受理が続いても K=2 に下がりません");

    for (int k = 2; k < 5; ++k) {
        const int accepted = k == 2 ? 1 : k;
        for (int i = 0; i < 16 && controller.current_k == k; ++i)
            controller.observe(accepted, 1);
        if (controller.current_k != k + 1)
            throw std::runtime_error("受理数が回復しても K=" + std::to_string(k + 1) + " に上がりません");
    }
}

static int test_adaptive_controller() {
    Hy4DflyAdaptiveController ctrl;
    if (ctrl.current_k != 4 || ctrl.switches != 0 || ctrl.rounds != 0)
        return 1;
    try {
        Hy4DflyAdaptiveController bad_interval(4, 0, 0.25f);
        return 1;
    } catch (const std::invalid_argument &) {}
    try {
        Hy4DflyAdaptiveController bad_alpha(4, 8, 0.f);
        return 1;
    } catch (const std::invalid_argument &) {}
    try {
        ctrl.observe(1.0, 0);
        return 1;
    } catch (const std::invalid_argument &) {}
    try {
        ctrl.observe(-0.1, 1);
        return 1;
    } catch (const std::invalid_argument &) {}

    try {
        ctrl.observe(5.0, 1);
        return 1;
    } catch (const std::invalid_argument &) {}

    for (int batch : {1, 16, 32}) {
        Hy4DflyAdaptiveController c;
        for (int r = 0; r < 7; ++r)
            if (c.observe(0, batch) != 4)
                return 1;
        if (c.observe(0, batch) != 2 || c.switches != 1)
            return 1;

        const int cap = batch > 16 ? 4 : 5;
        for (int k = 2; k < cap; ++k) {
            for (int r = 0; r < 7; ++r)
                if (c.observe(k, batch) != k || c.acc_ema != k)
                    return 1;
            if (c.observe(k, batch) != k + 1)
                return 1;
        }
        for (int r = 0; r < 16; ++r)
            if (c.observe(cap, batch) != cap)
                return 1;
        for (int r = 0; r < 24; ++r)
            c.observe(0, batch);
        if (c.current_k != 2)
            return 1;
    }

    struct BandCase {
        int start_k;
        int batch;
        double a;
        double b;
    };
    const std::vector<BandCase> bands{
        {2, 16, 0.75, 0.875},
        {3, 16, 0.75, 0.875},
        {3, 16, 1.75, 1.875},
        {4, 16, 1.75, 1.875},
        {4, 16, 3.125, 3.25},
        {5, 16, 3.125, 3.25},
        {2, 32, 1.25, 1.375},
        {3, 32, 1.25, 1.375},
        {3, 32, 2.375, 2.50},
        {4, 32, 2.375, 2.50},
    };
    for (const auto &band : bands) {
        Hy4DflyAdaptiveController c(band.start_k, 8, 0.25f);
        for (int r = 0; r < 64; ++r) {
            int got = c.observe((r % 2) ? band.a : band.b, band.batch);
            if (got != band.start_k || c.switches != 0)
                return 1;
        }
    }
    return 0;
}

int main() {
    unsetenv("SHELF_HY4_DFLY_DRAFTS");
    unsetenv("SHELF_HY4_DFLY_ADAPTIVE");
    Checkpoint model;
    model.tp = 2;
    Residency parallel(model, 32, 2);
    if (parallel.reserve().code() != shelf::Status::Code::unsupported)
        return 1;
    model.tp = 1;
    Residency gpu(model, 32, 2);
    if (gpu.draft_tokens != 4)
        return 1;
    for (int k : {2, 3, 4, 5}) {
        Residency configured(model, 32, 2, 0, k);
        if (configured.draft_tokens != k)
            return 1;
        Hy4BatchSession scoped(configured, {}, {}, k);
        if (scoped.draft_tokens() != k)
            return 1;
    }
    for (int bad : {-1, 1, 6}) {
        try {
            Residency rejected(model, 32, 2, 0, bad);
            return 1;
        } catch (const std::invalid_argument &) {}
    }
    setenv("SHELF_HY4_DFLY_DRAFTS", "3", 1);
    Residency env_gpu(model, 32, 2);
    if (env_gpu.draft_tokens != 3)
        return 1;
    setenv("SHELF_HY4_DFLY_DRAFTS", "6", 1);
    try {
        Residency env_bad(model, 32, 2);
        return 1;
    } catch (const std::invalid_argument &) {}
    unsetenv("SHELF_HY4_DFLY_DRAFTS");
    if (gpu.adaptive_draft)
        return 1;
    setenv("SHELF_HY4_DFLY_ADAPTIVE", "1", 1);
    Residency adaptive_env_gpu(model, 32, 2);
    if (!adaptive_env_gpu.adaptive_draft)
        return 1;
    Hy4BatchSession adaptive_session(adaptive_env_gpu, {}, {});
    if (!adaptive_session.adaptive() || adaptive_session.k_switches() != 0)
        return 1;
    setenv("SHELF_HY4_DFLY_ADAPTIVE", "2", 1);
    try {
        Residency adaptive_bad(model, 32, 2);
        return 1;
    } catch (const std::invalid_argument &) {}
    unsetenv("SHELF_HY4_DFLY_ADAPTIVE");
    Residency adaptive_explicit(model, 32, 2, 0, 4, true);
    if (!adaptive_explicit.adaptive_draft)
        return 1;
    test_adaptive_recovery();
    if (test_adaptive_controller())
        return 1;
    std::vector<int> cancelled;
    Hy4BatchSession session(gpu, {}, [&](int id, const std::vector<int> &ids, const char *reason) {
        if (!ids.empty() || std::string_view(reason) != "cancelled")
            throw std::runtime_error("キャンセルの通知が不正です");
        cancelled.push_back(id);
    });
    if (session.draft_tokens() != 4)
        return 1;
    session.set_draft_tokens(5);
    if (session.draft_tokens() != 5)
        return 1;
    try {
        session.set_draft_tokens(1);
        return 1;
    } catch (const std::invalid_argument &) {}
    for (int id = 0; id < 5; ++id) session.admit({id, {100}, 3, {}});
    if (session.active() != 2 || session.queued() != 3) return 1;
    for (int id : {0, 3}) {
        try {
            session.admit({id, {200}, 2, {}});
            return 1;
        } catch (const std::invalid_argument &) {}
    }
    session.cancel(3);
    session.cancel(0);
    session.cancel(1);
    if (session.active() != 2 || session.queued() != 0) return 1;
    session.cancel(4);
    session.cancel(2);
    session.step();
    if (session.active() || cancelled != std::vector<int>{3, 0, 1, 4, 2})
        return 1;
    if (session.prefill_decode_interval() != 1)
        return 1;
    session.set_prefill_decode_interval(0);
    if (session.prefill_decode_interval() != 0)
        return 1;
    session.set_prefill_decode_interval(2);
    if (session.prefill_decode_interval() != 2)
        return 1;
    try {
        session.set_prefill_decode_interval(-1);
        return 1;
    } catch (const std::invalid_argument &) {}
    if (gpu.pages_for_tokens(0) != 0 || gpu.pages_for_tokens(1) != 1 ||
        gpu.pages_for_tokens(kv_page_tokens) != 1 || gpu.pages_for_tokens(kv_page_tokens + 1) != 2)
        return 1;
    return 0;
}
