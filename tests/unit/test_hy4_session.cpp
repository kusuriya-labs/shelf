#include <shelf/models/hy4_flash/generate.h>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace shelf::hy4;

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

    for (int batch : {1, 16}) {
        Hy4DflyAdaptiveController c(4, 8, 0.25f);
        for (int r = 0; r < 7; ++r) {
            if (c.observe(0.5, batch) != 4 || c.switches != 0)
                return 1;
        }
        if (c.observe(0.5, batch) != 2 || c.switches != 1)
            return 1;

        for (int r = 0; r < 7; ++r) {
            if (c.observe(2.5, batch) != 2 || c.switches != 1)
                return 1;
        }
        int k_after_std = c.current_k;
        for (int r = 0; r < 16 && k_after_std != 4; ++r)
            k_after_std = c.observe(2.5, batch);
        if (k_after_std != 4)
            return 1;

        const size_t switches_before_high = c.switches;
        int k_after_high = c.current_k;
        for (int r = 0; r < 24 && k_after_high != 5; ++r)
            k_after_high = c.observe(4.0, batch);
        if (k_after_high != 5 || c.switches != switches_before_high + 1)
            return 1;

        int k_after_low = c.current_k;
        for (int r = 0; r < 24 && k_after_low != 3; ++r)
            k_after_low = c.observe(1.4, batch);
        if (k_after_low != 3)
            return 1;

        int k_after_vlow = c.current_k;
        for (int r = 0; r < 24 && k_after_vlow != 2; ++r)
            k_after_vlow = c.observe(0.6, batch);
        if (k_after_vlow != 2)
            return 1;
    }

    {
        Hy4DflyAdaptiveController c32(4, 8, 0.25f);
        for (int r = 0; r < 8; ++r)
            c32.observe(0.8, 32);
        if (c32.current_k != 2)
            return 1;
        int k_std = c32.current_k;
        for (int r = 0; r < 24 && k_std != 3; ++r)
            k_std = c32.observe(2.4, 32);
        if (k_std != 3)
            return 1;
        int k_high = c32.current_k;
        for (int r = 0; r < 24 && k_high != 4; ++r)
            k_high = c32.observe(4.0, 32);
        if (k_high != 4)
            return 1;
        for (int r = 0; r < 24; ++r)
            if (c32.observe(5.0, 32) != 4)
                return 1;
        int k_low = c32.current_k;
        for (int r = 0; r < 24 && k_low != 2; ++r)
            k_low = c32.observe(1.0, 32);
        if (k_low != 2)
            return 1;
    }

    struct BandCase {
        int start_k;
        int batch;
        double a;
        double b;
    };
    const std::vector<BandCase> bands{
        {2, 1, 1.10, 1.20},
        {3, 1, 1.10, 1.20},
        {3, 1, 1.70, 1.90},
        {4, 1, 1.70, 1.90},
        {4, 16, 3.10, 3.30},
        {5, 16, 3.10, 3.30},
        {2, 32, 1.70, 1.90},
        {3, 32, 1.70, 1.90},
        {3, 32, 3.10, 3.30},
        {4, 32, 3.10, 3.30},
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
