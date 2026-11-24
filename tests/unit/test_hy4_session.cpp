#include <shelf/models/hy4_flash/generate.h>
#include <cstdlib>
#include <stdexcept>

using namespace shelf::hy4;

int main() {
    unsetenv("SHELF_HY4_DFLY_DRAFTS");
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
