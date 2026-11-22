#include <shelf/models/hy4_flash/generate.h>
#include <stdexcept>

using namespace shelf::hy4;

int main() {
    Checkpoint model;
    model.tp = 2;
    Residency parallel(model, 32, 2);
    if (parallel.reserve().code() != shelf::Status::Code::unsupported)
        return 1;
    model.tp = 1;
    Residency gpu(model, 32, 2);
    std::vector<int> cancelled;
    Hy4BatchSession session(gpu, {}, [&](int id, const std::vector<int> &ids, const char *reason) {
        if (!ids.empty() || std::string_view(reason) != "cancelled")
            throw std::runtime_error("キャンセルの通知が不正です");
        cancelled.push_back(id);
    });
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
    return session.active() || cancelled != std::vector<int>{3, 0, 1, 4, 2};
}
