#pragma once

#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace shelf::hy4 {
class CollectiveAborted : public std::runtime_error {
public:
    CollectiveAborted() : std::runtime_error("collective が中断されました") {}
};

class SyncBarrier {
    std::mutex mutex_;
    std::condition_variable condition_;
    const int participants_;
    int waiting_ = 0;
    unsigned generation_ = 0;
    bool aborted_ = false;
public:
    explicit SyncBarrier(int participants) : participants_(participants) {}
    void arrive_and_wait() {
        std::unique_lock lock(mutex_);
        if (aborted_)
            throw CollectiveAborted();
        const auto generation = generation_;
        if (++waiting_ == participants_) {
            waiting_ = 0;
            ++generation_;
            condition_.notify_all();
        } else {
            condition_.wait(lock, [&] { return aborted_ || generation_ != generation; });
            if (aborted_)
                throw CollectiveAborted();
        }
    }
    void abort() {
        std::lock_guard lock(mutex_);
        aborted_ = true;
        condition_.notify_all();
    }
    void reset() {
        std::lock_guard lock(mutex_);
        waiting_ = 0;
        ++generation_;
        aborted_ = false;
    }
};

enum class ReduceTransport { push, peer, host };
inline ReduceTransport select_reduce_transport(bool push, bool source_access, bool destination_access) {
    return push ? (source_access ? ReduceTransport::push : ReduceTransport::host)
                : (destination_access ? ReduceTransport::peer : ReduceTransport::host);
}
}
