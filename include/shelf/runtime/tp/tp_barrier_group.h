#pragma once

#include <shelf/runtime/tp/tp_execution.h>

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>

namespace shelf::runtime {

class TpBarrierGroup final : public TpBarrierHook {
public:
    TpBarrierGroup(int rank_count, std::vector<int> devices, TpTransport* transport);
    ~TpBarrierGroup() override;

    Status initialize();
    void shutdown() noexcept;

    void abort(const Status& reason) noexcept;
    bool aborted() const noexcept;
    Status abort_status() const noexcept;

    Status arrive(int rank, const TpSumTarget& target, hipStream_t stream,
                    bool peer_written) override;
    Status peer_write_target(int rank, std::size_t bytes, void*& peer_out) override;

private:
    int rank_count_;
    std::vector<int> devices_;
    TpTransport* transport_;
    std::vector<hipEvent_t> ready_;
    std::vector<TpSumTarget> targets_;
    std::vector<hipStream_t> streams_;
    std::vector<char> peer_written_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int arrived_ = 0;
    uint64_t open_barrier_ = 0;
    uint64_t completed_barriers_ = 0;
    bool aborted_ = false;
    Status abort_status_;
};

}  // namespace shelf::runtime
