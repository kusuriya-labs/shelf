#include <shelf/runtime/tp/tp_barrier_group.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>

#include "tp_transport_internal.h"

namespace shelf::runtime {

TpBarrierGroup::TpBarrierGroup(int rank_count, std::vector<int> devices,
                               TpTransport* transport)
    : rank_count_(rank_count),
      devices_(std::move(devices)),
      transport_(transport),
      ready_(static_cast<std::size_t>(rank_count), nullptr),
      targets_(static_cast<std::size_t>(rank_count)),
      streams_(static_cast<std::size_t>(rank_count), nullptr),
      peer_written_(static_cast<std::size_t>(rank_count), 0),
      abort_status_(Status::make_ok()) {}

TpBarrierGroup::~TpBarrierGroup() {
    shutdown();
}

Status TpBarrierGroup::initialize() {
    if (rank_count_ < 2) {
        return Status::invalid_argument("tp barrier group requires at least 2 ranks",
                                        __FILE__, __LINE__);
    }
    if (static_cast<int>(devices_.size()) != rank_count_) {
        return Status::invalid_argument("tp barrier device count mismatch", __FILE__,
                                        __LINE__);
    }
    if (transport_ == nullptr) {
        return Status::invalid_argument("tp barrier group has no transport", __FILE__,
                                        __LINE__);
    }
    for (int r = 0; r < rank_count_; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_[static_cast<std::size_t>(r)]);
        if (!scope.ok()) return scope.status();
        hipEvent_t event = nullptr;
        const hipError_t e = hipEventCreate(&event);
        if (e != hipSuccess) {
            return Status::hip_error("hipEventCreate tp ready", hipGetErrorString(e),
                                     __FILE__, __LINE__);
        }
        ready_[static_cast<std::size_t>(r)] = event;
    }
    return Status::make_ok();
}

void TpBarrierGroup::shutdown() noexcept {
    for (std::size_t r = 0; r < ready_.size(); ++r) {
        if (ready_[r] == nullptr) continue;
        auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (scope.ok()) {
            shelf::gpu::discard_cleanup_result(hipEventDestroy(ready_[r]));
        }
        ready_[r] = nullptr;
    }
}

void TpBarrierGroup::abort(const Status& reason) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (aborted_) return;
    aborted_ = true;
    abort_status_ = reason;
    arrived_ = 0;
    cv_.notify_all();
}

bool TpBarrierGroup::aborted() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return aborted_;
}

Status TpBarrierGroup::abort_status() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!aborted_) return Status::make_ok();
    return abort_status_;
}

Status TpBarrierGroup::arrive(int rank, const TpSumTarget& target, hipStream_t stream,
                                bool peer_written) {
    if (rank < 0 || rank >= rank_count_) {
        return Status::invalid_argument("tp barrier rank out of range", __FILE__, __LINE__);
    }
    const std::size_t index = static_cast<std::size_t>(rank);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (aborted_) return abort_status_;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (aborted_) return abort_status_;
    targets_[index] = target;
    streams_[index] = stream;
    peer_written_[index] = peer_written ? 1 : 0;
    const uint64_t seq = open_barrier_;
    ++arrived_;
    if (arrived_ == rank_count_) {
        bool all_targets_empty = true;
        for (const TpSumTarget& t : targets_) {
            if (tp_target_bytes(t) != 0) {
                all_targets_empty = false;
                break;
            }
        }
        Status st = Status::make_ok();
        if (!all_targets_empty) {
            bool peer_write_complete = true;
            for (char flag : peer_written_) {
                if (flag == 0) {
                    peer_write_complete = false;
                    break;
                }
            }
            TpSumInvocation invocation{targets_, streams_, ready_};
            invocation.peer_write_complete = peer_write_complete;
            st = transport_->sum_hidden(invocation);
        }
        if (!st.ok()) {
            aborted_ = true;
            abort_status_ = st;
            arrived_ = 0;
            cv_.notify_all();
            return st;
        }
        arrived_ = 0;
        ++open_barrier_;
        completed_barriers_ = open_barrier_;
        cv_.notify_all();
        return Status::make_ok();
    }
    cv_.wait(lock, [&] { return completed_barriers_ > seq || aborted_; });
    return aborted_ ? abort_status_ : Status::make_ok();
}

bool TpBarrierGroup::supports_combine_fusion() const noexcept {
    return transport_ != nullptr && transport_->supports_combine_fusion();
}

Status TpBarrierGroup::peer_write_target(int rank, std::size_t bytes, void*& peer_out) {
    if (rank < 0 || rank >= rank_count_) {
        return Status::invalid_argument("tp barrier rank out of range", __FILE__, __LINE__);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (aborted_) return abort_status_;
    if (transport_ == nullptr) {
        return Status::invalid_state("tp barrier group has no transport", __FILE__, __LINE__);
    }
    return transport_->peer_write_target(rank, bytes, peer_out);
}

}  // namespace shelf::runtime
