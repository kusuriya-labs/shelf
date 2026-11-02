#pragma once

#include <phaseshift/runtime/tp/tp_execution.h>

#include <array>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace ps::runtime {

class HipPeerTpTransport final : public TpTransport {
public:
    static Result<std::unique_ptr<HipPeerTpTransport>> create(
        const std::vector<int>& devices);
    ~HipPeerTpTransport() override;

    const char* name() const noexcept override { return "hip-p2p"; }
    Status verify() override;
    Status sum_hidden(const TpSumInvocation& invocation) override;
    Status peer_write_target(int rank, std::size_t bytes, void*& peer_out) override;

private:
    explicit HipPeerTpTransport(std::vector<int> devices);

    Status enable_peer_access();
    Status ensure_buffers(std::size_t bytes, std::size_t rank_count);
    Status enqueue_sum(const TpSumInvocation& invocation);

    std::vector<int> devices_;
    std::vector<std::vector<hipEvent_t>> sent_;
    std::vector<std::vector<std::array<void*, 2>>> inbox_;
    std::vector<std::array<void*, 2>> inbox_ptrs_dev_;
    std::size_t inbox_bytes_ = 0;
    std::size_t parity_ = 0;
    std::vector<void*> probe_buffers_;
    std::vector<hipStream_t> probe_streams_;
    std::vector<hipEvent_t> probe_ready_;
    bool peer_access_enabled_ = false;
    bool verified_ = false;
    std::mutex mutex_;
};

class HostMediatedTpTransport final : public TpTransport {
public:
    explicit HostMediatedTpTransport(std::vector<int> devices);
    ~HostMediatedTpTransport() override;

    const char* name() const noexcept override { return "host-mediated"; }
    Status sum_hidden(const TpSumInvocation& invocation) override;

private:
    Status ensure_staging(std::size_t bytes, std::size_t count);

    std::vector<int> devices_;
    std::vector<void*> staging_;
    std::vector<float> accum_;
    std::size_t staging_bytes_ = 0;
};

struct TpTransportSelection {
    std::unique_ptr<TpTransport> transport;
    bool peer_access_available = false;
};

Result<TpTransportSelection> create_tp_transport(const std::vector<int>& devices);

}  // namespace ps::runtime
