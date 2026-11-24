#pragma once

#include <shelf/core/gpu/debug_ring.h>
#include <shelf/runtime/tp/tp_execution.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace shelf::runtime {

class HipPeerTpTransport final : public TpTransport {
public:
    static Result<std::unique_ptr<HipPeerTpTransport>> create(
        const std::vector<int>& devices);
    ~HipPeerTpTransport() override;

    const char* name() const noexcept override { return "hip-p2p"; }
    Status verify() override;
    Status sum_hidden(const TpSumInvocation& invocation) override;
    bool supports_combine_fusion() const noexcept override;
    bool fused_barrier_ready() const noexcept override;
    std::size_t inbox_capacity() const noexcept override;
    void advance_barrier(int rank) override;
    Status reserve(std::size_t bytes) override;
    Status sum_hidden_fused(int rank, const TpSumTarget& target, hipStream_t stream,
                            bool peer_written) override;
    Status peer_write_target(int rank, std::size_t bytes, void*& peer_out) override;

private:
    explicit HipPeerTpTransport(std::vector<int> devices);

    Status enable_peer_access();
    Status ensure_buffers(std::size_t bytes, std::size_t rank_count);
    Status ensure_fused_state();
    Status enqueue_sum(const TpSumInvocation& invocation);
    shelf::gpu::DebugRingDevice debug_ring_view(std::size_t rank) const noexcept;
    void report_debug_rings() noexcept;

    std::vector<int> devices_;
    std::vector<std::vector<std::array<hipEvent_t, 2>>> sent_;
    std::vector<std::vector<std::array<hipEvent_t, 2>>> consumed_;
    std::vector<std::vector<std::array<char, 2>>> consumed_recorded_;
    std::vector<std::vector<std::array<void*, 2>>> inbox_;
    std::vector<std::array<void*, 2>> inbox_ptrs_dev_;
    std::atomic<std::size_t> inbox_bytes_{0};
    std::vector<std::uint64_t> barrier_seq_;
    std::vector<std::pair<int, void*>> retired_inbox_;
    int* fused_flags_ = nullptr;
    std::array<int*, 2> fused_arrive_{{nullptr, nullptr}};
    std::array<int, 2> fused_epoch_{{0, 0}};
    bool fused_allocated_ = false;
    std::vector<void*> probe_buffers_;
    std::vector<hipStream_t> probe_streams_;
    std::vector<hipEvent_t> probe_ready_;
    bool peer_access_enabled_ = false;
    bool verified_ = false;
    std::vector<std::unique_ptr<shelf::gpu::DebugRing>> debug_rings_;
    std::mutex mutex_;
};

class HostMediatedTpTransport final : public TpTransport {
public:
    explicit HostMediatedTpTransport(std::vector<int> devices);
    ~HostMediatedTpTransport() override;

    const char* name() const noexcept override { return "host-mediated"; }
    Status sum_hidden(const TpSumInvocation& invocation) override;
    Status reserve(std::size_t bytes) override;

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

}  // namespace shelf::runtime
