#pragma once

#include <shelf/core/gpu/debug_ring.h>
#include <shelf/runtime/tp/tp_argmax.h>
#include <shelf/runtime/tp/tp_execution.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace shelf::runtime {

using TpPeerLinkMap = std::vector<std::vector<uint8_t>>;

bool tp_peer_link_map_all(const TpPeerLinkMap& map) noexcept;
bool tp_peer_link_map_none(const TpPeerLinkMap& map) noexcept;
std::size_t tp_peer_link_map_disabled(const TpPeerLinkMap& map) noexcept;
Result<TpPeerLinkMap> probe_tp_peer_links(const std::vector<int>& devices);

class HipPeerTpTransport final : public TpTransport {
public:
    static Result<std::unique_ptr<HipPeerTpTransport>> create(
        const std::vector<int>& devices, TpCombineBackend backend);
    static Result<std::unique_ptr<HipPeerTpTransport>> create(
        const std::vector<int>& devices, TpCombineBackend backend,
        const TpPeerLinkMap& links);
    ~HipPeerTpTransport() override;

    const char* name() const noexcept override { return "hip-p2p"; }
    Status verify() override;
    Status sum_hidden(const TpSumInvocation& invocation) override;
    bool supports_combine_fusion() const noexcept override;
    std::size_t inbox_capacity() const noexcept override;
    void advance_barrier(int rank) override;
    bool stream_signal_ready() const noexcept override;
    Status sum_hidden_stream_signal(int rank, const TpSumTarget& target,
                                    hipStream_t stream, bool peer_written) override;
    Status reserve(std::size_t bytes) override;
    Status peer_write_target(int rank, std::size_t bytes, void*& peer_out) override;
    bool has_peer_links() const noexcept { return !tp_peer_link_map_none(peer_link_); }
    bool supports_argmax_reduce() const noexcept override;
    Status begin_argmax(int rank, const TpArgmaxInvocation& invocation) override;
    Status finish_argmax(int rank, const TpArgmaxInvocation& invocation) override;

private:
    HipPeerTpTransport(std::vector<int> devices, TpCombineBackend backend);

    Status enable_peer_access();
    Status ensure_link_staging(std::size_t bytes, std::size_t rank_count);
    Status ensure_buffers(std::size_t bytes, std::size_t rank_count);
    Status ensure_stream_signal_state();
    Status ensure_lm_buffers(std::size_t rows, std::size_t rank_count);
    Status enqueue_sum(const TpSumInvocation& invocation);
    shelf::gpu::DebugRingDevice debug_ring_view(std::size_t rank) const noexcept;
    void report_debug_rings() noexcept;

    std::vector<int> devices_;
    TpCombineBackend backend_ = TpCombineBackend::Event;
    std::vector<std::vector<std::array<hipEvent_t, 2>>> sent_;
    std::vector<std::vector<std::array<hipEvent_t, 2>>> consumed_;
    std::vector<std::vector<std::array<char, 2>>> consumed_recorded_;
    std::vector<std::vector<std::array<void*, 2>>> inbox_;
    std::vector<std::array<void*, 2>> inbox_ptrs_dev_;
    std::atomic<std::size_t> inbox_bytes_{0};
    std::vector<std::uint64_t> barrier_seq_;
    std::vector<std::pair<int, void*>> retired_inbox_;
    void* stream_signal_host_ = nullptr;
    std::array<uint32_t*, 2> stream_signal_dev_{{nullptr, nullptr}};
    std::array<uint32_t*, 2> stream_arrive_{{nullptr, nullptr}};
    bool stream_signal_allocated_ = false;
    std::vector<void*> probe_buffers_;
    std::vector<hipStream_t> probe_streams_;
    std::vector<hipEvent_t> probe_ready_;
    TpPeerLinkMap peer_link_;
    std::vector<void*> link_staging_;
    std::size_t link_staging_bytes_ = 0;
    std::vector<void*> lm_recv_;
    std::vector<void*> lm_host_;
    std::vector<hipEvent_t> lm_sent_;
    std::vector<hipEvent_t> lm_done_;
    std::size_t lm_rows_ = 0;
    bool lm_published_ = false;
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

Result<TpTransportSelection> create_tp_transport(
    const std::vector<int>& devices,
    TpCombineBackend backend = TpCombineBackend::Event);
Result<TpTransportSelection> create_tp_transport(
    const std::vector<int>& devices,
    TpCombineBackend backend,
    const TpPeerLinkMap& links);

}  // namespace shelf::runtime
