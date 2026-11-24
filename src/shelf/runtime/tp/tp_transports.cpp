#include <shelf/runtime/tp/tp_transports.h>
#include <shelf/runtime/tp/tp_argmax.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>

#include "tp_transport_internal.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace shelf::runtime {

bool tp_peer_link_map_all(const TpPeerLinkMap& map) noexcept {
    for (std::size_t i = 0u; i < map.size(); ++i) {
        for (std::size_t j = 0u; j < map[i].size(); ++j) {
            if (i != j && map[i][j] == 0u) return false;
        }
    }
    return true;
}

bool tp_peer_link_map_none(const TpPeerLinkMap& map) noexcept {
    for (std::size_t i = 0u; i < map.size(); ++i) {
        for (std::size_t j = 0u; j < map[i].size(); ++j) {
            if (i != j && map[i][j] != 0u) return false;
        }
    }
    return true;
}

std::size_t tp_peer_link_map_disabled(const TpPeerLinkMap& map) noexcept {
    std::size_t count = 0u;
    for (std::size_t i = 0u; i < map.size(); ++i) {
        for (std::size_t j = 0u; j < map[i].size(); ++j) {
            if (i != j && map[i][j] == 0u) ++count;
        }
    }
    return count;
}

Result<TpPeerLinkMap> probe_tp_peer_links(const std::vector<int>& devices) {
    const std::size_t rank_count = devices.size();
    TpPeerLinkMap links(rank_count, std::vector<std::uint8_t>(rank_count, 0u));
    if (rank_count < 2u) return links;

    std::vector<void*> src_buffers(rank_count, nullptr);
    std::vector<void*> dst_buffers(rank_count, nullptr);
    std::vector<hipStream_t> streams(rank_count, nullptr);
    Status init_status = Status::make_ok();
    for (std::size_t r = 0; r < rank_count; ++r) {
        links[r][r] = 1u;
        auto scope = shelf::gpu::ScopedDevice::create(devices[r]);
        if (!scope.ok()) {
            init_status = scope.status();
            break;
        }
        if (hipMalloc(&src_buffers[r], kTpPeerProbeBytes) != hipSuccess ||
            hipMalloc(&dst_buffers[r], kTpPeerProbeBytes) != hipSuccess ||
            hipStreamCreate(&streams[r]) != hipSuccess) {
            init_status = Status::hip_error("tp directional probe allocation",
                                            hipGetErrorString(hipGetLastError()), __FILE__,
                                            __LINE__);
            break;
        }
    }
    if (!init_status.ok()) {
        for (std::size_t r = 0; r < rank_count; ++r) {
            auto scope = shelf::gpu::ScopedDevice::create(devices[r]);
            if (!scope.ok()) continue;
            if (src_buffers[r] != nullptr) shelf::gpu::discard_cleanup_result(hipFree(src_buffers[r]));
            if (dst_buffers[r] != nullptr) shelf::gpu::discard_cleanup_result(hipFree(dst_buffers[r]));
            if (streams[r] != nullptr)
                shelf::gpu::discard_cleanup_result(hipStreamDestroy(streams[r]));
        }
        return init_status;
    }

    constexpr std::size_t words = kTpPeerProbeBytes / sizeof(std::uint32_t);
    std::vector<std::uint32_t> expected(words);
    std::vector<std::uint32_t> actual(words);
    for (std::size_t src = 0u; src < rank_count; ++src) {
        for (std::size_t dst = 0u; dst < rank_count; ++dst) {
            if (src == dst) continue;
            for (std::size_t i = 0u; i < words; ++i) {
                expected[i] = static_cast<std::uint32_t>(i + 1u) * 0x9e3779b9u ^
                              static_cast<std::uint32_t>(src * 17u + dst * 31u + 1u);
            }
            bool ok = true;
            {
                auto scope = shelf::gpu::ScopedDevice::create(devices[src]);
                if (!scope.ok()) { ok = false; }
                else if (hipMemcpy(src_buffers[src], expected.data(), kTpPeerProbeBytes,
                                   hipMemcpyHostToDevice) != hipSuccess) {
                    ok = false;
                    (void)hipGetLastError();
                }
            }
            if (ok) {
                auto scope = shelf::gpu::ScopedDevice::create(devices[dst]);
                if (!scope.ok()) { ok = false; }
                else if (hipMemset(dst_buffers[dst], 0, kTpPeerProbeBytes) != hipSuccess) {
                    ok = false;
                    (void)hipGetLastError();
                }
            }
            if (ok) {
                int accessible = 0;
                const hipError_t can = hipDeviceCanAccessPeer(&accessible, devices[src],
                                                               devices[dst]);
                if (can != hipSuccess || accessible == 0) {
                    ok = false;
                } else {
                    auto scope = shelf::gpu::ScopedDevice::create(devices[src]);
                    if (!scope.ok()) { ok = false; }
                    else {
                        const hipError_t enable = hipDeviceEnablePeerAccess(devices[dst], 0);
                        if (enable != hipSuccess && enable != hipErrorPeerAccessAlreadyEnabled)
                            ok = false;
                        (void)hipGetLastError();
                    }
                }
            }
            if (ok) {
                auto scope = shelf::gpu::ScopedDevice::create(devices[src]);
                if (!scope.ok()) { ok = false; }
                else if (launch_tp_peer_probe_copy(dst_buffers[dst], src_buffers[src],
                                                    kTpPeerProbeBytes,
                                                    streams[src]) != hipSuccess ||
                         hipStreamSynchronize(streams[src]) != hipSuccess) {
                    ok = false;
                    (void)hipGetLastError();
                }
            }
            if (ok) {
                auto scope = shelf::gpu::ScopedDevice::create(devices[dst]);
                if (!scope.ok()) { ok = false; }
                else if (hipMemcpy(actual.data(), dst_buffers[dst], kTpPeerProbeBytes,
                                   hipMemcpyDeviceToHost) != hipSuccess) {
                    ok = false;
                    (void)hipGetLastError();
                } else {
                    ok = actual == expected;
                }
            }
            links[src][dst] = ok ? 1u : 0u;
        }
    }

    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices[r]);
        if (scope.ok()) {
            if (src_buffers[r] != nullptr) shelf::gpu::discard_cleanup_result(hipFree(src_buffers[r]));
            if (dst_buffers[r] != nullptr) shelf::gpu::discard_cleanup_result(hipFree(dst_buffers[r]));
            if (streams[r] != nullptr)
                shelf::gpu::discard_cleanup_result(hipStreamDestroy(streams[r]));
        }
    }
    return links;
}

HostMediatedTpTransport::HostMediatedTpTransport(std::vector<int> devices)
    : devices_(std::move(devices)) {}

HostMediatedTpTransport::~HostMediatedTpTransport() {
    for (void* buffer : staging_) {
        if (buffer != nullptr) shelf::gpu::discard_cleanup_result(hipHostFree(buffer));
    }
}

Status HostMediatedTpTransport::ensure_staging(std::size_t bytes, std::size_t count) {
    if (staging_bytes_ >= bytes && staging_.size() == devices_.size()) return Status::make_ok();
    for (void* buffer : staging_) {
        if (buffer != nullptr) shelf::gpu::discard_cleanup_result(hipHostFree(buffer));
    }
    staging_.assign(devices_.size(), nullptr);
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        void* buffer = nullptr;
        const hipError_t host_err =
            hipHostMalloc(&buffer, bytes, hipHostMallocDefault);
        if (host_err != hipSuccess) {
            return Status::hip_error("host transport staging alloc",
                                     hipGetErrorString(host_err), __FILE__, __LINE__);
        }
        staging_[r] = buffer;
    }
    accum_.assign(count, 0.0f);
    staging_bytes_ = bytes;
    return Status::make_ok();
}

Status HostMediatedTpTransport::reserve(std::size_t bytes) {
    return ensure_staging(bytes, bytes / sizeof(std::uint16_t));
}

Status HostMediatedTpTransport::sum_hidden(const TpSumInvocation& invocation) {
    std::size_t bytes = 0;
    Status st = validate_tp_sum_invocation(invocation, devices_.size(), bytes);
    if (!st.ok()) return st;
    if (staging_bytes_ < bytes) {
        return Status::unsupported("host transport staging is not reserved", __FILE__,
                                   __LINE__);
    }

    const ValueDType dtype = invocation.targets[0].dtype;
    const std::size_t elem = value_dtype_bytes(dtype);
    const std::size_t count = bytes / elem;

    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto device = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!device.ok()) return device.status();
        const hipError_t copy = hipMemcpyAsync(staging_[r], invocation.targets[r].ptr,
                                               bytes, hipMemcpyDeviceToHost,
                                               invocation.streams[r]);
        if (copy != hipSuccess) {
            return Status::hip_error("host transport D2H", hipGetErrorString(copy),
                                     __FILE__, __LINE__);
        }
    }
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto device = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!device.ok()) return device.status();
        if (hipStreamSynchronize(invocation.streams[r]) != hipSuccess) {
            return Status::hip_error("host transport D2H sync",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }

    std::vector<const void*> srcs(devices_.size());
    for (std::size_t r = 0; r < devices_.size(); ++r) srcs[r] = staging_[r];
    tp_reduce_f32_accumulate(srcs, count, dtype, staging_[0]);

    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto device = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!device.ok()) return device.status();
        const hipError_t copy = hipMemcpyAsync(invocation.targets[r].ptr, staging_[0],
                                               bytes, hipMemcpyHostToDevice,
                                               invocation.streams[r]);
        if (copy != hipSuccess) {
            return Status::hip_error("host transport H2D", hipGetErrorString(copy),
                                     __FILE__, __LINE__);
        }
    }
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto device = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!device.ok()) return device.status();
        if (hipStreamSynchronize(invocation.streams[r]) != hipSuccess) {
            return Status::hip_error("host transport H2D sync",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    return Status::make_ok();
}

Result<TpTransportSelection> create_tp_transport(const std::vector<int>& devices,
                                                 TpCombineBackend backend) {
    if (devices.size() < 2) {
        return Status::invalid_argument("tp transport requires at least 2 devices",
                                        __FILE__, __LINE__);
    }
    const char* forced = std::getenv("SHELF_TP_TRANSPORT");
    const std::string force = forced != nullptr ? forced : "";
    if (force == "host") {
        TpTransportSelection selection;
        selection.peer_access_available = false;
        selection.transport = std::make_unique<HostMediatedTpTransport>(devices);
        return selection;
    }
    const char* probe_env = std::getenv("SHELF_TP_P2P_PROBE");
    const bool probing = probe_env == nullptr || std::strtoul(probe_env, nullptr, 10) != 0ul;
    if (!probing) return create_tp_transport(devices, backend, {});
    auto links = probe_tp_peer_links(devices);
    if (!links.ok()) return links.status();
    if (force == "p2p") {
        for (auto& row : links.value()) {
            for (std::uint8_t& usable : row) usable = 1u;
        }
    }
    if (tp_peer_link_map_none(links.value())) {
        TpTransportSelection selection;
        selection.peer_access_available = false;
        selection.transport = std::make_unique<HostMediatedTpTransport>(devices);
        return selection;
    }
    return create_tp_transport(devices, backend, links.value());
}

Result<TpTransportSelection> create_tp_transport(const std::vector<int>& devices,
                                                  TpCombineBackend backend,
                                                  const TpPeerLinkMap& links) {
    if (devices.size() < 2) {
        return Status::invalid_argument("tp transport requires at least 2 devices",
                                         __FILE__, __LINE__);
    }
    TpPeerLinkMap effective = links;
    if (effective.empty()) {
        effective.assign(devices.size(), std::vector<std::uint8_t>(devices.size(), 1u));
        for (std::size_t r = 0; r < devices.size(); ++r) effective[r][r] = 0u;
    }
    if (effective.size() != devices.size()) {
        return Status::invalid_argument("tp peer link map rank count mismatch", __FILE__,
                                         __LINE__);
    }
    auto peer = HipPeerTpTransport::create(devices, backend, effective);
    if (!peer.ok()) {
        TpTransportSelection selection;
        selection.peer_access_available = false;
        selection.transport = std::make_unique<HostMediatedTpTransport>(devices);
        return selection;
    }
    TpTransportSelection selection;
    selection.peer_access_available = !tp_peer_link_map_none(effective);
    selection.transport = peer.release();
    return selection;
}

}  // namespace shelf::runtime
