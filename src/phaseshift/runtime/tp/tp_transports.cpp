#include <phaseshift/runtime/tp/tp_transports.h>

#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/gpu/scoped_device.h>
#include <phaseshift/core/memory/types.h>

#include "tp_transport_internal.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ps::runtime {

HostMediatedTpTransport::HostMediatedTpTransport(std::vector<int> devices)
    : devices_(std::move(devices)) {}

HostMediatedTpTransport::~HostMediatedTpTransport() {
    for (void* buffer : staging_) {
        if (buffer != nullptr) ps::gpu::discard_cleanup_result(hipHostFree(buffer));
    }
}

Status HostMediatedTpTransport::ensure_staging(std::size_t bytes, std::size_t count) {
    if (staging_bytes_ >= bytes && staging_.size() == devices_.size()) return Status::make_ok();
    for (void* buffer : staging_) {
        if (buffer != nullptr) ps::gpu::discard_cleanup_result(hipHostFree(buffer));
    }
    staging_.assign(devices_.size(), nullptr);
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        void* buffer = nullptr;
        if (hipHostMalloc(&buffer, bytes, hipHostMallocDefault) != hipSuccess) {
            ps::gpu::discard_cleanup_result(hipGetLastError());
            if (hipMallocHost(&buffer, bytes) != hipSuccess) {
                return Status::hip_error("host transport staging alloc",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        staging_[r] = buffer;
    }
    accum_.assign(count, 0.0f);
    staging_bytes_ = bytes;
    return Status::make_ok();
}

Status HostMediatedTpTransport::sum_hidden(const TpSumInvocation& invocation) {
    std::size_t bytes = 0;
    Status st = validate_tp_sum_invocation(invocation, devices_.size(), bytes);
    if (!st.ok()) return st;

    const ValueDType dtype = invocation.targets[0].dtype;
    const std::size_t elem = value_dtype_bytes(dtype);
    const std::size_t count = bytes / elem;
    st = ensure_staging(bytes, count);
    if (!st.ok()) return st;

    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto device = ps::gpu::ScopedDevice::create(devices_[r]);
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
        auto device = ps::gpu::ScopedDevice::create(devices_[r]);
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
        auto device = ps::gpu::ScopedDevice::create(devices_[r]);
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
        auto device = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!device.ok()) return device.status();
        if (hipStreamSynchronize(invocation.streams[r]) != hipSuccess) {
            return Status::hip_error("host transport H2D sync",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    return Status::make_ok();
}

Result<TpTransportSelection> create_tp_transport(const std::vector<int>& devices) {
    if (devices.size() < 2) {
        return Status::invalid_argument("tp transport requires at least 2 devices",
                                        __FILE__, __LINE__);
    }
    const char* forced = std::getenv("PHASESHIFT_TP_TRANSPORT");
    const std::string force = forced != nullptr ? forced : "";
    if (force == "host") {
        TpTransportSelection selection;
        selection.peer_access_available = false;
        selection.transport = std::make_unique<HostMediatedTpTransport>(devices);
        return selection;
    }
    for (std::size_t i = 0; i < devices.size(); ++i) {
        for (std::size_t j = 0; j < devices.size(); ++j) {
            if (i == j) continue;
            int can_peer = 0;
            const hipError_t e = hipDeviceCanAccessPeer(&can_peer, devices[i], devices[j]);
            if (e != hipSuccess) {
                return Status::hip_error("hipDeviceCanAccessPeer", hipGetErrorString(e),
                                         __FILE__, __LINE__);
            }
            if (can_peer == 0 && force != "p2p") {
                TpTransportSelection selection;
                selection.peer_access_available = false;
                selection.transport = std::make_unique<HostMediatedTpTransport>(devices);
                return selection;
            }
        }
    }
    auto peer = HipPeerTpTransport::create(devices);
    if (!peer.ok()) return peer.status();
    TpTransportSelection selection;
    selection.peer_access_available = true;
    selection.transport = peer.release();
    return selection;
}

}  // namespace ps::runtime
