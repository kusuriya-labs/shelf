#include <shelf/runtime/tp/tp_transports.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>

#include "tp_transport_internal.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace ps::runtime {

namespace {

constexpr std::size_t kProbeMaxBytes = 1u << 20;
constexpr std::size_t kProbeSizes[] = {8192u, 131072u, 1048576u};
constexpr int kProbeRounds = 4;

__device__ inline float tp_bf16_to_f32(uint16_t v) {
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

__device__ inline uint16_t tp_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        if ((u & 0x007FFFFFu) != 0u) return 0x7FC0u;
        return static_cast<uint16_t>(u >> 16);
    }
    const uint32_t rounding_bias = 0x7FFFu + ((u >> 16) & 1u);
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

__global__ void tp_add_peers_bf16_kernel(const uint16_t* own,
                                         const uint16_t* const* peers,
                                         uint32_t npeers, uint16_t* out, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        float acc = tp_bf16_to_f32(own[i]);
        for (uint32_t p = 0; p < npeers; ++p) {
            acc += tp_bf16_to_f32(peers[p][i]);
        }
        out[i] = tp_f32_to_bf16(acc);
    }
}

__global__ void tp_add_peers_f32_kernel(const float* own, const float* const* peers,
                                        uint32_t npeers, float* out, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        float acc = own[i];
        for (uint32_t p = 0; p < npeers; ++p) {
            acc += peers[p][i];
        }
        out[i] = acc;
    }
}

__global__ void tp_copy_bf16_kernel(uint16_t* out, const uint16_t* in, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = in[i];
    }
}

__global__ void tp_copy_f32_kernel(float* out, const float* in, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = in[i];
    }
}

uint32_t tp_grid_for(uint64_t count) {
    const uint64_t blocks = (count + 255u) / 256u;
    if (blocks == 0) return 1;
    return static_cast<uint32_t>(blocks > 1024u ? 1024u : blocks);
}

Status launch_copy(void* dst, const void* src, std::size_t bytes, ValueDType dtype,
                   hipStream_t stream) {
    if (dtype == ValueDType::F32) {
        const uint64_t count = bytes / sizeof(float);
        tp_copy_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<float*>(dst), static_cast<const float*>(src), count);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_copy_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), count);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("tp copy kernel launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status launch_add(void* out, const void* own, const void* peers_device, uint32_t npeers,
                  std::size_t bytes, ValueDType dtype, hipStream_t stream) {
    if (dtype == ValueDType::F32) {
        const uint64_t count = bytes / sizeof(float);
        tp_add_peers_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const float*>(own),
            reinterpret_cast<const float* const*>(peers_device), npeers,
            static_cast<float*>(out), count);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_add_peers_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const uint16_t*>(own),
            reinterpret_cast<const uint16_t* const*>(peers_device), npeers,
            static_cast<uint16_t*>(out), count);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("tp add kernel launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

}

HipPeerTpTransport::HipPeerTpTransport(std::vector<int> devices)
    : devices_(std::move(devices)) {}

HipPeerTpTransport::~HipPeerTpTransport() {
    const std::size_t rank_count = devices_.size();
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) continue;
        if (r < inbox_.size()) {
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r) continue;
                for (void*& slot : inbox_[r][q]) {
                    if (slot != nullptr) {
                        ps::gpu::discard_cleanup_result(hipFree(slot));
                        slot = nullptr;
                    }
                }
            }
        }
        if (r < inbox_ptrs_dev_.size()) {
            for (void*& ptrs : inbox_ptrs_dev_[r]) {
                if (ptrs != nullptr) {
                    ps::gpu::discard_cleanup_result(hipFree(ptrs));
                    ptrs = nullptr;
                }
            }
        }
        if (r < sent_.size()) {
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r) continue;
                if (sent_[r][q] != nullptr) {
                    ps::gpu::discard_cleanup_result(hipEventDestroy(sent_[r][q]));
                    sent_[r][q] = nullptr;
                }
            }
        }
        if (r < probe_buffers_.size() && probe_buffers_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipFree(probe_buffers_[r]));
            probe_buffers_[r] = nullptr;
        }
        if (r < probe_streams_.size() && probe_streams_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipStreamDestroy(probe_streams_[r]));
            probe_streams_[r] = nullptr;
        }
        if (r < probe_ready_.size() && probe_ready_[r] != nullptr) {
            ps::gpu::discard_cleanup_result(hipEventDestroy(probe_ready_[r]));
            probe_ready_[r] = nullptr;
        }
    }
}

Result<std::unique_ptr<HipPeerTpTransport>> HipPeerTpTransport::create(
    const std::vector<int>& devices) {
    if (devices.size() < 2) {
        return Status::invalid_argument("hip peer transport requires at least 2 devices",
                                        __FILE__, __LINE__);
    }
    auto transport = std::unique_ptr<HipPeerTpTransport>(
        new HipPeerTpTransport(devices));
    Status enable_status = transport->enable_peer_access();
    if (!enable_status.ok()) return enable_status;

    const std::size_t rank_count = devices.size();
    transport->sent_.assign(rank_count, std::vector<hipEvent_t>(rank_count, nullptr));
    transport->inbox_.assign(rank_count, {});
    transport->inbox_ptrs_dev_.assign(rank_count, {});
    for (std::size_t r = 0; r < rank_count; ++r) {
        transport->inbox_[r].assign(rank_count, {nullptr, nullptr});
        transport->inbox_ptrs_dev_[r] = {nullptr, nullptr};
    }
    transport->probe_buffers_.assign(rank_count, nullptr);
    transport->probe_streams_.assign(rank_count, nullptr);
    transport->probe_ready_.assign(rank_count, nullptr);

    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices[r]);
        if (!scope.ok()) return scope.status();
        for (std::size_t q = 0; q < rank_count; ++q) {
            if (q == r) continue;
            if (hipEventCreateWithFlags(&transport->sent_[r][q], hipEventDisableTiming) !=
                hipSuccess) {
                return Status::hip_error("hipEventCreate tp send",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        if (hipMalloc(&transport->probe_buffers_[r], kProbeMaxBytes) != hipSuccess) {
            return Status::hip_error("hipMalloc tp probe", hipGetErrorString(hipGetLastError()),
                                     __FILE__, __LINE__);
        }
        if (hipStreamCreate(&transport->probe_streams_[r]) != hipSuccess) {
            return Status::hip_error("hipStreamCreate tp probe",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        if (hipEventCreateWithFlags(&transport->probe_ready_[r],
                                    hipEventDisableTiming) != hipSuccess) {
            return Status::hip_error("hipEventCreate tp probe",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
    }
    return transport;
}

Status HipPeerTpTransport::enable_peer_access() {
    if (peer_access_enabled_) return Status::make_ok();
    for (std::size_t i = 0; i < devices_.size(); ++i) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[i]);
        if (!scope.ok()) return scope.status();
        for (std::size_t j = 0; j < devices_.size(); ++j) {
            if (i == j) continue;
            const hipError_t e = hipDeviceEnablePeerAccess(devices_[j], 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                return Status::hip_error("hipDeviceEnablePeerAccess", hipGetErrorString(e),
                                         __FILE__, __LINE__);
            }
            ps::gpu::discard_cleanup_result(hipGetLastError());
        }
    }
    peer_access_enabled_ = true;
    return Status::make_ok();
}

Status HipPeerTpTransport::ensure_buffers(std::size_t bytes, std::size_t rank_count) {
    if (inbox_bytes_ >= bytes && inbox_.size() == rank_count) return Status::make_ok();
    const std::size_t peer_count = rank_count - 1;
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        for (std::size_t q = 0; q < rank_count; ++q) {
            if (q == r) continue;
            for (void*& slot : inbox_[r][q]) {
                if (slot != nullptr) {
                    ps::gpu::discard_cleanup_result(hipFree(slot));
                    slot = nullptr;
                }
                if (hipMalloc(&slot, bytes) != hipSuccess) {
                    return Status::hip_error("hipMalloc tp inbox",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
            }
        }
        for (void*& ptrs : inbox_ptrs_dev_[r]) {
            if (ptrs != nullptr) {
                ps::gpu::discard_cleanup_result(hipFree(ptrs));
                ptrs = nullptr;
            }
            if (hipMalloc(&ptrs, peer_count * sizeof(void*)) != hipSuccess) {
                return Status::hip_error("hipMalloc tp inbox pointers",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        std::vector<void*> host_ptrs(peer_count, nullptr);
        for (std::size_t parity = 0; parity < 2; ++parity) {
            std::size_t index = 0;
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r) continue;
                host_ptrs[index++] = inbox_[r][q][parity];
            }
            if (hipMemcpy(inbox_ptrs_dev_[r][parity], host_ptrs.data(),
                          peer_count * sizeof(void*), hipMemcpyHostToDevice) != hipSuccess) {
                return Status::hip_error("hipMemcpy tp inbox pointers",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
    }
    inbox_bytes_ = bytes;
    return Status::make_ok();
}

Status HipPeerTpTransport::enqueue_sum(const TpSumInvocation& invocation) {
    const std::size_t rank_count = devices_.size();
    const ValueDType dtype = invocation.targets[0].dtype;
    std::size_t bytes = 0;
    Status st = validate_tp_sum_invocation(invocation, rank_count, bytes);
    if (!st.ok()) return st;

    std::lock_guard<std::mutex> lock(mutex_);
    st = ensure_buffers(bytes, rank_count);
    if (!st.ok()) return st;
    const std::size_t parity = parity_;

    if (!invocation.peer_write_complete) {
        for (std::size_t r = 0; r < rank_count; ++r) {
            auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
            if (!scope.ok()) return scope.status();
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r) continue;
                st = launch_copy(inbox_[q][r][parity], invocation.targets[r].ptr, bytes,
                                 dtype, invocation.streams[r]);
                if (!st.ok()) return st;
                if (hipEventRecord(sent_[r][q], invocation.streams[r]) != hipSuccess) {
                    return Status::hip_error("hipEventRecord tp send",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
            }
        }
    } else {
        for (std::size_t r = 0; r < rank_count; ++r) {
            auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
            if (!scope.ok()) return scope.status();
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r) continue;
                if (hipEventRecord(sent_[r][q], invocation.streams[r]) != hipSuccess) {
                    return Status::hip_error("hipEventRecord tp send",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
            }
        }
    }
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        for (std::size_t q = 0; q < rank_count; ++q) {
            if (q == r) continue;
            if (hipStreamWaitEvent(invocation.streams[r], sent_[q][r], 0) != hipSuccess) {
                return Status::hip_error("hipStreamWaitEvent tp receive",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        st = launch_add(invocation.targets[r].ptr, invocation.targets[r].ptr,
                        inbox_ptrs_dev_[r][parity], rank_count - 1, bytes, dtype,
                        invocation.streams[r]);
        if (!st.ok()) return st;
    }
    parity_ ^= 1u;
    return Status::make_ok();
}

Status HipPeerTpTransport::peer_write_target(int rank, std::size_t bytes, void*& peer_out) {
    peer_out = nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t rank_count = devices_.size();
    if (rank_count != 2) return Status::make_ok();
    if (rank < 0 || static_cast<std::size_t>(rank) >= rank_count) {
        return Status::invalid_argument("tp peer write rank out of range", __FILE__,
                                        __LINE__);
    }
    if (bytes == 0) {
        return Status::invalid_argument("tp peer write target is empty", __FILE__, __LINE__);
    }
    Status st = ensure_buffers(bytes, rank_count);
    if (!st.ok()) return st;
    const std::size_t peer = static_cast<std::size_t>(rank) == 0 ? 1 : 0;
    peer_out = inbox_[peer][static_cast<std::size_t>(rank)][parity_];
    return Status::make_ok();
}

Status HipPeerTpTransport::verify() {
    if (verified_) return Status::make_ok();
    Status st = enable_peer_access();
    if (!st.ok()) return st;

    const std::size_t rank_count = devices_.size();
    st = ensure_buffers(kProbeMaxBytes, rank_count);
    if (!st.ok()) return st;

    for (std::size_t probe_size : kProbeSizes) {
        const std::size_t count = probe_size / sizeof(std::uint16_t);
        std::vector<std::vector<std::uint16_t>> host_in(rank_count,
                                                        std::vector<std::uint16_t>(count));
        std::vector<std::uint16_t> want(count);
        for (std::size_t r = 0; r < rank_count; ++r) {
            for (std::size_t i = 0; i < count; ++i) {
                host_in[r][i] = static_cast<std::uint16_t>(
                    0x4000u + ((static_cast<std::uint32_t>(r) * 7u + i) % 512u));
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            float acc = 0.0f;
            for (std::size_t r = 0; r < rank_count; ++r) {
                acc += tp_bf16_bits_to_f32(host_in[r][i]);
            }
            want[i] = ps::f32_to_bf16_rne(acc);
        }

        std::vector<TpSumTarget> targets(rank_count);
        std::vector<hipStream_t> streams(rank_count);
        std::vector<hipEvent_t> ready(rank_count);
        for (std::size_t r = 0; r < rank_count; ++r) {
            targets[r].ptr = probe_buffers_[r];
            targets[r].rows = 1;
            targets[r].row_stride = static_cast<std::uint32_t>(probe_size / 2u);
            targets[r].feature_count = static_cast<std::uint32_t>(probe_size / 2u);
            targets[r].dtype = ValueDType::BF16;
            streams[r] = probe_streams_[r];
            ready[r] = probe_ready_[r];
        }
        TpSumInvocation invocation{targets, streams, ready};

        for (int round = 0; round < kProbeRounds; ++round) {
            for (std::size_t r = 0; r < rank_count; ++r) {
                auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
                if (!scope.ok()) return scope.status();
                if (hipMemcpy(probe_buffers_[r], host_in[r].data(), probe_size,
                              hipMemcpyHostToDevice) != hipSuccess) {
                    return Status::hip_error("tp probe upload",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
                if (hipEventRecord(probe_ready_[r], probe_streams_[r]) != hipSuccess) {
                    return Status::hip_error("tp probe record",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
            }
            st = enqueue_sum(invocation);
            if (!st.ok()) return st;
            for (std::size_t r = 0; r < rank_count; ++r) {
                auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
                if (!scope.ok()) return scope.status();
                if (hipStreamSynchronize(probe_streams_[r]) != hipSuccess) {
                    return Status::hip_error("tp probe sync",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
            }
            std::vector<std::uint16_t> got(count);
            bool round_ok = true;
            for (std::size_t r = 0; r < rank_count && round_ok; ++r) {
                auto scope = ps::gpu::ScopedDevice::create(devices_[r]);
                if (!scope.ok()) return scope.status();
                if (hipMemcpy(got.data(), probe_buffers_[r], probe_size,
                              hipMemcpyDeviceToHost) != hipSuccess) {
                    return Status::hip_error("tp probe download",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
                for (std::size_t i = 0; i < count; ++i) {
                    if (got[i] != want[i]) {
                        round_ok = false;
                        break;
                    }
                }
            }
            if (!round_ok) {
                return Status::unsupported(
                    "hip peer transport self-test failed (peer writes are unreliable)",
                    __FILE__, __LINE__);
            }
        }
    }
    verified_ = true;
    return Status::make_ok();
}

Status HipPeerTpTransport::sum_hidden(const TpSumInvocation& invocation) {
    std::size_t bytes = 0;
    Status st = validate_tp_sum_invocation(invocation, devices_.size(), bytes);
    if (!st.ok()) return st;
    if (!verified_) {
        st = verify();
        if (!st.ok()) return st;
    }
    st = ensure_buffers(bytes, devices_.size());
    if (!st.ok()) return st;
    return enqueue_sum(invocation);
}

}  // namespace ps::runtime
