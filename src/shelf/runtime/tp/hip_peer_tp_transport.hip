#include <shelf/runtime/tp/tp_transports.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/fused/tp_combine.h>

#include "tp_transport_internal.h"

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <limits>
#include <cstring>
#include <vector>

namespace shelf::runtime {

namespace {

constexpr std::size_t kProbeMaxBytes = 1u << 20;
constexpr std::size_t kProbeSizes[] = {8192u, 131072u, 1048576u};
constexpr int kProbeRounds = 4;

bool peer_flag_enabled() {
    const char* value = std::getenv("SHELF_TP_PEER_FLAG");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

__global__ void tp_flag_send(const unsigned char* src, unsigned char* dst,
                             unsigned long long* flag, unsigned long long tag,
                             std::size_t bytes) {
    for (std::size_t i = threadIdx.x; i < bytes; i += blockDim.x) dst[i] = src[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
        __hip_atomic_store(flag, tag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void tp_flag_receive(void* own, const void* inbox,
                                const unsigned long long* flag, unsigned long long tag,
                                unsigned int* error, std::size_t count, bool bf16,
                                unsigned long long limit) {
    __shared__ int ready;
    if (threadIdx.x == 0) {
        const unsigned long long start = clock64();
        ready = 0;
        while (clock64() - start < limit) {
            if (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == tag) {
                ready = 1;
                break;
            }
        }
        if (!ready) *error = 1;
    }
    __syncthreads();
    if (!ready) return;
    for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (bf16) {
            auto* a = static_cast<unsigned short*>(own);
            const auto* b = static_cast<const unsigned short*>(inbox);
            float x = __uint_as_float(static_cast<unsigned int>(a[i]) << 16);
            float y = __uint_as_float(static_cast<unsigned int>(b[i]) << 16);
            unsigned int v = __float_as_uint(x + y);
            if ((v & 0x7f800000u) == 0x7f800000u && (v & 0x007fffffu)) v = 0x7fc00000u;
            else v += 0x7fffu + ((v >> 16) & 1u);
            a[i] = static_cast<unsigned short>(v >> 16);
        } else {
            static_cast<float*>(own)[i] += static_cast<const float*>(inbox)[i];
        }
    }
}

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
    const uint64_t vectors = count >> 3u;
    for (uint64_t v = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         v < vectors; v += stride) {
        const uint4 own4 = reinterpret_cast<const uint4*>(own)[v];
        float acc[8];
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t word = (j & 4u) == 0u
                ? (j & 2u) == 0u ? own4.x : own4.y
                : (j & 2u) == 0u ? own4.z : own4.w;
            acc[j] = tp_bf16_to_f32(static_cast<uint16_t>(
                (word >> (16u * (j & 1u))) & 0xFFFFu));
        }
        for (uint32_t p = 0; p < npeers; ++p) {
            const uint4 peer4 = reinterpret_cast<const uint4*>(peers[p])[v];
            #pragma unroll
            for (uint32_t j = 0; j < 8u; ++j) {
                const uint32_t word = (j & 4u) == 0u
                    ? (j & 2u) == 0u ? peer4.x : peer4.y
                    : (j & 2u) == 0u ? peer4.z : peer4.w;
                acc[j] += tp_bf16_to_f32(static_cast<uint16_t>(
                    (word >> (16u * (j & 1u))) & 0xFFFFu));
            }
        }
        uint4 result;
        uint32_t* words = reinterpret_cast<uint32_t*>(&result);
        #pragma unroll
        for (uint32_t w = 0; w < 4u; ++w) {
            words[w] = static_cast<uint32_t>(tp_f32_to_bf16(acc[2u * w])) |
                       (static_cast<uint32_t>(tp_f32_to_bf16(acc[2u * w + 1u])) << 16u);
        }
        reinterpret_cast<uint4*>(out)[v] = result;
    }
    const uint64_t tail = vectors << 3u;
    for (uint64_t i = tail + static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        float value = tp_bf16_to_f32(own[i]);
        for (uint32_t p = 0; p < npeers; ++p)
            value += tp_bf16_to_f32(peers[p][i]);
        out[i] = tp_f32_to_bf16(value);
    }
}

__global__ void tp_add_peers_f32_kernel(const float* own, const float* const* peers,
                                        uint32_t npeers, float* out, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t vectors = count >> 2u;
    for (uint64_t v = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         v < vectors; v += stride) {
        const uint4 own4 = reinterpret_cast<const uint4*>(own)[v];
        float acc[4] = {__uint_as_float(own4.x), __uint_as_float(own4.y),
                        __uint_as_float(own4.z), __uint_as_float(own4.w)};
        for (uint32_t p = 0; p < npeers; ++p) {
            const uint4 peer4 = reinterpret_cast<const uint4*>(peers[p])[v];
            acc[0] += __uint_as_float(peer4.x);
            acc[1] += __uint_as_float(peer4.y);
            acc[2] += __uint_as_float(peer4.z);
            acc[3] += __uint_as_float(peer4.w);
        }
        reinterpret_cast<uint4*>(out)[v] =
            make_uint4(__float_as_uint(acc[0]), __float_as_uint(acc[1]),
                       __float_as_uint(acc[2]), __float_as_uint(acc[3]));
    }
    const uint64_t tail = vectors << 2u;
    for (uint64_t i = tail + static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        float value = own[i];
        for (uint32_t p = 0; p < npeers; ++p) value += peers[p][i];
        out[i] = value;
    }
}

__global__ void tp_copy_bf16_kernel(uint16_t* out, const uint16_t* in, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t vectors = count >> 3u;
    for (uint64_t v = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         v < vectors; v += stride) {
        reinterpret_cast<uint4*>(out)[v] = reinterpret_cast<const uint4*>(in)[v];
    }
    const uint64_t tail = vectors << 3u;
    for (uint64_t i = tail + static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        out[i] = in[i];
    }
}

__global__ void tp_copy_f32_kernel(float* out, const float* in, uint64_t count) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t vectors = count >> 2u;
    for (uint64_t v = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         v < vectors; v += stride) {
        reinterpret_cast<uint4*>(out)[v] = reinterpret_cast<const uint4*>(in)[v];
    }
    const uint64_t tail = vectors << 2u;
    for (uint64_t i = tail + static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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

Status launch_add(const TpSumTarget& target, const void* peers_device, uint32_t npeers,
                  std::size_t bytes, hipStream_t stream) {
    if (target.dtype == ValueDType::BF16 &&
        target.fusion == TpCombineFusion::ResidualAdd) {
        ::shelf::kernel::TpSumResidualArgs args;
        args.out = target.fusion_output;
        args.own = target.ptr;
        args.peers = static_cast<const void* const*>(peers_device);
        args.peer_count = npeers;
        args.residual = target.residual;
        args.rows = target.rows;
        args.features = target.feature_count;
        args.combine_stride = target.row_stride;
        args.residual_stride = target.residual_row_stride;
        args.output_stride = target.output_row_stride;
        const hipError_t err = ::shelf::kernel::launch_tp_sum_residual_bf16(args, stream);
        if (err != hipSuccess) {
            return Status::hip_error("tp fused residual launch", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (target.dtype == ValueDType::BF16 &&
        target.fusion == TpCombineFusion::HyperApply) {
        ::shelf::kernel::TpSumHyperApplyArgs args;
        args.out = target.fusion_output;
        args.own = target.ptr;
        args.peers = static_cast<const void* const*>(peers_device);
        args.peer_count = npeers;
        args.raw = target.residual;
        args.injection = target.injection;
        args.rows = target.rows;
        args.hidden = target.hidden;
        args.hc = target.hc;
        args.combine_stride = target.row_stride;
        args.raw_stride = target.residual_row_stride;
        args.injection_stride = target.injection_row_stride;
        args.output_stride = target.output_row_stride;
        const hipError_t err =
            ::shelf::kernel::launch_tp_sum_hyper_apply_bf16(args, stream);
        if (err != hipSuccess) {
            return Status::hip_error("tp fused hyper apply launch", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (target.dtype == ValueDType::F32) {
        const uint64_t count = bytes / sizeof(float);
        tp_add_peers_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const float*>(target.ptr),
            reinterpret_cast<const float* const*>(peers_device), npeers,
            static_cast<float*>(target.ptr), count);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_add_peers_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const uint16_t*>(target.ptr),
            reinterpret_cast<const uint16_t* const*>(peers_device), npeers,
            static_cast<uint16_t*>(target.ptr), count);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("tp add kernel launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

}

hipError_t tp_flag_test_baseline(void* own, const void* peer_ptrs, std::size_t count,
                                 bool bf16, hipStream_t stream) {
    if (bf16) {
        tp_add_peers_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const uint16_t*>(own), static_cast<const uint16_t* const*>(peer_ptrs),
            1, static_cast<uint16_t*>(own), count);
    } else {
        tp_add_peers_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const float*>(own), static_cast<const float* const*>(peer_ptrs),
            1, static_cast<float*>(own), count);
    }
    return hipGetLastError();
}

hipError_t tp_flag_test_send(const void* src, void* dst, void* flag, std::uint64_t tag,
                             std::size_t bytes, hipStream_t stream) {
    tp_flag_send<<<1, 256, 0, stream>>>(static_cast<const unsigned char*>(src),
        static_cast<unsigned char*>(dst), static_cast<unsigned long long*>(flag), tag, bytes);
    return hipGetLastError();
}

hipError_t tp_flag_test_receive(void* own, const void* inbox, void* flag, std::uint64_t tag,
                                std::size_t count, bool bf16, std::uint64_t limit,
                                hipStream_t stream) {
    auto* word = static_cast<unsigned long long*>(flag);
    tp_flag_receive<<<1, 256, 0, stream>>>(own, inbox, word, tag,
        reinterpret_cast<unsigned int*>(word + 1), count, bf16, limit);
    return hipGetLastError();
}

HipPeerTpTransport::HipPeerTpTransport(std::vector<int> devices)
    : devices_(std::move(devices)),
      peer_flag_enabled_(peer_flag_enabled() && devices_.size() == 2) {}

HipPeerTpTransport::~HipPeerTpTransport() {
    const std::size_t rank_count = devices_.size();
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) continue;
        if (r < peer_flags_.size() && peer_flags_[r] != nullptr) {
            shelf::gpu::discard_cleanup_result(hipFree(peer_flags_[r]));
        }
        if (r < inbox_.size()) {
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r) continue;
                for (void*& slot : inbox_[r][q]) {
                    if (slot != nullptr) {
                        shelf::gpu::discard_cleanup_result(hipFree(slot));
                        slot = nullptr;
                    }
                }
            }
        }
        if (r < inbox_ptrs_dev_.size()) {
            for (void*& ptrs : inbox_ptrs_dev_[r]) {
                if (ptrs != nullptr) {
                    shelf::gpu::discard_cleanup_result(hipFree(ptrs));
                    ptrs = nullptr;
                }
            }
        }
        if (r < sent_.size()) {
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r) continue;
                if (sent_[r][q] != nullptr) {
                    shelf::gpu::discard_cleanup_result(hipEventDestroy(sent_[r][q]));
                    sent_[r][q] = nullptr;
                }
            }
        }
        if (r < probe_buffers_.size() && probe_buffers_[r] != nullptr) {
            shelf::gpu::discard_cleanup_result(hipFree(probe_buffers_[r]));
            probe_buffers_[r] = nullptr;
        }
        if (r < probe_streams_.size() && probe_streams_[r] != nullptr) {
            shelf::gpu::discard_cleanup_result(hipStreamDestroy(probe_streams_[r]));
            probe_streams_[r] = nullptr;
        }
        if (r < probe_ready_.size() && probe_ready_[r] != nullptr) {
            shelf::gpu::discard_cleanup_result(hipEventDestroy(probe_ready_[r]));
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
    transport->peer_flags_.assign(rank_count, nullptr);
    transport->probe_buffers_.assign(rank_count, nullptr);
    transport->probe_streams_.assign(rank_count, nullptr);
    transport->probe_ready_.assign(rank_count, nullptr);

    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices[r]);
        if (!scope.ok()) return scope.status();
        if (transport->peer_flag_enabled_) {
            if (hipExtMallocWithFlags(&transport->peer_flags_[r], 16, hipDeviceMallocFinegrained) != hipSuccess ||
                hipMemset(transport->peer_flags_[r], 0, 16) != hipSuccess) {
                return Status::hip_error("tp flag allocation", hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
            }
        }
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
        auto scope = shelf::gpu::ScopedDevice::create(devices_[i]);
        if (!scope.ok()) return scope.status();
        for (std::size_t j = 0; j < devices_.size(); ++j) {
            if (i == j) continue;
            const hipError_t e = hipDeviceEnablePeerAccess(devices_[j], 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                return Status::hip_error("hipDeviceEnablePeerAccess", hipGetErrorString(e),
                                         __FILE__, __LINE__);
            }
            shelf::gpu::discard_cleanup_result(hipGetLastError());
        }
    }
    peer_access_enabled_ = true;
    return Status::make_ok();
}

Status HipPeerTpTransport::ensure_buffers(std::size_t bytes, std::size_t rank_count) {
    if (inbox_bytes_ >= bytes && inbox_.size() == rank_count) return Status::make_ok();
    const std::size_t peer_count = rank_count - 1;
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        for (std::size_t q = 0; q < rank_count; ++q) {
            if (q == r) continue;
            for (void*& slot : inbox_[r][q]) {
                if (slot != nullptr) {
                    shelf::gpu::discard_cleanup_result(hipFree(slot));
                    slot = nullptr;
                }
                if ((peer_flag_enabled_ ?
                     hipExtMallocWithFlags(&slot, bytes, hipDeviceMallocFinegrained) :
                     hipMalloc(&slot, bytes)) != hipSuccess) {
                    return Status::hip_error("hipMalloc tp inbox",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
            }
        }
        for (void*& ptrs : inbox_ptrs_dev_[r]) {
            if (ptrs != nullptr) {
                shelf::gpu::discard_cleanup_result(hipFree(ptrs));
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

    if (peer_flag_enabled_) {
        if (peer_failed_) return Status::invalid_state("tp peer flag transport aborted", __FILE__, __LINE__);
        if (peer_generation_ == std::numeric_limits<std::uint64_t>::max()) {
            peer_failed_ = true;
            return Status::overflow("tp peer flag generation exhausted", __FILE__, __LINE__);
        }
        if (invocation.peer_write_complete || invocation.targets[0].fusion != TpCombineFusion::None ||
            invocation.targets[1].fusion != TpCombineFusion::None) {
            return Status::invalid_argument("tp peer flag does not support fused targets", __FILE__, __LINE__);
        }
        const auto tag = ++peer_generation_;
        for (std::size_t r = 0; r < 2; ++r) {
            auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
            if (!scope.ok()) { peer_failed_ = true; return scope.status(); }
            auto* flag = static_cast<unsigned long long*>(peer_flags_[1-r]);
            tp_flag_send<<<1, 256, 0, invocation.streams[r]>>>(
                static_cast<const unsigned char*>(invocation.targets[r].ptr),
                static_cast<unsigned char*>(inbox_[1-r][r][parity]), flag, tag, bytes);
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) { peer_failed_ = true; return Status::hip_error("tp flag send", hipGetErrorString(err), __FILE__, __LINE__); }
        }
        for (std::size_t r = 0; r < 2; ++r) {
            auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
            if (!scope.ok()) { peer_failed_ = true; return scope.status(); }
            hipDeviceProp_t props{};
            hipError_t err = hipGetDeviceProperties(&props, devices_[r]);
            if (err != hipSuccess) { peer_failed_ = true; return Status::hip_error("tp flag clock", hipGetErrorString(err), __FILE__, __LINE__); }
            auto* flag = static_cast<unsigned long long*>(peer_flags_[r]);
            auto* error = reinterpret_cast<unsigned int*>(flag + 1);
            tp_flag_receive<<<1, 256, 0, invocation.streams[r]>>>(invocation.targets[r].ptr,
                inbox_[r][1-r][parity], flag, tag, error,
                bytes / (dtype == ValueDType::BF16 ? 2 : 4), dtype == ValueDType::BF16,
                static_cast<unsigned long long>(props.clockRate) * 5000ull);
            err = hipGetLastError();
            if (err != hipSuccess) { peer_failed_ = true; return Status::hip_error("tp flag receive", hipGetErrorString(err), __FILE__, __LINE__); }
        }
        bool timed_out = false;
        for (std::size_t r = 0; r < 2; ++r) {
            auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
            if (!scope.ok()) { peer_failed_ = true; return scope.status(); }
            hipError_t err = hipStreamSynchronize(invocation.streams[r]);
            if (err != hipSuccess) { peer_failed_ = true; return Status::hip_error("tp flag synchronize", hipGetErrorString(err), __FILE__, __LINE__); }
            unsigned int error = 0;
            err = hipMemcpy(&error, static_cast<char*>(peer_flags_[r]) + 8, sizeof(error), hipMemcpyDeviceToHost);
            if (err != hipSuccess) { peer_failed_ = true; return Status::hip_error("tp flag error read", hipGetErrorString(err), __FILE__, __LINE__); }
            timed_out |= error != 0;
        }
        if (timed_out) { peer_failed_ = true; return Status::invalid_state("tp peer flag timeout", __FILE__, __LINE__); }
        parity_ ^= 1u;
        return Status::make_ok();
    }

    if (!invocation.peer_write_complete) {
        for (std::size_t r = 0; r < rank_count; ++r) {
            auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
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
            auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
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
        auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        for (std::size_t q = 0; q < rank_count; ++q) {
            if (q == r) continue;
            if (hipStreamWaitEvent(invocation.streams[r], sent_[q][r], 0) != hipSuccess) {
                return Status::hip_error("hipStreamWaitEvent tp receive",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        st = launch_add(invocation.targets[r], inbox_ptrs_dev_[r][parity],
                        rank_count - 1, bytes, invocation.streams[r]);
        if (!st.ok()) return st;
    }
    parity_ ^= 1u;
    return Status::make_ok();
}

Status HipPeerTpTransport::peer_write_target(int rank, std::size_t bytes, void*& peer_out) {
    peer_out = nullptr;
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t rank_count = devices_.size();
    if (rank_count != 2 || peer_flag_enabled_) return Status::make_ok();
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
            want[i] = shelf::f32_to_bf16_rne(acc);
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
                auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
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
                auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
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
                auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
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

bool HipPeerTpTransport::supports_combine_fusion() const noexcept {
    if (peer_flag_enabled_) return false;
    static const bool enabled = [] {
        const char* e = std::getenv("SHELF_TP_FUSE");
        return e == nullptr || std::strtoul(e, nullptr, 10) != 0ul;
    }();
    return enabled;
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

}  // namespace shelf::runtime
