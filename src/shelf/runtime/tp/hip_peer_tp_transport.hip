#include <shelf/runtime/tp/tp_transports.h>
#include <shelf/runtime/tp/tp_argmax.h>
#include <shelf/runtime/stream_bridge.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/debug_ring.h>
#include <shelf/core/gpu/scoped_device.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/fused/tp_combine.h>

#include "tp_transport_internal.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace shelf::runtime {

namespace {

constexpr std::size_t kProbeMaxBytes = 1u << 20;
constexpr std::size_t kProbeSizes[] = {8192u, 131072u, 1048576u};
constexpr int kProbeRounds = 4;

__global__ void tp_peer_probe_copy_kernel(uint32_t* dst, const uint32_t* src,
                                          uint64_t words) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
         i += stride) {
        dst[i] = src[i];
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

__device__ __forceinline__ void tp_publish_block_writes(uint32_t* arrive,
                                                        uint32_t* peer_signal,
                                                        uint32_t epoch) {
    if (arrive == nullptr || peer_signal == nullptr) return;
    __syncthreads();
    if (threadIdx.x != 0u) return;
    const int prior = __hip_atomic_fetch_add(arrive, 1, __ATOMIC_ACQ_REL,
                                             __HIP_MEMORY_SCOPE_SYSTEM);
    if (prior != static_cast<int>(gridDim.x) - 1) return;
    __hip_atomic_store(peer_signal, epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(arrive, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

template <class T>
struct ThreePeerInputs {
    const T* addresses[3];
    __device__ const T* operator[](uint32_t peer) const { return addresses[peer]; }
};

template <unsigned FixedPeers = 0u, class PeerList>
__global__ void tp_add_peers_bf16_kernel(const uint16_t* own, PeerList peers, uint32_t peer_count,
                                    uint16_t* out, uint64_t count, bool vectorized) {
    const uint32_t npeers = FixedPeers == 0u ? peer_count : FixedPeers;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t vectors = (FixedPeers == 0u || vectorized) ? count >> 3u : 0u;
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

template <unsigned FixedPeers = 0u, class PeerList>
__global__ void tp_add_peers_f32_kernel(const float* own, PeerList peers, uint32_t peer_count,
                                    float* out, uint64_t count, bool vectorized) {
    const uint32_t npeers = FixedPeers == 0u ? peer_count : FixedPeers;
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const uint64_t vectors = (FixedPeers == 0u || vectorized) ? count >> 2u : 0u;
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

Status launch_add_four(const TpSumTarget& target, const std::array<void*, 4>& inputs,
                       std::size_t bytes, hipStream_t stream) {
    hipError_t err;
    if (target.dtype == ValueDType::F32) {
        err = launch_tp_sum_f32_4(static_cast<float*>(target.ptr),
            static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
            static_cast<const float*>(inputs[2]), static_cast<const float*>(inputs[3]),
            bytes / sizeof(float), stream);
    } else {
        err = launch_tp_sum_bf16_4(static_cast<uint16_t*>(target.ptr),
            static_cast<const uint16_t*>(inputs[0]), static_cast<const uint16_t*>(inputs[1]),
            static_cast<const uint16_t*>(inputs[2]), static_cast<const uint16_t*>(inputs[3]),
            bytes / sizeof(uint16_t), stream);
    }
    if (err != hipSuccess)
        return Status::hip_error("tp four-input reduce", hipGetErrorString(err), __FILE__, __LINE__);
    return Status::make_ok();
}

__global__ void tp_copy_bf16_kernel(uint16_t* out, const uint16_t* in, uint64_t count,
                                    shelf::gpu::DebugRingDevice ring, uint32_t debug_rank,
                                    uint32_t debug_parity, uint32_t debug_rows,
                                    uint32_t debug_stride, uint32_t debug_features,
                                    uint32_t* arrive, uint32_t* peer_signal,
                                    uint32_t epoch) {
    if (ring.slots != nullptr && blockIdx.x == 0u && threadIdx.x == 0u) {
        shelf::gpu::DebugRingRecord record{};
        record.kind = 1u;
        record.rank = debug_rank;
        record.stage = debug_parity;
        record.value0 = debug_rows;
        record.value1 = debug_stride;
        record.value2 = debug_features;
        record.data[0] = count;
        (void)shelf::gpu::shelf_debug_ring_push(ring, record);
    }
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
    tp_publish_block_writes(arrive, peer_signal, epoch);
}

__global__ void tp_copy_f32_kernel(float* out, const float* in, uint64_t count,
                                   shelf::gpu::DebugRingDevice ring, uint32_t debug_rank,
                                   uint32_t debug_parity, uint32_t debug_rows,
                                   uint32_t debug_stride, uint32_t debug_features,
                                   uint32_t* arrive, uint32_t* peer_signal,
                                   uint32_t epoch) {
    if (ring.slots != nullptr && blockIdx.x == 0u && threadIdx.x == 0u) {
        shelf::gpu::DebugRingRecord record{};
        record.kind = 1u;
        record.rank = debug_rank;
        record.stage = debug_parity;
        record.value0 = debug_rows;
        record.value1 = debug_stride;
        record.value2 = debug_features;
        record.data[0] = count;
        (void)shelf::gpu::shelf_debug_ring_push(ring, record);
    }
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
    tp_publish_block_writes(arrive, peer_signal, epoch);
}

bool tp_debug_ring_enabled() {
    const char* env = std::getenv("SHELF_TP_DEBUG_RING");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool tp_reduce_legacy() {
    const char* env = std::getenv("SHELF_TP_REDUCE");
    return env != nullptr && std::strcmp(env, "legacy") == 0;
}

bool tp_debug_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHELF_TP_DEBUG");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}


uint32_t tp_grid_for(uint64_t count) {
    const uint64_t blocks = (count + 255u) / 256u;
    if (blocks == 0) return 1;
    return static_cast<uint32_t>(blocks > 1024u ? 1024u : blocks);
}

Status launch_copy(void* dst, const void* src, const TpSumTarget& target,
                   std::size_t bytes, hipStream_t stream,
                   const shelf::gpu::DebugRingDevice& ring, uint32_t debug_rank,
                   uint32_t debug_parity, uint32_t* arrive = nullptr,
                   uint32_t* peer_signal = nullptr, uint32_t epoch = 0u) {
    const ValueDType dtype = target.dtype;
    const uint32_t debug_rows = static_cast<uint32_t>(target.rows);
    const uint32_t debug_stride = target.row_stride;
    const uint32_t debug_features = target.feature_count;
    if (dtype == ValueDType::F32) {
        const uint64_t count = bytes / sizeof(float);
        tp_copy_f32_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<float*>(dst), static_cast<const float*>(src), count, ring,
            debug_rank, debug_parity, debug_rows, debug_stride, debug_features, arrive,
            peer_signal, epoch);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_copy_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<uint16_t*>(dst), static_cast<const uint16_t*>(src), count, ring,
            debug_rank, debug_parity, debug_rows, debug_stride, debug_features, arrive,
            peer_signal, epoch);
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
            static_cast<float*>(target.ptr), count, true);
    } else {
        const uint64_t count = bytes / sizeof(uint16_t);
        tp_add_peers_bf16_kernel<<<tp_grid_for(count), 256, 0, stream>>>(
            static_cast<const uint16_t*>(target.ptr),
            reinterpret_cast<const uint16_t* const*>(peers_device), npeers,
            static_cast<uint16_t*>(target.ptr), count, true);
    }
    const hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return Status::hip_error("tp add kernel launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

}

hipError_t launch_tp_sum_bf16_4(uint16_t* dst, const uint16_t* s0, const uint16_t* s1,
                                const uint16_t* s2, const uint16_t* s3, std::size_t count,
                                hipStream_t stream) {
    if (count == 0u) return hipSuccess;
    if (dst == nullptr || s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr)
        return hipErrorInvalidValue;
    const uintptr_t addresses = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(s0) |
        reinterpret_cast<uintptr_t>(s1) | reinterpret_cast<uintptr_t>(s2) | reinterpret_cast<uintptr_t>(s3);
    const ThreePeerInputs<uint16_t> peers{{s1, s2, s3}};
    tp_add_peers_bf16_kernel<3u><<<tp_grid_for(count), 256, 0, stream>>>(s0, peers, 3u, dst, count,
                                                                    (addresses & 15u) == 0u);
    return hipGetLastError();
}

hipError_t launch_tp_sum_f32_4(float* dst, const float* s0, const float* s1, const float* s2,
                               const float* s3, std::size_t count, hipStream_t stream) {
    if (count == 0u) return hipSuccess;
    if (dst == nullptr || s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr)
        return hipErrorInvalidValue;
    const uintptr_t addresses = reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(s0) |
        reinterpret_cast<uintptr_t>(s1) | reinterpret_cast<uintptr_t>(s2) | reinterpret_cast<uintptr_t>(s3);
    const ThreePeerInputs<float> peers{{s1, s2, s3}};
    tp_add_peers_f32_kernel<3u><<<tp_grid_for(count), 256, 0, stream>>>(s0, peers, 3u, dst, count,
                                                                    (addresses & 15u) == 0u);
    return hipGetLastError();
}

hipError_t launch_tp_peer_probe_copy(void* dst, const void* src, std::size_t bytes,
                                     hipStream_t stream) {
    const std::size_t words = bytes / sizeof(uint32_t);
    if (words == 0u) return hipSuccess;
    if (dst == nullptr || src == nullptr || (bytes % sizeof(uint32_t)) != 0u)
        return hipErrorInvalidValue;
    const uint32_t threads = 256u;
    const uint32_t blocks = static_cast<uint32_t>((words + threads - 1u) / threads);
    tp_peer_probe_copy_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<uint32_t*>(dst), static_cast<const uint32_t*>(src), words);
    return hipGetLastError();
}

HipPeerTpTransport::HipPeerTpTransport(std::vector<int> devices,
                                       TpCombineBackend backend)
    : devices_(std::move(devices)), backend_(backend) {}

HipPeerTpTransport::~HipPeerTpTransport() {
    (void)shutdown();
}

Status HipPeerTpTransport::shutdown() {
    if (devices_.empty()) return Status::make_ok();
    Status first = Status::make_ok();
    auto record = [&](hipError_t err, const char* operation) {
        if (err != hipSuccess && first.ok())
            first = Status::hip_error(operation, hipGetErrorString(err), __FILE__, __LINE__);
    };
    report_debug_rings();
    debug_rings_.clear();
    for (std::size_t r = 0; r < devices_.size(); ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) {
            if (first.ok()) first = scope.status();
            continue;
        }
        if (r < inbox_.size())
            for (auto& direction : inbox_[r])
                for (void* slot : direction) if (slot) record(hipFree(slot), "tp inbox release");
        if (r < inbox_ptrs_dev_.size())
            for (void* ptrs : inbox_ptrs_dev_[r]) if (ptrs) record(hipFree(ptrs), "tp pointer release");
        for (const auto* events : {&sent_, &consumed_}) {
            if (r >= events->size()) continue;
            for (const auto& direction : (*events)[r])
                for (auto event : direction) if (event) record(hipEventDestroy(event), "tp event release");
        }
        if (r < probe_buffers_.size() && probe_buffers_[r]) record(hipFree(probe_buffers_[r]), "tp probe release");
        if (r < probe_streams_.size() && probe_streams_[r]) record(hipStreamDestroy(probe_streams_[r]), "tp probe stream release");
        if (r < probe_ready_.size() && probe_ready_[r]) record(hipEventDestroy(probe_ready_[r]), "tp probe event release");
        if (r < stream_arrive_.size() && stream_arrive_[r]) record(hipFree(stream_arrive_[r]), "tp arrival release");
        if (r < lm_recv_.size() && lm_recv_[r]) record(hipFree(lm_recv_[r]), "argmax buffer release");
        if (r < lm_sent_.size() && lm_sent_[r]) record(hipEventDestroy(lm_sent_[r]), "argmax send event release");
        if (r < lm_done_.size() && lm_done_[r]) record(hipEventDestroy(lm_done_[r]), "argmax receive event release");
    }
    for (const auto& retired : retired_inbox_) {
        auto scope = shelf::gpu::ScopedDevice::create(retired.first);
        if (scope.ok()) record(hipFree(retired.second), "retired tp inbox release");
        else if (first.ok()) first = scope.status();
    }
    for (void* p : link_staging_) if (p) record(hipHostFree(p), "tp staging release");
    for (void* p : lm_host_) if (p) record(hipHostFree(p), "argmax staging release");
    if (stream_signal_host_) record(hipHostFree(stream_signal_host_), "tp stream signal release");
    devices_.clear();
    verified_ = false;
    return first;
}

Result<std::unique_ptr<HipPeerTpTransport>> HipPeerTpTransport::create(
    const std::vector<int>& devices, TpCombineBackend backend) {
    const std::size_t rank_count = devices.size();
    TpPeerLinkMap links(rank_count, std::vector<uint8_t>(rank_count, 1u));
    for (std::size_t r = 0; r < rank_count; ++r) links[r][r] = 0u;
    return create(devices, backend, links);
}

Result<std::unique_ptr<HipPeerTpTransport>> HipPeerTpTransport::create(
    const std::vector<int>& devices, TpCombineBackend backend,
    const TpPeerLinkMap& links) {
    if (devices.size() < 2) {
        return Status::invalid_argument("hip peer transport requires at least 2 devices",
                                        __FILE__, __LINE__);
    }
    if (links.size() != devices.size()) {
        return Status::invalid_argument("tp peer link map rank count mismatch", __FILE__,
                                         __LINE__);
    }
    for (const auto& row : links) {
        if (row.size() != devices.size()) {
            return Status::invalid_argument("tp peer link map row count mismatch", __FILE__,
                                             __LINE__);
        }
    }
    auto transport = std::unique_ptr<HipPeerTpTransport>(
        new HipPeerTpTransport(devices, backend));
    transport->peer_link_ = links;
    Status enable_status = transport->enable_peer_access();
    if (!enable_status.ok()) return enable_status;

    const std::size_t rank_count = devices.size();
    transport->sent_.assign(
        rank_count,
        std::vector<std::array<hipEvent_t, 2>>(rank_count, {nullptr, nullptr}));
    transport->consumed_.assign(
        rank_count,
        std::vector<std::array<hipEvent_t, 2>>(rank_count, {nullptr, nullptr}));
    transport->consumed_recorded_.assign(
        rank_count,
        std::vector<std::array<char, 2>>(rank_count, {0, 0}));
    transport->inbox_.assign(rank_count, {});
    transport->inbox_ptrs_dev_.assign(rank_count, {});
    for (std::size_t r = 0; r < rank_count; ++r) {
        transport->inbox_[r].assign(rank_count, {nullptr, nullptr});
        transport->inbox_ptrs_dev_[r] = {nullptr, nullptr};
    }
    transport->barrier_seq_.assign(rank_count, 0);
    transport->probe_buffers_.assign(rank_count, nullptr);
    transport->probe_streams_.assign(rank_count, nullptr);
    transport->probe_ready_.assign(rank_count, nullptr);

    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices[r]);
        if (!scope.ok()) return scope.status();
        for (std::size_t q = 0; q < rank_count; ++q) {
            if (q == r) continue;
            unsigned event_flags = hipEventDisableTiming;
            if (!tp_reduce_legacy()) event_flags |= hipEventReleaseToSystem;
            for (int p = 0; p < 2; ++p) {
                if (hipEventCreateWithFlags(&transport->sent_[r][q][p], event_flags) !=
                    hipSuccess) {
                    return Status::hip_error("hipEventCreate tp send",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
                if (hipEventCreateWithFlags(&transport->consumed_[r][q][p], event_flags) !=
                    hipSuccess) {
                    return Status::hip_error("hipEventCreate tp consumed",
                                             hipGetErrorString(hipGetLastError()), __FILE__,
                                             __LINE__);
                }
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
    if (tp_debug_ring_enabled()) {
        std::size_t slots = 4096;
        if (const char* slots_env = std::getenv("SHELF_TP_DEBUG_RING_SLOTS")) {
            const unsigned long requested = std::strtoul(slots_env, nullptr, 10);
            if (requested >= 2ul) {
                slots = 1;
                while (slots < requested) slots <<= 1u;
            }
        }
        transport->debug_rings_.resize(rank_count);
        for (std::size_t r = 0; r < rank_count; ++r) {
            auto ring = shelf::gpu::DebugRing::create(devices[r], slots);
            if (!ring.ok()) return ring.status();
            transport->debug_rings_[r] = ring.release();
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
            if (i == j || peer_link_[i][j] == 0u) continue;
            const hipError_t e = hipDeviceEnablePeerAccess(devices_[j], 0);
            if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
                peer_link_[i][j] = 0u;
                shelf::gpu::discard_cleanup_result(hipGetLastError());
                continue;
            }
            shelf::gpu::discard_cleanup_result(hipGetLastError());
        }
    }
    peer_access_enabled_ = true;
    return Status::make_ok();
}

Status HipPeerTpTransport::ensure_link_staging(std::size_t bytes, std::size_t rank_count) {
    if (link_staging_bytes_ >= bytes && link_staging_.size() == 2u * rank_count * rank_count)
        return Status::make_ok();
    for (std::size_t slot = 0; slot < link_staging_.size(); ++slot) {
        if (!link_staging_[slot]) continue;
        const auto dst = slot / 2u % rank_count;
        const auto src = slot / 2u / rank_count;
        const auto parity = slot % 2u;
        auto scope = shelf::gpu::ScopedDevice::create(devices_[dst]);
        if (!scope.ok()) return scope.status();
        if (consumed_recorded_[dst][src][parity]) {
            const auto err = hipEventSynchronize(consumed_[dst][src][parity]);
            if (err != hipSuccess) return Status::hip_error("tp staging drain", hipGetErrorString(err), __FILE__, __LINE__);
        }
        const auto err = hipHostFree(link_staging_[slot]);
        if (err != hipSuccess) return Status::hip_error("tp staging release", hipGetErrorString(err), __FILE__, __LINE__);
        link_staging_[slot] = nullptr;
    }
    link_staging_bytes_ = 0u;
    link_staging_.assign(2u * rank_count * rank_count, nullptr);
    for (std::size_t r = 0; r < link_staging_.size(); ++r) {
        const auto src = r / 2u / rank_count;
        const auto dst = r / 2u % rank_count;
        if (src == dst || peer_link_[src][dst]) continue;
        void* buffer = nullptr;
        const hipError_t e = hipHostMalloc(&buffer, bytes, hipHostMallocPortable);
        if (e != hipSuccess) {
            return Status::hip_error("hipHostMalloc tp link staging", hipGetErrorString(e),
                                      __FILE__, __LINE__);
        }
        link_staging_[r] = buffer;
    }
    link_staging_bytes_ = bytes;
    return Status::make_ok();
}

Status HipPeerTpTransport::ensure_buffers(std::size_t bytes, std::size_t rank_count) {
    if (inbox_bytes_.load(std::memory_order_relaxed) >= bytes && inbox_.size() == rank_count)
        return Status::make_ok();
    if (tp_debug_enabled())
        std::fprintf(stderr, "ENSURE_GROW bytes=%zu was=%zu\n", bytes,
                     inbox_bytes_.load(std::memory_order_relaxed));
    const std::size_t peer_count = rank_count - 1;
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        for (std::size_t q = 0; q < rank_count; ++q) {
            if (q == r) continue;
            for (void*& slot : inbox_[r][q]) {
                if (slot != nullptr) {
                    retired_inbox_.emplace_back(devices_[r], slot);
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
                retired_inbox_.emplace_back(devices_[r], ptrs);
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
    inbox_bytes_.store(bytes, std::memory_order_relaxed);
    return Status::make_ok();
}

shelf::gpu::DebugRingDevice HipPeerTpTransport::debug_ring_view(std::size_t rank) const noexcept {
    if (!verified_ || rank >= debug_rings_.size() || debug_rings_[rank] == nullptr)
        return shelf::gpu::DebugRingDevice{};
    return debug_rings_[rank]->device_view();
}

void HipPeerTpTransport::report_debug_rings() noexcept {
    for (std::size_t r = 0; r < debug_rings_.size(); ++r) {
        shelf::gpu::DebugRing* ring = debug_rings_[r].get();
        if (ring == nullptr) continue;
        shelf::gpu::DebugRingRecord record{};
        std::size_t total = 0;
        std::size_t parity[2] = {0, 0};
        std::size_t kinds[4] = {0, 0, 0, 0};
        std::size_t shape_changes = 0;
        uint64_t first = 0;
        uint64_t last = 0;
        uint32_t rows = 0;
        uint32_t stride = 0;
        uint32_t features = 0;
        while (ring->try_pop(record)) {
            if (total == 0) {
                first = record.sequence;
                rows = record.value0;
                stride = record.value1;
                features = record.value2;
            } else if (record.value0 != rows || record.value1 != stride ||
                       record.value2 != features) {
                ++shape_changes;
            }
            if (record.stage < 2u) ++parity[record.stage];
            if (record.kind < 4u) ++kinds[record.kind];
            last = record.sequence;
            ++total;
        }
        std::fprintf(stderr,
                     "[tp-debug-ring] rank=%zu records=%zu lost=%llu seq=%llu..%llu "
                     "parity0=%zu parity1=%zu copy=%zu rows=%u stride=%u "
                     "features=%u shapes=%zu\n",
                     r, total, static_cast<unsigned long long>(ring->lost()),
                     static_cast<unsigned long long>(first),
                     static_cast<unsigned long long>(last), parity[0], parity[1],
                     kinds[1], rows, stride, features, shape_changes);
    }
}

Status HipPeerTpTransport::enqueue_sum(const TpSumInvocation& invocation) {
    const std::size_t rank_count = devices_.size();
    std::size_t bytes = 0;
    Status st = validate_tp_sum_invocation(invocation, rank_count, bytes);
    if (!st.ok()) return st;

    std::lock_guard<std::mutex> lock(mutex_);
    if (inbox_bytes_.load(std::memory_order_relaxed) < bytes) {
        return Status::unsupported("tp inbox is not reserved", __FILE__, __LINE__);
    }
    const std::size_t parity = static_cast<std::size_t>(barrier_seq_[0] & 1u);
    const bool need_link_staging = !link_staging_.empty();
    if (!invocation.peer_write_complete) {
        for (std::size_t r = 0; r < rank_count; ++r) {
            auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
            if (!scope.ok()) return scope.status();
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r) continue;
                if ((!tp_reduce_legacy() || need_link_staging) && consumed_recorded_[q][r][parity] != 0) {
                    if (hipStreamWaitEvent(invocation.streams[r],
                                           consumed_[q][r][parity], 0) != hipSuccess) {
                        return Status::hip_error("hipStreamWaitEvent tp consumed",
                                                 hipGetErrorString(hipGetLastError()),
                                                 __FILE__, __LINE__);
                    }
                }
                if (peer_link_[r][q] == 0u) {
                    const hipError_t e =
                        hipMemcpyAsync(link_staging_[2u * (r * rank_count + q) + parity], invocation.targets[r].ptr, bytes,
                                       hipMemcpyDeviceToHost, invocation.streams[r]);
                    if (e != hipSuccess) {
                        return Status::hip_error("tp link D2H staging", hipGetErrorString(e),
                                                 __FILE__, __LINE__);
                    }
                } else {
                    st = launch_copy(inbox_[q][r][parity], invocation.targets[r].ptr,
                                     invocation.targets[r], bytes, invocation.streams[r],
                                     debug_ring_view(r), static_cast<uint32_t>(r),
                                     static_cast<uint32_t>(parity));
                    if (!st.ok()) return st;
                }
                if (hipEventRecord(sent_[r][q][parity], invocation.streams[r]) != hipSuccess) {
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
                if (hipEventRecord(sent_[r][q][parity], invocation.streams[r]) != hipSuccess) {
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
            if (hipStreamWaitEvent(invocation.streams[r], sent_[q][r][parity], 0) != hipSuccess) {
                return Status::hip_error("hipStreamWaitEvent tp receive",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
        }
        if (!invocation.peer_write_complete) {
            for (std::size_t q = 0; q < rank_count; ++q) {
                if (q == r || peer_link_[q][r] != 0u) continue;
                const hipError_t e = hipMemcpyAsync(inbox_[r][q][parity], link_staging_[2u * (q * rank_count + r) + parity],
                                                    bytes, hipMemcpyHostToDevice,
                                                    invocation.streams[r]);
                if (e != hipSuccess) {
                    return Status::hip_error("tp link H2D staging", hipGetErrorString(e),
                                             __FILE__, __LINE__);
                }
            }
        }
        const auto& target = invocation.targets[r];
        if (rank_count == 4u && target.fusion == TpCombineFusion::None) {
            std::array<void*, 4> inputs;
            for (std::size_t q = 0; q < inputs.size(); ++q)
                inputs[q] = q == r ? target.ptr : inbox_[r][q][parity];
            st = launch_add_four(target, inputs, bytes, invocation.streams[r]);
        } else {
            st = launch_add(target, inbox_ptrs_dev_[r][parity], rank_count - 1, bytes,
                            invocation.streams[r]);
        }
        if (!st.ok()) return st;
        for (std::size_t q = 0; q < rank_count; ++q) {
            if (q == r) continue;
            if (hipEventRecord(consumed_[r][q][parity], invocation.streams[r]) != hipSuccess) {
                return Status::hip_error("hipEventRecord tp consumed",
                                         hipGetErrorString(hipGetLastError()), __FILE__,
                                         __LINE__);
            }
            consumed_recorded_[r][q][parity] = 1;
        }
    }
    return Status::make_ok();
}


std::size_t HipPeerTpTransport::inbox_capacity() const noexcept {
    return inbox_bytes_.load(std::memory_order_relaxed);
}

void HipPeerTpTransport::advance_barrier(int rank) {
    if (rank < 0 || static_cast<std::size_t>(rank) >= barrier_seq_.size()) return;
    ++barrier_seq_[static_cast<std::size_t>(rank)];
}

Status HipPeerTpTransport::reserve(std::size_t bytes) {
    if (tp_debug_enabled())
        std::fprintf(stderr, "TP_RESERVE bytes=%zu\n", bytes);
    std::lock_guard<std::mutex> lock(mutex_);
    Status st = ensure_buffers(bytes, devices_.size());
    if (!st.ok()) return st;
    if (!tp_peer_link_map_all(peer_link_)) {
        st = ensure_link_staging(bytes, devices_.size());
        if (!st.ok()) return st;
    }
    if (stream_signal_ready()) return ensure_stream_signal_state();
    return Status::make_ok();
}

Status HipPeerTpTransport::ensure_stream_signal_state() {
    if (stream_signal_allocated_) return Status::make_ok();
    const std::size_t rank_count = devices_.size();
    if (rank_count != 2)
        return Status::unsupported("stream signal barrier requires two ranks", __FILE__,
                                   __LINE__);
    const std::size_t words = rank_count * 2u;
    if (hipHostMalloc(&stream_signal_host_, words * sizeof(uint32_t),
                      hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) {
        stream_signal_host_ = nullptr;
        return Status::hip_error("hipHostMalloc stream signal",
                                 hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
    }
    std::memset(stream_signal_host_, 0, words * sizeof(uint32_t));
    for (std::size_t d = 0; d < rank_count; ++d) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_[d]);
        if (!scope.ok()) return scope.status();
        void* dev = nullptr;
        if (hipHostGetDevicePointer(&dev, stream_signal_host_, 0) != hipSuccess) {
            return Status::hip_error("hipHostGetDevicePointer stream signal",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        stream_signal_dev_[d] = static_cast<uint32_t*>(dev);
        void* arrive = nullptr;
        if (hipMalloc(&arrive, sizeof(uint32_t)) != hipSuccess) {
            return Status::hip_error("hipMalloc stream arrive",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        if (hipMemset(arrive, 0, sizeof(uint32_t)) != hipSuccess) {
            shelf::gpu::discard_cleanup_result(hipFree(arrive));
            return Status::hip_error("hipMemset stream arrive",
                                     hipGetErrorString(hipGetLastError()), __FILE__,
                                     __LINE__);
        }
        stream_arrive_[d] = static_cast<uint32_t*>(arrive);
    }
    stream_signal_allocated_ = true;
    return Status::make_ok();
}

bool HipPeerTpTransport::stream_signal_ready() const noexcept {
    if (devices_.size() != 2) return false;
    if (peer_link_.size() == 2u && (peer_link_[0][1] == 0u || peer_link_[1][0] == 0u))
        return false;
    const char* env = std::getenv("SHELF_TP_SIGNAL_BARRIER");
    if (env != nullptr && env[0] != '\0') return env[0] != '0';
    return backend_ == TpCombineBackend::StreamSignal;
}

Status HipPeerTpTransport::sum_hidden_stream_signal(int rank, const TpSumTarget& target,
                                                    hipStream_t stream, bool peer_written) {
    const std::size_t rank_count = devices_.size();
    if (rank_count != 2)
        return Status::unsupported("stream signal barrier requires two ranks", __FILE__,
                                   __LINE__);
    if (rank < 0 || static_cast<std::size_t>(rank) >= rank_count)
        return Status::invalid_argument("stream signal rank out of range", __FILE__,
                                        __LINE__);
    if (target.rows != 0u &&
        (target.ptr == nullptr || target.row_stride == 0u || target.feature_count == 0u))
        return Status::invalid_argument("stream signal target is empty", __FILE__, __LINE__);
    const bool fusion_ok =
        target.fusion == TpCombineFusion::None ||
        target.fusion == TpCombineFusion::ResidualAdd ||
        (target.fusion == TpCombineFusion::HyperApply &&
         target.dtype == ValueDType::BF16);
    if (!fusion_ok)
        return Status::unsupported("stream signal fusion is not supported", __FILE__,
                                   __LINE__);
    const std::size_t bytes = static_cast<std::size_t>(target.rows) * target.row_stride *
                              ::shelf::runtime::value_dtype_bytes(target.dtype);
    if (bytes != 0u && inbox_bytes_.load(std::memory_order_relaxed) < bytes)
        return Status::unsupported("stream signal inbox is not reserved", __FILE__,
                                   __LINE__);
    std::size_t parity = 0;
    uint32_t epoch = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!stream_signal_allocated_)
            return Status::invalid_state("stream signal state is not initialized", __FILE__,
                                         __LINE__);
        parity = static_cast<std::size_t>(barrier_seq_[static_cast<std::size_t>(rank)] & 1u);
        epoch = static_cast<uint32_t>(barrier_seq_[static_cast<std::size_t>(rank)] + 1u);
    }
    auto scope = shelf::gpu::ScopedDevice::create(devices_[static_cast<std::size_t>(rank)]);
    if (!scope.ok()) return scope.status();
    const std::size_t peer_index = rank_count - 1u - static_cast<std::size_t>(rank);
    uint32_t* signals = stream_signal_dev_[static_cast<std::size_t>(rank)];
    if (signals == nullptr)
        return Status::invalid_state("stream signal pointer is null", __FILE__, __LINE__);
    uint32_t* peer_signal = signals + peer_index * 2u + parity;
    if (!peer_written) {
        Status st = launch_copy(inbox_[peer_index][static_cast<std::size_t>(rank)][parity],
                                target.ptr, target, bytes, stream,
                                shelf::gpu::DebugRingDevice{}, static_cast<uint32_t>(rank),
                                static_cast<uint32_t>(parity),
                                stream_arrive_[static_cast<std::size_t>(rank)],
                                peer_signal, epoch);
        if (!st.ok()) return st;
    } else {
        const hipError_t write_error =
            hipStreamWriteValue32(stream, peer_signal, epoch, 0u);
        if (write_error != hipSuccess) {
            return Status::hip_error("hipStreamWriteValue32 stream signal",
                                     hipGetErrorString(write_error), __FILE__, __LINE__);
        }
    }
    const hipError_t wait_error = hipStreamWaitValue32(
        stream, signals + static_cast<std::size_t>(rank) * 2u + parity, epoch,
        hipStreamWaitValueGte, 0xffffffffu);
    if (wait_error != hipSuccess) {
        return Status::hip_error("hipStreamWaitValue32 stream signal",
                                 hipGetErrorString(wait_error), __FILE__, __LINE__);
    }
    return launch_add(target, inbox_ptrs_dev_[static_cast<std::size_t>(rank)][parity],
                      rank_count - 1u, bytes, stream);
}

Status HipPeerTpTransport::ensure_lm_buffers(std::size_t rows, std::size_t rank_count) {
    if (lm_rows_ >= rows && lm_recv_.size() == rank_count) return Status::make_ok();
    if (lm_sent_.empty()) {
        lm_recv_.resize(rank_count);
        lm_host_.resize(rank_count);
        lm_sent_.resize(rank_count);
        lm_done_.resize(rank_count);
    }
    const std::size_t bytes = rows * sizeof(TpArgmaxPair);
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        if (lm_done_[r]) {
            const auto err = hipEventSynchronize(lm_done_[r]);
            if (err != hipSuccess) return Status::hip_error("argmax drain", hipGetErrorString(err), __FILE__, __LINE__);
        }
    }
    lm_rows_ = 0;
    for (std::size_t r = 0; r < rank_count; ++r) {
        auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        hipError_t err = hipSuccess;
        if (lm_recv_[r]) err = hipFree(lm_recv_[r]);
        if (err == hipSuccess) {
            lm_recv_[r] = nullptr;
            err = hipMalloc(&lm_recv_[r], bytes * (rank_count + kTpArgmaxPartitions));
        }
        if (err != hipSuccess) return Status::hip_error("argmax device storage", hipGetErrorString(err), __FILE__, __LINE__);
        if (lm_host_[r]) err = hipHostFree(lm_host_[r]);
        if (err == hipSuccess) lm_host_[r] = nullptr;
        bool needs_host = false;
        for (std::size_t q = 0; q < rank_count; ++q)
            needs_host |= q != r && peer_link_[r][q] == 0u;
        if (err == hipSuccess && needs_host)
            err = hipHostMalloc(&lm_host_[r], bytes, hipHostMallocPortable);
        const unsigned flags = hipEventDisableTiming | hipEventReleaseToSystem;
        if (err == hipSuccess && !lm_sent_[r]) err = hipEventCreateWithFlags(&lm_sent_[r], flags);
        if (err == hipSuccess && !lm_done_[r]) err = hipEventCreateWithFlags(&lm_done_[r], flags);
        if (err != hipSuccess) return Status::hip_error("argmax communication storage", hipGetErrorString(err), __FILE__, __LINE__);
    }
    lm_rows_ = rows;
    return Status::make_ok();
}

Status HipPeerTpTransport::begin_argmax(int rank, const TpArgmaxInvocation& invocation) {
    const std::size_t rank_count = devices_.size();
    if (rank < 0 || static_cast<std::size_t>(rank) >= rank_count) {
        return Status::invalid_argument("argmax rank out of range", __FILE__, __LINE__);
    }
    if (invocation.rows == 0u || invocation.local_logits == nullptr ||
        invocation.local_cols == 0u) {
        return Status::invalid_argument("argmax invocation is empty", __FILE__, __LINE__);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    Status st = ensure_lm_buffers(invocation.rows, rank_count);
    if (!st.ok()) return st;
    const std::size_t r = static_cast<std::size_t>(rank);
    auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
    if (!scope.ok()) return scope.status();
    for (std::size_t q = 0; q < rank_count && lm_published_; ++q) {
        const auto done = lm_done_[q];
        const auto err = hipStreamWaitEvent(invocation.stream, done, 0);
        if (err != hipSuccess) return Status::hip_error("argmax reuse", hipGetErrorString(err), __FILE__, __LINE__);
    }
    auto* own = static_cast<TpArgmaxPair*>(lm_recv_[r]) + r * invocation.rows;
    const hipError_t pack_err =
        launch_tp_argmax_pack(invocation.local_logits, own, invocation.rows,
                              invocation.row_stride, invocation.local_cols,
                              invocation.local_offset, invocation.stream,
                              static_cast<TpArgmaxPair*>(lm_recv_[r]) + rank_count * lm_rows_);
    if (pack_err != hipSuccess) {
        return Status::hip_error("argmax pack launch", hipGetErrorString(pack_err), __FILE__,
                                  __LINE__);
    }
    const std::size_t bytes = static_cast<std::size_t>(invocation.rows) * kTpArgmaxPairBytes;
    if (lm_host_[r]) {
        const auto err = hipMemcpyAsync(lm_host_[r], own, bytes, hipMemcpyDeviceToHost, invocation.stream);
        if (err != hipSuccess) return Status::hip_error("argmax staging", hipGetErrorString(err), __FILE__, __LINE__);
    }
    for (std::size_t q = 0; q < rank_count; ++q) {
        if (q == r || !peer_link_[r][q]) continue;
        const auto err = launch_tp_peer_probe_copy(static_cast<char*>(lm_recv_[q]) + r * bytes,
                                                   own, bytes, invocation.stream);
        if (err != hipSuccess) return Status::hip_error("argmax peer copy", hipGetErrorString(err), __FILE__, __LINE__);
    }
    const auto err = hipEventRecord(lm_sent_[r], invocation.stream);
    if (err != hipSuccess) return Status::hip_error("argmax publication", hipGetErrorString(err), __FILE__, __LINE__);
    return Status::make_ok();
}

Status HipPeerTpTransport::finish_argmax(int rank, const TpArgmaxInvocation& invocation) {
    const std::size_t rank_count = devices_.size();
    if (rank < 0 || static_cast<std::size_t>(rank) >= rank_count) {
        return Status::invalid_argument("argmax rank out of range", __FILE__, __LINE__);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t r = static_cast<std::size_t>(rank);
    auto scope = shelf::gpu::ScopedDevice::create(devices_[r]);
    if (!scope.ok()) return scope.status();
    const std::size_t bytes = static_cast<std::size_t>(invocation.rows) * kTpArgmaxPairBytes;
    for (std::size_t q = 0; q < rank_count; ++q) {
        if (q == r) continue;
        if (hipStreamWaitEvent(invocation.stream, lm_sent_[q], 0) != hipSuccess) {
            return Status::hip_error("hipStreamWaitEvent lm sent",
                                      hipGetErrorString(hipGetLastError()), __FILE__, __LINE__);
        }
        if (peer_link_[q][r] == 0u) {
            const hipError_t copy_err = hipMemcpyAsync(
                static_cast<char*>(lm_recv_[r]) + q * bytes,
                lm_host_[q], bytes,
                hipMemcpyHostToDevice, invocation.stream);
            if (copy_err != hipSuccess) {
                return Status::hip_error("argmax host restore", hipGetErrorString(copy_err),
                                          __FILE__, __LINE__);
            }
        }
    }
    const hipError_t reduce_err = launch_tp_argmax_reduce(
        static_cast<const TpArgmaxPair*>(lm_recv_[r]), invocation.rows,
        static_cast<uint32_t>(rank_count), invocation.rows, invocation.out_ids,
        invocation.stream);
    if (reduce_err != hipSuccess) {
        return Status::hip_error("argmax reduce launch", hipGetErrorString(reduce_err),
                                  __FILE__, __LINE__);
    }
    const auto err = hipEventRecord(lm_done_[r], invocation.stream);
    if (err != hipSuccess) return Status::hip_error("argmax consumed", hipGetErrorString(err), __FILE__, __LINE__);
    if (r + 1u == rank_count) lm_published_ = true;
    return Status::make_ok();
}

bool HipPeerTpTransport::supports_argmax_reduce() const noexcept {
    return verified_;
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
    if (inbox_bytes_.load(std::memory_order_relaxed) < bytes) return Status::make_ok();
    const std::size_t peer = static_cast<std::size_t>(rank) == 0 ? 1 : 0;
    if (!peer_link_[0][1] || !peer_link_[1][0]) return Status::make_ok();
    peer_out = inbox_[peer][static_cast<std::size_t>(rank)]
                        [static_cast<std::size_t>(barrier_seq_[static_cast<std::size_t>(rank)] & 1u)];
    return Status::make_ok();
}

Status HipPeerTpTransport::verify() {
    if (verified_) return Status::make_ok();
    Status st = enable_peer_access();
    if (!st.ok()) return st;

    const std::size_t rank_count = devices_.size();
    st = ensure_buffers(kProbeMaxBytes, rank_count);
    if (!st.ok()) return st;

    if (!tp_peer_link_map_all(peer_link_)) {
        st = ensure_link_staging(kProbeMaxBytes, rank_count);
        if (!st.ok()) return st;
    }

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
    if (stream_signal_ready()) return ensure_stream_signal_state();
    return Status::make_ok();
}

bool HipPeerTpTransport::supports_combine_fusion() const noexcept {
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
    if (!verified_)
        return Status::invalid_state("tp transport is not verified", __FILE__, __LINE__);
    if (inbox_bytes_.load(std::memory_order_relaxed) < bytes) {
        return Status::unsupported("tp inbox is not reserved", __FILE__, __LINE__);
    }
    return enqueue_sum(invocation);
}

}  // namespace shelf::runtime
