#include <shelf/runtime/dflash2/tp_proposal.h>

#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/gpu/scoped_device.h>

#include <cstddef>
#include <limits>
#include <utility>

namespace shelf::runtime::dflash2 {

namespace {

template <typename T>
Status allocate_buffer(gpu::GpuArena& arena, std::size_t count, gpu::Tensor& out) {
    if (count == 0u) {
        return Status::invalid_argument("tp proposal buffer is empty", __FILE__, __LINE__);
    }
    auto view = arena.allocate_aligned(count * sizeof(T), 128);
    if (!view.ok()) return view.status();
    auto backing = view.release();
    const std::size_t shape[1] = {count};
    const std::size_t strides[1] = {1};
    auto tensor = gpu::Tensor::create<T>(backing, shape, 1u, strides);
    if (!tensor.ok()) return tensor.status();
    out = tensor.release();
    return Status::make_ok();
}

__global__ void proposal_copy_kernel(
    std::uint8_t* __restrict__ dst,
    const std::uint8_t* __restrict__ src,
    std::uint64_t bytes) {
    const std::uint64_t stride =
        static_cast<std::uint64_t>(blockDim.x) * gridDim.x * 16u;
    std::uint64_t i =
        (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 16u;
    for (; i + 16u <= bytes; i += stride) {
        *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + i);
    }
    const std::uint64_t tail = (bytes / 16u) * 16u;
    for (std::uint64_t j = tail +
                           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < bytes; j += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        dst[j] = src[j];
    }
}

void launch_copy(void* dst, const void* src, std::size_t bytes, hipStream_t stream) {
    if (bytes == 0u) return;
    const std::uint64_t blocks = (bytes + 256u * 16u - 1u) / (256u * 16u);
    const std::uint32_t grid = static_cast<std::uint32_t>(blocks < 256u ? blocks : 256u);
    proposal_copy_kernel<<<grid, 256u, 0, stream>>>(
        static_cast<std::uint8_t*>(dst), static_cast<const std::uint8_t*>(src), bytes);
}

Status create_event(int device, hipEvent_t& event) {
    auto scope = gpu::ScopedDevice::create(device);
    if (!scope.ok()) return scope.status();
    const hipError_t err = hipEventCreate(&event);
    if (err != hipSuccess) {
        return Status::hip_error("hipEventCreate tp proposal", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Result<DFlash2TpProposal> DFlash2TpProposal::create(const DFlash2TpProposalConfig& config) {
    if (config.coordinator == nullptr) {
        return Status::invalid_argument("tp proposal needs a coordinator", __FILE__,
                                        __LINE__);
    }
    const std::size_t rank_count = static_cast<std::size_t>(config.coordinator->tp_size());
    if (rank_count < 2u) {
        return Status::invalid_argument("tp proposal needs at least two ranks", __FILE__,
                                        __LINE__);
    }
    if (config.heads.size() != rank_count || config.vocab_offsets.size() != rank_count) {
        return Status::invalid_argument("tp proposal head count mismatch", __FILE__,
                                        __LINE__);
    }
    if (config.local_vocab == 0u || config.top_k == 0u || config.hidden_stride == 0u ||
        config.max_rows == 0u) {
        return Status::invalid_argument("tp proposal geometry is invalid", __FILE__,
                                        __LINE__);
    }

    DFlash2TpProposal out;
    out.config_ = config;
    out.devices_.resize(rank_count);
    out.streams_.resize(rank_count);
    out.proposal_streams_.assign(rank_count, nullptr);
    out.shards_.resize(rank_count);
    out.peer_hidden_.resize(rank_count);
    out.peer_done_.assign(rank_count, nullptr);
    out.hidden_sent_.assign(rank_count, nullptr);

    for (std::size_t r = 0; r < rank_count; ++r) {
        if (config.heads[r] == nullptr) {
            return Status::invalid_argument("tp proposal head is null", __FILE__, __LINE__);
        }
        if (config.heads[r]->rows != config.local_vocab || config.heads[r]->cols == 0u) {
            return Status::invalid_argument("tp proposal head does not match the shard",
                                            __FILE__, __LINE__);
        }
        TpRankRuntime& rank = config.coordinator->rank(r);
        out.devices_[r] = rank.device_id();
        out.streams_[r] = rank.stream();

        auto scope = gpu::ScopedDevice::create(rank.device_id());
        if (!scope.ok()) return scope.status();
        auto buffers = create_proposal_shard_buffers(rank.arena(), config.max_rows,
                                                     config.local_vocab, config.top_k);
        if (!buffers.ok()) return buffers.status();
        out.shards_[r] = buffers.release();
        if (r == 0u) {
            continue;
        }
        auto peer = allocate_buffer<bf16_t>(
            rank.arena(),
            static_cast<std::size_t>(config.max_rows) * config.hidden_stride,
            out.peer_hidden_[r]);
        if (!peer.ok()) return peer;
        hipStream_t proposal_stream = nullptr;
        hipError_t stream_err = hipStreamCreate(&proposal_stream);
        if (stream_err != hipSuccess) {
            return Status::hip_error("tp proposal stream", hipGetErrorString(stream_err),
                                     __FILE__, __LINE__);
        }
        out.proposal_streams_[r] = proposal_stream;
        std::uint8_t warm = 0u;
        (void)hipMemsetAsync(out.peer_hidden_[r].data<void>(), 0, sizeof(warm),
                             proposal_stream);
        Status event = create_event(rank.device_id(), out.peer_done_[r]);
        if (!event.ok()) return event;
    }

    auto scope = gpu::ScopedDevice::create(out.devices_[0]);
    if (!scope.ok()) return scope.status();
    TpRankRuntime& leader = config.coordinator->rank(0u);
    out.ranked_stride_ = config.max_rows * config.top_k;
    out.stats_stride_ = config.max_rows * 2u;
    Status st = allocate_buffer<int32_t>(
        leader.arena(), rank_count * out.ranked_stride_, out.ranked_ids_);
    if (!st.ok()) return st;
    st = allocate_buffer<float>(leader.arena(), rank_count * out.ranked_stride_,
                                out.ranked_logits_);
    if (!st.ok()) return st;
    st = allocate_buffer<float>(leader.arena(), rank_count * out.stats_stride_,
                                out.ranked_stats_);
    if (!st.ok()) return st;
    for (std::size_t r = 1; r < rank_count; ++r) {
        st = create_event(out.devices_[0], out.hidden_sent_[r]);
        if (!st.ok()) return st;
    }

    out.initialized_ = true;
    return out;
}

DFlash2TpProposal::DFlash2TpProposal(DFlash2TpProposal&& other) noexcept {
    *this = std::move(other);
}

DFlash2TpProposal& DFlash2TpProposal::operator=(DFlash2TpProposal&& other) noexcept {
    if (this == &other) return *this;
    shutdown();
    config_ = other.config_;
    devices_ = std::move(other.devices_);
    streams_ = std::move(other.streams_);
    shards_ = std::move(other.shards_);
    peer_hidden_ = std::move(other.peer_hidden_);
    ranked_ids_ = std::move(other.ranked_ids_);
    ranked_logits_ = std::move(other.ranked_logits_);
    ranked_stats_ = std::move(other.ranked_stats_);
    ranked_stride_ = other.ranked_stride_;
    stats_stride_ = other.stats_stride_;
    peer_done_ = std::move(other.peer_done_);
    hidden_sent_ = std::move(other.hidden_sent_);
    proposal_streams_ = std::move(other.proposal_streams_);
    initialized_ = other.initialized_;
    other.peer_done_.clear();
    other.initialized_ = false;
    return *this;
}

DFlash2TpProposal::~DFlash2TpProposal() noexcept { shutdown(); }

void DFlash2TpProposal::shutdown() noexcept {
    for (std::size_t r = 0; r < peer_done_.size(); ++r) {
        if (peer_done_[r] == nullptr) continue;
        auto scope = gpu::ScopedDevice::create(devices_[r]);
        if (scope.ok()) shelf::gpu::discard_cleanup_result(hipEventDestroy(peer_done_[r]));
        peer_done_[r] = nullptr;
    }
    for (std::size_t r = 0; r < hidden_sent_.size(); ++r) {
        if (hidden_sent_[r] == nullptr) continue;
        auto scope = gpu::ScopedDevice::create(devices_[0]);
        if (scope.ok()) shelf::gpu::discard_cleanup_result(hipEventDestroy(hidden_sent_[r]));
        hidden_sent_[r] = nullptr;
    }
    for (std::size_t r = 0; r < proposal_streams_.size(); ++r) {
        if (proposal_streams_[r] == nullptr) continue;
        auto scope = gpu::ScopedDevice::create(devices_[r]);
        if (scope.ok()) shelf::gpu::discard_cleanup_result(hipStreamDestroy(proposal_streams_[r]));
        proposal_streams_[r] = nullptr;
    }
    peer_hidden_.clear();
    shards_.clear();
    ranked_ids_ = gpu::Tensor{};
    ranked_logits_ = gpu::Tensor{};
    ranked_stats_ = gpu::Tensor{};
    proposal_streams_.clear();
    streams_.clear();
    devices_.clear();
    config_ = DFlash2TpProposalConfig{};
    ranked_stride_ = 0;
    stats_stride_ = 0;
    initialized_ = false;
}

Status DFlash2TpProposal::distribute(
    const bf16_t* hidden,
    uint32_t hidden_stride,
    uint32_t rows,
    int32_t* topk_ids,
    float* topk_logits,
    float* normalizers,
    hipStream_t stream) {
    if (!initialized_) {
        return Status::invalid_state("tp proposal is not initialized", __FILE__, __LINE__);
    }
    if (hidden == nullptr || topk_ids == nullptr || topk_logits == nullptr ||
        normalizers == nullptr) {
        return Status::invalid_argument("tp proposal buffers are null", __FILE__, __LINE__);
    }
    if (rows == 0u) return Status::make_ok();
    if (rows > config_.max_rows || hidden_stride != config_.hidden_stride) {
        return Status::invalid_argument("tp proposal rows out of range", __FILE__,
                                        __LINE__);
    }

    const std::size_t rank_count = devices_.size();
    const std::size_t hidden_bytes =
        static_cast<std::size_t>(rows) * hidden_stride * sizeof(bf16_t);
    const std::size_t candidate_bytes =
        static_cast<std::size_t>(rows) * config_.top_k * sizeof(int32_t);
    const std::size_t stat_bytes = static_cast<std::size_t>(rows) * 2u * sizeof(float);

    {
        auto scope = gpu::ScopedDevice::create(devices_[0]);
        if (!scope.ok()) return scope.status();
        for (std::size_t r = 1; r < rank_count; ++r) {
            launch_copy(peer_hidden_[r].data<void>(), hidden, hidden_bytes, stream);
            const hipError_t err = hipEventRecord(hidden_sent_[r], stream);
            if (err != hipSuccess) {
                return Status::hip_error("tp proposal hidden event", hipGetErrorString(err),
                                         __FILE__, __LINE__);
            }
        }
        Status st = proposal_shard_compute(*config_.heads[0], shards_[0], hidden,
                                           hidden_stride, rows, config_.vocab_offsets[0],
                                           stream);
        if (!st.ok()) return st;
        hipError_t err = hipMemcpyAsync(ranked_ids_.data<int32_t>(),
                                        shards_[0].topk_ids.data<int32_t>(),
                                        candidate_bytes, hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("tp proposal leader id copy", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        err = hipMemcpyAsync(ranked_logits_.data<float>(),
                             shards_[0].topk_logits.data<float>(), candidate_bytes,
                             hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("tp proposal leader logit copy", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        err = hipMemcpyAsync(ranked_stats_.data<float>(), shards_[0].stats.data<float>(),
                             stat_bytes, hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            return Status::hip_error("tp proposal leader stat copy", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
    }

    for (std::size_t r = 1; r < rank_count; ++r) {
        auto scope = gpu::ScopedDevice::create(devices_[r]);
        if (!scope.ok()) return scope.status();
        hipStream_t peer_stream = proposal_streams_[r];
        hipError_t err = hipStreamWaitEvent(peer_stream, hidden_sent_[r], 0);
        if (err != hipSuccess) {
            return Status::hip_error("tp proposal hidden wait", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        Status st = proposal_shard_compute(*config_.heads[r], shards_[r],
                                           peer_hidden_[r].data<bf16_t>(), hidden_stride,
                                           rows, config_.vocab_offsets[r], peer_stream);
        if (!st.ok()) return st;
        launch_copy(ranked_ids_.data<int32_t>() + r * ranked_stride_,
                    shards_[r].topk_ids.data<int32_t>(), candidate_bytes, peer_stream);
        launch_copy(ranked_logits_.data<float>() + r * ranked_stride_,
                    shards_[r].topk_logits.data<float>(), candidate_bytes, peer_stream);
        launch_copy(ranked_stats_.data<float>() + r * stats_stride_,
                    shards_[r].stats.data<float>(), stat_bytes, peer_stream);
        err = hipEventRecord(peer_done_[r], peer_stream);
        if (err != hipSuccess) {
            return Status::hip_error("tp proposal peer event", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
    }

    auto scope = gpu::ScopedDevice::create(devices_[0]);
    if (!scope.ok()) return scope.status();
    for (std::size_t r = 1; r < rank_count; ++r) {
        const hipError_t err = hipStreamWaitEvent(stream, peer_done_[r], 0);
        if (err != hipSuccess) {
            return Status::hip_error("tp proposal peer wait", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
    }
    Status st = proposal_merge_candidates(
        ranked_ids_.data<int32_t>(), ranked_logits_.data<float>(),
        static_cast<uint32_t>(rank_count), ranked_stride_, rows, config_.top_k,
        topk_ids, topk_logits, stream);
    if (!st.ok()) return st;
    return proposal_combine_denominators(
        ranked_stats_.data<float>(), static_cast<uint32_t>(rank_count), stats_stride_,
        rows, normalizers, stream);
}

}  // namespace shelf::runtime::dflash2
