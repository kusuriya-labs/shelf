#include <shelf/runtime/mtp/mtp_drafter.h>

#include <shelf/core/gpu/scoped_device.h>
#include <shelf/runtime/gdn_state_layout.h>
#include <shelf/runtime/executor_footprint.h>
#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/runtime/spec_decode.h>
#include <shelf/runtime/tp/tp_batch_hook.h>
#include <shelf/runtime/kv_pages.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace shelf::runtime::mtp {

namespace {

constexpr uint32_t kMtpKvLayers = 1u;


bool mtp_wavefront_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHELF_MTP_WAVEFRONT");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool mtp_fused_verify_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHELF_MTP_FUSED_VERIFY");
        return value == nullptr || (value[0] != '\0' && value[0] != '0');
    }();
    return enabled;
}


uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) noexcept {
    return (value + divisor - 1u) / divisor;
}

shelf::qwen_hybrid::QwenHybridConfig mtp_executor_config(
    const shelf::qwen_hybrid::QwenHybridConfig& source) {
    shelf::qwen_hybrid::QwenHybridConfig config = source;
    config.num_hidden_layers = 1;
    config.layer_types = {1};
    config.full_attention_interval = 1;
    return config;
}

ScheduledBatch make_batch(state::PagedSequenceState& sequence,
                          std::vector<ScheduledRequest>& requests,
                          const int32_t* tokens,
                          uint32_t num_tokens,
                          uint32_t prefix_tokens,
                          uint32_t num_output_rows,
                          bool compute_logits,
                          bool sample,
                          ::shelf::runtime::ExecutionClass execution_class,
                          bool speculative_verify,
                          const SamplingConfig* sampling,
                          uint64_t sampling_index) {
    ScheduledRequest request;
    request.sequence = &sequence;
    request.handle = sequence.request_handle();
    request.execution_class = execution_class;
    request.token_begin = 0u;
    request.num_tokens = num_tokens;
    request.prefix_tokens = prefix_tokens;
    request.compute_logits = compute_logits;
    request.sample = compute_logits && sample;
    request.num_output_rows = num_output_rows;
    if (sampling != nullptr) {
        request.sampling = *sampling;
        request.sampling_index = sampling_index;
    }
    requests.clear();
    requests.push_back(request);

    ScheduledBatch batch;
    batch.token_ids = tokens;
    batch.token_ids_location = ::shelf::runtime::TokenIdsLocation::Host;
    batch.requests = requests;
    batch.num_tokens = num_tokens;
    batch.num_requests = 1u;
    batch.num_decode_requests =
        execution_class == ::shelf::runtime::ExecutionClass::DECODE ? 1u : 0u;
    batch.num_verify_requests =
        (speculative_verify && num_tokens > 1u) ? 1u : 0u;
    batch.num_prefill_requests =
        (execution_class == ::shelf::runtime::ExecutionClass::PREFILL && num_tokens > 1u)
            ? 1u
            : 0u;
    batch.speculative_verify = speculative_verify;
    return batch;
}

Executor& draft_executor(MtpDrafter& drafter) {
    return drafter.shared_draft != nullptr ? drafter.shared_draft->executor
                                           : drafter.executor;
}

const Executor& draft_executor(const MtpDrafter& drafter) {
    return drafter.shared_draft != nullptr ? drafter.shared_draft->executor
                                           : drafter.executor;
}

state::SequenceSlotPool& draft_slot_pool(MtpDrafter& drafter) {
    return drafter.shared_draft != nullptr ? *drafter.shared_draft->slot_pool
                                           : *drafter.slot_pool;
}

state::GdnStatePool& draft_gdn_pool(MtpDrafter& drafter) {
    return drafter.shared_draft != nullptr ? *drafter.shared_draft->gdn_pool
                                           : *drafter.gdn_pool;
}

state::PagedKVPool& draft_kv_pool(MtpDrafter& drafter) {
    return drafter.shared_draft != nullptr ? *drafter.shared_draft->kv_pool
                                           : *drafter.kv_pool;
}

const state::PagedKVPool& draft_kv_pool(const MtpDrafter& drafter) {
    return drafter.shared_draft != nullptr ? *drafter.shared_draft->kv_pool
                                           : *drafter.kv_pool;
}

const gpu::Tensor& drafter_hidden_output(const MtpDrafter& drafter) {
    const Executor& executor = draft_executor(drafter);
    return drafter.wide_hidden ? executor.normed_token_hidden : executor.token_hidden;
}


bool mtp_top2_enabled() {
    static const bool enabled = std::getenv("SHELF_MTP_TOP2") != nullptr;
    return enabled;
}

void log_top2(const gpu::Tensor& logits, uint32_t row, const char* tag,
              uint32_t position, uint32_t slot) {
    if (!mtp_top2_enabled() || logits.ndim() < 2 || row >= logits.dim(0)) return;
    const std::size_t vocab = logits.dim(1);
    std::vector<float> values(vocab);
    const float* src = logits.data<float>() + static_cast<std::size_t>(row) *
                                                   logits.strides_data()[0];
    if (hipMemcpy(values.data(), src, vocab * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
        return;
    int32_t first = -1;
    int32_t second = -1;
    float top1 = -INFINITY;
    float top2 = -INFINITY;
    for (std::size_t i = 0; i < vocab; ++i) {
        const float v = values[i];
        if (v > top1) {
            top2 = top1;
            second = first;
            top1 = v;
            first = static_cast<int32_t>(i);
        } else if (v > top2) {
            top2 = v;
            second = static_cast<int32_t>(i);
        }
    }
    std::fprintf(stderr,
                 "MTP_TOP2 tag=%s slot=%u pos=%u top1=%d %.6f top2=%d %.6f "
                 "margin=%.6f\n",
                 tag, slot, position, first, top1, second, top2, top1 - top2);
}

Result<int32_t> copy_sampled_token(const gpu::Tensor& sampled, hipStream_t stream) {
    int32_t token = -1;
    const hipError_t err =
        hipMemcpyAsync(&token, sampled.data<int32_t>(), sizeof(int32_t),
                       hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        return Status::hip_error("mtp sampled token copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        return Status::hip_error("mtp sampled token sync", hipGetErrorString(hipGetLastError()),
                                 __FILE__, __LINE__);
    }
    return token;
}

Status copy_hidden_to_input(MtpDrafter& drafter, const gpu::Tensor& source,
                            uint32_t source_row_offset, uint32_t rows,
                            hipStream_t stream, uint32_t dst_row = 0u) {
    if (rows == 0u) return Status::make_ok();
    const std::size_t hidden = drafter.hidden_width;
    void* dst = draft_executor(drafter).input_hidden.data<void>();
    if (dst == nullptr) {
        return Status::invalid_state("mtp drafter has no hidden input buffer", __FILE__,
                                     __LINE__);
    }
    const std::size_t offset =
        static_cast<std::size_t>(source_row_offset) * hidden * sizeof(bf16_t);
    void* dst_ptr = static_cast<std::uint8_t*>(dst) +
                    static_cast<std::size_t>(dst_row) * hidden * sizeof(bf16_t);
    const hipError_t err = hipMemcpyAsync(
        dst_ptr, static_cast<const std::uint8_t*>(source.data<void>()) + offset,
        static_cast<std::size_t>(rows) * hidden * sizeof(bf16_t),
        hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("mtp hidden input copy", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status copy_hidden_to_peer_inputs(MtpDrafter& drafter, const gpu::Tensor& source,
                                  uint32_t source_row_offset, uint32_t rows,
                                  hipStream_t stream, uint32_t dst_row) {
    if (rows == 0u || drafter.peers.empty()) return Status::make_ok();
    if (source.ndim() == 0) {
        return Status::invalid_state("mtp peer hidden source is empty", __FILE__, __LINE__);
    }
    const std::size_t hidden = drafter.hidden_width;
    const std::size_t offset =
        static_cast<std::size_t>(source_row_offset) * hidden * sizeof(bf16_t);
    const std::size_t bytes = static_cast<std::size_t>(rows) * hidden * sizeof(bf16_t);
    std::vector<bf16_t> host(bytes / sizeof(bf16_t));
    auto source_scope = shelf::gpu::ScopedDevice::create(drafter.target_device);
    if (!source_scope.ok()) return source_scope.status();
    const hipError_t sync_err = hipStreamSynchronize(stream);
    if (sync_err != hipSuccess) {
        return Status::hip_error("mtp peer hidden sync", hipGetErrorString(sync_err),
                                 __FILE__, __LINE__);
    }
    const hipError_t read_err = hipMemcpy(
        host.data(), static_cast<const std::uint8_t*>(source.data<void>()) + offset,
        bytes, hipMemcpyDeviceToHost);
    if (read_err != hipSuccess) {
        return Status::hip_error("mtp peer hidden read", hipGetErrorString(read_err),
                                 __FILE__, __LINE__);
    }
    for (const MtpDraftPeer& peer : drafter.peers) {
        auto scope = shelf::gpu::ScopedDevice::create(peer.device);
        if (!scope.ok()) return scope.status();
        void* dst = peer.executor->input_hidden.data<void>();
        if (dst == nullptr) {
            return Status::invalid_state("mtp peer has no hidden input buffer", __FILE__,
                                         __LINE__);
        }
        void* dst_ptr = static_cast<std::uint8_t*>(dst) +
                        static_cast<std::size_t>(dst_row) * hidden * sizeof(bf16_t);
        const hipError_t err =
            hipMemcpy(dst_ptr, host.data(), bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            return Status::hip_error("mtp peer hidden input copy", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
    }
    (void)stream;
    return Status::make_ok();
}

Status copy_hidden_to_all(MtpDrafter& drafter, const gpu::Tensor& source,
                          uint32_t source_row_offset, uint32_t rows,
                          hipStream_t stream, uint32_t dst_row = 0u) {
    Status st = copy_hidden_to_input(drafter, source, source_row_offset, rows, stream,
                                     dst_row);
    if (!st.ok()) return st;
    return copy_hidden_to_peer_inputs(drafter, source, source_row_offset, rows, stream,
                                      dst_row);
}

Status cache_target_hidden(MtpDrafter& drafter, const Executor& target,
                           uint32_t row_base, uint32_t rows, uint32_t dst_row,
                           hipStream_t stream) {
    if (rows == 0u) return Status::make_ok();
    if (drafter.hidden_cache.ndim() == 0) {
        return Status::invalid_state("mtp drafter has no hidden cache", __FILE__,
                                     __LINE__);
    }
    if (rows > drafter.hidden_cache.dim(0) ||
        dst_row > drafter.hidden_cache.dim(0) - rows) {
        return Status::out_of_range("mtp hidden cache rows exceeded", __FILE__, __LINE__);
    }
    auto scope = shelf::gpu::ScopedDevice::create(drafter.target_device);
    if (!scope.ok()) return scope.status();
    const std::size_t width = drafter.hidden_width;
    const hipError_t err = hipMemcpyAsync(
        static_cast<std::uint8_t*>(drafter.hidden_cache.data<void>()) +
            static_cast<std::size_t>(dst_row) * width * sizeof(bf16_t),
        static_cast<const std::uint8_t*>(target.normed_token_hidden.data<void>()) +
            static_cast<std::size_t>(row_base) * width * sizeof(bf16_t),
        static_cast<std::size_t>(rows) * width * sizeof(bf16_t),
        hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("mtp hidden cache copy", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status rewind_one_sequence(state::PagedSequenceState& sequence, uint32_t position,
                           uint32_t required_blocks, hipStream_t stream) {
    if (required_blocks < sequence.block_table.size()) {
        Status st = rollback_sequence_append(sequence, required_blocks, stream);
        if (!st.ok()) return st;
    }
    sequence.position = position;
    return Status::make_ok();
}

Status rewind_sequences(MtpDrafter& drafter, uint32_t position,
                        uint32_t required_blocks, hipStream_t stream) {
    Status st = rewind_one_sequence(*drafter.sequence, position, required_blocks, stream);
    if (!st.ok()) return st;
    for (const MtpDraftPeer& peer : drafter.peers) {
        auto scope = shelf::gpu::ScopedDevice::create(peer.device);
        if (!scope.ok()) return scope.status();
        st = rewind_one_sequence(*peer.sequence, position, required_blocks, peer.stream);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

}  // namespace

uint64_t mtp_prefix_key(const std::vector<uint32_t>& tokens, uint32_t position) {
    uint64_t hash = 1469598103934665603ull;
    const uint32_t count = std::min<uint32_t>(position,
                                              static_cast<uint32_t>(tokens.size()));
    for (uint32_t i = 0u; i < count; ++i) {
        hash ^= static_cast<uint64_t>(tokens[i]);
        hash *= 1099511628211ull;
    }
    hash ^= static_cast<uint64_t>(position);
    hash *= 1099511628211ull;
    return hash;
}

namespace {

std::size_t mtp_ring_kv_bytes(const state::PagedKVPool& pool) {
    return static_cast<std::size_t>(pool.page_tokens()) * pool.kv_heads() *
           pool.head_dim() * sizeof(bf16_t);
}

Status save_mtp_ring(const MtpDrafter& drafter, std::vector<uint8_t>& k,
                     std::vector<uint8_t>& v, hipStream_t stream) {
    if (drafter.ring_pages == 0u || drafter.sequence == nullptr) {
        return Status::unsupported("mtp prefix snapshot needs a ring sequence",
                                   __FILE__, __LINE__);
    }
    const state::PagedKVPool& pool = draft_kv_pool(drafter);
    if (drafter.sequence->block_table.size() < drafter.ring_pages) {
        return Status::invalid_state("mtp ring block table is too short", __FILE__,
                                     __LINE__);
    }
    const std::size_t bytes = mtp_ring_kv_bytes(pool);
    k.resize(bytes * drafter.ring_pages);
    v.resize(bytes * drafter.ring_pages);
    const hipError_t sync = hipStreamSynchronize(stream);
    if (sync != hipSuccess) {
        return Status::hip_error("mtp prefix ring sync", hipGetErrorString(sync),
                                 __FILE__, __LINE__);
    }
    for (uint32_t i = 0u; i < drafter.ring_pages; ++i) {
        const state::PageId page = drafter.sequence->block_table[i];
        const auto kv = pool.k_view(0u, page);
        const auto vv = pool.v_view(0u, page);
        const hipError_t kerr = hipMemcpy(k.data() + static_cast<std::size_t>(i) * bytes,
                                          kv.data<void>(), bytes,
                                          hipMemcpyDeviceToHost);
        if (kerr != hipSuccess) {
            return Status::hip_error("mtp prefix ring K", hipGetErrorString(kerr),
                                     __FILE__, __LINE__);
        }
        const hipError_t verr = hipMemcpy(v.data() + static_cast<std::size_t>(i) * bytes,
                                          vv.data<void>(), bytes,
                                          hipMemcpyDeviceToHost);
        if (verr != hipSuccess) {
            return Status::hip_error("mtp prefix ring V", hipGetErrorString(verr),
                                     __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Status load_mtp_ring(const MtpDrafter& drafter, const std::vector<uint8_t>& k,
                     const std::vector<uint8_t>& v, hipStream_t stream) {
    if (drafter.ring_pages == 0u || drafter.sequence == nullptr) {
        return Status::unsupported("mtp prefix snapshot needs a ring sequence",
                                   __FILE__, __LINE__);
    }
    const state::PagedKVPool& pool = draft_kv_pool(drafter);
    const std::size_t bytes = mtp_ring_kv_bytes(pool);
    if (k.size() != bytes * drafter.ring_pages ||
        v.size() != bytes * drafter.ring_pages) {
        return Status::invalid_argument("mtp prefix ring size mismatch", __FILE__,
                                        __LINE__);
    }
    for (uint32_t i = 0u; i < drafter.ring_pages; ++i) {
        const state::PageId page = drafter.sequence->block_table[i];
        const auto kv = pool.k_view(0u, page);
        const auto vv = pool.v_view(0u, page);
        const hipError_t kerr = hipMemcpy(kv.data<void>(),
                                          k.data() + static_cast<std::size_t>(i) * bytes,
                                          bytes, hipMemcpyHostToDevice);
        if (kerr != hipSuccess) {
            return Status::hip_error("mtp prefix restore K", hipGetErrorString(kerr),
                                     __FILE__, __LINE__);
        }
        const hipError_t verr = hipMemcpy(vv.data<void>(),
                                          v.data() + static_cast<std::size_t>(i) * bytes,
                                          bytes, hipMemcpyHostToDevice);
        if (verr != hipSuccess) {
            return Status::hip_error("mtp prefix restore V", hipGetErrorString(verr),
                                     __FILE__, __LINE__);
        }
    }
    const hipError_t sync = hipStreamSynchronize(stream);
    if (sync != hipSuccess) {
        return Status::hip_error("mtp prefix restore sync", hipGetErrorString(sync),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Result<MtpDraftSnapshot> mtp_save_draft_snapshot(const MtpDrafter& drafter,
                                                 uint32_t prefix_position,
                                                 hipStream_t stream) {
    if (!drafter.initialized || drafter.sequence == nullptr) {
        return Status::invalid_state("mtp snapshot: drafter is not initialized",
                                     __FILE__, __LINE__);
    }
    MtpDraftSnapshot snapshot;
    snapshot.prefix_position = prefix_position;
    snapshot.mtp_position = drafter.sequence->position;
    snapshot.last_position = drafter.last_position;
    snapshot.last_tokens = drafter.last_tokens;
    snapshot.sample_index = drafter.sample_index;
    if (drafter.hidden_cache.ndim() != 0u) {
        const std::size_t bytes =
            static_cast<std::size_t>(drafter.hidden_width) * sizeof(bf16_t);
        snapshot.hidden.resize(bytes);
        auto scope = shelf::gpu::ScopedDevice::create(drafter.target_device);
        if (!scope.ok()) return scope.status();
        const hipError_t copy = hipMemcpy(snapshot.hidden.data(),
                                          drafter.hidden_cache.data<void>(), bytes,
                                          hipMemcpyDeviceToHost);
        if (copy != hipSuccess) {
            return Status::hip_error("mtp snapshot hidden", hipGetErrorString(copy),
                                     __FILE__, __LINE__);
        }
    }
    snapshot.ring_k.resize(1u + drafter.peers.size());
    snapshot.ring_v.resize(1u + drafter.peers.size());
    Status st = save_mtp_ring(drafter, snapshot.ring_k[0], snapshot.ring_v[0], stream);
    if (!st.ok()) return st;
    for (std::size_t r = 0u; r < drafter.peers.size(); ++r) {
        const MtpDraftPeer& peer = drafter.peers[r];
        if (peer.owner == nullptr) {
            return Status::invalid_state("mtp snapshot peer is incomplete", __FILE__,
                                         __LINE__);
        }
        auto scope = shelf::gpu::ScopedDevice::create(peer.device);
        if (!scope.ok()) return scope.status();
        st = save_mtp_ring(*peer.owner, snapshot.ring_k[r + 1u],
                           snapshot.ring_v[r + 1u], peer.stream);
        if (!st.ok()) return st;
    }
    return snapshot;
}

Status mtp_restore_draft_snapshot(MtpDrafter& drafter,
                                  const MtpDraftSnapshot& snapshot,
                                  hipStream_t stream) {
    if (!drafter.initialized || drafter.sequence == nullptr) {
        return Status::invalid_state("mtp restore: drafter is not initialized",
                                     __FILE__, __LINE__);
    }
    if (snapshot.ring_k.size() != 1u + drafter.peers.size() ||
        snapshot.ring_v.size() != snapshot.ring_k.size()) {
        return Status::invalid_argument("mtp restore rank count mismatch", __FILE__,
                                        __LINE__);
    }
    drafter.sequence->position = snapshot.mtp_position;
    drafter.last_position = snapshot.last_position;
    drafter.last_tokens = snapshot.last_tokens;
    drafter.sample_index = snapshot.sample_index;
    if (!snapshot.hidden.empty()) {
        const std::size_t bytes =
            static_cast<std::size_t>(drafter.hidden_width) * sizeof(bf16_t);
        if (snapshot.hidden.size() != bytes || drafter.hidden_cache.ndim() == 0u) {
            return Status::invalid_argument("mtp restore hidden size mismatch", __FILE__,
                                            __LINE__);
        }
        auto scope = shelf::gpu::ScopedDevice::create(drafter.target_device);
        if (!scope.ok()) return scope.status();
        const hipError_t copy = hipMemcpy(drafter.hidden_cache.data<void>(),
                                          snapshot.hidden.data(), bytes,
                                          hipMemcpyHostToDevice);
        if (copy != hipSuccess) {
            return Status::hip_error("mtp restore hidden", hipGetErrorString(copy),
                                     __FILE__, __LINE__);
        }
    }
    Status st = load_mtp_ring(drafter, snapshot.ring_k[0], snapshot.ring_v[0], stream);
    if (!st.ok()) return st;
    for (std::size_t r = 0u; r < drafter.peers.size(); ++r) {
        const MtpDraftPeer& peer = drafter.peers[r];
        if (peer.owner == nullptr) {
            return Status::invalid_state("mtp restore peer is incomplete", __FILE__,
                                         __LINE__);
        }
        auto scope = shelf::gpu::ScopedDevice::create(peer.device);
        if (!scope.ok()) return scope.status();
        st = load_mtp_ring(*peer.owner, snapshot.ring_k[r + 1u],
                           snapshot.ring_v[r + 1u], peer.stream);
        if (!st.ok()) return st;
        peer.owner->sequence->position = snapshot.mtp_position;
        peer.owner->last_position = snapshot.last_position;
        peer.owner->last_tokens = snapshot.last_tokens;
        peer.owner->sample_index = snapshot.sample_index;
    }
    return Status::make_ok();
}

namespace {

constexpr uint32_t kDraftWindowBlockTokens = 4u;

bool draft_ring_enabled() noexcept {
    const char* env = std::getenv("SHELF_MTP_RING");
    return env == nullptr || (env[0] != '\0' && std::strcmp(env, "0") != 0);
}

uint32_t draft_ring_pages(const MtpDrafterConfig& options) noexcept {
    if (!draft_ring_enabled()) return 0u;
    if (options.window_tokens == 0u || options.page_tokens == 0u) return 0u;
    const uint64_t tokens = static_cast<uint64_t>(options.sink_tokens) +
                            options.window_tokens + options.num_drafts + 1u;
    const uint32_t ring = static_cast<uint32_t>(ceil_div_u32(
                              static_cast<uint32_t>(tokens), options.page_tokens)) + 1u;
    const uint32_t logical =
        ::shelf::runtime::kv_pages_for_context(options.max_seq_len, options.page_tokens);
    return ring < logical ? ring : logical;
}

uint32_t draft_mask_bytes(uint32_t max_seq_len) noexcept {
    const uint32_t blocks =
        (max_seq_len + 1u + kDraftWindowBlockTokens - 1u) / kDraftWindowBlockTokens;
    return ((blocks + 7u) / 8u + 3u) & ~3u;
}

Status fill_draft_window_mask(MtpDrafter& drafter, uint32_t prefix,
                              uint32_t rows, hipStream_t stream) {
    if (drafter.mask_bytes == 0u) return Status::make_ok();
    void* dst = draft_executor(drafter).input_mask.data<void>();
    if (dst == nullptr || rows == 0u) return Status::make_ok();
    const uint32_t bytes = drafter.mask_bytes;
    if (drafter.window_tokens == 0u) {
        const hipError_t err = hipMemsetAsync(
            dst, 0xFF, static_cast<std::size_t>(rows) * bytes, stream);
        if (err != hipSuccess) {
            return Status::hip_error("mtp window mask clear", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        for (const MtpDraftPeer& peer : drafter.peers) {
            auto scope = shelf::gpu::ScopedDevice::create(peer.device);
            if (!scope.ok()) return scope.status();
            void* peer_dst = peer.executor->input_mask.data<void>();
            if (peer_dst == nullptr) continue;
            const hipError_t peer_err = hipMemsetAsync(
                peer_dst, 0xFF, static_cast<std::size_t>(rows) * bytes, peer.stream);
            if (peer_err != hipSuccess) {
                return Status::hip_error("mtp peer window mask clear",
                                         hipGetErrorString(peer_err), __FILE__, __LINE__);
            }
        }
        return Status::make_ok();
    }
    std::vector<std::uint8_t> host(static_cast<std::size_t>(rows) * bytes, 0u);
    const uint32_t sink_blocks =
        (drafter.sink_tokens + kDraftWindowBlockTokens - 1u) / kDraftWindowBlockTokens;
    for (uint32_t r = 0u; r < rows; ++r) {
        const uint32_t position = prefix + r;
        const uint32_t first_token =
            position + 1u > drafter.window_tokens ? position + 1u - drafter.window_tokens
                                                  : 0u;
        const uint32_t first_block = first_token / kDraftWindowBlockTokens;
        const uint32_t last_block = position / kDraftWindowBlockTokens;
        std::uint8_t* row = host.data() + static_cast<std::size_t>(r) * bytes;
        for (uint32_t b = 0u; b < sink_blocks; ++b) {
            const uint32_t byte = b >> 3u;
            if (byte < bytes) row[byte] |= static_cast<std::uint8_t>(1u << (b & 7u));
        }
        for (uint32_t b = first_block; b <= last_block; ++b) {
            const uint32_t byte = b >> 3u;
            if (byte < bytes) row[byte] |= static_cast<std::uint8_t>(1u << (b & 7u));
        }
    }
    const hipError_t err =
        hipMemcpy(dst, host.data(), host.size(), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        return Status::hip_error("mtp window mask upload", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    for (const MtpDraftPeer& peer : drafter.peers) {
        auto scope = shelf::gpu::ScopedDevice::create(peer.device);
        if (!scope.ok()) return scope.status();
        void* peer_dst = peer.executor->input_mask.data<void>();
        if (peer_dst == nullptr) continue;
        const hipError_t peer_err =
            hipMemcpy(peer_dst, host.data(), host.size(), hipMemcpyHostToDevice);
        if (peer_err != hipSuccess) {
            return Status::hip_error("mtp peer window mask upload",
                                     hipGetErrorString(peer_err), __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Result<BatchExecutionOutput> run_mtp_forward(MtpDrafter& drafter,
                                             std::vector<int32_t>& tokens,
                                             uint32_t prefix,
                                             bool compute_logits,
                                             hipStream_t stream) {
    if (tokens.empty()) {
        return Status::invalid_argument("mtp forward has no tokens", __FILE__, __LINE__);
    }
    if (prefix != drafter.sequence->position) {
        // A windowed drafter skips the prompt spans outside sink+window, so the prefix can
        // jump. The ring block table aliases the physical pages and the mask hides the
        // skipped span.
        if (drafter.window_tokens == 0u || drafter.ring_pages == 0u) {
            return Status::invalid_state("mtp sequence position diverged from prefix",
                                         __FILE__, __LINE__);
        }
        drafter.sequence->position = prefix;
        for (const MtpDraftPeer& peer : drafter.peers) peer.sequence->position = prefix;
    }
    const uint32_t rows = static_cast<uint32_t>(tokens.size());
    const ::shelf::runtime::ExecutionClass execution_class =
        rows == 1u ? ::shelf::runtime::ExecutionClass::DECODE
                   : ::shelf::runtime::ExecutionClass::PREFILL;
    std::vector<ScheduledRequest> requests;
    ScheduledBatch batch = make_batch(
        *drafter.sequence, requests, tokens.data(), rows, prefix, 1u, compute_logits,
        true, execution_class, false, nullptr, 0u);
    Status mask = fill_draft_window_mask(drafter, prefix, rows, stream);
    if (!mask.ok()) return mask;
    if (drafter.peers.empty()) {
        return execute_batch(draft_executor(drafter), batch, stream);
    }
    if (drafter.tp_hook == nullptr) {
        return Status::invalid_state("mtp peers need a tp coordinator", __FILE__,
                                     __LINE__);
    }
    std::vector<::shelf::runtime::TpAuxRank> aux;
    aux.reserve(1u + drafter.peers.size());
    aux.push_back(::shelf::runtime::TpAuxRank{
        &draft_executor(drafter), {drafter.sequence.get()}});
    for (const MtpDraftPeer& peer : drafter.peers) {
        aux.push_back(::shelf::runtime::TpAuxRank{peer.executor, {peer.sequence}});
    }
    return drafter.tp_hook->on_execute_aux(batch, aux, nullptr);
}

MtpGdnRankState rank0_state_view(const MtpDrafter& drafter) {
    MtpGdnRankState state;
    state.pool = drafter.target_pool;
    state.stream = drafter.target_stream;
    state.device = drafter.target_device;
    state.conv_snapshot = drafter.gdn_conv_snapshot;
    state.recurrent_snapshot = drafter.gdn_recurrent_snapshot;
    state.conv_bytes = drafter.gdn_conv_bytes;
    state.recurrent_bytes = drafter.gdn_recurrent_bytes;
    state.history = drafter.gdn_history;
    state.compact_delta = drafter.compact_delta;
    state.compact_k = drafter.compact_k;
    state.compact_a = drafter.compact_a;
    state.compact_delta_layer_stride = drafter.compact_delta_layer_stride;
    state.compact_k_layer_stride = drafter.compact_k_layer_stride;
    state.compact_a_layer_stride = drafter.compact_a_layer_stride;
    state.commit_request = drafter.commit_request;
    state.commit_request_host = drafter.commit_request_host;
    state.gdn_key_heads = drafter.gdn_key_heads;
    state.target = drafter.target_executor;
    state.qsa_partial_history = drafter.qsa_partial_history;
    state.qsa_count_history = drafter.qsa_count_history;
    state.qsa_partial_state_stride = drafter.qsa_partial_state_stride;
    state.qsa_count_state_stride = drafter.qsa_count_state_stride;
    state.qsa_history_rows = drafter.qsa_history_rows;
    state.ple_conv_history = drafter.ple_conv_history;
    state.ple_conv_state_stride = drafter.ple_conv_state_stride;
    state.ple_conv_history_rows = drafter.ple_conv_history_rows;
    return state;
}

std::vector<MtpGdnRankState> all_rank_states(const MtpDrafter& drafter) {
    std::vector<MtpGdnRankState> states;
    states.reserve(1u + drafter.extra_ranks.size());
    states.push_back(rank0_state_view(drafter));
    for (const MtpGdnRankState& state : drafter.extra_ranks) states.push_back(state);
    return states;
}

ExecuteBatchOptions verify_options_for(const MtpGdnRankState& state, uint32_t rows) {
    ExecuteBatchOptions options;
    options.gdn_spec_history = gdn_spec_history_view(state.history, rows);
    options.gdn_compact.delta = state.compact_delta;
    options.gdn_compact.k = state.compact_k;
    options.gdn_compact.a = state.compact_a;
    options.gdn_compact.delta_layer_stride = state.compact_delta_layer_stride;
    options.gdn_compact.k_layer_stride = state.compact_k_layer_stride;
    options.gdn_compact.a_layer_stride = state.compact_a_layer_stride;
    options.qsa_spec_history.partial = state.qsa_partial_history;
    options.qsa_spec_history.counts = state.qsa_count_history;
    options.qsa_spec_history.rows = state.qsa_history_rows;
    options.qsa_spec_history.partial_state_stride = state.qsa_partial_state_stride;
    options.qsa_spec_history.count_state_stride = state.qsa_count_state_stride;
    options.ple_conv_history.store = state.ple_conv_history;
    options.ple_conv_history.rows = state.ple_conv_history_rows;
    options.ple_conv_history.state_stride = state.ple_conv_state_stride;
    return options;
}

std::vector<ExecuteBatchOptions> build_verify_options(const MtpDrafter& drafter,
                                                      uint32_t rows) {
    std::vector<MtpGdnRankState> states = all_rank_states(drafter);
    std::vector<ExecuteBatchOptions> options;
    options.reserve(states.size());
    for (const MtpGdnRankState& state : states) {
        options.push_back(verify_options_for(state, rows));
    }
    return options;
}

Result<BatchExecutionOutput> run_target(MtpDrafter& drafter,
                                        Executor& target,
                                        state::PagedSequenceState& sequence,
                                        std::vector<int32_t>& tokens,
                                        uint32_t prefix,
                                        bool compute_logits,
                                        bool sample,
                                        uint32_t output_rows,
                                        ::shelf::runtime::ExecutionClass execution_class,
                                        bool speculative_verify,
                                        const SamplingConfig& sampling,
                                        uint64_t sampling_index,
                                        const std::vector<ExecuteBatchOptions>* per_rank,
                                        hipStream_t stream) {
    if (tokens.empty()) {
        return Status::invalid_argument("target forward has no tokens", __FILE__, __LINE__);
    }
    if (prefix != sequence.position) {
        return Status::invalid_state("target sequence position diverged from prefix",
                                     __FILE__, __LINE__);
    }
    std::vector<ScheduledRequest> requests;
    ScheduledBatch batch = make_batch(
        sequence, requests, tokens.data(), static_cast<uint32_t>(tokens.size()), prefix,
        output_rows, compute_logits, sample, execution_class, speculative_verify,
        compute_logits ? &sampling : nullptr, sampling_index);
    if (drafter.tp_hook != nullptr) {
        if (per_rank != nullptr && !per_rank->empty()) {
            return drafter.tp_hook->on_execute(batch, per_rank->data());
        }
        return drafter.tp_hook->on_execute(batch);
    }
    if (per_rank != nullptr && !per_rank->empty()) {
        return execute_batch(target, batch, stream, (*per_rank)[0]);
    }
    return execute_batch(target, batch, stream);
}

Status allocate_qsa_history(Executor* target, gpu::GpuArena& arena, uint32_t rows,
                           bf16_t*& partial, std::uint32_t*& counts,
                           std::uint64_t& partial_state_stride,
                           std::uint64_t& count_state_stride) {
    if (target == nullptr || target->qsa_state_pool == nullptr || rows == 0u) {
        return Status::make_ok();
    }
    const auto view = target->qsa_state_pool->device_view();
    if (view.num_qsa_states == 0u || view.compress_ratio == 0u || view.head_dim == 0u) {
        return Status::make_ok();
    }
    partial_state_stride =
        static_cast<std::uint64_t>(rows) * view.compress_ratio * view.head_dim;
    count_state_stride = static_cast<std::uint64_t>(rows) * 2u;
    const std::size_t partial_bytes = static_cast<std::size_t>(view.num_qsa_states) *
        partial_state_stride * sizeof(bf16_t);
    const std::size_t count_bytes = static_cast<std::size_t>(view.num_qsa_states) *
        count_state_stride * sizeof(std::uint32_t);
    auto pv = arena.allocate_aligned(partial_bytes, 256u);
    if (!pv.ok()) return pv.status();
    partial = static_cast<bf16_t*>(pv.release().data());
    auto cv = arena.allocate_aligned(count_bytes, 256u);
    if (!cv.ok()) return cv.status();
    counts = static_cast<std::uint32_t*>(cv.release().data());
    return Status::make_ok();
}

Status restore_rank_qsa(const MtpGdnRankState& state, SequenceSlotId slot, uint32_t row) {
    if (state.target == nullptr || state.target->qsa_state_pool == nullptr ||
        state.qsa_partial_history == nullptr || state.qsa_count_history == nullptr) {
        return Status::make_ok();
    }
    ::shelf::execution::QsaSpecHistoryDeviceView view;
    view.partial = state.qsa_partial_history;
    view.counts = state.qsa_count_history;
    view.rows = state.qsa_history_rows;
    view.partial_state_stride = state.qsa_partial_state_stride;
    view.count_state_stride = state.qsa_count_state_stride;
    auto scope = shelf::gpu::ScopedDevice::create(state.device);
    if (!scope.ok()) return scope.status();
    return restore_qsa_spec_state(*state.target->qsa_state_pool, slot, view, row,
                                  state.stream);
}

Status allocate_ple_conv_history(Executor* target, gpu::GpuArena& arena, uint32_t rows,
                                bf16_t*& store, std::uint64_t& state_stride) {
    if (target == nullptr || target->ple_conv_state_pool == nullptr || rows == 0u) {
        return Status::make_ok();
    }
    const auto view = target->ple_conv_state_pool->device_view();
    if (view.channels == 0u || view.history_stride == 0u) return Status::make_ok();
    const std::size_t history_elems = static_cast<std::size_t>(view.history_stride) *
                                      ::shelf::kernel::kPleConvHistory;
    state_stride = static_cast<std::uint64_t>(rows) * history_elems;
    const std::size_t bytes = static_cast<std::size_t>(
        target->ple_conv_state_pool->num_ple_states()) * state_stride * sizeof(bf16_t);
    auto view_alloc = arena.allocate_aligned(bytes, 256u);
    if (!view_alloc.ok()) return view_alloc.status();
    store = static_cast<bf16_t*>(view_alloc.release().data());
    return Status::make_ok();
}

Status restore_rank_ple(const MtpGdnRankState& state, SequenceSlotId slot, uint32_t row) {
    if (state.target == nullptr || state.target->ple_conv_state_pool == nullptr ||
        state.ple_conv_history == nullptr) {
        return Status::make_ok();
    }
    ::shelf::execution::PleConvSpecHistoryDeviceView view;
    view.store = state.ple_conv_history;
    view.rows = state.ple_conv_history_rows;
    view.state_stride = state.ple_conv_state_stride;
    auto scope = shelf::gpu::ScopedDevice::create(state.device);
    if (!scope.ok()) return scope.status();
    return restore_ple_conv_spec_state(*state.target->ple_conv_state_pool, slot, view,
                                       row, state.stream);
}

}  // namespace

std::size_t mtp_drafter_reserve_bytes(const shelf::qwen_hybrid::QwenHybridConfig& config,
                                      const MtpDrafterConfig& options,
                                      std::size_t token_row_intermediate,
                                      std::size_t grouped_executor_bytes,
                                      bool exl3_linear_workspace,
                                      const shelf::weights::Exl3LinearShape&
                                          exl3_linear_shape,
                                      const state::GdnStatePoolLayout* target_gdn) noexcept {
    if (options.max_seq_len == 0u || options.page_tokens == 0u) {
        return 0u;
    }
    shelf::qwen_hybrid::QwenHybridConfig mtp = mtp_executor_config(config);
    if (options.local_kv_heads != 0u) mtp.num_key_value_heads = options.local_kv_heads;
    const uint32_t full_pages =
        ::shelf::runtime::kv_pages_for_context(options.max_seq_len, options.page_tokens);
    const uint32_t ring = draft_ring_pages(options);
    const uint32_t pages = ring != 0u ? ring : full_pages;
    const std::size_t kv = state::PagedKVPool::page_bytes(
        options.page_tokens, kMtpKvLayers,
        static_cast<uint32_t>(mtp.num_key_value_heads),
        static_cast<uint32_t>(mtp.attention_head_dim), options.kv_dtype) *
        static_cast<std::size_t>(pages);

    ExecutorConfig executor_config;
    executor_config.max_batch_rows = mtp_drafter_rows(options);
    executor_config.max_scheduled_requests = 1u;
    executor_config.max_scheduled_output_rows = 1u;
    executor_config.external_hidden_input = true;
    executor_config.external_mask_bytes = draft_mask_bytes(options.max_seq_len);
    executor_config.exl3_linear_shape = exl3_linear_shape;
    const std::size_t executor = executor_reserve_bytes(
        mtp, token_row_intermediate, executor_config, 1u, exl3_linear_workspace) +
        grouped_executor_bytes;

    const state::GdnStatePoolLayout gdn =
        target_gdn != nullptr ? *target_gdn : make_gdn_state_layout(config);
    std::size_t history = 0u;
    std::size_t compact = 0u;
    if (options.allocate_spec_buffers) {
        history = static_cast<std::size_t>(options.num_drafts) * gdn.num_gdn_states *
            gdn.conv_history_stride * gdn.conv_history * sizeof(bf16_t);
        const std::uint64_t log_rows = static_cast<std::uint64_t>(options.num_drafts) + 1u;
        const std::uint64_t vdim =
            static_cast<std::uint64_t>(gdn.num_v_heads) * gdn.head_v;
        const std::uint64_t qkdim =
            static_cast<std::uint64_t>(gdn.num_v_heads) * gdn.head_k;
        compact = static_cast<std::size_t>(gdn.num_gdn_states) * log_rows *
            (vdim + qkdim + gdn.num_v_heads) * sizeof(float);
    }
    constexpr std::size_t kMargin = 64ull * 1024ull * 1024ull;
    return kv + executor + history + compact + kMargin;
}

std::size_t mtp_shared_draft_reserve_bytes(
    const shelf::qwen_hybrid::QwenHybridConfig& config,
    const MtpDrafterConfig& options,
    std::size_t token_row_intermediate,
    std::size_t grouped_executor_bytes,
    bool exl3_linear_workspace,
    const shelf::weights::Exl3LinearShape& exl3_linear_shape,
    std::uint32_t max_requests) noexcept {
    if (options.max_seq_len == 0u || options.page_tokens == 0u ||
        max_requests == 0u) {
        return 0u;
    }
    shelf::qwen_hybrid::QwenHybridConfig mtp = mtp_executor_config(config);
    if (options.local_kv_heads != 0u) mtp.num_key_value_heads = options.local_kv_heads;
    const uint32_t full_pages =
        ::shelf::runtime::kv_pages_for_context(options.max_seq_len, options.page_tokens);
    const uint32_t ring = draft_ring_pages(options);
    const uint32_t pages = ring != 0u ? ring : full_pages;
    const std::size_t kv = state::PagedKVPool::page_bytes(
        options.page_tokens, kMtpKvLayers,
        static_cast<uint32_t>(mtp.num_key_value_heads),
        static_cast<uint32_t>(mtp.attention_head_dim), options.kv_dtype) *
        static_cast<std::size_t>(pages) * static_cast<std::size_t>(max_requests);

    ExecutorConfig executor_config;
    const uint32_t batch_rows = max_requests * (options.num_drafts + 1u);
    executor_config.max_batch_rows =
        options.max_batch_rows > batch_rows ? options.max_batch_rows : batch_rows;
    executor_config.max_scheduled_requests = max_requests;
    executor_config.max_scheduled_output_rows = max_requests;
    executor_config.external_hidden_input = true;
    executor_config.exl3_linear_shape = exl3_linear_shape;
    const std::size_t executor = executor_reserve_bytes(
        mtp, token_row_intermediate, executor_config, 1u, exl3_linear_workspace) +
        grouped_executor_bytes;

    constexpr std::size_t kMargin = 64ull * 1024ull * 1024ull;
    return kv + executor + kMargin;
}

Result<std::unique_ptr<MtpDrafter>> create_mtp_drafter(
    const MtpDrafterInput& input,
    Executor* target,
    const state::GdnStatePool& target_gdn_pool,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const MtpDrafterConfig& options,
    const std::vector<MtpGdnRankBinding>* extra_ranks,
    TpBatchHook* tp_hook,
    MtpDraftShared* shared_draft) {

    if (input.config == nullptr || input.embed_tokens == nullptr) {
        return Status::invalid_argument("mtp drafter input is incomplete", __FILE__,
                                        __LINE__);
    }
    if (input.lowered.graph.external_inputs.size() != 2u &&
        input.lowered.graph.external_inputs.size() != 3u) {
        return Status::invalid_argument(
            "the drafter graph takes a token id, the previous hidden and an optional "
            "window mask",
            __FILE__, __LINE__);
    }
    if (options.num_drafts == 0u || options.num_drafts > kMtpMaxDrafts) {
        return Status::invalid_argument("mtp num_drafts is out of range", __FILE__,
                                        __LINE__);
    }
    if (options.max_seq_len == 0u) {
        return Status::invalid_argument("mtp max_seq_len must be positive", __FILE__,
                                        __LINE__);
    }
    if (options.page_tokens == 0u) {
        return Status::invalid_argument("mtp page_tokens must be positive", __FILE__,
                                        __LINE__);
    }
    if (options.max_batch_rows == 0u) {
        return Status::invalid_argument("mtp max_batch_rows must be positive", __FILE__,
                                        __LINE__);
    }

    std::unique_ptr<MtpDrafter> drafter(new MtpDrafter());
    drafter->target_executor = target;
    drafter->config = mtp_executor_config(*input.config);
    if (options.local_kv_heads != 0u)
        drafter->config.num_key_value_heads = options.local_kv_heads;
    drafter->max_seq_len = options.max_seq_len;
    drafter->num_drafts = options.num_drafts;
    drafter->hidden_width = input.hidden_width != 0u
        ? input.hidden_width
        : static_cast<std::uint32_t>(input.config->hidden_size);
    drafter->wide_hidden = input.wide_hidden;
    drafter->target_pool = &target_gdn_pool;
    drafter->target_stream = stream;
    drafter->serial_verify =
        (target->qsa_state_pool != nullptr || target->ple_conv_state_pool != nullptr) &&
        !mtp_fused_verify_enabled();
    const hipError_t device_err = hipGetDevice(&drafter->target_device);
    if (device_err != hipSuccess) {
        return Status::hip_error("hipGetDevice", hipGetErrorString(device_err), __FILE__,
                                 __LINE__);
    }
    drafter->tp_hook = tp_hook;
    drafter->shared_draft = shared_draft;
    drafter->window_tokens = options.window_tokens;
    drafter->sink_tokens = options.sink_tokens;
    drafter->mask_bytes = draft_mask_bytes(options.max_seq_len);
    drafter->ring_pages = draft_ring_pages(options);

    const uint32_t pages =
        ::shelf::runtime::kv_pages_for_context(options.max_seq_len, options.page_tokens);
    const uint32_t ring_pages = draft_ring_pages(options);
    if (shared_draft == nullptr) {
        auto slot_pool = state::SequenceSlotPool::create(arena, 1u, pages);
        if (!slot_pool.ok()) return slot_pool.status();
        drafter->slot_pool =
            std::make_unique<state::SequenceSlotPool>(slot_pool.release());

        const state::GdnStatePoolLayout gdn_layout =
            make_gdn_state_layout(drafter->config);
        auto gdn_pool = state::GdnStatePool::create(arena, 1u, gdn_layout);
        if (!gdn_pool.ok()) return gdn_pool.status();
        drafter->gdn_pool = std::make_unique<state::GdnStatePool>(gdn_pool.release());

        auto kv_pool = state::PagedKVPool::create(
            arena, ring_pages != 0u ? ring_pages : pages, options.page_tokens,
            kMtpKvLayers, static_cast<uint32_t>(drafter->config.num_key_value_heads),
            static_cast<uint32_t>(drafter->config.attention_head_dim), options.kv_dtype);
        if (!kv_pool.ok()) return kv_pool.status();
        drafter->kv_pool = std::make_unique<state::PagedKVPool>(kv_pool.release());
    }

    auto sequence = state::create_paged_sequence_state(
        draft_slot_pool(*drafter), draft_gdn_pool(*drafter), draft_kv_pool(*drafter),
        options.max_seq_len, stream, ring_pages);
    if (!sequence.ok()) return sequence.status();
    drafter->sequence =
        std::make_unique<state::PagedSequenceState>(sequence.release());

    if (shared_draft == nullptr) {
        ExecutorModelInput model_input;
        model_input.model.config = &drafter->config;
        model_input.model.embed_tokens = input.embed_tokens;
        model_input.lowered = input.lowered;

        ExecutorConfig executor_config;
        executor_config.max_batch_rows = mtp_drafter_rows(options);
        executor_config.max_scheduled_requests = 1u;
        executor_config.max_scheduled_output_rows = 1u;
        executor_config.external_hidden_input = true;
        executor_config.external_hidden_width = drafter->hidden_width;
        executor_config.external_mask_bytes = drafter->mask_bytes;
        executor_config.exl3_linear_shape = input.exl3_linear_shape;
        executor_config.emit_normed_token_hidden = drafter->wide_hidden;
        executor_config.normed_token_hidden_width = drafter->hidden_width;
        executor_config.rope_profile = options.rope_profile;
        executor_config.backend = ::shelf::runtime::DecodeBackend::Hip;

        auto executor = create_executor(std::move(model_input), *drafter->slot_pool,
                                        *drafter->gdn_pool, *drafter->kv_pool, arena,
                                        executor_config);
        if (!executor.ok()) {
            Status st = state::release_paged_sequence_state(*drafter->sequence, stream);
            (void)st;
            return executor.status();
        }
        drafter->executor = executor.release();
    }

    auto fail_with_sequence = [&](const Status& status) {
        Status st = shutdown_gdn_spec_history(drafter->gdn_history);
        (void)st;
        if (shared_draft == nullptr) {
            st = executor_shutdown(drafter->executor);
            (void)st;
        }
        st = state::release_paged_sequence_state(*drafter->sequence, stream);
        (void)st;
        return status;
    };
    {
        const std::uint32_t cache_rows = options.num_drafts + 1u;
        auto cache = arena.allocate_aligned(
            static_cast<std::size_t>(cache_rows) * drafter->hidden_width *
                sizeof(bf16_t),
            256u);
        if (!cache.ok()) return fail_with_sequence(cache.status());
        auto view = cache.release();
        auto tensor = gpu::Tensor::create<bf16_t>(
            view, {cache_rows, static_cast<std::size_t>(drafter->hidden_width)},
            {static_cast<std::size_t>(drafter->hidden_width), 1u});
        if (!tensor.ok()) return fail_with_sequence(tensor.status());
        drafter->hidden_cache = tensor.release();
    }

    if (options.allocate_spec_buffers) {
        Status qsa = allocate_qsa_history(
            target, arena, options.num_drafts + 1u, drafter->qsa_partial_history,
            drafter->qsa_count_history, drafter->qsa_partial_state_stride,
            drafter->qsa_count_state_stride);
        if (!qsa.ok()) return fail_with_sequence(qsa);
        drafter->qsa_history_rows = options.num_drafts + 1u;
        Status ple = allocate_ple_conv_history(
            target, arena, options.num_drafts + 1u, drafter->ple_conv_history,
            drafter->ple_conv_state_stride);
        if (!ple.ok()) return fail_with_sequence(ple);
        drafter->ple_conv_history_rows = options.num_drafts + 1u;
    }

    drafter->gdn_conv_bytes = spec_gdn_conv_bytes(target_gdn_pool);
    drafter->gdn_recurrent_bytes = spec_gdn_recurrent_bytes(target_gdn_pool);
    if (!drafter->serial_verify && drafter->gdn_conv_bytes != 0u) {
        auto view = arena.allocate_aligned(drafter->gdn_conv_bytes, 256u);
        if (!view.ok()) return fail_with_sequence(view.status());
        drafter->gdn_conv_snapshot = view.release().data();
    }
    if (!drafter->serial_verify && drafter->gdn_recurrent_bytes != 0u) {
        auto view = arena.allocate_aligned(drafter->gdn_recurrent_bytes, 256u);
        if (!view.ok()) return fail_with_sequence(view.status());
        drafter->gdn_recurrent_snapshot = view.release().data();
    }

    const GdnStatePoolDeviceView pool = target_gdn_pool.device_view();
    if (options.allocate_spec_buffers && !drafter->serial_verify) {
        auto history = create_gdn_spec_history(arena, target_gdn_pool,
                                               options.num_drafts, false);
        if (!history.ok()) return fail_with_sequence(history.status());
        drafter->gdn_history = history.release();

        const std::uint64_t log_rows = static_cast<std::uint64_t>(options.num_drafts) + 1u;
        const std::uint64_t vdim =
            static_cast<std::uint64_t>(pool.num_v_heads) * pool.head_v;
        const std::uint64_t qkdim =
            static_cast<std::uint64_t>(pool.num_v_heads) * pool.head_k;
        drafter->compact_delta_layer_stride = log_rows * vdim;
        drafter->compact_k_layer_stride = log_rows * qkdim;
        drafter->compact_a_layer_stride = log_rows * pool.num_v_heads;
        const std::size_t delta_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
            drafter->compact_delta_layer_stride * sizeof(float);
        const std::size_t k_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
            drafter->compact_k_layer_stride * sizeof(float);
        const std::size_t a_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
            drafter->compact_a_layer_stride * sizeof(float);
        {
            auto view = arena.allocate_aligned(delta_bytes, 256u);
            if (!view.ok()) return fail_with_sequence(view.status());
            drafter->compact_delta = static_cast<float*>(view.release().data());
        }
        {
            auto view = arena.allocate_aligned(k_bytes, 256u);
            if (!view.ok()) return fail_with_sequence(view.status());
            drafter->compact_k = static_cast<float*>(view.release().data());
        }
        {
            auto view = arena.allocate_aligned(a_bytes, 256u);
            if (!view.ok()) return fail_with_sequence(view.status());
            drafter->compact_a = static_cast<float*>(view.release().data());
        }
    }
    {
        auto view = arena.allocate_aligned(
            sizeof(::shelf::runtime::DeviceRequestDescriptor), 256u);
        if (!view.ok()) return fail_with_sequence(view.status());
        drafter->commit_request =
            static_cast<::shelf::runtime::DeviceRequestDescriptor*>(view.release().data());
    }
    drafter->commit_request_host.row_begin = 0u;
    drafter->commit_request_host.row_count = 0u;
    drafter->gdn_key_heads = static_cast<std::uint32_t>(
        input.config->linear_num_key_heads != 0u ? input.config->linear_num_key_heads : 16u);

    if (extra_ranks != nullptr && !drafter->serial_verify) {
        for (const MtpGdnRankBinding& binding : *extra_ranks) {
            if (binding.pool == nullptr || binding.arena == nullptr) {
                return fail_with_sequence(Status::invalid_argument(
                    "mtp extra rank binding is incomplete", __FILE__, __LINE__));
            }
            auto scope = shelf::gpu::ScopedDevice::create(binding.device);
            if (!scope.ok()) return fail_with_sequence(scope.status());
            MtpGdnRankState state;
            state.pool = binding.pool;
            state.stream = binding.stream;
            state.device = binding.device;
            state.target = binding.executor;
            if (options.allocate_spec_buffers) {
                Status qsa = allocate_qsa_history(
                    binding.executor, *binding.arena, options.num_drafts + 1u,
                    state.qsa_partial_history, state.qsa_count_history,
                    state.qsa_partial_state_stride, state.qsa_count_state_stride);
                if (!qsa.ok()) return fail_with_sequence(qsa);
                state.qsa_history_rows = options.num_drafts + 1u;
                Status ple = allocate_ple_conv_history(
                    binding.executor, *binding.arena, options.num_drafts + 1u,
                    state.ple_conv_history, state.ple_conv_state_stride);
                if (!ple.ok()) return fail_with_sequence(ple);
                state.ple_conv_history_rows = options.num_drafts + 1u;
            }
            state.conv_bytes = spec_gdn_conv_bytes(*binding.pool);
            state.recurrent_bytes = spec_gdn_recurrent_bytes(*binding.pool);
            if (state.conv_bytes != 0u) {
                auto view = binding.arena->allocate_aligned(state.conv_bytes, 256u);
                if (!view.ok()) return fail_with_sequence(view.status());
                state.conv_snapshot = view.release().data();
            }
            if (state.recurrent_bytes != 0u) {
                auto view = binding.arena->allocate_aligned(state.recurrent_bytes, 256u);
                if (!view.ok()) return fail_with_sequence(view.status());
                state.recurrent_snapshot = view.release().data();
            }
            const auto pool = binding.pool->device_view();
            if (!options.allocate_spec_buffers) {
                auto request = binding.arena->allocate_aligned(
                    sizeof(::shelf::runtime::DeviceRequestDescriptor), 256u);
                if (!request.ok()) return fail_with_sequence(request.status());
                state.commit_request = static_cast<::shelf::runtime::DeviceRequestDescriptor*>(
                    request.release().data());
                state.commit_request_host.row_begin = 0u;
                state.commit_request_host.row_count = 0u;
                state.gdn_key_heads = drafter->gdn_key_heads;
                drafter->extra_ranks.push_back(std::move(state));
                continue;
            }
            auto history = create_gdn_spec_history(*binding.arena, *binding.pool,
                                                   options.num_drafts, false);
            if (!history.ok()) return fail_with_sequence(history.status());
            state.history = history.release();
            const std::uint64_t log_rows = static_cast<std::uint64_t>(options.num_drafts) + 1u;
            const std::uint64_t vdim =
                static_cast<std::uint64_t>(pool.num_v_heads) * pool.head_v;
            const std::uint64_t qkdim =
                static_cast<std::uint64_t>(pool.num_v_heads) * pool.head_k;
            state.compact_delta_layer_stride = log_rows * vdim;
            state.compact_k_layer_stride = log_rows * qkdim;
            state.compact_a_layer_stride = log_rows * pool.num_v_heads;
            const std::size_t delta_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
                state.compact_delta_layer_stride * sizeof(float);
            const std::size_t k_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
                state.compact_k_layer_stride * sizeof(float);
            const std::size_t a_bytes = static_cast<std::size_t>(pool.num_gdn_states) *
                state.compact_a_layer_stride * sizeof(float);
            auto delta = binding.arena->allocate_aligned(delta_bytes, 256u);
            if (!delta.ok()) return fail_with_sequence(delta.status());
            state.compact_delta = static_cast<float*>(delta.release().data());
            auto kb = binding.arena->allocate_aligned(k_bytes, 256u);
            if (!kb.ok()) return fail_with_sequence(kb.status());
            state.compact_k = static_cast<float*>(kb.release().data());
            auto ab = binding.arena->allocate_aligned(a_bytes, 256u);
            if (!ab.ok()) return fail_with_sequence(ab.status());
            state.compact_a = static_cast<float*>(ab.release().data());
            auto request = binding.arena->allocate_aligned(
                sizeof(::shelf::runtime::DeviceRequestDescriptor), 256u);
            if (!request.ok()) return fail_with_sequence(request.status());
            state.commit_request =
                static_cast<::shelf::runtime::DeviceRequestDescriptor*>(request.release().data());
            state.commit_request_host.row_begin = 0u;
            state.commit_request_host.row_count = 0u;
            state.gdn_key_heads = drafter->gdn_key_heads;
            drafter->extra_ranks.push_back(std::move(state));
        }
    }

    drafter->last_tokens.reserve(options.num_drafts + 1u);
    drafter->initialized = true;
    return drafter;
}

Status mtp_drafter_shutdown(MtpDrafter& drafter, hipStream_t stream) noexcept {
    Status first_error = Status::make_ok();
    if (drafter.sequence != nullptr) {
        Status st = state::release_paged_sequence_state(*drafter.sequence,
                                                        stream);
        if (first_error.ok() && !st.ok()) first_error = st;
        drafter.sequence.reset();
    }
    Status st = shutdown_gdn_spec_history(drafter.gdn_history);
    if (first_error.ok() && !st.ok()) first_error = st;
    if (drafter.shared_draft == nullptr) {
        st = executor_shutdown(drafter.executor);
        if (first_error.ok() && !st.ok()) first_error = st;
        drafter.kv_pool.reset();
        drafter.gdn_pool.reset();
        drafter.slot_pool.reset();
    }
    drafter.shared_draft = nullptr;
    drafter.hidden_cache = gpu::Tensor{};
    drafter.gdn_conv_snapshot = nullptr;
    drafter.gdn_recurrent_snapshot = nullptr;
    drafter.compact_delta = nullptr;
    drafter.compact_k = nullptr;
    drafter.compact_a = nullptr;
    drafter.commit_request = nullptr;
    drafter.qsa_partial_history = nullptr;
    drafter.qsa_count_history = nullptr;
    drafter.qsa_history_rows = 0u;
    drafter.ple_conv_history = nullptr;
    drafter.ple_conv_state_stride = 0u;
    drafter.ple_conv_history_rows = 0u;
    drafter.initialized = false;
    return first_error;
}

Status mtp_drafter_reset_sequence(MtpDrafter& drafter, hipStream_t stream) noexcept {
    if (!drafter.initialized || drafter.sequence == nullptr) {
        return Status::invalid_state("mtp reset: drafter is not initialized", __FILE__,
                                     __LINE__);
    }
    auto reset_one = [](MtpDrafter& target, hipStream_t target_stream) -> Status {
        Status st =
            state::release_paged_sequence_state(*target.sequence, target_stream);
        if (!st.ok()) return st;
        auto sequence = state::create_paged_sequence_state(
            draft_slot_pool(target), draft_gdn_pool(target), draft_kv_pool(target),
            target.max_seq_len, target_stream, target.ring_pages);
        if (!sequence.ok()) {
            target.sequence.reset();
            return sequence.status();
        }
        target.sequence =
            std::make_unique<state::PagedSequenceState>(sequence.release());
        target.last_tokens.clear();
        target.last_position = 0u;
        target.last_hidden_offset = 0u;
        target.sample_index = 0u;
        target.timing = MtpTiming{};
        return Status::make_ok();
    };
    Status st = reset_one(drafter, stream);
    if (!st.ok()) return st;
    for (MtpDraftPeer& peer : drafter.peers) {
        if (peer.owner == nullptr) continue;
        auto scope = shelf::gpu::ScopedDevice::create(peer.device);
        if (!scope.ok()) return scope.status();
        st = reset_one(*peer.owner, peer.stream);
        if (!st.ok()) return st;
        peer.sequence = peer.owner->sequence.get();
    }
    return Status::make_ok();
}

// The draft attention only reads the attention sink and the last window of the prompt, so
// the MTP block's K/V for the middle of a long prompt is never read. Skip those chunks.
bool mtp_prefill_needed(const MtpDrafter& drafter, uint32_t chunk_begin, uint32_t chunk_end,
                        uint32_t prompt_count) {
    if (drafter.window_tokens == 0u) return true;
    const uint32_t window_begin =
        prompt_count > drafter.window_tokens ? prompt_count - drafter.window_tokens : 0u;
    return chunk_begin < drafter.sink_tokens || chunk_end > window_begin;
}

Result<MtpPrefillOutput> mtp_spec_prefill(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    const int32_t* prompt_tokens,
    uint32_t prompt_count,
    const SamplingConfig& sampling,
    const MtpPrefillProgress* progress,
    const MtpPrefillBoundary* boundary,
    uint32_t start_position,
    uint32_t checkpoint_position) {

    if (!drafter.initialized) {
        return Status::invalid_state("mtp_spec_prefill: drafter is not initialized",
                                     __FILE__, __LINE__);
    }
    if (prompt_tokens == nullptr || prompt_count == 0u) {
        return Status::invalid_argument("mtp_spec_prefill: empty prompt", __FILE__,
                                        __LINE__);
    }
    if (prompt_count >= drafter.max_seq_len) {
        return Status::out_of_range("mtp_spec_prefill: prompt leaves no room", __FILE__,
                                    __LINE__);
    }
    if (start_position >= prompt_count) {
        return Status::invalid_argument("mtp_spec_prefill: empty suffix", __FILE__,
                                        __LINE__);
    }
    if (target_sequence.position != start_position) {
        return Status::invalid_state("mtp_spec_prefill: target position is not the start",
                                     __FILE__, __LINE__);
    }
    if (start_position == 0u && drafter.sequence->position != 0u) {
        return Status::invalid_state("mtp_spec_prefill: drafter sequence is not empty",
                                     __FILE__, __LINE__);
    }

    const hipStream_t stream = drafter.target_stream;
    if (start_position > 0u && drafter.sequence->position + 1u == start_position) {
        std::vector<int32_t> seed_token{prompt_tokens[start_position]};
        Status copy = copy_hidden_to_all(drafter, drafter.hidden_cache, 0u, 1u, stream);
        if (!copy.ok()) return copy;
        auto seeded = run_mtp_forward(drafter, seed_token, start_position - 1u, false,
                                      stream);
        if (!seeded.ok()) return seeded.status();
    }
    if (drafter.sequence->position != start_position) {
        return Status::invalid_state(
            ("mtp_spec_prefill: drafter position is not the start (" +
             std::to_string(drafter.sequence->position) + " != " +
             std::to_string(start_position) + ")")
                .c_str(),
            __FILE__, __LINE__);
    }
    const uint32_t chunk_limit =
        target.config.max_batch_rows != 0u ? target.config.max_batch_rows : prompt_count;
    const uint32_t mtp_limit = draft_executor(drafter).config.max_batch_rows;
    drafter.sampling = sampling;
    int32_t pending = -1;
    std::vector<int32_t> chunk_tokens;
    std::vector<int32_t> mtp_tokens;
    const auto prefill_started = std::chrono::steady_clock::now();
    auto report_progress = [&](uint32_t processed_tokens) {
        if (progress == nullptr || progress->report == nullptr) return;
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - prefill_started)
                                      .count();
        progress->report(progress->context, processed_tokens, elapsed_ms);
    };
    auto save_boundary = [&](uint32_t position) -> Status {
        if (boundary == nullptr || boundary->save == nullptr) return Status::make_ok();
        return boundary->save(boundary->context, position, stream);
    };

    for (uint32_t begin = start_position; begin < prompt_count;) {
        if (progress != nullptr && progress->should_stop != nullptr &&
            progress->should_stop(progress->stop_context)) {
            return Status::invalid_state("mtp prefill cancelled", __FILE__, __LINE__);
        }
        uint32_t end = std::min(prompt_count, begin + chunk_limit);
        if (checkpoint_position > begin && checkpoint_position < end)
            end = checkpoint_position;
        const uint32_t rows = end - begin;
        const bool final_chunk = end == prompt_count;
        chunk_tokens.assign(prompt_tokens + begin, prompt_tokens + end);
        auto target_output = run_target(
            drafter, target, target_sequence, chunk_tokens, begin, final_chunk,
            final_chunk, 1u, ::shelf::runtime::ExecutionClass::PREFILL, false, sampling,
            drafter.sample_index, nullptr, stream);
        if (!target_output.ok()) return target_output.status();
        if (final_chunk) {
            auto sampled = copy_sampled_token(target_output.value().sampled_tokens, stream);
            if (!sampled.ok()) return sampled.status();
            pending = sampled.value();
            drafter.sample_index += 1u;
            const uint32_t last_row = end - 1u;
            const uint32_t build_end = rows > 1u ? last_row : begin;
            for (uint32_t mtp_begin = begin; mtp_begin < build_end; mtp_begin += mtp_limit) {
                const uint32_t mtp_end = std::min(build_end, mtp_begin + mtp_limit);
                if (!mtp_prefill_needed(drafter, mtp_begin, mtp_end, prompt_count)) continue;
                const uint32_t mtp_rows = mtp_end - mtp_begin;
                mtp_tokens.resize(mtp_rows);
                for (uint32_t i = 0u; i < mtp_rows; ++i) {
                    mtp_tokens[i] = prompt_tokens[mtp_begin + i + 1u];
                }
                Status copy = copy_hidden_to_all(
                    drafter, target.normed_token_hidden, mtp_begin - begin, mtp_rows, stream);
                if (!copy.ok()) return copy;
                auto mtp_output = run_mtp_forward(drafter, mtp_tokens, mtp_begin, false,
                                                  stream);
                if (!mtp_output.ok()) return mtp_output.status();
            }
            Status cache = cache_target_hidden(drafter, target, last_row - begin, 1u, 0u,
                                               stream);
            if (!cache.ok()) return cache;
            drafter.last_tokens.assign(1u, prompt_tokens[last_row]);
            drafter.last_position = last_row;
            drafter.last_hidden_offset = 0u;
            Status saved = save_boundary(end);
            if (!saved.ok()) return saved;
            report_progress(end);
            begin = end;
            continue;
        }

        for (uint32_t mtp_begin = begin; mtp_begin < end; mtp_begin += mtp_limit) {
            const uint32_t mtp_end = std::min(end, mtp_begin + mtp_limit);
            if (!mtp_prefill_needed(drafter, mtp_begin, mtp_end, prompt_count)) continue;
            const uint32_t mtp_rows = mtp_end - mtp_begin;
            mtp_tokens.resize(mtp_rows);
            for (uint32_t i = 0u; i < mtp_rows; ++i) {
                mtp_tokens[i] = prompt_tokens[mtp_begin + i + 1u];
            }
            Status copy = copy_hidden_to_all(
                drafter, target.normed_token_hidden, mtp_begin - begin, mtp_rows, stream);
            if (!copy.ok()) return copy;
            auto mtp_output = run_mtp_forward(drafter, mtp_tokens, mtp_begin, false,
                                              stream);
            if (!mtp_output.ok()) return mtp_output.status();
        }
        Status saved = save_boundary(end);
        if (!saved.ok()) return saved;
        report_progress(end);
        begin = end;
    }

    MtpPrefillOutput output;
    output.pending_token = pending;
    return output;
}

namespace {

Result<MtpIterationOutput> mtp_decode_only(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& sequence,
    int32_t pending_token,
    uint32_t remaining_tokens,
    hipStream_t stream) {
    (void)remaining_tokens;
    const uint32_t position = sequence.position;
    std::vector<int32_t> tokens{pending_token};
    auto executed = run_target(
        drafter, target, sequence, tokens, position, true, true, 1u,
        ::shelf::runtime::ExecutionClass::DECODE, false, drafter.sampling,
        drafter.sample_index, nullptr, stream);
    if (!executed.ok()) return executed.status();
    drafter.sample_index += 1u;
    auto sampled = copy_sampled_token(executed.value().sampled_tokens, stream);
    if (!sampled.ok()) return sampled.status();
    Status cache = cache_target_hidden(drafter, target, 0u, 1u, 0u, stream);
    if (!cache.ok()) return cache;
    drafter.last_tokens.assign(1u, pending_token);
    drafter.last_position = position;
    drafter.last_hidden_offset = 0u;
    MtpIterationOutput output;
    output.emitted[0] = sampled.value();
    output.emitted_count = 1u;
    output.pending_token = sampled.value();
    output.finished = qwen_hybrid::is_stop_token(target.model.config->stop_tokens,
                                                 sampled.value());
    return output;
}

Result<MtpIterationOutput> mtp_verify_serial(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& sequence,
    int32_t pending_token,
    uint32_t remaining_tokens,
    const MtpDraftResult& plan,
    hipStream_t stream) {
    const uint32_t drafts = plan.drafts;
    std::vector<int32_t> verify_tokens(drafts + 1u);
    verify_tokens[0] = pending_token;
    for (uint32_t i = 0u; i < drafts; ++i) verify_tokens[i + 1u] = plan.proposals[i];

    MtpIterationOutput output;
    output.pending_token = pending_token;
    uint32_t accepted = 0u;
    uint32_t emitted = 0u;
    uint32_t committed = 0u;
    int32_t pending = pending_token;
    bool finished = false;
    for (uint32_t i = 0u; i <= drafts; ++i) {
        std::vector<int32_t> row{verify_tokens[i]};
        auto step = run_target(
            drafter, target, sequence, row, plan.position + i, true, true, 1u,
            ::shelf::runtime::ExecutionClass::DECODE, false, drafter.sampling,
            drafter.sample_index + i, nullptr, stream);
        if (!step.ok()) return step.status();
        Status cache = cache_target_hidden(drafter, target, 0u, 1u, i, stream);
        if (!cache.ok()) return cache;
        auto sampled = copy_sampled_token(step.value().sampled_tokens, stream);
        if (!sampled.ok()) return sampled.status();
        log_top2(step.value().logits, 0u, "safe", plan.position + i, 0u);
        committed = i + 1u;
        bool keep_verifying = false;
        if (i == drafts) {
            accepted = drafts;
            output.emitted[emitted++] = sampled.value();
            pending = sampled.value();
        } else if (sampled.value() == plan.proposals[i]) {
            accepted = i + 1u;
            output.emitted[emitted++] = plan.proposals[i];
            pending = plan.proposals[i];
            keep_verifying = true;
        } else {
            output.emitted[emitted++] = sampled.value();
            pending = sampled.value();
        }
        finished = qwen_hybrid::is_stop_token(target.model.config->stop_tokens,
                                              output.emitted[emitted - 1u]);
        if (!keep_verifying || finished || emitted >= remaining_tokens) break;
    }
    output.emitted_count = emitted;
    output.num_accepted = accepted;
    output.pending_token = pending;
    output.finished = finished;
    drafter.sample_index += committed;
    drafter.last_tokens.assign(verify_tokens.begin(), verify_tokens.begin() + committed);
    drafter.last_position = plan.position;
    drafter.last_hidden_offset = 0u;
    sequence.position = plan.position + committed;
    return output;
}

Result<MtpIterationOutput> accept_verified(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& sequence,
    uint32_t position,
    const int32_t* proposals,
    uint32_t drafts,
    const std::vector<int32_t>& verify_tokens,
    const int32_t* sampled,
    uint32_t remaining_tokens,
    std::vector<MtpGdnRankState>& request_ranks,
    const std::vector<MtpGdnRankState>& log_ranks,
    uint32_t row_base,
    hipStream_t stream) {
    SpecVerifyResult accepted_result;
    Status accepted = spec_greedy_accept(proposals, drafts, sampled, true, accepted_result);
    if (!accepted.ok()) return accepted;
    // Cut at the first stop token before the state commit, so rows past it are not committed.
    const bool stopped = spec_bound_result(accepted_result, remaining_tokens,
        target.model.config->stop_tokens);
    const uint32_t accepted_drafts = accepted_result.num_accepted_drafts;
    MtpIterationOutput output;
    output.finished = stopped;
    output.num_accepted = accepted_drafts;
    output.emitted_count = std::min(
        static_cast<uint32_t>(accepted_result.emitted_tokens.size()), remaining_tokens);
    for (uint32_t i = 0u; i < output.emitted_count; ++i) {
        output.emitted[i] = accepted_result.emitted_tokens[i];
    }
    output.pending_token = accepted_result.correction_valid
        ? accepted_result.correction_token
        : accepted_result.bonus_token;

    const uint32_t page_tokens = sequence.kv_pool()->page_tokens();
    if (page_tokens == 0u) {
        return Status::invalid_state("mtp accept: page_tokens is zero", __FILE__, __LINE__);
    }
    if (accepted_drafts == drafts) {
        const uint32_t committed = drafts + 1u;
        Status cache = cache_target_hidden(drafter, target, row_base, committed, 0u, stream);
        if (!cache.ok()) return cache;
        drafter.last_tokens = verify_tokens;
        drafter.last_position = position;
        drafter.last_hidden_offset = 0u;
        const uint32_t required = ceil_div_u32(sequence.position, page_tokens);
        Status rollback = rollback_sequence_append(sequence, required, stream);
        if (!rollback.ok()) return rollback;
        drafter.timing.full_accepts += 1u;
    } else {
        const uint32_t committed = accepted_drafts + 1u;
        const auto restore_started = std::chrono::steady_clock::now();
        for (std::size_t ri = 0u; ri < request_ranks.size(); ++ri) {
            MtpGdnRankState& state = request_ranks[ri];
            auto scope = shelf::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            Status restore = spec_gdn_restore(
                *state.pool, sequence.slot, state.conv_snapshot,
                state.recurrent_snapshot, state.conv_bytes, state.recurrent_bytes,
                state.stream);
            if (!restore.ok()) return restore;
            Status conv = restore_gdn_spec_conv(*state.pool, sequence.slot,
                                                log_ranks[ri].history,
                                                row_base + accepted_drafts, state.stream);
            if (!conv.ok()) return conv;
            Status qsa = restore_rank_qsa(log_ranks[ri], sequence.slot,
                                          row_base + accepted_drafts + 1u);
            if (!qsa.ok()) return qsa;
            Status ple = restore_rank_ple(log_ranks[ri], sequence.slot,
                                          row_base + accepted_drafts + 1u);
            if (!ple.ok()) return ple;
        }
        const auto restore_done = std::chrono::steady_clock::now();
        for (std::size_t ri = 0u; ri < request_ranks.size(); ++ri) {
            MtpGdnRankState& state = request_ranks[ri];
            auto scope = shelf::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            ::shelf::runtime::DeviceRequestDescriptor request = state.commit_request_host;
            request.request_handle = sequence.request_handle();
            request.row_begin = row_base;
            request.row_count = 0u;
            hipError_t upload = hipMemcpyAsync(state.commit_request, &request,
                                               sizeof(request), hipMemcpyHostToDevice,
                                               state.stream);
            if (upload != hipSuccess) {
                return Status::hip_error("mtp commit request upload",
                                         hipGetErrorString(upload), __FILE__, __LINE__);
            }
            GdnCompactLogDeviceView compact{};
            compact.delta = log_ranks[ri].compact_delta;
            compact.k = log_ranks[ri].compact_k;
            compact.a = log_ranks[ri].compact_a;
            compact.delta_layer_stride = log_ranks[ri].compact_delta_layer_stride;
            compact.k_layer_stride = log_ranks[ri].compact_k_layer_stride;
            compact.a_layer_stride = log_ranks[ri].compact_a_layer_stride;
            Status st = commit_gdn_compact_log(
                *state.pool, sequence.slot, compact, state.commit_request, committed,
                state.gdn_key_heads, state.stream);
            if (!st.ok()) return st;
        }
        const auto compact_done = std::chrono::steady_clock::now();
        Status cache = cache_target_hidden(drafter, target, row_base, committed, 0u, stream);
        if (!cache.ok()) return cache;
        sequence.position = position + committed;
        const uint32_t required = ceil_div_u32(sequence.position, page_tokens);
        Status rollback = rollback_sequence_append(sequence, required, stream);
        if (!rollback.ok()) return rollback;
        drafter.last_tokens.assign(verify_tokens.begin(),
                                   verify_tokens.begin() + committed);
        drafter.last_position = position;
        drafter.last_hidden_offset = 0u;
        drafter.timing.partial_accepts += 1u;
        drafter.timing.gdn_restores += 1u;
        drafter.timing.gdn_restore_ms +=
            std::chrono::duration<double, std::milli>(restore_done - restore_started).count();
        drafter.timing.compact_replay_ms +=
            std::chrono::duration<double, std::milli>(compact_done - restore_done).count();
    }

    for (uint32_t i = 0u; i < output.emitted_count; ++i) {
        if (qwen_hybrid::is_stop_token(target.model.config->stop_tokens,
                                       output.emitted[i])) {
            output.emitted_count = i + 1u;
            output.finished = true;
            break;
        }
    }
    if (qwen_hybrid::is_stop_token(target.model.config->stop_tokens,
                                   output.pending_token)) {
        output.finished = true;
    }
    return output;
}

}  // namespace

Result<MtpDraftResult> mtp_draft(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    int32_t pending_token,
    uint32_t remaining_tokens) {
    (void)target;
    if (!drafter.initialized) {
        return Status::invalid_state("mtp_draft: drafter is not initialized", __FILE__,
                                     __LINE__);
    }
    if (remaining_tokens == 0u) {
        return Status::invalid_argument("mtp_draft: remaining_tokens is zero", __FILE__,
                                        __LINE__);
    }
    if (target_sequence.position != drafter.last_position + drafter.last_tokens.size()) {
        return Status::invalid_state("mtp_draft: target sequence diverged from drafter",
                                     __FILE__, __LINE__);
    }

    const hipStream_t stream = drafter.target_stream;
    const uint32_t position = target_sequence.position;
    const uint32_t room = position < drafter.max_seq_len
        ? drafter.max_seq_len - position
        : 0u;
    uint32_t drafts = std::min(drafter.num_drafts, remaining_tokens - 1u);
    if (room < 2u) drafts = 0u;
    else drafts = std::min(drafts, room - 1u);

    MtpDraftResult result;
    result.position = position;
    result.drafts = drafts;
    if (drafts == 0u) {
        result.decode_only = true;
        return result;
    }

    const uint32_t prefix_rows = target_sequence.position - drafter.last_position;
    if (prefix_rows == 0u || prefix_rows > drafter.last_tokens.size()) {
        return Status::invalid_state("mtp_draft: invalid target prefix", __FILE__,
                                     __LINE__);
    }
    const uint32_t required_rewind = ceil_div_u32(drafter.last_position,
                                                  drafter.sequence->kv_pool()->page_tokens());
    Status rewind = rewind_sequences(drafter, drafter.last_position,
                                     required_rewind, stream);
    if (!rewind.ok()) return rewind;

    std::vector<int32_t> first_pass(prefix_rows);
    for (uint32_t i = 0u; i + 1u < prefix_rows; ++i) {
        first_pass[i] = drafter.last_tokens[i + 1u];
    }
    first_pass[prefix_rows - 1u] = pending_token;
    Status copy = copy_hidden_to_all(drafter, drafter.hidden_cache, 0u, prefix_rows,
                                       stream);
    if (!copy.ok()) return copy;
    auto first = run_mtp_forward(drafter, first_pass, drafter.last_position, true,
                                 stream);
    if (!first.ok()) return first.status();
    auto first_sampled = copy_sampled_token(first.value().sampled_tokens, stream);
    if (!first_sampled.ok()) return first_sampled.status();

    result.proposals[0] = first_sampled.value();
    copy = copy_hidden_to_all(drafter, drafter_hidden_output(drafter), 0u, 1u, stream);
    if (!copy.ok()) return copy;
    for (uint32_t i = 1u; i < drafts; ++i) {
        std::vector<int32_t> step_tokens{result.proposals[i - 1u]};
        auto step = run_mtp_forward(drafter, step_tokens, position + i - 1u, true,
                                    stream);
        if (!step.ok()) return step.status();
        auto sampled = copy_sampled_token(step.value().sampled_tokens, stream);
        if (!sampled.ok()) return sampled.status();
        result.proposals[i] = sampled.value();
        copy = copy_hidden_to_all(drafter, drafter_hidden_output(drafter), 0u, 1u,
                                    stream);
        if (!copy.ok()) return copy;
    }

    if (!drafter.serial_verify) {
        std::vector<MtpGdnRankState> rank_states = all_rank_states(drafter);
        for (const MtpGdnRankState& state : rank_states) {
            auto scope = shelf::gpu::ScopedDevice::create(state.device);
            if (!scope.ok()) return scope.status();
            Status snapshot = spec_gdn_snapshot(
                *state.pool, target_sequence.slot, state.conv_snapshot,
                state.recurrent_snapshot, state.conv_bytes, state.recurrent_bytes,
                state.stream);
            if (!snapshot.ok()) return snapshot;
        }
    }
    return result;
}

Result<MtpIterationOutput> mtp_spec_step(
    MtpDrafter& drafter,
    Executor& target,
    state::PagedSequenceState& target_sequence,
    int32_t pending_token,
    uint32_t remaining_tokens) {
    if (qwen_hybrid::is_stop_token(target.model.config->stop_tokens, pending_token)) {
        // The stop token was already emitted by the caller; do not draft past it.
        MtpIterationOutput stopped;
        stopped.pending_token = pending_token;
        stopped.finished = true;
        return stopped;
    }
    const auto started = std::chrono::steady_clock::now();
    auto draft = mtp_draft(drafter, target, target_sequence, pending_token,
                           remaining_tokens);
    if (!draft.ok()) return draft.status();
    const MtpDraftResult& plan = draft.value();
    const hipStream_t stream = drafter.target_stream;
    if (plan.decode_only) {
        auto output = mtp_decode_only(drafter, target, target_sequence, pending_token,
                                      remaining_tokens, stream);
        if (output.ok()) {
            drafter.timing.rounds += 1u;
            drafter.timing.generated_tokens += output.value().emitted_count;
            drafter.timing.verify_rows += 1u;
            drafter.timing.round_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
        }
        return output;
    }

    const auto draft_done = std::chrono::steady_clock::now();
    if (drafter.serial_verify) {
        auto output = mtp_verify_serial(drafter, target, target_sequence, pending_token,
                                        remaining_tokens, plan, stream);
        if (!output.ok()) return output.status();
        drafter.timing.rounds += 1u;
        drafter.timing.accepted_drafts += output.value().num_accepted;
        drafter.timing.generated_tokens += output.value().emitted_count;
        drafter.timing.verify_rows += static_cast<uint32_t>(drafter.last_tokens.size());
        if (output.value().num_accepted == plan.drafts)
            drafter.timing.full_accepts += 1u;
        else
            drafter.timing.partial_accepts += 1u;
        const auto finished_at = std::chrono::steady_clock::now();
        drafter.timing.draft_ms +=
            std::chrono::duration<double, std::milli>(draft_done - started).count();
        drafter.timing.verify_ms += std::chrono::duration<double, std::milli>(
            finished_at - draft_done).count();
        drafter.timing.round_ms += std::chrono::duration<double, std::milli>(
            finished_at - started).count();
        return output;
    }
    std::vector<MtpGdnRankState> rank_states = all_rank_states(drafter);
    std::vector<int32_t> verify_tokens(plan.drafts + 1u);
    verify_tokens[0] = pending_token;
    for (uint32_t i = 0u; i < plan.drafts; ++i) {
        verify_tokens[i + 1u] = plan.proposals[i];
    }
    std::vector<ExecuteBatchOptions> verify_options =
        build_verify_options(drafter, plan.drafts);
    auto verified = run_target(
        drafter, target, target_sequence, verify_tokens, plan.position, true, true,
        plan.drafts + 1u,
        plan.drafts + 1u == 1u ? ::shelf::runtime::ExecutionClass::DECODE
                               : ::shelf::runtime::ExecutionClass::SPEC_VERIFY,
        true, drafter.sampling, drafter.sample_index, &verify_options, stream);
    if (!verified.ok()) return verified.status();
    drafter.sample_index += static_cast<uint64_t>(plan.drafts) + 1u;

    std::vector<int32_t> sampled(plan.drafts + 1u, -1);
    const hipError_t copy_err = hipMemcpy(
        sampled.data(), verified.value().sampled_tokens.data<int32_t>(),
        sampled.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
    if (copy_err != hipSuccess) {
        return Status::hip_error("mtp verify sample copy", hipGetErrorString(copy_err),
                                 __FILE__, __LINE__);
    }
    if (sampled[0] < 0) {
        return Status::invalid_state("mtp verify produced no sampled token", __FILE__,
                                     __LINE__);
    }
    const auto verify_done = std::chrono::steady_clock::now();
    auto output = accept_verified(
        drafter, target, target_sequence, plan.position, plan.proposals.data(),
        plan.drafts, verify_tokens, sampled.data(), remaining_tokens, rank_states,
        rank_states, 0u, stream);
    if (!output.ok()) return output.status();

    drafter.timing.rounds += 1u;
    drafter.timing.accepted_drafts += output.value().num_accepted;
    drafter.timing.generated_tokens += output.value().emitted_count;
    drafter.timing.verify_rows += plan.drafts + 1u;
    const auto finished_at = std::chrono::steady_clock::now();
    drafter.timing.draft_ms +=
        std::chrono::duration<double, std::milli>(draft_done - started).count();
    drafter.timing.verify_ms +=
        std::chrono::duration<double, std::milli>(verify_done - draft_done).count();
    drafter.timing.round_ms +=
        std::chrono::duration<double, std::milli>(finished_at - started).count();
    return output;
}

std::size_t mtp_batch_rank_state_reserve_bytes(
    const state::GdnStatePoolLayout& gdn, std::uint32_t spec_rows,
    std::uint32_t drafter_count) noexcept {
    if (spec_rows == 0u) {
        return 0u;
    }
    const std::uint64_t rows = spec_rows;
    const std::size_t history = static_cast<std::size_t>(rows) * gdn.num_gdn_states *
        gdn.conv_history_stride * gdn.conv_history * sizeof(bf16_t);
    const std::uint64_t vdim = static_cast<std::uint64_t>(gdn.num_v_heads) * gdn.head_v;
    const std::uint64_t qkdim = static_cast<std::uint64_t>(gdn.num_v_heads) * gdn.head_k;
    const std::size_t compact = static_cast<std::size_t>(gdn.num_gdn_states) * rows *
        (vdim + qkdim + gdn.num_v_heads) * sizeof(float);
    const std::size_t snapshot = static_cast<std::size_t>(gdn.num_gdn_states) *
        (static_cast<std::size_t>(gdn.conv_history_stride) * gdn.conv_history *
             sizeof(bf16_t) +
         static_cast<std::size_t>(gdn.num_v_heads) * gdn.head_k * gdn.head_v *
             sizeof(::shelf::state::GdnStateScalar));
    return history + compact + static_cast<std::size_t>(drafter_count) * snapshot;
}

Result<MtpSpecBatchContext> create_mtp_spec_batch_context(
    Executor& target,
    const MtpDrafterInput& input,
    const state::GdnStatePool& target_gdn_pool,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const MtpDrafterConfig& options,
    uint32_t max_requests,
    const std::vector<MtpGdnRankBinding>* extra_ranks,
    TpBatchHook* tp_hook) {
    if (max_requests == 0u) {
        return Status::invalid_argument("mtp batch context needs a positive slot count",
                                        __FILE__, __LINE__);
    }
    if (options.max_batch_rows < options.num_drafts + 1u) {
        return Status::invalid_argument("mtp batch rows are below one verify round",
                                        __FILE__, __LINE__);
    }
    MtpSpecBatchContext context;
    context.target = &target;
    const std::uint32_t spec_rows = mtp_spec_rows(max_requests, options.num_drafts);
    context.max_rows = spec_rows;
    context.max_requests = max_requests;
    context.num_drafts = options.num_drafts;

    MtpDrafterConfig per_request = options;
    per_request.allocate_spec_buffers = false;
    const bool share_draft =
        target.qsa_state_pool != nullptr || target.ple_conv_state_pool != nullptr;

    auto make_shared = [&](const MtpDrafterInput& rank_input, gpu::GpuArena& rank_arena,
                           uint32_t local_kv_heads,
                           uint32_t tp_rank) -> Result<std::unique_ptr<MtpDraftShared>> {
        auto shared = std::make_unique<MtpDraftShared>();
        shared->config = mtp_executor_config(*rank_input.config);
        if (local_kv_heads != 0u)
            shared->config.num_key_value_heads = local_kv_heads;
        const uint32_t pages =
            ::shelf::runtime::kv_pages_for_context(options.max_seq_len, options.page_tokens);
        const uint32_t ring_pages = draft_ring_pages(options);
        auto slot_pool =
            state::SequenceSlotPool::create(rank_arena, max_requests, pages);
        if (!slot_pool.ok()) return slot_pool.status();
        shared->slot_pool =
            std::make_unique<state::SequenceSlotPool>(slot_pool.release());
        const state::GdnStatePoolLayout gdn_layout =
            make_gdn_state_layout(shared->config);
        auto gdn_pool = state::GdnStatePool::create(rank_arena, max_requests, gdn_layout);
        if (!gdn_pool.ok()) return gdn_pool.status();
        shared->gdn_pool = std::make_unique<state::GdnStatePool>(gdn_pool.release());
        const uint32_t ring_scale = ring_pages != 0u ? max_requests + 1u : max_requests;
        auto kv_pool = state::PagedKVPool::create(
            rank_arena, (ring_pages != 0u ? ring_pages : pages) * ring_scale,
            options.page_tokens, kMtpKvLayers,
            static_cast<uint32_t>(shared->config.num_key_value_heads),
            static_cast<uint32_t>(shared->config.attention_head_dim), options.kv_dtype);
        if (!kv_pool.ok()) return kv_pool.status();
        shared->kv_pool = std::make_unique<state::PagedKVPool>(kv_pool.release());

        ExecutorModelInput model_input;
        model_input.model.config = &shared->config;
        model_input.model.embed_tokens = rank_input.embed_tokens;
        model_input.lowered = rank_input.lowered;
        ExecutorConfig executor_config;
        const uint32_t batch_rows = max_requests * (options.num_drafts + 1u);
        executor_config.max_batch_rows =
            options.max_batch_rows > batch_rows ? options.max_batch_rows : batch_rows;
        executor_config.max_scheduled_requests = max_requests;
        executor_config.max_scheduled_output_rows = max_requests;
        executor_config.external_hidden_input = true;
        executor_config.external_hidden_width =
            rank_input.hidden_width != 0u
                ? rank_input.hidden_width
                : static_cast<std::uint32_t>(rank_input.config->hidden_size);
        executor_config.external_mask_bytes = draft_mask_bytes(options.max_seq_len);
        executor_config.exl3_linear_shape = rank_input.exl3_linear_shape;
        executor_config.emit_normed_token_hidden = rank_input.wide_hidden;
        executor_config.normed_token_hidden_width = executor_config.external_hidden_width;
        executor_config.rope_profile = options.rope_profile;
        executor_config.backend = ::shelf::runtime::DecodeBackend::Hip;
        auto executor = create_executor(std::move(model_input), *shared->slot_pool,
                                        *shared->gdn_pool, *shared->kv_pool, rank_arena,
                                        executor_config);
        if (!executor.ok()) return executor.status();
        shared->executor = executor.release();
        if (tp_hook != nullptr) {
            Status attached = tp_hook->attach_aux_executor(shared->executor, tp_rank);
            if (!attached.ok()) return attached;
        }
        shared->initialized = true;
        return shared;
    };

    if (share_draft) {
        auto shared = make_shared(input, arena, options.local_kv_heads, 0u);
        if (!shared.ok()) return shared.status();
        context.shared_draft = shared.release();
    }
    if (extra_ranks != nullptr) {
        for (const MtpGdnRankBinding& binding : *extra_ranks) {
            if (binding.mtp_input == nullptr) continue;
            auto peer_scope = shelf::gpu::ScopedDevice::create(binding.device);
            if (!peer_scope.ok()) {
                (void)shutdown_mtp_spec_batch_context(context);
                return peer_scope.status();
            }
            auto peer_shared = make_shared(*binding.mtp_input, *binding.arena,
                                           binding.local_kv_heads, binding.tp_rank);
            if (!peer_shared.ok()) {
                (void)shutdown_mtp_spec_batch_context(context);
                return peer_shared.status();
            }
            context.peer_shared_drafts.push_back(peer_shared.release());
            std::vector<std::unique_ptr<MtpDrafter>> peer_slots;
            MtpDrafterConfig peer_options = per_request;
            peer_options.local_kv_heads = binding.local_kv_heads;
            peer_slots.reserve(max_requests);
            for (uint32_t i = 0u; i < max_requests; ++i) {
                auto peer = create_mtp_drafter(
                    *binding.mtp_input, binding.executor, *binding.pool, *binding.arena,
                    binding.stream, peer_options, nullptr, tp_hook,
                    context.peer_shared_drafts.back().get());
                if (!peer.ok()) {
                    (void)shutdown_mtp_spec_batch_context(context);
                    return peer.status();
                }
                peer_slots.push_back(peer.release());
            }
            context.peer_drafters.push_back(std::move(peer_slots));
        }
    }
    context.drafters.reserve(max_requests);
    for (uint32_t i = 0u; i < max_requests; ++i) {
        auto drafter = create_mtp_drafter(input, &target, target_gdn_pool, arena,
                                          stream, per_request, extra_ranks, tp_hook,
                                          context.shared_draft.get());
        if (!drafter.ok()) {
            (void)shutdown_mtp_spec_batch_context(context);
            return drafter.status();
        }
        for (const auto& peer_slots : context.peer_drafters) {
            MtpDrafter& peer = *peer_slots[i];
            drafter.value()->peers.push_back(MtpDraftPeer{
                &draft_executor(peer), peer.sequence.get(), peer.target_stream,
                peer.target_device, &peer});
        }
        context.drafters.push_back(std::move(drafter.value()));
    }

    int rank0_device = 0;
    const hipError_t rank0_device_err = hipGetDevice(&rank0_device);
    if (rank0_device_err != hipSuccess) {
        return Status::hip_error("hipGetDevice", hipGetErrorString(rank0_device_err),
                                 __FILE__, __LINE__);
    }
    if (context.shared_draft != nullptr && !mtp_fused_verify_enabled()) {
        MtpGdnRankState rank0;
        rank0.pool = &target_gdn_pool;
        rank0.stream = stream;
        rank0.device = rank0_device;
        rank0.target = &target;
        context.ranks.push_back(std::move(rank0));
        context.initialized = true;
        return context;
    }
    auto make_rank = [&](const state::GdnStatePool& pool, gpu::GpuArena& rank_arena,
                         hipStream_t rank_stream, int device,
                         Executor* executor) -> Result<MtpGdnRankState> {
        MtpGdnRankState state;
        state.pool = &pool;
        state.stream = rank_stream;
        state.device = device;
        state.target = executor;
        Status qsa = allocate_qsa_history(
            executor, rank_arena, spec_rows, state.qsa_partial_history,
            state.qsa_count_history, state.qsa_partial_state_stride,
            state.qsa_count_state_stride);
        if (!qsa.ok()) return qsa;
        state.qsa_history_rows = spec_rows;
        Status ple = allocate_ple_conv_history(
            executor, rank_arena, spec_rows, state.ple_conv_history,
            state.ple_conv_state_stride);
        if (!ple.ok()) return ple;
        state.ple_conv_history_rows = spec_rows;
        auto history = create_gdn_spec_history(rank_arena, pool, spec_rows, false);
        if (!history.ok()) return history.status();
        state.history = history.release();
        const auto view = pool.device_view();
        const std::uint64_t rows = spec_rows;
        const std::uint64_t vdim = static_cast<std::uint64_t>(view.num_v_heads) * view.head_v;
        const std::uint64_t qkdim = static_cast<std::uint64_t>(view.num_v_heads) * view.head_k;
        state.compact_delta_layer_stride = rows * vdim;
        state.compact_k_layer_stride = rows * qkdim;
        state.compact_a_layer_stride = rows * view.num_v_heads;
        const std::size_t delta_bytes = static_cast<std::size_t>(view.num_gdn_states) *
            state.compact_delta_layer_stride * sizeof(float);
        const std::size_t k_bytes = static_cast<std::size_t>(view.num_gdn_states) *
            state.compact_k_layer_stride * sizeof(float);
        const std::size_t a_bytes = static_cast<std::size_t>(view.num_gdn_states) *
            state.compact_a_layer_stride * sizeof(float);
        auto delta = rank_arena.allocate_aligned(delta_bytes, 256u);
        if (!delta.ok()) return delta.status();
        state.compact_delta = static_cast<float*>(delta.release().data());
        auto kb = rank_arena.allocate_aligned(k_bytes, 256u);
        if (!kb.ok()) return kb.status();
        state.compact_k = static_cast<float*>(kb.release().data());
        auto ab = rank_arena.allocate_aligned(a_bytes, 256u);
        if (!ab.ok()) return ab.status();
        state.compact_a = static_cast<float*>(ab.release().data());
        return state;
    };

    auto rank0 = make_rank(target_gdn_pool, arena, stream, rank0_device, &target);
    if (!rank0.ok()) {
        (void)shutdown_mtp_spec_batch_context(context);
        return rank0.status();
    }
    context.ranks.push_back(std::move(rank0.value()));
    if (extra_ranks != nullptr) {
        for (const MtpGdnRankBinding& binding : *extra_ranks) {
            if (binding.pool == nullptr || binding.arena == nullptr) {
                (void)shutdown_mtp_spec_batch_context(context);
                return Status::invalid_argument("mtp batch rank binding is incomplete",
                                                __FILE__, __LINE__);
            }
            auto state = make_rank(*binding.pool, *binding.arena, binding.stream,
                                   binding.device, binding.executor);
            if (!state.ok()) {
                (void)shutdown_mtp_spec_batch_context(context);
                return state.status();
            }
            context.ranks.push_back(std::move(state.value()));
        }
    }
    context.initialized = true;
    return context;
}

Status shutdown_mtp_spec_batch_context(MtpSpecBatchContext& context) noexcept {
    Status first_error = Status::make_ok();
    for (MtpGdnRankState& rank : context.ranks) {
        Status st = shutdown_gdn_spec_history(rank.history);
        if (first_error.ok() && !st.ok()) first_error = st;
        rank.compact_delta = nullptr;
        rank.compact_k = nullptr;
        rank.compact_a = nullptr;
    }
    context.ranks.clear();
    for (std::size_t i = context.drafters.size(); i-- > 0u;) {
        if (context.drafters[i] == nullptr) continue;
        Status st = mtp_drafter_shutdown(*context.drafters[i], nullptr);
        if (first_error.ok() && !st.ok()) first_error = st;
        context.drafters[i].reset();
    }
    for (auto& slots : context.peer_drafters) {
        for (std::size_t i = slots.size(); i-- > 0u;) {
            if (slots[i] == nullptr) continue;
            Status st = mtp_drafter_shutdown(*slots[i], slots[i]->target_stream);
            if (first_error.ok() && !st.ok()) first_error = st;
            slots[i].reset();
        }
    }
    context.peer_drafters.clear();
    for (auto& shared : context.peer_shared_drafts) {
        if (shared == nullptr) continue;
        Status st = executor_shutdown(shared->executor);
        if (first_error.ok() && !st.ok()) first_error = st;
        shared->kv_pool.reset();
        shared->gdn_pool.reset();
        shared->slot_pool.reset();
        shared.reset();
    }
    context.peer_shared_drafts.clear();
    context.shared_draft.reset();
    context.initialized = false;
    return first_error;
}

Status mtp_draft_batch(
    MtpSpecBatchContext& context,
    std::span<const MtpSpecBatchSlot> slots,
    std::vector<MtpDraftResult>& plans,
    std::vector<char>& stopped,
    uint32_t& draft_forwards) {
    if (context.shared_draft == nullptr) {
        return Status::invalid_state("mtp batch draft needs a shared executor", __FILE__,
                                     __LINE__);
    }
    const std::size_t count = slots.size();
    plans.assign(count, MtpDraftResult{});
    stopped.assign(count, 0);
    Executor& executor = context.shared_draft->executor;
    const hipStream_t stream = context.ranks.empty() ? nullptr
                                                     : context.ranks.front().stream;
    std::vector<std::vector<int32_t>> first_tokens(count);
    std::vector<uint32_t> prefix_rows(count, 0u);
    std::vector<uint32_t> first_prefix(count, 0u);
    std::vector<uint32_t> row_begin(count, 0u);
    uint32_t max_drafts = 0u;
    uint32_t token_offset = 0u;

    for (std::size_t i = 0u; i < count; ++i) {
        const MtpSpecBatchSlot& slot = slots[i];
        MtpDrafter& drafter = *slot.drafter;
        if (!slot.ignore_eos && qwen_hybrid::is_stop_token(
                                   context.target->model.config->stop_tokens,
                                   slot.pending_token)) {
            stopped[i] = 1;
            plans[i].decode_only = true;
            continue;
        }
        const uint32_t position = slot.sequence->position;
        if (position != drafter.last_position + drafter.last_tokens.size()) {
            return Status::invalid_state("mtp draft: target sequence diverged from drafter",
                                         __FILE__, __LINE__);
        }
        const uint32_t room = position < drafter.max_seq_len
            ? drafter.max_seq_len - position
            : 0u;
        uint32_t drafts = std::min(drafter.num_drafts, slot.remaining_tokens - 1u);
        if (room < 2u) drafts = 0u;
        else drafts = std::min(drafts, room - 1u);
        if (slot.page_tokens != 0u) {
            const uint32_t to_boundary =
                slot.page_tokens - position % slot.page_tokens - 1u;
            drafts = std::min(drafts, to_boundary);
        }
        plans[i].position = position;
        plans[i].drafts = drafts;
        if (drafts == 0u) {
            plans[i].decode_only = true;
            continue;
        }
        const uint32_t rows = position - drafter.last_position;
        if (rows == 0u || rows > drafter.last_tokens.size()) {
            return Status::invalid_state("mtp draft: invalid target prefix", __FILE__,
                                         __LINE__);
        }
        const uint32_t required_rewind = ceil_div_u32(
            drafter.last_position, drafter.sequence->kv_pool()->page_tokens());
        Status rewind = rewind_sequences(drafter, drafter.last_position,
                                         required_rewind, stream);
        if (!rewind.ok()) return rewind;
        prefix_rows[i] = rows;
        first_prefix[i] = drafter.last_position;
        row_begin[i] = token_offset;
        token_offset += rows;
        first_tokens[i].resize(rows);
        for (uint32_t r = 0u; r + 1u < rows; ++r)
            first_tokens[i][r] = drafter.last_tokens[r + 1u];
        first_tokens[i][rows - 1u] = slot.pending_token;
        Status copy = copy_hidden_to_all(drafter, drafter.hidden_cache, 0u, rows,
                                           stream, row_begin[i]);
        if (!copy.ok()) return copy;
        if (drafts > max_drafts) max_drafts = drafts;
    }

    auto execute_depth = [&](const std::vector<std::size_t>& active,
                             const std::vector<int32_t>& tokens,
                             const std::vector<uint32_t>& row_counts,
                             const std::vector<uint32_t>& prefixes,
                             bool force_prefill)
        -> Result<std::vector<int32_t>> {
        std::vector<ScheduledRequest> requests;
        requests.reserve(active.size());
        uint32_t token_begin = 0u;
        for (std::size_t b = 0u; b < active.size(); ++b) {
            const std::size_t i = active[b];
            ScheduledRequest req;
            req.sequence = slots[i].drafter->sequence.get();
            req.handle = slots[i].drafter->sequence->request_handle();
            req.execution_class = !force_prefill && row_counts[b] == 1u
                ? ::shelf::runtime::ExecutionClass::DECODE
                : ::shelf::runtime::ExecutionClass::PREFILL;
            req.token_begin = token_begin;
            req.num_tokens = row_counts[b];
            req.prefix_tokens = prefixes[b];
            req.compute_logits = true;
            req.sample = true;
            req.num_output_rows = 1u;
            requests.push_back(req);
            token_begin += row_counts[b];
        }
        ScheduledBatch batch;
        batch.token_ids = tokens.data();
        batch.token_ids_location = ::shelf::runtime::TokenIdsLocation::Host;
        batch.requests = requests;
        batch.num_tokens = static_cast<uint32_t>(tokens.size());
        batch.num_requests = static_cast<uint32_t>(requests.size());
        batch.num_decode_requests = 0u;
        batch.num_prefill_requests = 0u;
        for (const auto& req : requests) {
            if (req.execution_class == ::shelf::runtime::ExecutionClass::DECODE)
                ++batch.num_decode_requests;
            else
                ++batch.num_prefill_requests;
        }
        batch.num_verify_requests = 0u;
        Result<BatchExecutionOutput> output = Status::make_ok();
        if (context.peer_shared_drafts.empty()) {
            output = execute_batch(executor, batch, stream);
        } else {
            auto* hook = context.drafters.empty()
                ? nullptr
                : context.drafters.front()->tp_hook;
            if (hook == nullptr) {
                return Status::invalid_state("mtp batch peers need a tp coordinator",
                                             __FILE__, __LINE__);
            }
            std::vector<::shelf::runtime::TpAuxRank> aux;
            aux.reserve(1u + context.peer_shared_drafts.size());
            ::shelf::runtime::TpAuxRank leader;
            leader.executor = &executor;
            leader.sequences.reserve(active.size());
            for (std::size_t i : active)
                leader.sequences.push_back(slots[i].drafter->sequence.get());
            aux.push_back(std::move(leader));
            for (std::size_t r = 0u; r < context.peer_shared_drafts.size(); ++r) {
                ::shelf::runtime::TpAuxRank peer;
                peer.executor = &context.peer_shared_drafts[r]->executor;
                peer.sequences.reserve(active.size());
                for (std::size_t i : active) {
                    const MtpDrafter& leader = *slots[i].drafter;
                    peer.sequences.push_back(leader.peers[r].sequence);
                }
                aux.push_back(std::move(peer));
            }
            output = hook->on_execute_aux(batch, aux, nullptr);
        }
        if (!output.ok()) return output.status();
        std::vector<int32_t> sampled(active.size(), -1);
        const hipError_t copy = hipMemcpy(
            sampled.data(), output.value().sampled_tokens.data<int32_t>(),
            sampled.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
        if (copy != hipSuccess) {
            return Status::hip_error("mtp batch draft sample copy",
                                     hipGetErrorString(copy), __FILE__, __LINE__);
        }
        return sampled;
    };

    std::vector<std::size_t> active;
    std::vector<int32_t> tokens;
    std::vector<uint32_t> row_counts;
    std::vector<uint32_t> prefixes;
    for (std::size_t i = 0u; i < count; ++i) {
        if (plans[i].decode_only) continue;
        active.push_back(i);
        tokens.insert(tokens.end(), first_tokens[i].begin(), first_tokens[i].end());
        row_counts.push_back(prefix_rows[i]);
        prefixes.push_back(first_prefix[i]);
    }
    if (active.empty()) return Status::make_ok();
    auto first = execute_depth(active, tokens, row_counts, prefixes, true);
    if (!first.ok()) return first.status();
    ++draft_forwards;
    for (std::size_t b = 0u; b < active.size(); ++b) {
        const std::size_t i = active[b];
        plans[i].proposals[0] = first.value()[b];
        const uint32_t source_row = row_begin[i] + prefix_rows[i] - 1u;
        Status copy = copy_hidden_to_all(
            *slots[i].drafter, executor.normed_token_hidden, source_row, 1u, stream,
            static_cast<uint32_t>(i));
        if (!copy.ok()) return copy;
    }
    for (std::size_t i = 0u; i < count; ++i) {
        MtpBoundaryCapture* boundary = slots[i].boundary_capture;
        if (boundary == nullptr || boundary->capture == nullptr) continue;
        if (stopped[i] != 0 || plans[i].decode_only) continue;
        if (slots[i].page_tokens == 0u ||
            plans[i].position % slots[i].page_tokens != 0u) {
            continue;
        }
        auto snapshot =
            mtp_save_draft_snapshot(*slots[i].drafter, plans[i].position, stream);
        if (!snapshot.ok()) continue;
        boundary->capture(boundary->context, plans[i].position, snapshot.value());
    }
    for (uint32_t depth = 1u; depth < max_drafts; ++depth) {
        active.clear();
        tokens.clear();
        row_counts.clear();
        prefixes.clear();
        for (std::size_t i = 0u; i < count; ++i) {
            if (plans[i].decode_only || depth >= plans[i].drafts) continue;
            active.push_back(i);
            tokens.push_back(plans[i].proposals[depth - 1u]);
            row_counts.push_back(1u);
            prefixes.push_back(plans[i].position + depth - 1u);
        }
        if (active.empty()) break;
        for (std::size_t b = 0u; b < active.size(); ++b) {
            const std::size_t i = active[b];
            if (i == b) continue;
            Status compact = copy_hidden_to_all(
                *slots[i].drafter, executor.input_hidden, static_cast<uint32_t>(i), 1u,
                stream, static_cast<uint32_t>(b));
            if (!compact.ok()) return compact;
        }
        auto sampled = execute_depth(active, tokens, row_counts, prefixes, false);
        if (!sampled.ok()) return sampled.status();
        ++draft_forwards;
        for (std::size_t b = 0u; b < active.size(); ++b) {
            const std::size_t i = active[b];
            plans[i].proposals[depth] = sampled.value()[b];
            Status copy = copy_hidden_to_all(
                *slots[i].drafter, executor.normed_token_hidden, static_cast<uint32_t>(b),
                1u, stream, static_cast<uint32_t>(i));
            if (!copy.ok()) return copy;
        }
    }
    if (!slots.front().drafter->serial_verify) {
        for (std::size_t i = 0u; i < count; ++i) {
            if (plans[i].decode_only) continue;
            std::vector<MtpGdnRankState> rank_states =
                all_rank_states(*slots[i].drafter);
            for (const MtpGdnRankState& state : rank_states) {
                auto scope = shelf::gpu::ScopedDevice::create(state.device);
                if (!scope.ok()) return scope.status();
                Status snapshot = spec_gdn_snapshot(
                    *state.pool, slots[i].sequence->slot, state.conv_snapshot,
                    state.recurrent_snapshot, state.conv_bytes, state.recurrent_bytes,
                    state.stream);
                if (!snapshot.ok()) return snapshot;
            }
        }
    }
    return Status::make_ok();
}

Result<MtpSpecBatchOutput> mtp_verify_wavefront(
    MtpSpecBatchContext& context,
    std::span<const MtpSpecBatchSlot> slots) {
    const std::size_t count = slots.size();
    MtpSpecBatchOutput out;
    out.slots.resize(count);
    std::vector<MtpDraftResult> plans(count);
    std::vector<std::vector<int32_t>> verify_tokens(count);
    std::vector<MtpIterationOutput> results(count);
    std::vector<uint32_t> accepted(count, 0u);
    std::vector<uint32_t> committed(count, 0u);
    std::vector<char> active(count, 0);
    std::vector<char> stopped(count, 0);
    uint32_t max_drafts = 0u;

    for (std::size_t i = 0u; i < count; ++i) {
        if (slots[i].drafter == nullptr || slots[i].sequence == nullptr) {
            return Status::invalid_argument("mtp batch: slot is incomplete", __FILE__,
                                            __LINE__);
        }
    }
    uint32_t draft_forwards = 0u;
    Status drafted = mtp_draft_batch(context, slots, plans, stopped, draft_forwards);
    if (!drafted.ok()) return drafted;
    for (std::size_t i = 0u; i < count; ++i) {
        const MtpSpecBatchSlot& slot = slots[i];
        results[i].pending_token = slot.pending_token;
        if (stopped[i] != 0) {
            results[i].finished = true;
            continue;
        }
        if (plans[i].decode_only) {
            verify_tokens[i].assign(1u, slot.pending_token);
        } else {
            verify_tokens[i].resize(plans[i].drafts + 1u);
            verify_tokens[i][0] = slot.pending_token;
            for (uint32_t d = 0u; d < plans[i].drafts; ++d)
                verify_tokens[i][d + 1u] = plans[i].proposals[d];
        }
        if (plans[i].drafts > max_drafts) max_drafts = plans[i].drafts;
        active[i] = 1;
    }

    const hipStream_t stream = context.ranks.empty() ? nullptr
                                                     : context.ranks.front().stream;
    const auto started = std::chrono::steady_clock::now();
    const std::size_t kWavefrontChunk = mtp_wavefront_enabled() ? 4u : 1u;
    for (uint32_t depth = 0u; depth <= max_drafts; ++depth) {
        std::vector<std::size_t> depth_slots;
        for (std::size_t i = 0u; i < count; ++i) {
            if (active[i] != 0 && depth <= plans[i].drafts) depth_slots.push_back(i);
        }
        for (std::size_t begin = 0u; begin < depth_slots.size();
             begin += kWavefrontChunk) {
            const std::size_t end = std::min(begin + kWavefrontChunk, depth_slots.size());
            std::vector<int32_t> token_ids;
            std::vector<ScheduledRequest> requests;
            for (std::size_t k = begin; k < end; ++k) {
                const std::size_t i = depth_slots[k];
                ScheduledRequest req;
                req.sequence = slots[i].sequence;
                req.handle = slots[i].sequence->request_handle();
                req.execution_class = ::shelf::runtime::ExecutionClass::DECODE;
                req.token_begin = static_cast<uint32_t>(token_ids.size());
                req.num_tokens = 1u;
                req.prefix_tokens = plans[i].position + depth;
                req.compute_logits = true;
                req.sample = true;
                req.num_output_rows = 1u;
                req.sampling = slots[i].drafter->sampling;
                req.sampling_index = slots[i].drafter->sample_index + depth;
                token_ids.push_back(verify_tokens[i][depth]);
                requests.push_back(req);
            }

            ScheduledBatch batch;
            batch.token_ids = token_ids.data();
            batch.token_ids_location = ::shelf::runtime::TokenIdsLocation::Host;
            batch.requests = requests;
            batch.num_tokens = static_cast<uint32_t>(token_ids.size());
            batch.num_requests = static_cast<uint32_t>(requests.size());
            batch.num_decode_requests = batch.num_requests;
            batch.num_verify_requests = 0u;
            batch.num_prefill_requests = 0u;
            batch.speculative_verify = false;

            MtpDrafter& leader = *slots[depth_slots[begin]].drafter;
            auto executed = leader.tp_hook != nullptr
                ? leader.tp_hook->on_execute(batch)
                : execute_batch(*context.target, batch, stream);
            if (!executed.ok()) return executed.status();
            const BatchExecutionOutput output = executed.release();
            if (output.num_outputs < requests.size()) {
                return Status::invalid_state("mtp wavefront output rows mismatch",
                                             __FILE__, __LINE__);
            }
            std::vector<int32_t> sampled(requests.size(), -1);
            const hipError_t copy = hipMemcpy(
                sampled.data(), output.sampled_tokens.data<int32_t>(),
                sampled.size() * sizeof(int32_t), hipMemcpyDeviceToHost);
            if (copy != hipSuccess) {
                return Status::hip_error("mtp wavefront sample copy",
                                         hipGetErrorString(copy), __FILE__, __LINE__);
            }
            for (std::size_t b = 0u; b < requests.size(); ++b) {
                const std::size_t i = depth_slots[begin + b];
                log_top2(output.logits, static_cast<uint32_t>(b), "safe",
                         plans[i].position + depth, static_cast<uint32_t>(i));
            }
            ++out.verify_chunks;
            for (std::size_t b = 0u; b < requests.size(); ++b) {
                const std::size_t i = depth_slots[begin + b];
                MtpDrafter& drafter = *slots[i].drafter;
                Status cache = cache_target_hidden(drafter, *context.target, b, 1u, depth,
                                                   stream);
                if (!cache.ok()) return cache;
                committed[i] = depth + 1u;
                bool keep_verifying = false;
                if (depth == plans[i].drafts) {
                    accepted[i] = plans[i].drafts;
                    results[i].emitted[results[i].emitted_count++] = sampled[b];
                    results[i].pending_token = sampled[b];
                } else if (sampled[b] == plans[i].proposals[depth]) {
                    accepted[i] = depth + 1u;
                    results[i].emitted[results[i].emitted_count++] =
                        plans[i].proposals[depth];
                    results[i].pending_token = plans[i].proposals[depth];
                    keep_verifying = true;
                } else {
                    results[i].emitted[results[i].emitted_count++] = sampled[b];
                    results[i].pending_token = sampled[b];
                }
                results[i].finished = !slots[i].ignore_eos && qwen_hybrid::is_stop_token(
                    context.target->model.config->stop_tokens,
                    results[i].emitted[results[i].emitted_count - 1u]);
                if (!keep_verifying || results[i].finished ||
                    results[i].emitted_count >= slots[i].remaining_tokens) {
                    active[i] = 0;
                }
            }
        }
    }

    for (std::size_t i = 0u; i < count; ++i) {
        MtpDrafter& drafter = *slots[i].drafter;
        results[i].num_accepted = accepted[i];
        if (stopped[i] == 0) {
            drafter.sample_index += committed[i];
            drafter.last_tokens.assign(verify_tokens[i].begin(),
                                       verify_tokens[i].begin() + committed[i]);
            drafter.last_position = plans[i].position;
            drafter.last_hidden_offset = 0u;
            slots[i].sequence->position = plans[i].position + committed[i];
            drafter.timing.rounds += 1u;
            drafter.timing.accepted_drafts += accepted[i];
            drafter.timing.generated_tokens += results[i].emitted_count;
            drafter.timing.verify_rows += committed[i];
            if (accepted[i] == plans[i].drafts)
                drafter.timing.full_accepts += 1u;
            else
                drafter.timing.partial_accepts += 1u;
        }
        out.slots[i] = results[i];
        out.total_rows += committed[i];
    }
    out.rounds = 1u;
    out.draft_forwards = draft_forwards;
    out.round_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - started)
                       .count();
    return out;
}

Result<MtpSpecBatchOutput> mtp_spec_step_batch(
    MtpSpecBatchContext& context,
    std::span<const MtpSpecBatchSlot> slots) {
    if (!context.initialized || context.target == nullptr) {
        return Status::invalid_state("mtp batch: context is not initialized", __FILE__,
                                     __LINE__);
    }
    if (slots.empty()) {
        return Status::invalid_argument("mtp batch: no slots", __FILE__, __LINE__);
    }
    if (slots.size() > context.max_requests) {
        return Status::out_of_range("mtp batch: more slots than the context holds",
                                    __FILE__, __LINE__);
    }
    if (slots.front().drafter != nullptr && slots.front().drafter->serial_verify) {
        return mtp_verify_wavefront(context, slots);
    }
    const auto started = std::chrono::steady_clock::now();
    MtpSpecBatchOutput out;
    out.slots.resize(slots.size());
    std::vector<MtpDraftResult> drafts(slots.size());
    std::vector<uint32_t> row_begin(slots.size(), 0u);
    std::vector<char> stopped_slot(slots.size(), 0);
    uint32_t total_rows = 0u;
    uint32_t draft_forwards = 0u;
    std::vector<char> batch_stopped(slots.size(), 0);
    if (context.shared_draft != nullptr) {
        Status drafted = mtp_draft_batch(context, slots, drafts, batch_stopped,
                                         draft_forwards);
        if (!drafted.ok()) return drafted;
        for (std::size_t i = 0u; i < slots.size(); ++i) {
            row_begin[i] = total_rows;
            if (batch_stopped[i] != 0) {
                stopped_slot[i] = 1;
                out.slots[i].pending_token = slots[i].pending_token;
                out.slots[i].finished = true;
                continue;
            }
            if (!drafts[i].decode_only) total_rows += drafts[i].drafts + 1u;
        }
    } else {
        for (std::size_t i = 0u; i < slots.size(); ++i) {
            const MtpSpecBatchSlot& slot = slots[i];
            if (slot.drafter == nullptr || slot.sequence == nullptr) {
                return Status::invalid_argument("mtp batch: slot is incomplete",
                                                __FILE__, __LINE__);
            }
            row_begin[i] = total_rows;
            if (qwen_hybrid::is_stop_token(context.target->model.config->stop_tokens,
                                           slot.pending_token)) {
                stopped_slot[i] = 1;
                drafts[i].decode_only = true;
                out.slots[i].pending_token = slot.pending_token;
                out.slots[i].finished = true;
                continue;
            }
            auto draft = mtp_draft(*slot.drafter, *context.target, *slot.sequence,
                                   slot.pending_token, slot.remaining_tokens);
            if (!draft.ok()) return draft.status();
            drafts[i] = draft.value();
            if (!drafts[i].decode_only) total_rows += drafts[i].drafts + 1u;
        }
    }
    if (total_rows > context.max_rows) {
        return Status::out_of_range("mtp batch: rows exceed the context capacity",
                                    __FILE__, __LINE__);
    }
    const hipStream_t stream = context.ranks.empty() ? nullptr
                                                     : context.ranks.front().stream;
    std::vector<std::size_t> pending;
    pending.reserve(slots.size());
    for (std::size_t i = 0u; i < slots.size(); ++i) {
        if (!drafts[i].decode_only) pending.push_back(i);
    }

    std::vector<std::size_t> chunk;
    std::vector<uint32_t> chunk_row_begin;
    std::vector<int32_t> token_ids;
    std::vector<ScheduledRequest> requests;
    std::vector<int32_t> sampled;
    std::size_t cursor = 0u;
    while (cursor < pending.size()) {
        chunk.clear();
        chunk_row_begin.clear();
        uint32_t chunk_rows = 0u;
        while (cursor < pending.size() &&
               chunk.size() < ::shelf::kernel::kGdnRecurrenceVerifyMaxRequests) {
            const std::size_t i = pending[cursor];
            const uint32_t rows = drafts[i].drafts + 1u;
            if (!chunk.empty() &&
                chunk_rows + rows > ::shelf::kernel::kGdnRecurrenceVerifyBatchRows) {
                break;
            }
            chunk.push_back(i);
            chunk_row_begin.push_back(chunk_rows);
            chunk_rows += rows;
            ++cursor;
        }
        if (chunk.empty()) {
            return Status::invalid_state("mtp batch verify chunk is empty", __FILE__,
                                         __LINE__);
        }
        token_ids.clear();
        token_ids.reserve(chunk_rows);
        requests.clear();
        requests.reserve(chunk.size());
        for (std::size_t c = 0u; c < chunk.size(); ++c) {
            const std::size_t i = chunk[c];
            const MtpSpecBatchSlot& slot = slots[i];
            std::vector<MtpGdnRankState> request_ranks = all_rank_states(*slot.drafter);
            for (MtpGdnRankState& state : request_ranks) {
                auto scope = shelf::gpu::ScopedDevice::create(state.device);
                if (!scope.ok()) return scope.status();
                ::shelf::runtime::DeviceRequestDescriptor request =
                    state.commit_request_host;
                request.request_handle = slot.sequence->request_handle();
                request.row_begin = chunk_row_begin[c];
                request.row_count = 0u;
                if (state.commit_request != nullptr) {
                    const hipError_t err = hipMemcpy(
                        state.commit_request, &request, sizeof(request),
                        hipMemcpyHostToDevice);
                    if (err != hipSuccess) {
                        return Status::hip_error("mtp batch commit descriptor",
                                                 hipGetErrorString(err), __FILE__,
                                                 __LINE__);
                    }
                }
            }
            token_ids.push_back(slot.pending_token);
            for (uint32_t j = 0u; j < drafts[i].drafts; ++j) {
                token_ids.push_back(drafts[i].proposals[j]);
            }
            ScheduledRequest req;
            req.sequence = slot.sequence;
            req.handle = slot.sequence->request_handle();
            req.execution_class = ::shelf::runtime::ExecutionClass::SPEC_VERIFY;
            req.token_begin = chunk_row_begin[c];
            req.num_tokens = drafts[i].drafts + 1u;
            req.prefix_tokens = drafts[i].position;
            req.compute_logits = true;
            req.sample = true;
            req.num_output_rows = drafts[i].drafts + 1u;
            req.sampling = slot.drafter->sampling;
            req.sampling_index = slot.drafter->sample_index;
            requests.push_back(req);
        }

        ScheduledBatch batch;
        batch.token_ids = token_ids.data();
        batch.token_ids_location = ::shelf::runtime::TokenIdsLocation::Host;
        batch.requests = requests;
        batch.num_tokens = chunk_rows;
        batch.num_requests = static_cast<uint32_t>(requests.size());
        batch.num_decode_requests = 0u;
        batch.num_verify_requests = static_cast<uint32_t>(requests.size());
        batch.num_prefill_requests = 0u;
        batch.speculative_verify = true;

        std::vector<ExecuteBatchOptions> options(context.ranks.size());
        for (std::size_t r = 0u; r < options.size(); ++r) {
            options[r].gdn_spec_history =
                gdn_spec_history_view(context.ranks[r].history, chunk_rows);
            options[r].gdn_compact.delta = context.ranks[r].compact_delta;
            options[r].gdn_compact.k = context.ranks[r].compact_k;
            options[r].gdn_compact.a = context.ranks[r].compact_a;
            options[r].gdn_compact.delta_layer_stride =
                context.ranks[r].compact_delta_layer_stride;
            options[r].gdn_compact.k_layer_stride = context.ranks[r].compact_k_layer_stride;
            options[r].gdn_compact.a_layer_stride = context.ranks[r].compact_a_layer_stride;
            options[r].qsa_spec_history.partial = context.ranks[r].qsa_partial_history;
            options[r].qsa_spec_history.counts = context.ranks[r].qsa_count_history;
            options[r].qsa_spec_history.rows = context.ranks[r].qsa_history_rows;
            options[r].qsa_spec_history.partial_state_stride =
                context.ranks[r].qsa_partial_state_stride;
            options[r].qsa_spec_history.count_state_stride =
                context.ranks[r].qsa_count_state_stride;
            options[r].ple_conv_history.store = context.ranks[r].ple_conv_history;
            options[r].ple_conv_history.rows = context.ranks[r].ple_conv_history_rows;
            options[r].ple_conv_history.state_stride = context.ranks[r].ple_conv_state_stride;
        }
        MtpDrafter& leader = *slots[chunk.front()].drafter;
        auto executed = leader.tp_hook != nullptr
            ? leader.tp_hook->on_execute(batch, options.data())
            : execute_batch(*context.target, batch, stream, options[0]);
        if (!executed.ok()) return executed.status();
        const BatchExecutionOutput output = executed.release();
        ++out.verify_chunks;
        if (output.num_outputs < chunk_rows) {
            return Status::invalid_state("mtp batch verify output rows mismatch", __FILE__,
                                         __LINE__);
        }
        sampled.assign(chunk_rows, -1);
        const hipError_t err = hipMemcpy(
            sampled.data(), output.sampled_tokens.data<int32_t>(),
            static_cast<std::size_t>(chunk_rows) * sizeof(int32_t),
            hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            return Status::hip_error("mtp batch sample copy", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        for (std::size_t c = 0u; c < chunk.size(); ++c) {
            const std::size_t i = chunk[c];
            for (uint32_t r = 0u; r <= drafts[i].drafts; ++r) {
                log_top2(output.logits, chunk_row_begin[c] + r, "fused",
                         drafts[i].position + r, static_cast<uint32_t>(i));
                if (mtp_top2_enabled()) {
                    std::fprintf(stderr,
                                 "MTP_SAMPLED slot=%u pos=%u token=%d\n",
                                 static_cast<uint32_t>(i), drafts[i].position + r,
                                 sampled[chunk_row_begin[c] + r]);
                }
            }
        }

        for (std::size_t c = 0u; c < chunk.size(); ++c) {
            const std::size_t i = chunk[c];
            const MtpSpecBatchSlot& slot = slots[i];
            std::vector<MtpGdnRankState> request_ranks = all_rank_states(*slot.drafter);
            std::vector<int32_t> verify_tokens(drafts[i].drafts + 1u);
            verify_tokens[0] = slot.pending_token;
            for (uint32_t j = 0u; j < drafts[i].drafts; ++j) {
                verify_tokens[j + 1u] = drafts[i].proposals[j];
            }
            auto accepted = accept_verified(
                *slot.drafter, *context.target, *slot.sequence, drafts[i].position,
                drafts[i].proposals.data(), drafts[i].drafts, verify_tokens,
                sampled.data() + chunk_row_begin[c], slot.remaining_tokens, request_ranks,
                context.ranks, chunk_row_begin[c], stream);
            if (!accepted.ok()) return accepted.status();
            slot.drafter->sample_index += static_cast<uint64_t>(drafts[i].drafts) + 1u;
            out.slots[i] = accepted.value();
        }
    }

    for (std::size_t i = 0u; i < slots.size(); ++i) {
        if (!drafts[i].decode_only || stopped_slot[i]) continue;
        const MtpSpecBatchSlot& slot = slots[i];
        auto output = mtp_decode_only(*slot.drafter, *context.target, *slot.sequence,
                                      slot.pending_token, slot.remaining_tokens, stream);
        if (!output.ok()) return output.status();
        out.slots[i] = output.value();
    }
    out.total_rows = total_rows;
    out.draft_forwards = draft_forwards;
    out.rounds = 1u;
    out.round_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return out;
}

}  // namespace shelf::runtime::mtp
