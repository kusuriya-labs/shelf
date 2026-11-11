#pragma once

#include <shelf/inference/prefix/prefix_kv_cache.h>
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/inference/state/ple_conv_state_pool.h>
#include <shelf/inference/state/qsa_state_pool.h>
#include <shelf/models/qwen35/runtime/tp_batch_hook.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace shelf {
namespace qwen35 {
namespace runtime {

using ::shelf::inference::prefix::PrefixKVStore;
using ::shelf::state::PageId;

struct PrefixCacheConfig {
    uint32_t max_concurrent = 4;
    uint32_t min_prefix_tokens = 64;
    uint32_t spec_ring_bytes = 0;
};

struct PrefixCacheBytes {
    std::size_t state = 0;
    std::size_t extra_state = 0;
    std::size_t ring = 0;
    std::size_t state_slot = 0;

    std::size_t total() const noexcept { return state + extra_state + ring; }
};

PrefixCacheBytes prefix_cache_bytes(
    const PrefixCacheConfig& config,
    const GdnStatePoolLayout& layout,
    const ::shelf::state::QsaKeyStatePoolLayout* qsa = nullptr,
    const ::shelf::state::PleConvStatePoolLayout* ple_conv = nullptr) noexcept;

struct SpecRingRef {
    const bf16_t* k = nullptr;
    const bf16_t* v = nullptr;
    uint32_t length = 0;
    uint32_t next_position = 0;

    bool valid() const noexcept { return k != nullptr && v != nullptr; }
};

struct SpecRingOut {
    bf16_t* k = nullptr;
    bf16_t* v = nullptr;
    uint32_t length = 0;
    uint32_t next_position = 0;

    bool valid() const noexcept { return k != nullptr && v != nullptr; }
};

struct PrefixHit {
    uint64_t entry_id = 0;
    uint32_t state_position = 0;
    uint32_t match_tokens = 0;
    uint32_t ring_length = 0;
    uint32_t ring_next_position = 0;

    bool hit() const noexcept { return entry_id != 0; }

    bool ring_covers_state() const noexcept {
        return ring_length != 0 && ring_next_position == state_position;
    }
};

struct PrefixCacheStats {
    uint64_t saves = 0;
    uint64_t input_boundary_saves = 0;
    uint64_t output_tail_saves = 0;
    uint64_t skipped_too_short = 0;
    uint64_t skipped_too_large = 0;
    uint64_t lookups = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t restores = 0;
    uint64_t captures_started = 0;
    uint64_t captures_cancelled = 0;
    uint64_t entries_evicted = 0;
    uint64_t pages_evicted = 0;
    uint32_t entries = 0;
    uint32_t used_pages = 0;
    uint32_t radix_nodes = 0;
    uint32_t captures_active = 0;
};

using CaptureId = uint32_t;

inline constexpr CaptureId kInvalidCapture = 0;

class PrefixCache;

struct PrefixCacheRankContext {
    PagedKVPool* active_kv = nullptr;
    GdnStatePool* active_state = nullptr;
    gpu::GpuArena* arena = nullptr;
    hipStream_t stream = nullptr;
    int device = 0;
    ::shelf::state::QsaKeyStatePool* active_qsa = nullptr;
    ::shelf::state::PleConvStatePool* active_ple_conv = nullptr;
};

class Qwen35PrefixState;

class PrefixCache {
 public:
    static Result<PrefixCache> create(
        gpu::GpuArena& arena,
        PagedKVPool& active_kv_pool,
        GdnStatePool& active_state_pool,
        const PrefixCacheConfig& config,
        const std::vector<PrefixCacheRankContext>& extra_ranks = {});

    static Result<PrefixCache> create(
        gpu::GpuArena& arena,
        PagedKVPool& active_kv_pool,
        GdnStatePool& active_state_pool,
        ::shelf::state::QsaKeyStatePool* active_qsa_pool,
        ::shelf::state::PleConvStatePool* active_ple_conv_pool,
        const PrefixCacheConfig& config,
        const std::vector<PrefixCacheRankContext>& extra_ranks = {});

    PrefixCache(PrefixCache&&) noexcept;
    PrefixCache& operator=(PrefixCache&&) noexcept;
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;
    ~PrefixCache();

    Status set_tp_hook(TpBatchHook* hook) noexcept;

    Status shutdown() noexcept;

    PrefixHit lookup(const std::vector<uint32_t>& tokens);

    Status save_input_boundary(
        const PagedSequenceState& seq,
        const std::vector<uint32_t>& tokens,
        uint32_t input_len,
        hipStream_t stream,
        const SpecRingRef* ring = nullptr);

    Result<CaptureId> begin_capture(const PagedSequenceState& seq);

    Status capture_boundary(
        CaptureId capture,
        const PagedSequenceState& seq,
        hipStream_t stream);

    Status cancel_capture(CaptureId capture);

    Status save_output_tail(
        CaptureId capture,
        const PagedSequenceState& seq,
        const std::vector<uint32_t>& tokens,
        uint32_t position,
        hipStream_t stream,
        const SpecRingRef* ring = nullptr);

    Status restore(
        uint64_t entry_id,
        PagedSequenceState& dst,
        hipStream_t stream,
        SpecRingOut* ring = nullptr);

    PrefixCacheStats stats() const noexcept;

    std::size_t reserved_bytes() const noexcept;

 private:
    PrefixCache() = default;

    struct CaptureState {
        bool active = false;
        uint32_t generation = 0;
        SequenceSlotId seq_slot = kInvalidSlot;
        uint32_t seq_generation = 0;
        uint32_t state_slot = 0;
        uint32_t last_boundary = 0;
    };

    CaptureState* resolve_capture(CaptureId capture,
                                  const PagedSequenceState* seq);

    Result<uint32_t> acquire_state_slot(hipStream_t stream);

    Result<uint32_t> acquire_ring_slot(hipStream_t stream);

    struct AuxSlot {
        uint32_t hot_slot = 0;
        uint32_t ring_slot = 0xFFFFFFFFu;
    };

    struct AuxTable {
        std::unordered_map<uint64_t, AuxSlot> slots;
        uint64_t next = 1;
    };

    std::unique_ptr<Qwen35PrefixState> state_;
    std::unique_ptr<PrefixKVStore> kv_;
    std::shared_ptr<AuxTable> aux_;
    TpBatchHook* tp_hook_ = nullptr;
    PrefixCacheConfig config_;
    uint32_t page_tokens_ = 16;
    std::vector<CaptureState> captures_;
    PrefixCacheStats stats_;
};

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
