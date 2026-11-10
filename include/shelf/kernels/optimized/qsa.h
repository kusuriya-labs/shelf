#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/runtime/batch/device_batch_context.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::kernel {

// QSA key pooling: every compress_ratio complete raw keys become one block key.
// out[b, d] = rmsnorm(mean_t raw[b*cr+t, d], 1 + weight[d]).
struct QsaPoolArgs {
    const bf16_t* raw_keys = nullptr;
    const bf16_t* norm_weight = nullptr;
    bf16_t* pooled = nullptr;
    uint32_t rows = 0;
    uint32_t blocks = 0;
    uint32_t head_dim = 0;
    uint32_t compress_ratio = 0;
    float eps = 1e-6f;
};

inline constexpr uint32_t kQsaHeadDim = 128u;
inline constexpr uint32_t kQsaThreads = 128u;
inline constexpr uint32_t kQsaTopkMaxThreads = 512u;
inline constexpr uint32_t kQsaTopkSmallRows = 32u;

inline constexpr uint32_t kQsaSelectChunk = 64u;
inline constexpr uint32_t kQsaSelectMaxHeads = 8u;
inline constexpr uint32_t kQsaSelectMaxTopK = 1024u;
inline constexpr uint32_t kQsaScoreRowTile = 2048u;
inline constexpr uint32_t kQsaScoreBlockTile = 8192u;
inline constexpr uint32_t kQsaScoreRowsPerCta = 16u;
inline constexpr uint32_t kQsaScoreBlocksPerCta = 1024u;

inline constexpr std::size_t qsa_select_workspace_bytes() noexcept {
    const std::size_t scores = static_cast<std::size_t>(kQsaScoreRowTile) *
                               kQsaScoreBlockTile * sizeof(float);
    const std::size_t state = static_cast<std::size_t>(kQsaScoreRowTile) *
                              2u * kQsaSelectMaxTopK *
                              (sizeof(float) + sizeof(std::int32_t));
    return scores + state;
}

struct QsaSelectBatchArgs {
    const bf16_t* q = nullptr;
    const bf16_t* pooled_base = nullptr;
    int32_t* selected = nullptr;
    uint8_t* block_mask = nullptr;
    const uint32_t* row_sequence_slots = nullptr;
    const uint32_t* positions = nullptr;
    uint32_t rows = 0;
    uint32_t blocks = 0;
    uint32_t n_heads = 0;
    uint32_t head_dim = 0;
    uint32_t compress_ratio = 0;
    uint32_t state_index = 0;
    uint32_t max_sequences = 0;
    uint32_t top_k = 0;
    uint32_t q_row_stride = 0;
    uint32_t selected_row_stride = 0;
    uint32_t mask_row_bytes = 0;
    uint64_t pooled_slot_stride = 0;
    uint64_t pooled_layer_stride = 0;
    float scale = 0.0f;
};

// Partial rotary in place over the first rotary_dim lanes of each head:
// out[i] = x[i] * cos - x[i + rotary_dim/2] * sin
// out[i + rotary_dim/2] = x[i + rotary_dim/2] * cos + x[i] * sin
struct QsaRopeArgs {
    bf16_t* x = nullptr;
    const uint32_t* positions = nullptr;
    // Optional YaRN-corrected frequencies (half = rotary_dim / 2 entries). When null,
    // theta and position_scale derive the frequency.
    const double* inv_freq = nullptr;
    uint32_t rows = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 0;
    uint32_t rotary_dim = 0;
    float theta = 10000000.0f;
    // Position interpolation / YaRN factor. The QSA path supports up to 2x.
    float position_scale = 1.0f;
    float attention_factor = 1.0f;
};

struct QsaScoreArgs {
    const bf16_t* q = nullptr;
    const bf16_t* pooled_base = nullptr;
    float* scores = nullptr;
    const uint32_t* row_sequence_slots = nullptr;
    uint32_t rows = 0;
    uint32_t row_begin = 0;
    uint32_t block_begin = 0;
    uint32_t blocks = 0;
    uint32_t block_tile = kQsaScoreBlocksPerCta;
    uint32_t score_row_stride = 0;
    uint32_t n_heads = 0;
    uint32_t head_dim = 0;
    uint32_t compress_ratio = 0;
    uint32_t state_index = 0;
    uint32_t max_sequences = 0;
    uint32_t q_row_stride = 0;
    uint64_t pooled_slot_stride = 0;
    uint64_t pooled_layer_stride = 0;
    float scale = 0.0f;
};

struct QsaTopkArgs {
    const float* scores = nullptr;
    float* state_scores = nullptr;
    int32_t* state_indices = nullptr;
    uint8_t* block_mask = nullptr;
    int32_t* selected = nullptr;
    const uint32_t* positions = nullptr;
    uint32_t rows = 0;
    uint32_t block_begin = 0;
    uint32_t block_count = 0;
    uint32_t score_row_stride = 0;
    uint32_t blocks = 0;
    uint32_t compress_ratio = 0;
    uint32_t top_k = 0;
    uint32_t mask_row_bytes = 0;
    uint32_t selected_row_stride = 0;
    uint32_t first = 0;
    uint32_t last = 0;
};

hipError_t launch_qsa_score_wmma(const QsaScoreArgs& args, hipStream_t stream);
hipError_t launch_qsa_topk(const QsaTopkArgs& args, hipStream_t stream);

hipError_t launch_qsa_pool(const QsaPoolArgs& args, hipStream_t stream);
hipError_t launch_qsa_select_batch(const QsaSelectBatchArgs& args, hipStream_t stream);
hipError_t launch_qsa_rope(const QsaRopeArgs& args, hipStream_t stream);

// Append raw keys to one sequence's partial block and emit a pooled key when the block
// completes. `partial` is [compress_ratio][head_dim], `pooled` is [max_blocks][head_dim].
// Both counts live on the device so decode can continue from the previous call.
struct QsaKeyCacheArgs {
    const bf16_t* raw_keys = nullptr;
    const bf16_t* norm_weight = nullptr;
    const uint32_t* positions = nullptr;
    bf16_t* pooled = nullptr;
    bf16_t* partial = nullptr;
    uint32_t* pool_count = nullptr;
    uint32_t* partial_count = nullptr;
    uint32_t rows = 0;
    uint32_t max_blocks = 0;
    uint32_t head_dim = 0;
    uint32_t compress_ratio = 0;
    uint32_t rotary_dim = 0;
    float theta = 10000000.0f;
    float position_scale = 1.0f;
    const double* inv_freq = nullptr;
    float eps = 1e-6f;
};

hipError_t launch_qsa_key_cache(const QsaKeyCacheArgs& args, hipStream_t stream);

// Multi-sequence key cache. One block owns one request, appends that request's rows in
// order, and resolves the per-sequence slot from the descriptor. `positions` and the
// request descriptors are device memory, so a decode step does not bake a row count in.
struct QsaKeyCacheBatchArgs {
    const bf16_t* raw_keys = nullptr;
    const bf16_t* norm_weight = nullptr;
    const uint32_t* positions = nullptr;
    const ::shelf::runtime::DeviceRequestDescriptor* requests = nullptr;
    bf16_t* pooled_base = nullptr;
    bf16_t* partial_base = nullptr;
    uint32_t* pool_count_base = nullptr;
    uint32_t* partial_count_base = nullptr;
    bf16_t* partial_history = nullptr;
    uint32_t* count_history = nullptr;
    uint32_t history_rows = 0;
    uint64_t history_partial_state_stride = 0;
    uint64_t history_count_state_stride = 0;
    uint32_t num_requests = 0;
    uint32_t actual_rows = 0;
    uint32_t max_sequences = 0;
    uint32_t state_index = 0;
    uint32_t max_blocks = 0;
    uint32_t head_dim = 0;
    uint32_t compress_ratio = 0;
    uint32_t rotary_dim = 0;
    uint32_t raw_row_stride = 0;
    uint64_t pooled_slot_stride = 0;
    uint64_t pooled_layer_stride = 0;
    uint64_t partial_slot_stride = 0;
    uint64_t partial_layer_stride = 0;
    uint64_t count_slot_stride = 0;
    uint64_t count_layer_stride = 0;
    float theta = 10000000.0f;
    float position_scale = 1.0f;
    const double* inv_freq = nullptr;
    float eps = 1e-6f;
};

hipError_t launch_qsa_key_cache_batch(const QsaKeyCacheBatchArgs& args,
                                      hipStream_t stream);
hipError_t launch_qsa_key_cache_blocks(const QsaKeyCacheBatchArgs& args,
                                       hipStream_t stream);

}  // namespace shelf::kernel
