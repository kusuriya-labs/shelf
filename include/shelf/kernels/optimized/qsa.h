#pragma once

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

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

// QSA block scores:
// score[r, b] = sum_h relu(dot(q[r,h], pooled[b])) / sqrt(head_dim).
// Blocks at or beyond visible[r] are written as -inf.
struct QsaScoresArgs {
    const bf16_t* q = nullptr;
    const bf16_t* pooled = nullptr;
    float* scores = nullptr;
    const uint32_t* visible = nullptr;
    uint32_t rows = 0;
    uint32_t blocks = 0;
    uint32_t n_heads = 0;
    uint32_t head_dim = 0;
    float scale = 0.0f;
};

inline constexpr uint32_t kQsaHeadDim = 128u;
inline constexpr uint32_t kQsaThreads = 128u;

// Per-row top-k over block scores. Ties go to the smaller block index. Entries past the
// visible block count, or past the number of selected blocks, are -1.
struct QsaTopkArgs {
    const float* scores = nullptr;
    const uint32_t* visible = nullptr;
    int32_t* indices = nullptr;
    uint32_t rows = 0;
    uint32_t blocks = 0;
    uint32_t k = 0;
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

hipError_t launch_qsa_pool(const QsaPoolArgs& args, hipStream_t stream);
hipError_t launch_qsa_scores(const QsaScoresArgs& args, hipStream_t stream);
hipError_t launch_qsa_topk(const QsaTopkArgs& args, hipStream_t stream);
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

// Expand selected block indices to a token mask. A token is visible when it is in the
// incomplete tail block (or at/before the query position) or in a selected complete block,
// and never after the query position.
struct QsaExpandMaskArgs {
    const int32_t* selected_blocks = nullptr;
    const uint32_t* positions = nullptr;
    uint8_t* mask = nullptr;
    uint32_t rows = 0;
    uint32_t selected_per_row = 0;
    uint32_t compress_ratio = 0;
    uint32_t total_len = 0;
};

hipError_t launch_qsa_expand_mask(const QsaExpandMaskArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
