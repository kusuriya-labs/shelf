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

hipError_t launch_qsa_pool(const QsaPoolArgs& args, hipStream_t stream);
hipError_t launch_qsa_scores(const QsaScoresArgs& args, hipStream_t stream);
hipError_t launch_qsa_topk(const QsaTopkArgs& args, hipStream_t stream);

}  // namespace shelf::kernel
