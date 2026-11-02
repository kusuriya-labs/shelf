#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

constexpr uint32_t kQsaIndexerNumHeads = 4u;
constexpr uint32_t kQsaIndexerHeadDim = 128u;
constexpr uint32_t kQsaIndexerKvHeads = 1u;
constexpr uint32_t kQsaIndexerCompressRatio = 4u;
constexpr uint32_t kQsaIndexerBlockTopK = 512u;
constexpr uint32_t kQsaIndexerRotaryDim = 64u;

struct QsaIndexerArgs {
    const bf16_t* q_raw = nullptr;
    const bf16_t* key_cache = nullptr;
    const uint32_t* key_positions = nullptr;
    const bf16_t* q_weight = nullptr;
    const bf16_t* k_weight = nullptr;

    uint32_t query_begin = 0;
    uint32_t query_rows = 0;
    uint32_t key_tokens = 0;

    uint32_t num_heads = kQsaIndexerNumHeads;
    uint32_t head_dim = kQsaIndexerHeadDim;
    uint32_t compress_ratio = kQsaIndexerCompressRatio;
    uint32_t block_topk = kQsaIndexerBlockTopK;

    float rope_theta = 10000000.0f;
    float eps = 1e-6f;
    float score_scale = 0.0f;

    float* scores = nullptr;
    uint32_t scores_stride = 0;
    uint32_t* selected_blocks = nullptr;
    uint32_t* selected_block_counts = nullptr;
    uint8_t* mask = nullptr;

    void* scratch = nullptr;
    size_t scratch_bytes = 0;
    hipStream_t stream = nullptr;
};

uint32_t qsa_indexer_block_count(uint32_t key_tokens, uint32_t compress_ratio) noexcept;

uint32_t qsa_indexer_scores_stride(uint32_t key_tokens, uint32_t compress_ratio) noexcept;

size_t qsa_indexer_scratch_bytes(uint32_t query_rows, uint32_t key_tokens,
                                 uint32_t num_heads, uint32_t head_dim,
                                 uint32_t compress_ratio) noexcept;

hipError_t launch_qsa_indexer(const QsaIndexerArgs& args);

}
