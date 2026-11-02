#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

inline constexpr const char* kEmbeddingBf16Symbol =
    "shelf_embedding_bf16";

inline constexpr uint32_t kEmbeddingThreads = 256u;

struct EmbeddingBf16Args {
    const bf16_t* table = nullptr;
    const int32_t* token_ids = nullptr;
    bf16_t* output = nullptr;
    uint32_t* error_word = nullptr;
    uint32_t rows = 0;
    uint32_t vocab_size = 0;
    uint32_t hidden_size = 0;
    uint32_t output_row_stride = 0;
    uint32_t vocab_offset = 0;
    uint32_t partial_vocab = 0;
};

static_assert(sizeof(EmbeddingBf16Args) == 56);
static_assert(alignof(EmbeddingBf16Args) == 8);
static_assert(offsetof(EmbeddingBf16Args, table) == 0);
static_assert(offsetof(EmbeddingBf16Args, token_ids) == 8);
static_assert(offsetof(EmbeddingBf16Args, output) == 16);
static_assert(offsetof(EmbeddingBf16Args, error_word) == 24);
static_assert(offsetof(EmbeddingBf16Args, rows) == 32);
static_assert(offsetof(EmbeddingBf16Args, vocab_size) == 36);
static_assert(offsetof(EmbeddingBf16Args, hidden_size) == 40);
static_assert(offsetof(EmbeddingBf16Args, output_row_stride) == 44);
static_assert(offsetof(EmbeddingBf16Args, vocab_offset) == 48);
static_assert(offsetof(EmbeddingBf16Args, partial_vocab) == 52);

hipError_t launch_embedding_bf16(
    const bf16_t* table,
    const int32_t* token_ids,
    bf16_t* output,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t hidden_size,
    uint32_t output_row_stride,
    uint32_t* error_word,
    hipStream_t stream,
    uint32_t vocab_offset = 0u,
    uint32_t partial_vocab = 0u);
}  // namespace ps::kernel
