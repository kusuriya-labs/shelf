// vendored from PhaseShift (own work); namespace/type adapted only.
// (SYCL port of backend/hip kernels/elementwise/embedding.h; hipError_t ->
// shelf::Status.)
#pragma once

#include <cstdint>

#include <shelf/backend/sycl/sycl_api.hpp>
#include <shelf/foundation/numeric.hpp>
#include <shelf/status.hpp>

namespace shelf::syclk {

using shelf::bf16_t;

// Mirrors PhaseShift's ps::runtime::ProgramStatus wire values; the embedding
// kernel only writes COMPLETE -> INVALID_BINDING through `error_word`.
// (shelf has no runtime/program module yet, so the enum is vendored here.)
enum class ProgramStatus : uint32_t {
    COMPLETE = 0,
    INVALID_PROGRAM = 1,
    INVALID_OPCODE = 2,
    INVALID_BINDING = 3,
    DISPATCH_FAILED = 4,
    UNSUPPORTED_P2P = 5,
};

inline constexpr uint32_t kEmbeddingThreads = 256u;

// vocab_offset / partial_vocab support a TP-sharded table: rows outside
// [vocab_offset, vocab_offset + vocab_size) are either zero-filled
// (partial_vocab != 0) or flagged INVALID_BINDING in *error_word.
::shelf::Status launch_embedding_bf16(
    const bf16_t* table,
    const int32_t* token_ids,
    bf16_t* output,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t hidden_size,
    uint32_t output_row_stride,
    uint32_t* error_word,
    const ::shelf::sycl::Stream& stream,
    uint32_t vocab_offset = 0u,
    uint32_t partial_vocab = 0u);

} // namespace shelf::syclk
