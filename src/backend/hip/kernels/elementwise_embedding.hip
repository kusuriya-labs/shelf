// vendored from PhaseShift (own work); namespace/type adapted only.
//
// PhaseShift included <phaseshift/runtime/program/device_program.h> here for
// ProgramStatus (the device error-word protocol) and an unused e4m3 codec
// header; shelf vendors the enum into <shelf/backend/hip/kernels/embedding.h>.
#include <shelf/backend/hip/kernels/embedding.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cstdint>

namespace shelf::hipk {

namespace {
namespace op_embedding_bf16 {

constexpr uint32_t kThreads = kEmbeddingThreads;
static_assert(kThreads == 256u);
constexpr uint32_t kVec = 8u;

__device__ __forceinline__ void body(const EmbeddingBf16Args& args) {
    const uint32_t row = blockIdx.x;
    if (row >= args.rows) return;
    const int32_t token = args.token_ids[row];
    const uint32_t first = args.vocab_offset;
    const uint32_t last = first + args.vocab_size;
    const bool valid = token >= 0 && static_cast<uint32_t>(token) >= first &&
                       static_cast<uint32_t>(token) < last;
    bf16_t* __restrict__ dst =
        args.output + static_cast<size_t>(row) * args.output_row_stride;
    if (!valid) {
        if (args.partial_vocab != 0u) {
            for (uint32_t base = threadIdx.x * kVec; base < args.hidden_size;
                 base += static_cast<uint32_t>(blockDim.x) * kVec) {
                const uint32_t end = min(base + kVec, args.hidden_size);
                for (uint32_t i = base; i < end; ++i) dst[i] = bf16_t{};
            }
            return;
        }
        if (threadIdx.x == 0u && args.error_word != nullptr) {
            atomicCAS(args.error_word,
                      static_cast<uint32_t>(::shelf::hipk::ProgramStatus::COMPLETE),
                      static_cast<uint32_t>(::shelf::hipk::ProgramStatus::INVALID_BINDING));
        }
        return;
    }
    const int32_t local_token = token - static_cast<int32_t>(first);
    const bf16_t* __restrict__ src =
        args.table + static_cast<size_t>(local_token) * args.hidden_size;
    for (uint32_t base = threadIdx.x * kVec; base < args.hidden_size;
         base += static_cast<uint32_t>(blockDim.x) * kVec) {
        if (base + kVec <= args.hidden_size &&
            (reinterpret_cast<uintptr_t>(src + base) & 15u) == 0u &&
            (reinterpret_cast<uintptr_t>(dst + base) & 15u) == 0u) {
            const uint4 v = *reinterpret_cast<const uint4*>(src + base);
            *reinterpret_cast<uint4*>(dst + base) = v;
        } else {
            const uint32_t end = min(base + kVec, args.hidden_size);
            for (uint32_t i = base; i < end; ++i)
                dst[i] = src[i];
        }
    }
}

}  // namespace op_embedding_bf16
}  // namespace

extern "C" __global__ void phaseshift_embedding_bf16(
    EmbeddingBf16Args args) {
    op_embedding_bf16::body(args);
}

hipError_t launch_embedding_bf16(
    const bf16_t* table, const int32_t* token_ids, bf16_t* output,
    uint32_t rows, uint32_t vocab_size, uint32_t hidden_size,
    uint32_t output_row_stride, uint32_t* error_word, hipStream_t stream,
    uint32_t vocab_offset, uint32_t partial_vocab) {
    if (rows == 0u) return hipSuccess;
    if (hidden_size == 0u) return hipErrorInvalidValue;
    if (table == nullptr || token_ids == nullptr || output == nullptr)
        return hipErrorInvalidValue;
    EmbeddingBf16Args args{};
    args.table = table;
    args.token_ids = token_ids;
    args.output = output;
    args.error_word = error_word;
    args.rows = rows;
    args.vocab_size = vocab_size;
    args.hidden_size = hidden_size;
    args.output_row_stride = output_row_stride;
    args.vocab_offset = vocab_offset;
    args.partial_vocab = partial_vocab;
    phaseshift_embedding_bf16<<<rows, op_embedding_bf16::kThreads, 0, stream>>>(
        args);
    return hipGetLastError();
}

namespace {
}  // namespace

}  // namespace shelf::hipk
