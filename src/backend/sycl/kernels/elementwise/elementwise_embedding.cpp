// vendored from PhaseShift (own work); namespace/type adapted only.
//
// PhaseShift included <phaseshift/runtime/program/device_program.h> here for
// ProgramStatus (the device error-word protocol) and an unused e4m3 codec
// header; shelf vendors the enum into the public header.
// (SYCL port of backend/hip kernels/elementwise/elementwise_embedding.hip.)
#include <shelf/backend/sycl/kernels/elementwise/embedding.h>
#include <shelf/backend/sycl/kernels/detail/sycl_util.h>

#include <algorithm>
#include <cstdint>

namespace shelf::syclk {

namespace {
namespace op_embedding_bf16 {

constexpr uint32_t kThreads = kEmbeddingThreads;
static_assert(kThreads == 256u);
constexpr uint32_t kVec = 8u;

struct Args {
    const bf16_t* table;
    const int32_t* token_ids;
    bf16_t* output;
    uint32_t* error_word;
    uint32_t rows, vocab_size, hidden_size, output_row_stride;
    uint32_t vocab_offset, partial_vocab;
};

inline void body(const Args& args, uint32_t row, uint32_t tid, uint32_t bdim) {
    if (row >= args.rows) return;
    const int32_t token = args.token_ids[row];
    const uint32_t first = args.vocab_offset;
    const uint32_t last = first + args.vocab_size;
    const bool valid = token >= 0 && static_cast<uint32_t>(token) >= first &&
                       static_cast<uint32_t>(token) < last;
    bf16_t* dst = args.output + static_cast<size_t>(row) * args.output_row_stride;
    if (!valid) {
        if (args.partial_vocab != 0u) {
            for (uint32_t base = tid * kVec; base < args.hidden_size;
                 base += bdim * kVec) {
                const uint32_t end = std::min(base + kVec, args.hidden_size);
                for (uint32_t i = base; i < end; ++i) dst[i] = bf16_t{};
            }
            return;
        }
        if (tid == 0u && args.error_word != nullptr) {
            uint32_t expected = static_cast<uint32_t>(ProgramStatus::COMPLETE);
            ::sycl::atomic_ref<uint32_t, ::sycl::memory_order::relaxed,
                               ::sycl::memory_scope::device>
                ref(*args.error_word);
            ref.compare_exchange_strong(
                expected, static_cast<uint32_t>(ProgramStatus::INVALID_BINDING));
        }
        return;
    }
    const int32_t local_token = token - static_cast<int32_t>(first);
    const bf16_t* src =
        args.table + static_cast<size_t>(local_token) * args.hidden_size;
    for (uint32_t base = tid * kVec; base < args.hidden_size;
         base += bdim * kVec) {
        if (base + kVec <= args.hidden_size &&
            (reinterpret_cast<uintptr_t>(src + base) & 15u) == 0u &&
            (reinterpret_cast<uintptr_t>(dst + base) & 15u) == 0u) {
            const U4 v = [&] { U4 r; std::memcpy(&r, src + base, 16u); return r; }();
            std::memcpy(dst + base, &v, 16u);
        } else {
            const uint32_t end = std::min(base + kVec, args.hidden_size);
            for (uint32_t i = base; i < end; ++i)
                dst[i] = src[i];
        }
    }
}

} // namespace op_embedding_bf16
} // namespace

::shelf::Status launch_embedding_bf16(
    const bf16_t* table, const int32_t* token_ids, bf16_t* output,
    uint32_t rows, uint32_t vocab_size, uint32_t hidden_size,
    uint32_t output_row_stride, uint32_t* error_word, const ::shelf::sycl::Stream& stream,
    uint32_t vocab_offset, uint32_t partial_vocab) {
    if (rows == 0u) return ::shelf::Status::make_ok();
    if (hidden_size == 0u)
        return ::shelf::Status::invalid_argument("hidden_size == 0", __FILE__, __LINE__);
    if (table == nullptr || token_ids == nullptr || output == nullptr)
        return ::shelf::Status::invalid_argument("null pointer argument", __FILE__, __LINE__);
    op_embedding_bf16::Args args{table, token_ids, output, error_word,
                                 rows, vocab_size, hidden_size, output_row_stride,
                                 vocab_offset, partial_vocab};
    ::sycl::queue& q = stream.queue();
    return ::shelf::syclk::sycl_submit(q, "sycl-submit", __FILE__, __LINE__, [&](::sycl::handler& cgh) {
        cgh.parallel_for(
            ::sycl::nd_range<1>{::sycl::range<1>{rows * op_embedding_bf16::kThreads},
                                ::sycl::range<1>{op_embedding_bf16::kThreads}},
            [=](::sycl::nd_item<1> it) {
                op_embedding_bf16::body(
                    args,
                    static_cast<uint32_t>(it.get_group(0)),
                    static_cast<uint32_t>(it.get_local_linear_id()),
                    static_cast<uint32_t>(it.get_local_range(0)));
            });
    });
}

} // namespace shelf::syclk
