#include <shelf/runtime/dflash2/proposal_shard.h>

#include <shelf/kernels/dflash2/proposal_stats.h>
#include <shelf/kernels/dflash2/topk.h>
#include <shelf/kernels/optimized/linear/lm_head_int8.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace shelf::runtime::dflash2 {

namespace {

template <typename T>
Status allocate_contiguous(
    gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    gpu::Tensor& out) {
    std::size_t count = 1;
    for (std::size_t d : shape) {
        if (d == 0u) {
            return Status::invalid_argument("proposal shard shape has a zero dim", __FILE__,
                                            __LINE__);
        }
        if (std::numeric_limits<std::size_t>::max() / count < d) {
            return Status::invalid_argument("proposal shard element count overflow",
                                            __FILE__, __LINE__);
        }
        count *= d;
    }
    auto view = arena.allocate_aligned(count * sizeof(T), 128);
    if (!view.ok()) return view.status();
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto result = gpu::Tensor::create<T>(backing, shape, strides);
    if (!result.ok()) return result.status();
    out = result.release();
    return Status::make_ok();
}

}  // namespace

Result<ProposalShardBuffers> create_proposal_shard_buffers(
    gpu::GpuArena& arena,
    uint32_t max_rows,
    uint32_t local_vocab,
    uint32_t top_k) {
    if (max_rows == 0u || local_vocab == 0u || top_k == 0u ||
        top_k > shelf::kernel::kDflash2TopKMaxK) {
        return Status::invalid_argument("proposal shard geometry is invalid", __FILE__,
                                        __LINE__);
    }
    ProposalShardBuffers buffers;
    buffers.max_rows = max_rows;
    buffers.local_vocab = local_vocab;
    buffers.top_k = top_k;
    Status st = allocate_contiguous<float>(arena, {max_rows, local_vocab}, buffers.logits);
    if (!st.ok()) return st;
    st = allocate_contiguous<int32_t>(arena, {max_rows, top_k}, buffers.topk_ids);
    if (!st.ok()) return st;
    st = allocate_contiguous<float>(arena, {max_rows, top_k}, buffers.topk_logits);
    if (!st.ok()) return st;
    st = allocate_contiguous<int32_t>(
        arena, {max_rows, shelf::kernel::kDflash2TopKScratchPerRow}, buffers.scratch_ids);
    if (!st.ok()) return st;
    st = allocate_contiguous<float>(
        arena, {max_rows, shelf::kernel::kDflash2TopKScratchPerRow}, buffers.scratch_logits);
    if (!st.ok()) return st;
    st = allocate_contiguous<float>(arena, {max_rows, 2u}, buffers.stats);
    if (!st.ok()) return st;
    return buffers;
}

Status proposal_shard_compute(
    const shelf::weights::MatrixWeight& head,
    const ProposalShardBuffers& buffers,
    const bf16_t* hidden,
    uint32_t hidden_stride,
    uint32_t rows,
    uint32_t vocab_offset,
    hipStream_t stream) {
    if (hidden == nullptr) {
        return Status::invalid_argument("proposal shard hidden is null", __FILE__, __LINE__);
    }
    if (rows == 0u || rows > buffers.max_rows) {
        return Status::invalid_argument("proposal shard rows out of range", __FILE__,
                                        __LINE__);
    }
    if (head.encoding != shelf::weights::MatrixEncoding::Int8W32) {
        return Status::unsupported("proposal shard head must be fp8 w32", __FILE__,
                                   __LINE__);
    }
    if (head.rows != buffers.local_vocab || head.cols == 0u ||
        hidden_stride < head.cols) {
        return Status::invalid_argument("proposal shard head geometry mismatch", __FILE__,
                                        __LINE__);
    }
    if (head.codes.ndim() == 0 || head.scales.ndim() == 0) {
        return Status::invalid_argument("proposal shard head payload missing", __FILE__,
                                        __LINE__);
    }
    hipError_t err = shelf::kernel::launch_lm_head_int8_w32(
        hidden, hidden_stride, head.codes.data<uint8_t>(), head.scales.data<shelf::fp16_t>(),
        buffers.logits.data<float>(), buffers.local_vocab, rows, buffers.local_vocab,
        head.cols, stream);
    if (err != hipSuccess) {
        return Status::hip_error("proposal shard lm_head launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    err = shelf::kernel::launch_dflash2_topk_f32(
        buffers.logits.data<float>(), rows, buffers.local_vocab, buffers.local_vocab,
        buffers.top_k, buffers.topk_ids.data<int32_t>(), buffers.topk_logits.data<float>(),
        buffers.scratch_ids.data<int32_t>(), buffers.scratch_logits.data<float>(),
        static_cast<int32_t>(vocab_offset), stream);
    if (err != hipSuccess) {
        return Status::hip_error("proposal shard topk launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    err = shelf::kernel::launch_dflash2_row_logsumexp(
        buffers.logits.data<float>(), rows, buffers.local_vocab, buffers.local_vocab,
        buffers.stats.data<float>(), stream);
    if (err != hipSuccess) {
        return Status::hip_error("proposal shard logsumexp launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status proposal_merge_candidates(
    const int32_t* candidate_ids,
    const float* candidate_logits,
    uint32_t ranks,
    uint32_t rank_stride,
    uint32_t rows,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    hipStream_t stream) {
    const hipError_t err = shelf::kernel::launch_dflash2_merge_ranked_topk(
        candidate_ids, candidate_logits, ranks, rank_stride, rows, k, out_ids, out_logits,
        stream);
    if (err != hipSuccess) {
        return Status::hip_error("proposal merge topk launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status proposal_combine_denominators(
    const float* rank_stats,
    uint32_t ranks,
    uint32_t rank_stride,
    uint32_t rows,
    float* out,
    hipStream_t stream) {
    const hipError_t err = shelf::kernel::launch_dflash2_combine_denominators(
        rank_stats, ranks, rank_stride, rows, out, stream);
    if (err != hipSuccess) {
        return Status::hip_error("proposal denominator launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace shelf::runtime::dflash2
