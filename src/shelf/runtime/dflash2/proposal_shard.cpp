#include <shelf/runtime/dflash2/proposal_shard.h>

#include <shelf/kernels/dflash2/proposal_stats.h>
#include <shelf/kernels/dflash2/topk.h>
#include <shelf/kernels/fused/exl3/decode.h>
#include <shelf/kernels/fused/exl3/rotate_quantize.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/kernels/optimized/linear/lm_head_int8.h>
#include <shelf/kernels/optimized/linear/exl3_mul24.h>
#include <shelf/runtime/program/int8_activation_workspace.h>

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

bool exl3_head_layout_valid(const shelf::weights::MatrixWeight& head) {
    using shelf::weights::MatrixEncoding;
    if (head.encoding != MatrixEncoding::Exl3) return true;
    if (head.preshuffled || head.codes.ndim() == 0 || head.exl3_suh.ndim() == 0 ||
        head.exl3_svh.ndim() == 0) {
        return false;
    }
    if (head.k_padded != head.cols || (head.rows % 128u) != 0u ||
        (head.cols % 128u) != 0u) {
        return false;
    }
    if (head.exl3_ka < 1u || head.exl3_ka > 8u ||
        (head.exl3_half && head.exl3_ka > 3u)) {
        return false;
    }
    return head.quant_spec == shelf::quantization::QuantSpecId::EXL3_MUL24_INT8_V1;
}

Status proposal_shard_exl3(
    const shelf::weights::MatrixWeight& head,
    const ProposalShardBuffers& buffers,
    const bf16_t* hidden,
    uint32_t hidden_stride,
    uint32_t rows,
    hipStream_t stream) {
    if (!exl3_head_layout_valid(head)) {
        return Status::invalid_argument("proposal shard exl3 head layout is invalid",
                                        __FILE__, __LINE__);
    }
    if (head.quant_spec == shelf::quantization::QuantSpecId::EXL3_MUL24_INT8_V1) {
        if (head.exl3_half || !buffers.exl3_mul24 || buffers.exl3_codes.ndim() == 0 ||
            buffers.exl3_scales.ndim() == 0 || buffers.exl3_accum.ndim() == 0) {
            return Status::invalid_argument("proposal shard exl3 int8 workspace is missing",
                                            __FILE__, __LINE__);
        }
        const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(
            head.k_padded, buffers.max_rows);
        if (layout.k_padded != head.k_padded) {
            return Status::invalid_state("proposal shard activation k mismatch", __FILE__,
                                         __LINE__);
        }
        auto* codes = buffers.exl3_codes.data<int8_t>();
        auto* scales = buffers.exl3_scales.data<float>();
        auto* accum = buffers.exl3_accum.data<float>();
        hipError_t err = shelf::kernel::launch_exl3_mul24_rotate_quantize(
            hidden, hidden_stride, head.exl3_suh.data<uint16_t>(), codes, scales, rows,
            head.cols, layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream);
        if (err == hipSuccess) {
            err = shelf::kernel::launch_exl3_mul24_gemm(
                codes, scales, layout.code_row_stride_bytes, layout.scale_row_stride_bytes,
                head.codes.data<uint16_t>(), head.exl3_ka, accum, rows, head.rows,
                head.cols, 1u, stream);
        }
        if (err == hipSuccess) {
            err = shelf::kernel::launch_exl3_mul24_rotate_out(
                accum, head.exl3_svh.data<uint16_t>(), buffers.logits.data<float>(),
                shelf::kernel::Exl3OutputDType::F32, rows, head.rows, buffers.local_vocab,
                1u, stream);
        }
        if (err != hipSuccess) {
            return Status::hip_error("proposal shard exl3 int8 launch",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        return Status::make_ok();
    }

    return Status::unsupported("proposal shard exl3 format unsupported", __FILE__, __LINE__);
}

}  // namespace

Result<ProposalShardBuffers> create_proposal_shard_buffers(
    gpu::GpuArena& arena,
    const shelf::weights::MatrixWeight& head,
    uint32_t max_rows,
    uint32_t local_vocab,
    uint32_t top_k) {
    using shelf::weights::MatrixEncoding;
    if (max_rows == 0u || local_vocab == 0u || top_k == 0u ||
        top_k > shelf::kernel::kDflash2TopKMaxK) {
        return Status::invalid_argument("proposal shard geometry is invalid", __FILE__,
                                        __LINE__);
    }
    if (head.rows != local_vocab || head.cols == 0u) {
        return Status::invalid_argument("proposal shard head does not match the shard",
                                        __FILE__, __LINE__);
    }
    if (head.encoding != MatrixEncoding::Int8W32 && head.encoding != MatrixEncoding::Exl3 &&
        head.encoding != MatrixEncoding::Bf16) {
        return Status::unsupported("proposal shard head encoding is unsupported", __FILE__,
                                   __LINE__);
    }
    if (head.encoding == MatrixEncoding::Exl3 && !exl3_head_layout_valid(head)) {
        return Status::invalid_argument("proposal shard exl3 head layout is invalid",
                                        __FILE__, __LINE__);
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

    if (head.encoding == MatrixEncoding::Exl3 &&
        head.quant_spec == shelf::quantization::QuantSpecId::EXL3_MUL24_INT8_V1) {
        const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(
            head.k_padded, max_rows);
        if (layout.k_padded != head.k_padded) {
            return Status::invalid_argument("proposal shard activation k is not supported",
                                            __FILE__, __LINE__);
        }
        st = allocate_contiguous<int8_t>(arena, {layout.codes_bytes()}, buffers.exl3_codes);
        if (!st.ok()) return st;
        st = allocate_contiguous<float>(
            arena, {layout.scales_bytes() / sizeof(float)}, buffers.exl3_scales);
        if (!st.ok()) return st;
        st = allocate_contiguous<float>(arena, {max_rows, local_vocab}, buffers.exl3_accum);
        if (!st.ok()) return st;
        buffers.exl3_mul24 = true;
    }
    return buffers;
}

namespace {

Status proposal_shard_finish(
    const ProposalShardBuffers& buffers,
    uint32_t rows,
    uint32_t vocab_offset,
    hipStream_t stream) {
    hipError_t err = shelf::kernel::launch_dflash2_topk_f32(
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

}  // namespace

Status proposal_shard_compute(
    const shelf::weights::MatrixWeight& head,
    const ProposalShardBuffers& buffers,
    const bf16_t* hidden,
    uint32_t hidden_stride,
    uint32_t rows,
    uint32_t vocab_offset,
    hipStream_t stream) {
    using shelf::weights::MatrixEncoding;
    if (hidden == nullptr) {
        return Status::invalid_argument("proposal shard hidden is null", __FILE__, __LINE__);
    }
    if (rows == 0u || rows > buffers.max_rows) {
        return Status::invalid_argument("proposal shard rows out of range", __FILE__,
                                        __LINE__);
    }
    if (head.encoding != MatrixEncoding::Int8W32 && head.encoding != MatrixEncoding::Exl3 &&
        head.encoding != MatrixEncoding::Bf16) {
        return Status::unsupported("proposal shard head encoding is unsupported", __FILE__,
                                   __LINE__);
    }
    if (head.rows != buffers.local_vocab || head.cols == 0u ||
        hidden_stride < head.cols) {
        return Status::invalid_argument("proposal shard head geometry mismatch", __FILE__,
                                        __LINE__);
    }
    if (head.encoding == MatrixEncoding::Exl3) {
        Status st = proposal_shard_exl3(head, buffers, hidden, hidden_stride, rows, stream);
        if (!st.ok()) return st;
        return proposal_shard_finish(buffers, rows, vocab_offset, stream);
    }
    if (head.encoding == MatrixEncoding::Bf16) {
        const void* weight = head.bf16.ndim() != 0u ? head.bf16.data<void>() : nullptr;
        if (weight == nullptr) {
            return Status::invalid_argument("proposal shard head payload missing", __FILE__,
                                            __LINE__);
        }
        shelf::kernel::Bf16GemmConfig config;
        if (rows <= shelf::kernel::kBf16GemmExactRowsKernelMax) {
            config.id = shelf::kernel::Bf16GemmConfigId::ExactRows;
            config.exact_rows = static_cast<std::uint8_t>(rows);
        } else {
            config.id = shelf::kernel::Bf16GemmConfigId::Wmma;
        }
        const hipError_t herr = shelf::kernel::launch_gemm_bf16(
            config, static_cast<const bf16_t*>(weight), hidden, buffers.logits.data<float>(),
            shelf::kernel::Bf16GemmOutputDType::F32, rows, head.rows, head.cols,
            hidden_stride, buffers.local_vocab, stream);
        if (herr != hipSuccess) {
            return Status::hip_error("proposal shard bf16 launch", hipGetErrorString(herr),
                                     __FILE__, __LINE__);
        }
        return proposal_shard_finish(buffers, rows, vocab_offset, stream);
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
    return proposal_shard_finish(buffers, rows, vocab_offset, stream);
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
