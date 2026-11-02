#include <phaseshift/models/qwen35/kernels/dflash2/candidate_selector.h>
#include <phaseshift/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

using namespace ::ps::kernel::detail;

namespace {

constexpr uint32_t kSelectorThreads = kDflash2SelectorThreads;
constexpr uint32_t kSelectorMaxTopK = kDflash2SelectorMaxTopK;

__device__ __forceinline__ bool selector_better(float sa, int32_t ta, float sb, int32_t tb) {
    if (sa > sb) return true;
    if (sa < sb) return false;
    return ta < tb;
}

__device__ __forceinline__ void candidate_selector_body(
    const bf16* hidden_proj,
    const int32_t* topk_ids,
    const float* topk_logits,
    const bf16* predecessor_codebook,
    const bf16* successor_codebook,
    int32_t anchor_token,
    int32_t* output_tokens,
    float* output_scores,
    uint32_t draft_rows,
    uint32_t rank,
    uint32_t top_k,
    float* partial) {
    const uint32_t tid = threadIdx.x;
    const uint32_t per_candidate = kSelectorThreads / top_k;
    if (per_candidate == 0u) return;
    const uint32_t candidate = tid / per_candidate;
    const uint32_t lane = tid % per_candidate;

    int32_t predecessor = anchor_token;
    for (uint32_t row = 0u; row < draft_rows; ++row) {
        const bf16* hidden = hidden_proj + static_cast<std::size_t>(row) * rank;
        const bf16* pred_row =
            predecessor_codebook + static_cast<std::size_t>(predecessor) * rank;
        float acc = 0.0f;
        if (candidate < top_k) {
            const int32_t token = topk_ids[static_cast<std::size_t>(row) * top_k + candidate];
            const bf16* succ_row =
                successor_codebook + static_cast<std::size_t>(token) * rank;
            for (uint32_t d = lane; d < rank; d += per_candidate) {
                acc += bf16_to_f32(pred_row[d]) * bf16_to_f32(hidden[d]) *
                       bf16_to_f32(succ_row[d]);
            }
        }
        partial[tid] = acc;
        __syncthreads();
        for (uint32_t step = per_candidate / 2u; step > 0u; step >>= 1u) {
            if (candidate < top_k && lane < step) {
                partial[tid] += partial[tid + step];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            int32_t best_token = 0;
            float best_score = -INFINITY;
            for (uint32_t j = 0u; j < top_k; ++j) {
                const int32_t token = topk_ids[static_cast<std::size_t>(row) * top_k + j];
                const float score =
                    topk_logits[static_cast<std::size_t>(row) * top_k + j] +
                    partial[j * per_candidate];
                if (output_scores != nullptr) {
                    output_scores[static_cast<std::size_t>(row) * top_k + j] = score;
                }
                if (j == 0u || selector_better(score, token, best_score, best_token)) {
                    best_score = score;
                    best_token = token;
                }
            }
            output_tokens[row] = best_token;
            predecessor = best_token;
        }
        __syncthreads();
        predecessor = output_tokens[row];
        __syncthreads();
    }
}

__global__ void dflash2_candidate_selector_kernel(
    const bf16* hidden_proj,
    const int32_t* topk_ids,
    const float* topk_logits,
    const bf16* predecessor_codebook,
    const bf16* successor_codebook,
    int32_t anchor_token,
    int32_t* output_tokens,
    float* output_scores,
    uint32_t draft_rows,
    uint32_t rank,
    uint32_t top_k) {
    __shared__ float partial[kSelectorThreads];
    candidate_selector_body(hidden_proj, topk_ids, topk_logits, predecessor_codebook,
                            successor_codebook, anchor_token, output_tokens, output_scores,
                            draft_rows, rank, top_k, partial);
}

__global__ void dflash2_candidate_selector_requests_kernel(
    const bf16* hidden_proj,
    const int32_t* topk_ids,
    const float* topk_logits,
    const bf16* predecessor_codebook,
    const bf16* successor_codebook,
    const DFlash2RequestDescriptor* requests,
    float* output_scores,
    uint32_t rank,
    uint32_t top_k) {
    __shared__ float partial[kSelectorThreads];
    const DFlash2RequestDescriptor& request = requests[blockIdx.x];
    candidate_selector_body(
        hidden_proj + static_cast<std::size_t>(request.row_begin + 1u) * rank,
        topk_ids + static_cast<std::size_t>(request.row_begin + 1u) * top_k,
        topk_logits + static_cast<std::size_t>(request.row_begin + 1u) * top_k,
        predecessor_codebook, successor_codebook, request.anchor_token,
        request.output_drafts,
        output_scores != nullptr
            ? output_scores + static_cast<std::size_t>(request.row_begin + 1u) * top_k
            : nullptr,
        request.num_drafts, rank, top_k, partial);
}

}  // namespace

hipError_t launch_dflash2_candidate_selector(
    const DFlash2CandidateSelectorArgs& args,
    hipStream_t stream) {
    if (args.hidden_proj == nullptr || args.topk_ids == nullptr ||
        args.topk_logits == nullptr || args.predecessor_codebook == nullptr ||
        args.successor_codebook == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rank == 0u || args.top_k == 0u || args.top_k > kSelectorMaxTopK) {
        return hipErrorInvalidValue;
    }
    if (args.top_k > kSelectorThreads || (kSelectorThreads % args.top_k) != 0u) {
        return hipErrorInvalidValue;
    }
    if (args.requests != nullptr) {
        if (args.request_count == 0u) {
            return hipErrorInvalidValue;
        }
        dflash2_candidate_selector_requests_kernel<<<args.request_count, kSelectorThreads, 0,
                                                    stream>>>(
            reinterpret_cast<const bf16*>(args.hidden_proj), args.topk_ids, args.topk_logits,
            reinterpret_cast<const bf16*>(args.predecessor_codebook),
            reinterpret_cast<const bf16*>(args.successor_codebook), args.requests,
            args.output_scores, args.rank, args.top_k);
        return hipGetLastError();
    }
    if (args.output_tokens == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.draft_rows == 0u) return hipSuccess;
    if (args.anchor_token < 0 ||
        static_cast<uint32_t>(args.anchor_token) >= args.vocab_size) {
        return hipErrorInvalidValue;
    }
    dflash2_candidate_selector_kernel<<<1u, kSelectorThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(args.hidden_proj), args.topk_ids,
        args.topk_logits, reinterpret_cast<const bf16*>(args.predecessor_codebook),
        reinterpret_cast<const bf16*>(args.successor_codebook), args.anchor_token,
        args.output_tokens, args.output_scores, args.draft_rows, args.rank,
        args.top_k);
    return hipGetLastError();
}

}  // namespace ps::kernel
