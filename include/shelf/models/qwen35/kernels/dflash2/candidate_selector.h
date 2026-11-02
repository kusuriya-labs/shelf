#pragma once

#include <shelf/core/memory/types.h>
#include <shelf/models/qwen35/kernels/dflash2/batch.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

inline constexpr uint32_t kDflash2SelectorThreads = 256u;
inline constexpr uint32_t kDflash2SelectorMaxTopK = 16u;

struct DFlash2CandidateSelectorArgs {
    const bf16_t* hidden_proj = nullptr;
    const int32_t* topk_ids = nullptr;
    const float* topk_logits = nullptr;
    const bf16_t* predecessor_codebook = nullptr;
    const bf16_t* successor_codebook = nullptr;
    int32_t anchor_token = 0;
    int32_t* output_tokens = nullptr;
    float* output_scores = nullptr;
    uint32_t draft_rows = 0;
    uint32_t rank = 0;
    uint32_t top_k = 0;
    uint32_t vocab_size = 0;
    const DFlash2RequestDescriptor* requests = nullptr;
    uint32_t request_count = 0;
};

hipError_t launch_dflash2_candidate_selector(
    const DFlash2CandidateSelectorArgs& args,
    hipStream_t stream);

}  // namespace shelf::kernel
