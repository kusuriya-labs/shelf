#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/tensor_parallel_context.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

#include <cstdint>

namespace shelf::qwen35moe {

// One rank's geometry for a model with routed experts: the shared backbone split, plus the
// expert axis and the shared expert's width. The router keeps every expert, so
// global_num_experts is also the width of the router logits on this rank.
// local_intermediate_size carries the shared expert's local width, which is the only
// token-row FFN width this family has.
Result<shelf::qwen_hybrid::TensorParallelContext> make_qwen35moe_tensor_parallel_context(
    const Qwen35MoeTextConfig& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    bool vocab_parallel_lm_head = false);

}  // namespace shelf::qwen35moe
