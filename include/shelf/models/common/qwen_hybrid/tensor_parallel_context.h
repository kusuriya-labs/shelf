#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/config.h>

#include <cstdint>

namespace shelf {
namespace qwen_hybrid {

// One rank's local geometry: what the backbone and this family's FFN look like after the
// split. A dense model divides one intermediate width; a model with routed experts divides
// the expert axis plus the shared expert's width. The executor, the state pools and the
// footprint all read the same numbers, so a rank cannot state two different geometries.
struct TensorParallelContext {
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    std::uint32_t local_intermediate_size = 0;
    std::uint32_t local_attention_heads = 0;
    std::uint32_t local_key_value_heads = 0;
    std::uint32_t local_gdn_key_heads = 0;
    std::uint32_t local_gdn_value_heads = 0;
    std::uint32_t local_vocab_size = 0;
    bool vocab_parallel_lm_head = false;

    // The expert axis. A model without routed experts leaves global_num_experts 0. The router
    // keeps the whole expert count on every rank, so global_num_experts is also the width of
    // the router logits.
    std::uint32_t global_num_experts = 0;
    std::uint32_t local_num_experts = 0;
    std::uint32_t expert_begin = 0;
    std::uint32_t local_shared_expert_intermediate_size = 0;
};

// The backbone part: attention and GDN head counts, and the vocab split. It does not look at
// any FFN width; the family that owns one divides it in its own builder.
Result<TensorParallelContext> make_qwen_hybrid_tensor_parallel_context(
    const QwenHybridConfig& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    bool vocab_parallel_lm_head = true);

}  // namespace qwen_hybrid
}  // namespace shelf
