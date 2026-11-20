#include <shelf/models/qwen35moe/model/tensor_parallel_context.h>

namespace shelf::qwen35moe {

namespace {

Status require_divisible(std::uint64_t value, std::uint32_t divisor, const char* what) {
    if (divisor == 0u || (value % divisor) != 0) {
        return Status::unsupported(what, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Result<shelf::qwen_hybrid::TensorParallelContext> make_qwen35moe_tensor_parallel_context(
    const Qwen35MoeTextConfig& config, std::uint32_t tp_size, std::uint32_t tp_rank,
    std::uint32_t ep_size, std::uint32_t ep_rank, bool vocab_parallel_lm_head) {
    auto ctx = shelf::qwen_hybrid::make_qwen_hybrid_tensor_parallel_context(
        static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(config), tp_size, tp_rank,
        vocab_parallel_lm_head);
    if (!ctx.ok()) return ctx;
    if (ep_size == 0u || ep_rank >= ep_size) {
        return Status::invalid_argument("qwen35moe ep_size / ep_rank out of range",
                                        __FILE__, __LINE__);
    }
    if (config.ffn.num_experts == 0) {
        return Status::unsupported(
            "qwen35moe tp context: num_experts is 0 (not a MoE config)", __FILE__, __LINE__);
    }
    if (config.ffn.shared_expert_intermediate_size == 0) {
        return Status::unsupported(
            "qwen35moe tp context: shared_expert_intermediate_size is required", __FILE__,
            __LINE__);
    }
    Status st = require_divisible(
        config.ffn.num_experts, ep_size,
        "qwen35moe requires num_experts divisible by ep_size");
    if (!st.ok()) return st;
    if (tp_size > 1u) {
        st = require_divisible(
            config.ffn.shared_expert_intermediate_size, tp_size,
            "qwen35moe tp requires shared_expert_intermediate_size divisible by tp_size");
        if (!st.ok()) return st;
    }

    shelf::qwen_hybrid::TensorParallelContext& out = ctx.value();
    out.ep_size = ep_size;
    out.ep_rank = ep_rank;
    out.global_num_experts = static_cast<std::uint32_t>(config.ffn.num_experts);
    out.local_num_experts = static_cast<std::uint32_t>(config.ffn.num_experts / ep_size);
    out.expert_begin = ep_rank * out.local_num_experts;
    out.local_shared_expert_intermediate_size =
        static_cast<std::uint32_t>(config.ffn.shared_expert_intermediate_size / tp_size);
    out.local_intermediate_size = out.local_shared_expert_intermediate_size;
    return ctx;
}

Result<shelf::qwen_hybrid::TensorParallelContext> make_qwen35moe_tensor_parallel_context(
    const Qwen35MoeTextConfig& config, std::uint32_t tp_size, std::uint32_t tp_rank,
    bool vocab_parallel_lm_head) {
    return make_qwen35moe_tensor_parallel_context(
        config, tp_size, tp_rank, tp_size, tp_rank, vocab_parallel_lm_head);
}

}  // namespace shelf::qwen35moe
