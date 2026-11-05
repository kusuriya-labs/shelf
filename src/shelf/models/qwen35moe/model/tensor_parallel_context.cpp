#include <shelf/models/qwen35moe/model/tensor_parallel_context.h>

namespace shelf::qwen35moe {

namespace {

Status require_divisible(std::uint64_t value, std::uint32_t tp_size, const char* what) {
    if ((value % tp_size) != 0) {
        return Status::unsupported(what, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Result<shelf::qwen_hybrid::TensorParallelContext> make_qwen35moe_tensor_parallel_context(
    const Qwen35MoeTextConfig& config, std::uint32_t tp_size, std::uint32_t tp_rank,
    bool vocab_parallel_lm_head) {
    auto ctx = shelf::qwen_hybrid::make_qwen_hybrid_tensor_parallel_context(
        static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(config), tp_size, tp_rank,
        vocab_parallel_lm_head);
    if (!ctx.ok()) return ctx;
    if (config.ffn.num_experts == 0) {
        return Status::unsupported(
            "qwen35moe tp context: num_experts is 0 (not a MoE config)", __FILE__, __LINE__);
    }
    if (config.ffn.shared_expert_intermediate_size == 0) {
        return Status::unsupported(
            "qwen35moe tp context: shared_expert_intermediate_size is required", __FILE__,
            __LINE__);
    }
    if (tp_size > 1) {
        Status st = require_divisible(
            config.ffn.num_experts, tp_size,
            "qwen35moe tp requires num_experts divisible by tp_size");
        if (!st.ok()) return st;
        st = require_divisible(
            config.ffn.shared_expert_intermediate_size, tp_size,
            "qwen35moe tp requires shared_expert_intermediate_size divisible by tp_size");
        if (!st.ok()) return st;
    }

    shelf::qwen_hybrid::TensorParallelContext& out = ctx.value();
    out.global_num_experts = static_cast<std::uint32_t>(config.ffn.num_experts);
    out.local_num_experts = static_cast<std::uint32_t>(config.ffn.num_experts / tp_size);
    out.expert_begin = tp_rank * out.local_num_experts;
    out.local_shared_expert_intermediate_size =
        static_cast<std::uint32_t>(config.ffn.shared_expert_intermediate_size / tp_size);
    out.local_intermediate_size = out.local_shared_expert_intermediate_size;
    return ctx;
}

}  // namespace shelf::qwen35moe
