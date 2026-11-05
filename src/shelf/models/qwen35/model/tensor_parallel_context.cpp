#include <shelf/models/qwen35/model/tensor_parallel_context.h>

namespace shelf {
namespace qwen35 {

namespace {

Status require_divisible(std::uint64_t value, std::uint32_t tp_size, const char* what) {
    if ((value % tp_size) != 0) {
        return Status::unsupported(what, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}

Result<Qwen35TensorParallelContext> make_qwen35_tensor_parallel_context(
    const Qwen35TextConfig& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    bool vocab_parallel_lm_head) {
    auto ctx = shelf::qwen_hybrid::make_qwen_hybrid_tensor_parallel_context(
        static_cast<const shelf::qwen_hybrid::QwenHybridConfig&>(config), tp_size, tp_rank,
        vocab_parallel_lm_head);
    if (!ctx.ok()) return ctx;
    if (tp_size > 1) {
        Status st = require_divisible(
            config.ffn.intermediate_size, tp_size,
            "qwen35 tp requires intermediate_size divisible by tp_size");
        if (!st.ok()) return st;
    }
    ctx.value().local_intermediate_size =
        static_cast<std::uint32_t>(config.ffn.intermediate_size / tp_size);
    return ctx;
}

}
}
