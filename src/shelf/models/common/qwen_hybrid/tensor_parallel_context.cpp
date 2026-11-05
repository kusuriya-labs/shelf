#include <shelf/models/common/qwen_hybrid/tensor_parallel_context.h>

namespace shelf {
namespace qwen_hybrid {

namespace {

Status require_divisible(std::uint64_t value, std::uint32_t tp_size, const char* what) {
    if ((value % tp_size) != 0) {
        return Status::unsupported(what, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Result<TensorParallelContext> make_qwen_hybrid_tensor_parallel_context(
    const QwenHybridConfig& config, std::uint32_t tp_size, std::uint32_t tp_rank,
    bool vocab_parallel_lm_head) {
    if (tp_size == 0) {
        return Status::invalid_argument("tp_size must be at least 1", __FILE__, __LINE__);
    }
    if (tp_rank >= tp_size) {
        return Status::invalid_argument("tp_rank out of range", __FILE__, __LINE__);
    }

    TensorParallelContext ctx;
    ctx.tp_size = tp_size;
    ctx.tp_rank = tp_rank;

    if (tp_size > 1) {
        Status st = require_divisible(
            config.num_attention_heads, tp_size,
            "hybrid tp requires num_attention_heads divisible by tp_size");
        if (!st.ok()) return st;
        st = require_divisible(config.num_key_value_heads, tp_size,
                               "hybrid tp requires num_key_value_heads divisible by tp_size");
        if (!st.ok()) return st;
        st = require_divisible(config.linear_num_key_heads, tp_size,
                               "hybrid tp requires linear_num_key_heads divisible by tp_size");
        if (!st.ok()) return st;
        st = require_divisible(config.linear_num_value_heads, tp_size,
                               "hybrid tp requires linear_num_value_heads divisible by tp_size");
        if (!st.ok()) return st;
    }

    ctx.local_attention_heads = static_cast<std::uint32_t>(config.num_attention_heads / tp_size);
    ctx.local_key_value_heads = static_cast<std::uint32_t>(config.num_key_value_heads / tp_size);
    ctx.local_gdn_key_heads = static_cast<std::uint32_t>(config.linear_num_key_heads / tp_size);
    ctx.local_gdn_value_heads = static_cast<std::uint32_t>(config.linear_num_value_heads / tp_size);

    if (tp_size == 1 || config.tie_word_embeddings || !vocab_parallel_lm_head) {
        ctx.local_vocab_size = static_cast<std::uint32_t>(config.vocab_size);
        ctx.vocab_parallel_lm_head = false;
        return ctx;
    }

    if (config.vocab_size == 0) {
        return Status::invalid_argument("hybrid vocab_size must be non-zero", __FILE__, __LINE__);
    }
    Status vocab_status = require_divisible(config.vocab_size, tp_size,
                                            "hybrid tp requires vocab_size divisible by tp_size");
    if (!vocab_status.ok()) return vocab_status;

    ctx.local_vocab_size = static_cast<std::uint32_t>(config.vocab_size / tp_size);
    ctx.vocab_parallel_lm_head = true;
    return ctx;
}

}  // namespace qwen_hybrid
}  // namespace shelf
