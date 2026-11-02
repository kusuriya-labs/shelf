#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>

#include <cstdint>

namespace ps::qwen35 {

struct Qwen35TensorParallelContext {
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    std::uint32_t local_intermediate_size = 0;
    std::uint32_t local_attention_heads = 0;
    std::uint32_t local_key_value_heads = 0;
    std::uint32_t local_gdn_key_heads = 0;
    std::uint32_t local_gdn_value_heads = 0;
    std::uint32_t local_vocab_size = 0;
    bool vocab_parallel_lm_head = false;
};

Result<Qwen35TensorParallelContext> make_qwen35_tensor_parallel_context(
    const Qwen35TextConfig& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    bool vocab_parallel_lm_head = true);

}
