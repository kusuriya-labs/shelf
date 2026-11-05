#pragma once

#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/tensor_parallel_context.h>
#include <shelf/models/qwen35/model/qwen35_config.h>

#include <cstdint>

namespace shelf {
namespace qwen35 {

// The rank geometry a dense model runs with. The shared fields come from the hybrid
// backbone context; the dense FFN width is the one extra term.
using Qwen35TensorParallelContext = shelf::qwen_hybrid::TensorParallelContext;

Result<Qwen35TensorParallelContext> make_qwen35_tensor_parallel_context(
    const Qwen35TextConfig& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    bool vocab_parallel_lm_head = true);

}
}
