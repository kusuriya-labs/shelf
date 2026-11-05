#pragma once

#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/executor.h>

#include <cstddef>

namespace shelf::qwen35 {
namespace runtime {

std::size_t qwen35_padded_row_count(std::uint32_t max_batch_rows) noexcept;

// The bytes an arena must leave for the executor: its staging, output buffers, state
// descriptors and the program workspace estimate. `ffn_intermediate_size` is the model's
// dense FFN width, or 0 when the model has none. An FFN that runs on grouped rows sizes its
// own planes by the plan capacity instead, and the caller adds that part here; it does not
// scale with a dense intermediate width. `exl3_linear_workspace` says whether the model
// carries its linears in the rotated-domain trellis form: that form's staging planes are
// large and only a program which runs such a linear needs them.
std::size_t qwen35_executor_reserve_bytes(const shelf::qwen_hybrid::QwenHybridConfig& text_config,
                                          std::size_t ffn_intermediate_size,
                                          const ExecutorConfig& config,
                                          std::size_t tp_size,
                                          bool exl3_linear_workspace) noexcept;

}  // namespace runtime
}  // namespace shelf::qwen35
