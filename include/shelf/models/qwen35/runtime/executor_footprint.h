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
// scale with a dense intermediate width.
std::size_t qwen35_executor_reserve_bytes(const shelf::qwen_hybrid::QwenHybridConfig& text_config,
                                          std::size_t ffn_intermediate_size,
                                          const ExecutorConfig& config,
                                          std::size_t tp_size) noexcept;

}  // namespace runtime
}  // namespace shelf::qwen35
