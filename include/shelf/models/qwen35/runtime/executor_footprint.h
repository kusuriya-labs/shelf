#pragma once

#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/executor.h>

#include <cstddef>

namespace ps::qwen35 {
namespace runtime {

std::size_t qwen35_padded_row_count(std::uint32_t max_batch_rows) noexcept;

std::size_t qwen35_executor_reserve_bytes(const Qwen35TextConfig& text_config,
                                          const ExecutorConfig& config,
                                          std::size_t tp_size) noexcept;

}  // namespace runtime
}  // namespace ps::qwen35
