#pragma once

#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/runtime/executor.h>

#include <cstddef>

namespace shelf::runtime {

std::size_t padded_row_count(std::uint32_t max_batch_rows) noexcept;

std::size_t executor_reserve_bytes(const shelf::qwen_hybrid::QwenHybridConfig& text_config,
                                   std::size_t ffn_intermediate_size,
                                   const ExecutorConfig& config,
                                   std::size_t tp_size,
                                   bool exl3_linear_workspace,
                                   bool qsa_workspace = false) noexcept;

}  // namespace shelf::runtime
