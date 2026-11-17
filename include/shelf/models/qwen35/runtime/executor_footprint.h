#pragma once

#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/runtime/executor_footprint.h>

namespace shelf::qwen35 {

using ::shelf::runtime::ExecutorConfig;

namespace runtime {

inline std::size_t qwen35_padded_row_count(std::uint32_t max_batch_rows) noexcept {
    return ::shelf::runtime::padded_row_count(max_batch_rows);
}

inline std::size_t qwen35_executor_reserve_bytes(
    const shelf::qwen_hybrid::QwenHybridConfig& text_config,
    std::size_t ffn_intermediate_size,
    const ::shelf::runtime::ExecutorConfig& config,
    std::size_t tp_size,
    bool exl3_linear_workspace,
    bool qsa_workspace = false) noexcept {
    return ::shelf::runtime::executor_reserve_bytes(
        text_config, ffn_intermediate_size, config, tp_size, exl3_linear_workspace,
        qsa_workspace);
}

}  // namespace runtime
}  // namespace shelf::qwen35
