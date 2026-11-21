#pragma once

#include <shelf/models/qwen4exp/model/qwen4exp_config.h>

#include <cstddef>
#include <cstdint>

namespace shelf {
namespace qwen4exp {

std::size_t mtp_grouped_executor_bytes(const Qwen4ExpTextConfig& config,
                                       std::size_t max_batch_rows,
                                       std::uint32_t max_context_tokens,
                                       std::uint32_t local_experts = 0u) noexcept;

}
}
