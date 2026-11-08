#pragma once

#include <shelf/models/qwen4exp/model/qwen4exp_config.h>

#include <cstddef>
#include <cstdint>

namespace shelf {
namespace qwen4exp {

// The group-row planes and routing plan the MTP block's unsharded expert set needs on top
// of the shared executor estimate. The main model's formula slices the experts per rank;
// the draft block keeps all of them, so its grouped rows are wider.
std::size_t mtp_grouped_executor_bytes(const Qwen4ExpTextConfig& config,
                                       std::size_t max_batch_rows,
                                       std::uint32_t max_context_tokens) noexcept;

}
}
