#pragma once
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

#include <cstddef>

namespace shelf::qwen35moe {

// Bytes the routed experts add to the program workspace for a batch of `max_batch_rows`
// tokens: the group-row planes (fused gate / up, the swiglu output of one group row, and the
// expert down output) in bf16, the two int8 activation planes, and one routing plan. Those
// rows are counted by the plan capacity of `tokens * top_k` over `experts`, not by token
// rows, so they do not scale with a dense intermediate width and are not covered by the
// dense term of `qwen35_executor_reserve_bytes`.
std::size_t moe_executor_grouped_bytes(const Qwen35MoeTextConfig& config,
                                       std::size_t max_batch_rows) noexcept;

// The row count the executor actually uses for `max_batch_rows` (the bucket limit), which is
// what the plan capacity and the group-row planes are sized by.
std::size_t moe_padded_row_count(std::size_t max_batch_rows) noexcept;

}  // namespace shelf::qwen35moe
