#pragma once
#include <shelf/core/status.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/weights/tensor_partition.h>

#include <cstdint>

namespace shelf::qwen35moe {

// The rank's tensor partition for a routed-expert checkpoint: the shared hybrid backbone
// (GDN plus full attention) and the shared expert, split the way a dense FFN is. The
// routed experts are not in this plan. They split along the expert axis, which is a plane
// depth rather than a matrix axis, so they are described by MoeExpertGroupPlan with the
// plan's own tp_size / tp_rank.
//
// What stays replicated is what has to be rank-invariant: the router gate (every rank must
// select the same experts) and the shared expert gate.
Result<shelf::weights::TensorPartitionPlan> build_qwen35moe_tensor_partition_plan(
    const Qwen35MoeTextConfig& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    bool shard_lm_head = true);

}  // namespace shelf::qwen35moe
