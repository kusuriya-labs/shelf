#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/weights/tensor_partition.h>

#include <cstdint>

namespace shelf {
namespace qwen35 {

Result<shelf::weights::TensorPartitionPlan> build_qwen35_tensor_partition_plan(
    const Qwen35TextConfig& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    bool shard_lm_head = true);

}
}
