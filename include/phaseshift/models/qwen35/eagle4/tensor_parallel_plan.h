#pragma once

#include <phaseshift/core/status.h>
#include <phaseshift/models/qwen35/eagle4/config.h>
#include <phaseshift/weights/tensor_partition.h>

#include <cstdint>

namespace ps::qwen35::eagle4 {

Result<ps::weights::TensorPartitionPlan> build_eagle4_tensor_partition_plan(
    const Eagle4Config& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank);

}
