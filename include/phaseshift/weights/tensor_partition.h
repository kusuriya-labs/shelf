#pragma once

#include <phaseshift/core/status.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ps::weights {

struct TensorPartitionRange {
    uint64_t global_offset = 0;
    uint64_t extent = 0;

    bool operator==(const TensorPartitionRange&) const = default;
};

struct TensorPartitionDesc {
    int32_t axis = -1;
    uint32_t index = 0;
    uint32_t count = 1;
    std::vector<int64_t> global_shape;
    std::vector<TensorPartitionRange> ranges;

    bool operator==(const TensorPartitionDesc&) const = default;
};

struct TensorPartitionPlan {
    uint32_t tp_size = 1;
    uint32_t tp_rank = 0;
    std::map<std::string, TensorPartitionDesc> tensors;
};

Status validate_tensor_partition(const TensorPartitionDesc& partition);

Result<std::vector<int64_t>> tensor_partition_local_shape(const TensorPartitionDesc& partition);

Status validate_tensor_partition_block_alignment(
    const TensorPartitionDesc& partition, uint64_t block_elements);

Status validate_tensor_partition_plan(const TensorPartitionPlan& plan);

}  // namespace ps::weights
