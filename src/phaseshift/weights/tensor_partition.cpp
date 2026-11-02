#include <phaseshift/weights/tensor_partition.h>

#include <limits>

namespace ps::weights {

Status validate_tensor_partition(const TensorPartitionDesc& partition) {
    if (partition.count < 2) {
        return Status::invalid_argument(
            "tensor partition count must be at least 2", __FILE__, __LINE__);
    }
    if (partition.index >= partition.count) {
        return Status::invalid_argument(
            "tensor partition index out of range", __FILE__, __LINE__);
    }
    if (partition.global_shape.empty()) {
        return Status::invalid_argument(
            "tensor partition global_shape is empty", __FILE__, __LINE__);
    }
    for (const int64_t d : partition.global_shape) {
        if (d <= 0) {
            return Status::invalid_argument(
                "tensor partition global_shape has non-positive dim", __FILE__, __LINE__);
        }
    }
    if (partition.axis < 0 ||
        static_cast<std::size_t>(partition.axis) >= partition.global_shape.size()) {
        return Status::invalid_argument(
            "tensor partition axis out of range", __FILE__, __LINE__);
    }
    if (partition.ranges.empty()) {
        return Status::invalid_argument(
            "tensor partition ranges are empty", __FILE__, __LINE__);
    }

    const uint64_t axis_extent =
        static_cast<uint64_t>(partition.global_shape[static_cast<std::size_t>(partition.axis)]);
    for (const TensorPartitionRange& r : partition.ranges) {
        if (r.extent == 0) {
            return Status::invalid_argument(
                "tensor partition range extent must be positive", __FILE__, __LINE__);
        }
        if (r.global_offset >= axis_extent) {
            return Status::invalid_argument(
                "tensor partition range offset out of range", __FILE__, __LINE__);
        }
        if (r.extent > axis_extent - r.global_offset) {
            return Status::invalid_argument(
                "tensor partition range exceeds global axis", __FILE__, __LINE__);
        }
    }

    for (std::size_t i = 0; i < partition.ranges.size(); ++i) {
        const uint64_t a_begin = partition.ranges[i].global_offset;
        const uint64_t a_end = a_begin + partition.ranges[i].extent;
        for (std::size_t j = i + 1; j < partition.ranges.size(); ++j) {
            const uint64_t b_begin = partition.ranges[j].global_offset;
            const uint64_t b_end = b_begin + partition.ranges[j].extent;
            if (a_begin < b_end && b_begin < a_end) {
                return Status::invalid_argument(
                    "tensor partition ranges overlap", __FILE__, __LINE__);
            }
        }
    }
    return Status::make_ok();
}

Result<std::vector<int64_t>> tensor_partition_local_shape(
    const TensorPartitionDesc& partition) {
    Status valid = validate_tensor_partition(partition);
    if (!valid.ok()) return valid;

    std::vector<int64_t> local = partition.global_shape;
    uint64_t local_axis = 0;
    for (const TensorPartitionRange& r : partition.ranges) {
        local_axis += r.extent;
    }
    const std::size_t axis = static_cast<std::size_t>(partition.axis);
    if (local_axis > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return Status::overflow("tensor partition local axis overflow", __FILE__, __LINE__);
    }
    local[axis] = static_cast<int64_t>(local_axis);
    return local;
}

Status validate_tensor_partition_block_alignment(
    const TensorPartitionDesc& partition, uint64_t block_elements) {
    if (block_elements == 0) {
        return Status::make_ok();
    }
    Status valid = validate_tensor_partition(partition);
    if (!valid.ok()) return valid;
    if (static_cast<std::size_t>(partition.axis) + 1 != partition.global_shape.size()) {
        return Status::make_ok();
    }
    for (const TensorPartitionRange& r : partition.ranges) {
        if ((r.global_offset % block_elements) != 0 ||
            (r.extent % block_elements) != 0) {
            return Status::invalid_argument(
                "tensor partition K range is not aligned to the quantization block",
                __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Status validate_tensor_partition_plan(const TensorPartitionPlan& plan) {
    if (plan.tp_size < 1) {
        return Status::invalid_argument("tp_size must be at least 1", __FILE__, __LINE__);
    }
    if (plan.tp_rank >= plan.tp_size) {
        return Status::invalid_argument("tp_rank out of range", __FILE__, __LINE__);
    }
    for (const auto& kv : plan.tensors) {
        const TensorPartitionDesc& d = kv.second;
        Status valid = validate_tensor_partition(d);
        if (!valid.ok()) return valid;
        if (d.count != plan.tp_size || d.index != plan.tp_rank) {
            return Status::invalid_argument(
                ("tensor partition does not match plan geometry: " + kv.first).c_str(),
                __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

}  // namespace ps::weights
