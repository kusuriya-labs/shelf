#include <shelf/models/qwen35/weights/tensor_parallel_plan.h>

#include <shelf/models/common/qwen_hybrid/tensor_partition.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>

#include <string>
#include <vector>

namespace shelf {
namespace qwen35 {

namespace {

using shelf::weights::TensorPartitionPlan;
using shelf::weights::TensorPartitionRange;
using shelf::weights::validate_tensor_partition_plan;

std::string layer_prefix(std::size_t layer) {
    return shelf::qwen_hybrid::layer_tensor_prefix(layer);
}

shelf::weights::TensorPartitionRange range_of(uint64_t offset, uint64_t extent) {
    shelf::weights::TensorPartitionRange r;
    r.global_offset = offset;
    r.extent = extent;
    return r;
}

Status add_dense_ffn_partitions(TensorPartitionPlan& plan,
                                const shelf::qwen_hybrid::HybridPartition& body,
                                const std::string& prefix, uint64_t intermediate,
                                uint64_t local_intermediate, uint64_t offset) {
    const std::vector<int64_t> up_shape{static_cast<int64_t>(intermediate),
                                        static_cast<int64_t>(body.hidden)};
    const std::vector<int64_t> down_shape{static_cast<int64_t>(body.hidden),
                                          static_cast<int64_t>(intermediate)};
    const std::vector<TensorPartitionRange> rows{range_of(offset, local_intermediate)};

    auto put = [&](const std::string& name, std::int32_t axis,
                   const std::vector<int64_t>& shape) -> Status {
        shelf::weights::TensorPartitionDesc desc;
        desc.axis = axis;
        desc.index = body.tp_rank;
        desc.count = body.tp_size;
        desc.global_shape = shape;
        desc.ranges = rows;
        Status valid = shelf::weights::validate_tensor_partition(desc);
        if (!valid.ok()) return valid;
        plan.tensors.emplace(name, std::move(desc));
        return Status::make_ok();
    };

    Status st = put(prefix + "mlp.gate_proj.weight", 0, up_shape);
    if (!st.ok()) return st;
    st = put(prefix + "mlp.up_proj.weight", 0, up_shape);
    if (!st.ok()) return st;
    return put(prefix + "mlp.down_proj.weight", 1, down_shape);
}

}  // namespace

Result<TensorPartitionPlan> build_qwen35_tensor_partition_plan(
    const Qwen35TextConfig& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    bool shard_lm_head) {
    if (tp_size < 1) {
        return Status::invalid_argument("tp_size must be at least 1", __FILE__, __LINE__);
    }
    if (tp_rank >= tp_size) {
        return Status::invalid_argument("tp_rank out of range", __FILE__, __LINE__);
    }

    TensorPartitionPlan plan;
    plan.tp_size = tp_size;
    plan.tp_rank = tp_rank;
    if (tp_size == 1) {
        return plan;
    }
    if (config.ffn.intermediate_size == 0) {
        return Status::invalid_argument("invalid qwen35 config geometry", __FILE__, __LINE__);
    }

    auto partition = shelf::qwen_hybrid::build_hybrid_partition(config, tp_size, tp_rank);
    if (!partition.ok()) return partition.status();
    const shelf::qwen_hybrid::HybridPartition& body = partition.value();

    const uint64_t intermediate = config.ffn.intermediate_size;
    if ((intermediate % tp_size) != 0) {
        return Status::unsupported(
            "qwen35 tp requires intermediate_size divisible by tp_size", __FILE__, __LINE__);
    }
    const uint64_t local_intermediate = intermediate / tp_size;
    const uint64_t mlp_offset = static_cast<uint64_t>(tp_rank) * local_intermediate;

    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        const std::string prefix = layer_prefix(l);

        Status st = shelf::qwen_hybrid::add_hybrid_layer_partitions(
            plan, body, prefix, shelf::qwen_hybrid::layer_is_gdn(config, l));
        if (!st.ok()) return st;

        st = add_dense_ffn_partitions(plan, body, prefix, intermediate, local_intermediate,
                                      mlp_offset);
        if (!st.ok()) return st;
    }

    if (shard_lm_head && !config.tie_word_embeddings) {
        Status st = shelf::qwen_hybrid::add_lm_head_partitions(plan, config, body);
        if (!st.ok()) return st;
    }

    Status valid = validate_tensor_partition_plan(plan);
    if (!valid.ok()) return valid;
    return plan;
}

}  // namespace qwen35
}  // namespace shelf
