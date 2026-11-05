#include <shelf/models/qwen35moe/weights/moe_tensor_parallel_plan.h>

#include <shelf/models/common/qwen_hybrid/tensor_partition.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>

#include <string>
#include <vector>

namespace shelf::qwen35moe {

namespace {

using shelf::weights::TensorPartitionDesc;
using shelf::weights::TensorPartitionPlan;
using shelf::weights::TensorPartitionRange;

shelf::weights::TensorPartitionRange range_of(uint64_t offset, uint64_t extent) {
    TensorPartitionRange r;
    r.global_offset = offset;
    r.extent = extent;
    return r;
}

Status add_shared_expert_partitions(TensorPartitionPlan& plan,
                                    const shelf::qwen_hybrid::HybridPartition& body,
                                    const std::string& mlp, uint64_t shared_intermediate,
                                    uint64_t local_shared_intermediate, uint64_t offset) {
    const std::vector<int64_t> up_shape{static_cast<int64_t>(shared_intermediate),
                                        static_cast<int64_t>(body.hidden)};
    const std::vector<int64_t> down_shape{static_cast<int64_t>(body.hidden),
                                          static_cast<int64_t>(shared_intermediate)};
    const std::vector<TensorPartitionRange> rows{range_of(offset, local_shared_intermediate)};

    auto put = [&](const std::string& name, std::int32_t axis,
                   const std::vector<int64_t>& shape) -> Status {
        TensorPartitionDesc desc;
        desc.axis = axis;
        desc.index = body.tp_rank;
        desc.count = body.tp_size;
        desc.global_shape = shape;
        desc.ranges = rows;
        Status valid = shelf::weights::validate_tensor_partition(desc);
        if (!valid.ok()) return valid;
        plan.tensors.emplace(mlp + name, std::move(desc));
        return Status::make_ok();
    };

    Status st = put("shared_expert.gate_proj.weight", 0, up_shape);
    if (!st.ok()) return st;
    st = put("shared_expert.up_proj.weight", 0, up_shape);
    if (!st.ok()) return st;
    return put("shared_expert.down_proj.weight", 1, down_shape);
}

}  // namespace

Result<TensorPartitionPlan> build_qwen35moe_tensor_partition_plan(
    const Qwen35MoeTextConfig& config, std::uint32_t tp_size, std::uint32_t tp_rank,
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
    if (config.ffn.num_experts == 0) {
        return Status::unsupported(
            "build_qwen35moe_tensor_partition_plan: num_experts is 0 (not a MoE config)",
            __FILE__, __LINE__);
    }
    if ((config.ffn.num_experts % tp_size) != 0) {
        return Status::unsupported(
            "qwen35moe tp requires num_experts divisible by tp_size", __FILE__, __LINE__);
    }
    const uint64_t shared_intermediate = config.ffn.shared_expert_intermediate_size;
    if (shared_intermediate == 0) {
        return Status::invalid_argument("qwen35moe tp requires shared_expert_intermediate_size",
                                        __FILE__, __LINE__);
    }
    if ((shared_intermediate % tp_size) != 0) {
        return Status::unsupported(
            "qwen35moe tp requires shared_expert_intermediate_size divisible by tp_size",
            __FILE__, __LINE__);
    }

    auto body = shelf::qwen_hybrid::build_hybrid_partition(config, tp_size, tp_rank);
    if (!body.ok()) return body.status();
    const shelf::qwen_hybrid::HybridPartition& geometry = body.value();

    const uint64_t local_shared = shared_intermediate / tp_size;
    const uint64_t shared_offset = static_cast<uint64_t>(tp_rank) * local_shared;

    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        const std::string prefix = shelf::qwen_hybrid::layer_tensor_prefix(l);
        Status st = shelf::qwen_hybrid::add_hybrid_layer_partitions(
            plan, geometry, prefix, shelf::qwen_hybrid::layer_is_gdn(config, l));
        if (!st.ok()) return st;
        st = add_shared_expert_partitions(plan, geometry, prefix + "mlp.",
                                          shared_intermediate, local_shared, shared_offset);
        if (!st.ok()) return st;
    }

    if (shard_lm_head && !config.tie_word_embeddings) {
        Status st = shelf::qwen_hybrid::add_lm_head_partitions(plan, config, geometry);
        if (!st.ok()) return st;
    }

    Status valid = shelf::weights::validate_tensor_partition_plan(plan);
    if (!valid.ok()) return valid;
    return plan;
}

}  // namespace shelf::qwen35moe
