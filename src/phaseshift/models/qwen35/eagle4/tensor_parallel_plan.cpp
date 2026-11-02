#include <phaseshift/models/qwen35/eagle4/tensor_parallel_plan.h>

#include <cstdio>
#include <limits>
#include <string>

namespace ps::qwen35::eagle4 {

namespace {

using ps::weights::TensorPartitionDesc;
using ps::weights::TensorPartitionPlan;
using ps::weights::TensorPartitionRange;

Status require_divisible(std::uint64_t value, std::uint32_t tp_size, const char* what) {
    if ((value % tp_size) != 0) {
        return Status::unsupported(what, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

TensorPartitionDesc split_axis(std::size_t axis, std::uint64_t rows,
                               std::uint64_t cols, std::uint32_t tp_size,
                               std::uint32_t tp_rank) {
    TensorPartitionDesc desc;
    desc.axis = static_cast<int32_t>(axis);
    desc.index = tp_rank;
    desc.count = tp_size;
    desc.global_shape = {static_cast<std::int64_t>(rows),
                         static_cast<std::int64_t>(cols)};
    const std::uint64_t extent =
        (axis == 0 ? rows : cols) / static_cast<std::uint64_t>(tp_size);
    TensorPartitionRange range;
    range.global_offset = extent * static_cast<std::uint64_t>(tp_rank);
    range.extent = extent;
    desc.ranges.push_back(range);
    return desc;
}

std::string layer_prefix(std::size_t layer) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "layers.%zu.", layer);
    return buf;
}

}  // namespace

Result<TensorPartitionPlan> build_eagle4_tensor_partition_plan(
    const Eagle4Config& config,
    std::uint32_t tp_size,
    std::uint32_t tp_rank) {
    TensorPartitionPlan plan;
    plan.tp_size = tp_size;
    plan.tp_rank = tp_rank;
    if (tp_size < 1) {
        return Status::invalid_argument("tp_size must be at least 1", __FILE__, __LINE__);
    }
    if (tp_rank >= tp_size) {
        return Status::invalid_argument("tp_rank out of range", __FILE__, __LINE__);
    }
    if (tp_size == 1) {
        return plan;
    }

    const std::uint64_t hidden = config.hidden_size;
    const std::uint64_t vocab = config.vocab_size;
    const std::uint64_t intermediate = config.intermediate_size;
    const std::uint64_t candidate_rank = config.candidate_rank;
    const std::uint64_t q_rows = config.attention_q_rows();
    const std::uint64_t kv_rows = config.attention_kv_rows();
    if (hidden == 0 || vocab == 0 || intermediate == 0 || q_rows == 0 ||
        kv_rows == 0 || candidate_rank == 0 || config.num_hidden_layers == 0) {
        return Status::invalid_argument("invalid eagle4 config geometry", __FILE__, __LINE__);
    }

    Status st = require_divisible(hidden, tp_size,
                                  "eagle4 tp requires hidden_size divisible by tp_size");
    if (!st.ok()) return st;
    st = require_divisible(vocab, tp_size,
                           "eagle4 tp requires vocab_size divisible by tp_size");
    if (!st.ok()) return st;
    st = require_divisible(intermediate, tp_size,
                           "eagle4 tp requires intermediate_size divisible by tp_size");
    if (!st.ok()) return st;
    st = require_divisible(q_rows, tp_size,
                           "eagle4 tp requires attention q rows divisible by tp_size");
    if (!st.ok()) return st;
    st = require_divisible(kv_rows, tp_size,
                           "eagle4 tp requires attention kv rows divisible by tp_size");
    if (!st.ok()) return st;

    plan.tensors.emplace("embed_tokens.weight",
                         split_axis(0, vocab, hidden, tp_size, tp_rank));
    plan.tensors.emplace("lm_head.weight",
                         split_axis(0, vocab, hidden, tp_size, tp_rank));
    plan.tensors.emplace("efficient_lm_head.weight",
                         split_axis(0, vocab, candidate_rank, tp_size, tp_rank));

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        const std::string pre = layer_prefix(layer);
        const std::uint64_t qk_in = 2ull * hidden;
        plan.tensors.emplace(pre + "self_attn.q_proj.weight",
                             split_axis(0, hidden, qk_in, tp_size, tp_rank));
        plan.tensors.emplace(pre + "self_attn.k_proj.weight",
                             split_axis(0, hidden, qk_in, tp_size, tp_rank));
        plan.tensors.emplace(pre + "self_attn.v_proj.weight",
                             split_axis(0, hidden, qk_in, tp_size, tp_rank));
        plan.tensors.emplace(pre + "self_attn.o_proj.weight",
                             split_axis(1, hidden, hidden, tp_size, tp_rank));
        plan.tensors.emplace(pre + "mlp.gate_proj.weight",
                             split_axis(0, intermediate, hidden, tp_size, tp_rank));
        plan.tensors.emplace(pre + "mlp.up_proj.weight",
                             split_axis(0, intermediate, hidden, tp_size, tp_rank));
        plan.tensors.emplace(pre + "mlp.down_proj.weight",
                             split_axis(1, hidden, intermediate, tp_size, tp_rank));
    }
    return plan;
}

}
