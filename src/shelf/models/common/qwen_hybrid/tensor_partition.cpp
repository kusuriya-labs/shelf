#include <shelf/models/common/qwen_hybrid/tensor_partition.h>

#include <limits>
#include <vector>

namespace shelf {
namespace qwen_hybrid {

namespace {

using shelf::weights::TensorPartitionDesc;
using shelf::weights::TensorPartitionPlan;
using shelf::weights::TensorPartitionRange;

Status checked_dim_mul(uint64_t left, uint64_t right, uint64_t& out, const char* what) {
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right) {
        return Status::overflow(what, __FILE__, __LINE__);
    }
    out = left * right;
    return Status::make_ok();
}

Status require_divisible(uint64_t value, std::uint32_t tp_size, const char* what) {
    if ((value % tp_size) != 0) {
        return Status::unsupported(what, __FILE__, __LINE__);
    }
    return Status::make_ok();
}

std::vector<int64_t> shape1(uint64_t a) {
    return {static_cast<int64_t>(a)};
}

std::vector<int64_t> shape2(uint64_t a, uint64_t b) {
    return {static_cast<int64_t>(a), static_cast<int64_t>(b)};
}

std::vector<int64_t> shape3(uint64_t a, uint64_t b, uint64_t c) {
    return {static_cast<int64_t>(a), static_cast<int64_t>(b), static_cast<int64_t>(c)};
}

TensorPartitionRange range_of(uint64_t offset, uint64_t extent) {
    TensorPartitionRange r;
    r.global_offset = offset;
    r.extent = extent;
    return r;
}

Status add(TensorPartitionPlan& plan, const HybridPartition& partition,
           const std::string& name, std::int32_t axis, std::vector<int64_t> global_shape,
           std::vector<TensorPartitionRange> ranges) {
    TensorPartitionDesc desc;
    desc.axis = axis;
    desc.index = partition.tp_rank;
    desc.count = partition.tp_size;
    desc.global_shape = std::move(global_shape);
    desc.ranges = std::move(ranges);
    Status valid = weights::validate_tensor_partition(desc);
    if (!valid.ok()) return valid;
    plan.tensors.emplace(name, std::move(desc));
    return Status::make_ok();
}

}  // namespace

Result<HybridPartition> build_hybrid_partition(const QwenHybridConfig& config,
                                               std::uint32_t tp_size,
                                               std::uint32_t tp_rank) {
    if (tp_size < 1) {
        return Status::invalid_argument("tp_size must be at least 1", __FILE__, __LINE__);
    }
    if (tp_rank >= tp_size) {
        return Status::invalid_argument("tp_rank out of range", __FILE__, __LINE__);
    }
    if (config.num_hidden_layers == 0 || config.hidden_size == 0) {
        return Status::invalid_argument("invalid hybrid config geometry", __FILE__, __LINE__);
    }

    HybridPartition partition;
    partition.tp_size = tp_size;
    partition.tp_rank = tp_rank;
    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        if (layer_is_gdn(config, l)) partition.has_gdn = true;
        else partition.has_full_attention = true;
    }

    partition.hidden = config.hidden_size;
    partition.attention_head_dim = config.attention_head_dim;
    partition.query_heads = config.num_attention_heads;
    partition.key_value_heads = config.num_key_value_heads;
    partition.gdn_key_heads = config.linear_num_key_heads;
    partition.gdn_value_heads = config.linear_num_value_heads;
    partition.gdn_key_head_dim = config.linear_key_head_dim;
    partition.gdn_value_head_dim = config.linear_value_head_dim;
    partition.gdn_conv_kernel = config.linear_conv_kernel_dim;

    if (partition.has_full_attention &&
        (partition.query_heads == 0 || partition.key_value_heads == 0 ||
         partition.attention_head_dim == 0)) {
        return Status::invalid_argument("invalid hybrid attention geometry", __FILE__, __LINE__);
    }
    if (partition.has_gdn && (partition.gdn_key_heads == 0 || partition.gdn_value_heads == 0 ||
                              partition.gdn_key_head_dim == 0 ||
                              partition.gdn_value_head_dim == 0 ||
                              partition.gdn_conv_kernel == 0)) {
        return Status::invalid_argument("invalid hybrid GDN geometry", __FILE__, __LINE__);
    }

    if (partition.has_full_attention) {
        Status st = require_divisible(partition.query_heads, tp_size,
                                      "hybrid tp requires num_attention_heads divisible by "
                                      "tp_size");
        if (!st.ok()) return st;
        st = require_divisible(partition.key_value_heads, tp_size,
                               "hybrid tp requires num_key_value_heads divisible by tp_size");
        if (!st.ok()) return st;
    }
    if (partition.has_gdn) {
        Status st = require_divisible(partition.gdn_key_heads, tp_size,
                                      "hybrid tp requires linear_num_key_heads divisible by "
                                      "tp_size");
        if (!st.ok()) return st;
        st = require_divisible(partition.gdn_value_heads, tp_size,
                               "hybrid tp requires linear_num_value_heads divisible by tp_size");
        if (!st.ok()) return st;
    }

    Status st = checked_dim_mul(partition.query_heads, partition.attention_head_dim,
                                partition.query_features, "hybrid query feature overflow");
    if (!st.ok()) return st;
    st = checked_dim_mul(partition.key_value_heads, partition.attention_head_dim,
                         partition.key_value_features, "hybrid kv feature overflow");
    if (!st.ok()) return st;
    st = checked_dim_mul(partition.gdn_key_heads, partition.gdn_key_head_dim,
                         partition.gdn_key_features, "hybrid GDN key feature overflow");
    if (!st.ok()) return st;
    st = checked_dim_mul(partition.gdn_value_heads, partition.gdn_value_head_dim,
                         partition.gdn_value_features, "hybrid GDN value feature overflow");
    if (!st.ok()) return st;
    if (partition.gdn_key_features > std::numeric_limits<uint64_t>::max() / 2) {
        return Status::overflow("hybrid GDN conv feature overflow", __FILE__, __LINE__);
    }
    const uint64_t twice_qk = 2 * partition.gdn_key_features;
    if (twice_qk > std::numeric_limits<uint64_t>::max() - partition.gdn_value_features) {
        return Status::overflow("hybrid GDN conv feature overflow", __FILE__, __LINE__);
    }
    partition.gdn_conv_features = twice_qk + partition.gdn_value_features;

    partition.local_query_heads = partition.query_heads / tp_size;
    partition.local_key_value_heads = partition.key_value_heads / tp_size;
    partition.local_gdn_value_heads = partition.gdn_value_heads / tp_size;
    partition.local_gdn_key_heads = partition.gdn_key_heads / tp_size;
    partition.local_query_features = partition.local_query_heads * partition.attention_head_dim;
    partition.local_key_value_features =
        partition.local_key_value_heads * partition.attention_head_dim;
    partition.local_gdn_key_features =
        partition.local_gdn_key_heads * partition.gdn_key_head_dim;
    partition.local_gdn_value_features =
        partition.local_gdn_value_heads * partition.gdn_value_head_dim;
    return partition;
}

Status add_hybrid_layer_partitions(TensorPartitionPlan& plan,
                                   const HybridPartition& partition,
                                   const std::string& prefix,
                                   bool is_gdn) {
    Status st = Status::make_ok();
    if (is_gdn) {
        const uint64_t qk = partition.gdn_key_features;
        const uint64_t local_qk = partition.local_gdn_key_features;
        const uint64_t local_value = partition.local_gdn_value_features;
        const uint64_t q_offset = static_cast<uint64_t>(partition.tp_rank) * local_qk;
        const uint64_t k_offset = qk + q_offset;
        const uint64_t v_offset = 2 * qk +
                                  static_cast<uint64_t>(partition.tp_rank) * local_value;
        const uint64_t value_offset = static_cast<uint64_t>(partition.tp_rank) * local_value;
        const uint64_t value_head_offset =
            static_cast<uint64_t>(partition.tp_rank) * partition.local_gdn_value_heads;
        const std::vector<TensorPartitionRange> qkv_ranges = {
            range_of(q_offset, local_qk),
            range_of(k_offset, local_qk),
            range_of(v_offset, local_value),
        };

        st = add(plan, partition, prefix + "linear_attn.in_proj_qkv.weight", 0,
                 shape2(partition.gdn_conv_features, partition.hidden), qkv_ranges);
        if (!st.ok()) return st;
        st = add(plan, partition, prefix + "linear_attn.in_proj_z.weight", 0,
                 shape2(partition.gdn_value_features, partition.hidden),
                 {range_of(value_offset, local_value)});
        if (!st.ok()) return st;
        st = add(plan, partition, prefix + "linear_attn.in_proj_a.weight", 0,
                 shape2(partition.gdn_value_heads, partition.hidden),
                 {range_of(value_head_offset, partition.local_gdn_value_heads)});
        if (!st.ok()) return st;
        st = add(plan, partition, prefix + "linear_attn.in_proj_b.weight", 0,
                 shape2(partition.gdn_value_heads, partition.hidden),
                 {range_of(value_head_offset, partition.local_gdn_value_heads)});
        if (!st.ok()) return st;
        st = add(plan, partition, prefix + "linear_attn.out_proj.weight", 1,
                 shape2(partition.hidden, partition.gdn_value_features),
                 {range_of(value_offset, local_value)});
        if (!st.ok()) return st;
        st = add(plan, partition, prefix + "linear_attn.conv1d.weight", 0,
                 shape3(partition.gdn_conv_features, 1, partition.gdn_conv_kernel), qkv_ranges);
        if (!st.ok()) return st;
        st = add(plan, partition, prefix + "linear_attn.dt_bias", 0,
                 shape1(partition.gdn_value_heads),
                 {range_of(value_head_offset, partition.local_gdn_value_heads)});
        if (!st.ok()) return st;
        return add(plan, partition, prefix + "linear_attn.A_log", 0,
                   shape1(partition.gdn_value_heads),
                   {range_of(value_head_offset, partition.local_gdn_value_heads)});
    }

    const uint64_t head_pair = 2 * partition.attention_head_dim;
    const uint64_t q_row_offset =
        static_cast<uint64_t>(partition.tp_rank) * partition.local_query_heads * head_pair;
    const uint64_t kv_offset = static_cast<uint64_t>(partition.tp_rank) *
                               partition.local_key_value_features;
    const uint64_t query_offset = static_cast<uint64_t>(partition.tp_rank) *
                                  partition.local_query_features;

    st = add(plan, partition, prefix + "self_attn.q_proj.weight", 0,
             shape2(2 * partition.query_features, partition.hidden),
             {range_of(q_row_offset, partition.local_query_heads * head_pair)});
    if (!st.ok()) return st;
    st = add(plan, partition, prefix + "self_attn.k_proj.weight", 0,
             shape2(partition.key_value_features, partition.hidden),
             {range_of(kv_offset, partition.local_key_value_features)});
    if (!st.ok()) return st;
    st = add(plan, partition, prefix + "self_attn.v_proj.weight", 0,
             shape2(partition.key_value_features, partition.hidden),
             {range_of(kv_offset, partition.local_key_value_features)});
    if (!st.ok()) return st;
    return add(plan, partition, prefix + "self_attn.o_proj.weight", 1,
               shape2(partition.hidden, partition.query_features),
               {range_of(query_offset, partition.local_query_features)});
}

Status add_lm_head_partitions(TensorPartitionPlan& plan, const QwenHybridConfig& config,
                              const HybridPartition& partition) {
    if (config.tie_word_embeddings) {
        return Status::invalid_argument("lm_head partition requires untied embeddings",
                                        __FILE__, __LINE__);
    }
    Status st = require_divisible(config.vocab_size, partition.tp_size,
                                  "hybrid tp requires vocab_size divisible by tp_size");
    if (!st.ok()) return st;
    const uint64_t vocab = config.vocab_size;
    const uint64_t local_vocab = vocab / partition.tp_size;
    const uint64_t vocab_offset = static_cast<uint64_t>(partition.tp_rank) * local_vocab;
    const std::vector<TensorPartitionRange> vocab_range = {range_of(vocab_offset, local_vocab)};
    st = add(plan, partition, "model.language_model.lm_head.weight", 0,
             shape2(vocab, partition.hidden), vocab_range);
    if (!st.ok()) return st;
    return add(plan, partition, "lm_head.weight", 0, shape2(vocab, partition.hidden),
               vocab_range);
}

}  // namespace qwen_hybrid
}  // namespace shelf
