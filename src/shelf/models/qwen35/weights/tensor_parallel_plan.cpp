#include <shelf/models/qwen35/weights/tensor_parallel_plan.h>

#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <limits>
#include <string>
#include <vector>

namespace shelf {
namespace qwen35 {

namespace {

using shelf::weights::TensorPartitionDesc;
using shelf::weights::TensorPartitionPlan;
using shelf::weights::TensorPartitionRange;
using shelf::weights::validate_tensor_partition;
using shelf::weights::validate_tensor_partition_plan;

std::string layer_prefix(std::size_t layer) {
    return shelf::qwen_hybrid::layer_tensor_prefix(layer);
}

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

}

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
    if (config.num_hidden_layers == 0 || config.hidden_size == 0 ||
        config.ffn.intermediate_size == 0) {
        return Status::invalid_argument("invalid qwen35 config geometry", __FILE__, __LINE__);
    }

    std::vector<int> layer_types;
    layer_types.reserve(config.num_hidden_layers);
    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        layer_types.push_back(shelf::qwen_hybrid::layer_is_gdn(config, l) ? 0 : 1);
    }
    bool has_gdn = false;
    bool has_full_attention = false;
    for (const int t : layer_types) {
        if (t == 0) has_gdn = true;
        else has_full_attention = true;
    }

    const uint64_t hidden = config.hidden_size;
    const uint64_t intermediate = config.ffn.intermediate_size;
    const uint64_t query_heads = config.num_attention_heads;
    const uint64_t kv_heads = config.num_key_value_heads;
    const uint64_t head_dim = config.attention_head_dim;
    const uint64_t key_heads = config.linear_num_key_heads;
    const uint64_t value_heads = config.linear_num_value_heads;
    const uint64_t key_head_dim = config.linear_key_head_dim;
    const uint64_t value_head_dim = config.linear_value_head_dim;
    const uint64_t conv_kernel = config.linear_conv_kernel_dim;

    if (has_full_attention &&
        (query_heads == 0 || kv_heads == 0 || head_dim == 0)) {
        return Status::invalid_argument("invalid qwen35 attention geometry",
                                        __FILE__, __LINE__);
    }
    if (has_gdn && (key_heads == 0 || value_heads == 0 ||
                    key_head_dim == 0 || value_head_dim == 0 ||
                    conv_kernel == 0)) {
        return Status::invalid_argument("invalid qwen35 GDN geometry", __FILE__, __LINE__);
    }

    Status st = require_divisible(intermediate, tp_size,
                                  "qwen35 tp requires intermediate_size divisible by tp_size");
    if (!st.ok()) return st;
    if (has_full_attention) {
        st = require_divisible(query_heads, tp_size,
                               "qwen35 tp requires num_attention_heads divisible by tp_size");
        if (!st.ok()) return st;
        st = require_divisible(kv_heads, tp_size,
                               "qwen35 tp requires num_key_value_heads divisible by tp_size");
        if (!st.ok()) return st;
    }
    if (has_gdn) {
        st = require_divisible(key_heads, tp_size,
                               "qwen35 tp requires linear_num_key_heads divisible by tp_size");
        if (!st.ok()) return st;
        st = require_divisible(value_heads, tp_size,
                               "qwen35 tp requires linear_num_value_heads divisible by tp_size");
        if (!st.ok()) return st;
    }

    uint64_t query_features = 0;
    uint64_t kv_features = 0;
    uint64_t qk_features = 0;
    uint64_t value_features = 0;
    uint64_t conv_features = 0;
    st = checked_dim_mul(query_heads, head_dim, query_features,
                         "qwen35 query feature overflow");
    if (!st.ok()) return st;
    st = checked_dim_mul(kv_heads, head_dim, kv_features, "qwen35 kv feature overflow");
    if (!st.ok()) return st;
    st = checked_dim_mul(key_heads, key_head_dim, qk_features,
                         "qwen35 GDN key feature overflow");
    if (!st.ok()) return st;
    st = checked_dim_mul(value_heads, value_head_dim, value_features,
                         "qwen35 GDN value feature overflow");
    if (!st.ok()) return st;
    if (qk_features > std::numeric_limits<uint64_t>::max() / 2) {
        return Status::overflow("qwen35 GDN conv feature overflow", __FILE__, __LINE__);
    }
    const uint64_t twice_qk = 2 * qk_features;
    if (twice_qk > std::numeric_limits<uint64_t>::max() - value_features) {
        return Status::overflow("qwen35 GDN conv feature overflow", __FILE__, __LINE__);
    }
    conv_features = twice_qk + value_features;

    const uint64_t local_intermediate = intermediate / tp_size;
    const uint64_t local_query_heads = query_heads / tp_size;
    const uint64_t local_kv_heads = kv_heads / tp_size;
    const uint64_t local_value_heads = value_heads / tp_size;
    const uint64_t local_query_features = local_query_heads * head_dim;
    const uint64_t local_kv_features = local_kv_heads * head_dim;
    const uint64_t local_qk = qk_features / tp_size;
    const uint64_t local_value = value_features / tp_size;

    auto add = [&](const std::string& name, std::int32_t axis,
                   std::vector<int64_t> global_shape,
                   std::vector<TensorPartitionRange> ranges) -> Status {
        TensorPartitionDesc desc;
        desc.axis = axis;
        desc.index = tp_rank;
        desc.count = tp_size;
        desc.global_shape = std::move(global_shape);
        desc.ranges = std::move(ranges);
        Status valid = validate_tensor_partition(desc);
        if (!valid.ok()) return valid;
        plan.tensors.emplace(name, std::move(desc));
        return Status::make_ok();
    };

    for (std::size_t l = 0; l < config.num_hidden_layers; ++l) {
        const std::string prefix = layer_prefix(l);
        const uint64_t mlp_offset = static_cast<uint64_t>(tp_rank) * local_intermediate;

        st = add(prefix + "mlp.gate_proj.weight", 0, shape2(intermediate, hidden),
                 {range_of(mlp_offset, local_intermediate)});
        if (!st.ok()) return st;
        st = add(prefix + "mlp.up_proj.weight", 0, shape2(intermediate, hidden),
                 {range_of(mlp_offset, local_intermediate)});
        if (!st.ok()) return st;
        st = add(prefix + "mlp.down_proj.weight", 1, shape2(hidden, intermediate),
                 {range_of(mlp_offset, local_intermediate)});
        if (!st.ok()) return st;

        if (layer_types[l] == 0) {
            const uint64_t q_offset = static_cast<uint64_t>(tp_rank) * local_qk;
            const uint64_t k_offset = qk_features + q_offset;
            const uint64_t v_offset = 2 * qk_features +
                                      static_cast<uint64_t>(tp_rank) * local_value;
            const std::vector<TensorPartitionRange> qkv_ranges = {
                range_of(q_offset, local_qk),
                range_of(k_offset, local_qk),
                range_of(v_offset, local_value),
            };
            const uint64_t value_offset = static_cast<uint64_t>(tp_rank) * local_value;
            const uint64_t value_head_offset =
                static_cast<uint64_t>(tp_rank) * local_value_heads;

            st = add(prefix + "linear_attn.in_proj_qkv.weight", 0,
                     shape2(conv_features, hidden), qkv_ranges);
            if (!st.ok()) return st;
            st = add(prefix + "linear_attn.in_proj_z.weight", 0,
                     shape2(value_features, hidden),
                     {range_of(value_offset, local_value)});
            if (!st.ok()) return st;
            st = add(prefix + "linear_attn.in_proj_a.weight", 0,
                     shape2(value_heads, hidden),
                     {range_of(value_head_offset, local_value_heads)});
            if (!st.ok()) return st;
            st = add(prefix + "linear_attn.in_proj_b.weight", 0,
                     shape2(value_heads, hidden),
                     {range_of(value_head_offset, local_value_heads)});
            if (!st.ok()) return st;
            st = add(prefix + "linear_attn.out_proj.weight", 1,
                     shape2(hidden, value_features),
                     {range_of(value_offset, local_value)});
            if (!st.ok()) return st;
            st = add(prefix + "linear_attn.conv1d.weight", 0,
                     shape3(conv_features, 1, conv_kernel), qkv_ranges);
            if (!st.ok()) return st;
            st = add(prefix + "linear_attn.dt_bias", 0, shape1(value_heads),
                     {range_of(value_head_offset, local_value_heads)});
            if (!st.ok()) return st;
            st = add(prefix + "linear_attn.A_log", 0, shape1(value_heads),
                     {range_of(value_head_offset, local_value_heads)});
            if (!st.ok()) return st;
        } else {
            const uint64_t head_pair = 2 * head_dim;
            const uint64_t q_row_offset =
                static_cast<uint64_t>(tp_rank) * local_query_heads * head_pair;
            const uint64_t kv_offset = static_cast<uint64_t>(tp_rank) * local_kv_features;
            const uint64_t query_offset =
                static_cast<uint64_t>(tp_rank) * local_query_features;

            st = add(prefix + "self_attn.q_proj.weight", 0,
                     shape2(2 * query_features, hidden),
                     {range_of(q_row_offset, local_query_heads * head_pair)});
            if (!st.ok()) return st;
            st = add(prefix + "self_attn.k_proj.weight", 0, shape2(kv_features, hidden),
                     {range_of(kv_offset, local_kv_features)});
            if (!st.ok()) return st;
            st = add(prefix + "self_attn.v_proj.weight", 0, shape2(kv_features, hidden),
                     {range_of(kv_offset, local_kv_features)});
            if (!st.ok()) return st;
            st = add(prefix + "self_attn.o_proj.weight", 1,
                     shape2(hidden, query_features),
                     {range_of(query_offset, local_query_features)});
            if (!st.ok()) return st;
        }
    }

    if (shard_lm_head && !config.tie_word_embeddings) {
        st = require_divisible(config.vocab_size, tp_size,
                               "qwen35 tp requires vocab_size divisible by tp_size");
        if (!st.ok()) return st;
        const uint64_t vocab = config.vocab_size;
        const uint64_t local_vocab = vocab / tp_size;
        const uint64_t vocab_offset = static_cast<uint64_t>(tp_rank) * local_vocab;
        const std::vector<TensorPartitionRange> vocab_range = {
            range_of(vocab_offset, local_vocab),
        };
        st = add("model.language_model.lm_head.weight", 0, shape2(vocab, hidden),
                 vocab_range);
        if (!st.ok()) return st;
        st = add("lm_head.weight", 0, shape2(vocab, hidden), vocab_range);
        if (!st.ok()) return st;
    }

    Status valid = validate_tensor_partition_plan(plan);
    if (!valid.ok()) return valid;
    return plan;
}

}
}
