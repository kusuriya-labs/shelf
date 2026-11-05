#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/weights/tensor_partition.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace shelf {
namespace qwen_hybrid {

// The backbone part of a rank's tensor partition: GDN plus full attention. A model family
// owns its FFN, so a dense family splits one intermediate width and a routed-expert family
// splits experts. What the backbone requires to divide is the same for both.
struct HybridPartition {
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;

    std::uint64_t hidden = 0;
    std::uint64_t attention_head_dim = 0;

    std::uint64_t query_heads = 0;
    std::uint64_t key_value_heads = 0;
    std::uint64_t query_features = 0;
    std::uint64_t key_value_features = 0;
    std::uint64_t local_query_heads = 0;
    std::uint64_t local_key_value_heads = 0;
    std::uint64_t local_query_features = 0;
    std::uint64_t local_key_value_features = 0;

    std::uint64_t gdn_key_heads = 0;
    std::uint64_t gdn_value_heads = 0;
    std::uint64_t gdn_key_head_dim = 0;
    std::uint64_t gdn_value_head_dim = 0;
    std::uint64_t gdn_conv_kernel = 0;
    std::uint64_t gdn_key_features = 0;
    std::uint64_t gdn_value_features = 0;
    std::uint64_t gdn_conv_features = 0;
    std::uint64_t local_gdn_key_heads = 0;
    std::uint64_t local_gdn_value_heads = 0;
    std::uint64_t local_gdn_key_features = 0;
    std::uint64_t local_gdn_value_features = 0;

    bool has_gdn = false;
    bool has_full_attention = false;
};

// Checks that the backbone geometry divides by tp_size and returns the rank's local
// widths. tp_size == 1 returns the global widths with no partition requirement beyond
// the config's own geometry.
Result<HybridPartition> build_hybrid_partition(const QwenHybridConfig& config,
                                               std::uint32_t tp_size,
                                               std::uint32_t tp_rank);

// Adds one layer's GDN or full attention entries. The prefix comes from the caller, so a
// block that lives outside the layer array uses the same entries, and `is_gdn` says which
// body the checkpoint stores there.
Status add_hybrid_layer_partitions(weights::TensorPartitionPlan& plan,
                                   const HybridPartition& partition,
                                   const std::string& prefix,
                                   bool is_gdn);

// Adds the vocab-parallel lm_head entries under both names a checkpoint may use.
Status add_lm_head_partitions(weights::TensorPartitionPlan& plan,
                              const QwenHybridConfig& config,
                              const HybridPartition& partition);

}  // namespace qwen_hybrid
}  // namespace shelf
