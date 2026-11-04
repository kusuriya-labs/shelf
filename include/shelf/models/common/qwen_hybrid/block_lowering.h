#pragma once

#include <shelf/models/common/qwen_hybrid/attention_lowering.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/common/qwen_hybrid/gdn_lowering.h>

#include <cstdint>
#include <string>

namespace shelf {
namespace qwen_hybrid {

// One transformer block without its FFN: input layernorm, whichever attention the layer
// carries (GDN or full attention), the attention residual, and the post-attention
// layernorm. A dense model and an MoE model emit exactly this and then their own FFN, so
// neither one's FFN vocabulary reaches the other's.
struct HybridBlockLowering {
    bool is_gdn = false;

    // Both layernorms of the block.
    uint32_t weight_input_norm = 0;
    uint32_t weight_post_norm = 0;
    float eps = 1e-6f;

    GdnLoweringGeometry gdn_geometry;
    GdnLoweringWeights gdn_weights;
    GdnLoweringIndices gdn_indices;

    GatedAttentionLoweringGeometry attention_geometry;
    GatedAttentionLoweringWeights attention_weights;
    GatedAttentionLoweringIndices attention_indices;
};

struct HybridBlockOutput {
    // The value the block's FFN output is added to.
    RT::ValueId attention_residual;
    // The value the block's FFN reads.
    RT::ValueId post_norm;
};

inline HybridBlockOutput emit_hybrid_block(
    Emitter& em,
    const RT::RowwiseShapeKey& hidden_key,
    const HybridBlockLowering& block,
    RT::ValueId layer_input,
    const std::string& debug_prefix) {
    auto& pg = em.pg;
    const auto name_last = [&](const char* suffix) {
        pg.nodes.back().debug_name = debug_prefix + suffix;
    };

    RT::RmsNormNode input_rms;
    input_rms.shape = hidden_key;
    input_rms.eps = block.eps;
    input_rms.group_size = hidden_key.features;
    input_rms.parameter_index = block.weight_input_norm;
    input_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId normed = em.unary(RT::PrimitiveNode{input_rms}, layer_input,
                                  hidden_key.features);
    name_last("input_norm");

    RT::ValueId block_out{0};
    if (block.is_gdn) {
        block_out = emit_gdn(em, block.gdn_geometry, block.gdn_weights, block.gdn_indices,
                             normed, debug_prefix);
    } else {
        block_out = emit_gated_attention(em, block.attention_geometry, block.attention_weights,
                                         block.attention_indices, normed, debug_prefix);
    }

    RT::ValueId attention_residual{0};
    {
        RT::ResidualAddNode ra{hidden_key};
        attention_residual = em.value(hidden_key.features);
        em.emit(RT::PrimitiveNode{ra}, {block_out, layer_input}, {attention_residual});
        name_last("residual_attn");
    }

    RT::RmsNormNode post_rms;
    post_rms.shape = hidden_key;
    post_rms.eps = input_rms.eps;
    post_rms.group_size = hidden_key.features;
    post_rms.parameter_index = block.weight_post_norm;
    post_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
    RT::ValueId post_norm = em.unary(RT::PrimitiveNode{post_rms}, attention_residual,
                                     hidden_key.features);
    name_last("post_norm");

    return HybridBlockOutput{attention_residual, post_norm};
}

}
}
