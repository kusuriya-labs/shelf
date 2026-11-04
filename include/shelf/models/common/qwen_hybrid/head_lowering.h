#pragma once

#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/runtime/graph/primitive_node.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace shelf {
namespace qwen_hybrid {

// What a model does with the last block's output: gather the rows that produce tokens,
// apply the final norm, project to logits, and sample. This part does not depend on how
// any layer's FFN was built, so every family emits it the same way.
struct HybridHeadLowering {
    uint32_t hidden_features = 0;
    float eps = 1e-6f;
    uint32_t final_norm_param = UINT32_MAX;

    // The LM projection as it stands in the weight table. When the head is tied, the
    // caller names the embedding's index and shape.
    RT::MatrixwiseShapeKey lm_shape{};
    uint32_t lm_weight_index = 0;
    // Logit columns. It differs from lm_shape.output_features only when the head is
    // sharded across devices, in which case the caller also patches the weight slot.
    uint32_t out_vocab = 0;

    // The head's fourth external output is the last block's output for a model and the
    // normed hidden for a single drafter block. The order of the head's outputs is the
    // executor's staging order, so it is fixed here rather than at the call sites.
    bool tail_output_is_normed_hidden = false;
};

struct HybridHeadOutput {
    RT::ValueId selected_hidden{};
    RT::ValueId normed_hidden{};
    RT::ValueId selected_logits{};
    RT::ValueId sampled{};
    // The logits node, so a caller that shards the head can mark that node without
    // relying on it being the last one emitted.
    std::size_t logits_node_index = 0;
};

inline HybridHeadOutput emit_hybrid_head(
    Emitter& em,
    const HybridHeadLowering& head,
    RT::ValueId block_output,
    const std::string& debug_prefix = {}) {
    auto& pg = em.pg;
    const RT::RowwiseShapeKey hidden_key{head.hidden_features};

    RT::ValueId selected_hidden = em.value(
        head.hidden_features, RT::ValueDType::BF16, RT::ValueRowDomain::OUTPUT_ROWS);
    {
        RT::OutputGatherNode og{hidden_key};
        em.emit(RT::PrimitiveNode{og}, {block_output}, {selected_hidden});
        pg.nodes.back().debug_name = debug_prefix + "selected_hidden";
    }

    RT::ValueId normed_hidden = em.value(head.hidden_features);
    {
        RT::RmsNormNode head_rms;
        head_rms.shape = hidden_key;
        head_rms.eps = head.eps;
        head_rms.group_size = head.hidden_features;
        head_rms.parameter_index = head.final_norm_param;
        head_rms.weight_mode = RT::RmsNormWeightMode::ONE_PLUS;
        em.emit(RT::PrimitiveNode{head_rms}, {selected_hidden}, {normed_hidden});
        pg.nodes.back().debug_name = debug_prefix + "final_norm";
    }

    RT::ValueId selected_logits = em.value(
        head.out_vocab, RT::ValueDType::F32, RT::ValueRowDomain::OUTPUT_ROWS);
    {
        RT::LinearNode ln;
        ln.shape = head.lm_shape;
        ln.weight_index = head.lm_weight_index;
        ln.out_dtype = RT::ValueDType::F32;
        em.emit(RT::PrimitiveNode{ln}, {normed_hidden}, {selected_logits});
        pg.nodes.back().debug_name = debug_prefix + "logits";
    }
    const std::size_t logits_node = pg.nodes.size() - 1u;

    RT::ValueId sampled = em.value(
        1, RT::ValueDType::I32, RT::ValueRowDomain::OUTPUT_ROWS);
    {
        em.emit(RT::PrimitiveNode{RT::SamplingNode{head.out_vocab}}, {selected_logits},
                {sampled});
        pg.nodes.back().debug_name = debug_prefix + "sampled";
    }

    pg.external_outputs.push_back(selected_hidden);
    pg.external_outputs.push_back(selected_logits);
    pg.external_outputs.push_back(sampled);
    pg.external_outputs.push_back(head.tail_output_is_normed_hidden ? normed_hidden
                                                                   : block_output);

    return HybridHeadOutput{selected_hidden, normed_hidden, selected_logits, sampled,
                            logits_node};
}

}
}
