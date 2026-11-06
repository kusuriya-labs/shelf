#include <shelf/models/qwen4exp/model/layer_lowering.h>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;
using ::shelf::qwen_hybrid::Emitter;

Result<RT::ValueId> emit_qwen4exp_layer(Emitter& em,
                                        const Qwen4ExpLayerLowering& layer,
                                        RT::ValueId hyper,
                                        const char* debug_prefix) {
    ::shelf::runtime::ValueId input_hyper = hyper;
    if (layer.has_ple) {
        auto ple = emit_qwen4exp_ple(em, layer.ple_geometry, layer.ple_weights,
                                     layer.ple_indices, input_hyper,
                                     layer.ple_embeddings, debug_prefix);
        if (!ple.ok()) return ple.status();
        input_hyper = ple.value();
    }

    auto attn_mix = emit_hyper_connection_mix(
        em, layer.hyper, layer.attn_hyper_weights, layer.attn_hyper_indices, input_hyper,
        true, debug_prefix);
    if (!attn_mix.ok()) return attn_mix.status();

    if (layer.trace != nullptr) {
        layer.trace->emplace_back(std::string(debug_prefix) + "attn_mixed",
                                  attn_mix.value().mixed);
        layer.trace->emplace_back(std::string(debug_prefix) + "attn_inject",
                                  attn_mix.value().injection);
    }
    RT::ValueId attn_out{0};
    if (layer.is_gdn) {
        attn_out = ::shelf::qwen_hybrid::emit_gdn(
            em, layer.gdn_geometry, layer.gdn_weights, layer.gdn_indices,
            attn_mix.value().mixed, debug_prefix);
    } else if (layer.dense_attention) {
        ::shelf::qwen_hybrid::GatedAttentionLoweringIndices dense_indices =
            layer.qsa_indices.attention;
        dense_indices.has_token_mask = false;
        attn_out = ::shelf::qwen_hybrid::emit_gated_attention(
            em, layer.qsa_geometry.attention, layer.qsa_weights.attention, dense_indices,
            attn_mix.value().mixed, debug_prefix);
    } else {
        auto qsa = emit_qwen4exp_qsa(em, layer.qsa_geometry, layer.qsa_weights,
                                     layer.qsa_indices, attn_mix.value().mixed,
                                     debug_prefix);
        if (!qsa.ok()) return qsa.status();
        attn_out = qsa.value();
    }
    if (layer.trace != nullptr) {
        layer.trace->emplace_back(std::string(debug_prefix) + "attn_out", attn_out);
    }
    auto hyper_after_attn = emit_hyper_connection_apply(
        em, layer.hyper, input_hyper, attn_out, attn_mix.value().injection, debug_prefix);
    if (!hyper_after_attn.ok()) return hyper_after_attn.status();
    if (layer.trace != nullptr) {
        layer.trace->emplace_back(std::string(debug_prefix) + "hyper_attn",
                                  hyper_after_attn.value());
    }

    HyperConnectionSiteGeometry mlp_hyper = layer.hyper;
    if (layer.mlp_hyper_norm_parameter_index != UINT32_MAX)
        mlp_hyper.norm_parameter_index = layer.mlp_hyper_norm_parameter_index;
    auto mlp_mix = emit_hyper_connection_mix(
        em, mlp_hyper, layer.mlp_hyper_weights, layer.mlp_hyper_indices,
        hyper_after_attn.value(), true, debug_prefix);
    if (!mlp_mix.ok()) return mlp_mix.status();

    if (layer.trace != nullptr) {
        layer.trace->emplace_back(std::string(debug_prefix) + "mlp_mixed",
                                  mlp_mix.value().mixed);
    }
    if (layer.trace != nullptr) {
        layer.trace->emplace_back(std::string(debug_prefix) + "mlp_inject",
                                  mlp_mix.value().injection);
    }
    auto ffn = emit_qwen4exp_moe(em, layer.moe_geometry, layer.moe_weights,
                                 layer.moe_indices, mlp_mix.value().mixed, debug_prefix);
    if (!ffn.ok()) return ffn.status();
    if (layer.trace != nullptr) {
        layer.trace->emplace_back(std::string(debug_prefix) + "moe_out", ffn.value());
    }

    return emit_hyper_connection_apply(em, layer.hyper, hyper_after_attn.value(), ffn.value(),
                                       mlp_mix.value().injection, debug_prefix);
}

}  // namespace qwen4exp
}  // namespace shelf
