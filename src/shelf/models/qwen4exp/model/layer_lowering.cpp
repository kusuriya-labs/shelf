#include <shelf/models/qwen4exp/model/layer_lowering.h>

namespace shelf {
namespace qwen4exp {

namespace RT = ::shelf::runtime;
using ::shelf::qwen_hybrid::Emitter;

Result<RT::ValueId> emit_qwen4exp_layer(Emitter& em,
                                        const Qwen4ExpLayerLowering& layer,
                                        RT::ValueId hyper,
                                        const char* debug_prefix) {
    if (!layer.is_gdn) {
        return Status::unsupported(
            "Qwen4Exp QSA layer lowering is not implemented yet", __FILE__, __LINE__);
    }
    auto attn_mix = emit_hyper_connection_mix(
        em, layer.hyper, layer.attn_hyper_weights, layer.attn_hyper_indices, hyper, true,
        debug_prefix);
    if (!attn_mix.ok()) return attn_mix.status();

    const RT::ValueId attn_out = ::shelf::qwen_hybrid::emit_gdn(
        em, layer.gdn_geometry, layer.gdn_weights, layer.gdn_indices, attn_mix.value().mixed,
        debug_prefix);
    auto hyper_after_attn = emit_hyper_connection_apply(
        em, layer.hyper, hyper, attn_out, attn_mix.value().injection, debug_prefix);
    if (!hyper_after_attn.ok()) return hyper_after_attn.status();

    auto mlp_mix = emit_hyper_connection_mix(
        em, layer.hyper, layer.mlp_hyper_weights, layer.mlp_hyper_indices,
        hyper_after_attn.value(), true, debug_prefix);
    if (!mlp_mix.ok()) return mlp_mix.status();

    auto ffn = emit_qwen4exp_moe(em, layer.moe_geometry, layer.moe_weights,
                                 layer.moe_indices, mlp_mix.value().mixed, debug_prefix);
    if (!ffn.ok()) return ffn.status();

    return emit_hyper_connection_apply(em, layer.hyper, hyper_after_attn.value(), ffn.value(),
                                       mlp_mix.value().injection, debug_prefix);
}

}  // namespace qwen4exp
}  // namespace shelf
