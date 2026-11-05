#pragma once
#include <shelf/core/memory/tensor.h>
#include <shelf/weights/matrix_weight.h>

namespace shelf {
namespace qwen_hybrid {

// One transformer layer's weights that do not depend on how its FFN is built: the two
// layernorms plus whichever attention the layer carries (GDN linear attention or full
// attention with q / k norms). A dense model and an MoE model hold these identically;
// each adds its own FFN next to them.
struct HybridLayerWeights {
    bool is_gdn = false;

    gpu::Tensor input_layernorm_weight;
    gpu::Tensor post_attention_layernorm_weight;

    shelf::weights::MatrixWeight attn_q_proj;
    shelf::weights::MatrixWeight attn_k_proj;
    shelf::weights::MatrixWeight attn_v_proj;
    shelf::weights::MatrixWeight attn_o_proj;
    gpu::Tensor attn_q_norm_weight;
    gpu::Tensor attn_k_norm_weight;

    shelf::weights::MatrixWeight attn_in_proj_a;
    shelf::weights::MatrixWeight attn_in_proj_b;
    shelf::weights::MatrixWeight attn_in_proj_qkv;
    shelf::weights::MatrixWeight attn_in_proj_z;
    gpu::Tensor attn_conv1d_weight;
    shelf::weights::MatrixWeight attn_out_proj;
    gpu::Tensor attn_norm_weight;
    gpu::Tensor attn_dt_bias;
    gpu::Tensor attn_A_log;
};

// True when one of this layer's projections is carried in the rotated-domain trellis form.
// The execution workspace that form needs is sized per rank and per row bucket, so a model
// that uses it must say so before the KV pool takes the rest of the arena.
inline bool layer_uses_exl3_linear(const HybridLayerWeights& layer) noexcept {
    const shelf::weights::MatrixWeight* projections[] = {
        &layer.attn_q_proj, &layer.attn_k_proj,   &layer.attn_v_proj,
        &layer.attn_o_proj, &layer.attn_in_proj_a, &layer.attn_in_proj_b,
        &layer.attn_in_proj_qkv, &layer.attn_in_proj_z, &layer.attn_out_proj};
    for (const shelf::weights::MatrixWeight* w : projections) {
        if (w->encoding == shelf::weights::MatrixEncoding::Exl3) return true;
    }
    return false;
}

}  // namespace qwen_hybrid
}  // namespace shelf
