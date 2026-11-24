#include <shelf/models/hunyuan_dense/model/lower_to_primitives.h>

#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/common/qwen_hybrid/weight_slot.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace shelf::hunyuan_dense {
namespace RT = shelf::runtime;
namespace QH = shelf::qwen_hybrid;

Result<QH::HybridLoweredPrimitives> lower_to_primitives(const Config& c, const ModelWeights& w) {
    if (w.layers.size() != c.num_hidden_layers || c.head_dim == 0 ||
        c.num_key_value_heads == 0 || c.num_attention_heads == 0 ||
        c.num_attention_heads % c.num_key_value_heads != 0 ||
        c.hidden_size == 0 || c.intermediate_size == 0 || c.vocab_size == 0 ||
        c.hidden_size > UINT32_MAX || c.intermediate_size > UINT32_MAX / 2 ||
        c.vocab_size > UINT32_MAX || w.lm_head.rows < c.vocab_size ||
        c.num_attention_heads > UINT32_MAX / c.head_dim ||
        c.num_key_value_heads > UINT32_MAX / c.head_dim) {
        return Status::invalid_argument(
            "hunyuan_v1_dense: invalid lowering geometry", __FILE__, __LINE__);
    }

    const std::uint32_t h = static_cast<std::uint32_t>(c.hidden_size);
    const std::uint32_t d = static_cast<std::uint32_t>(c.head_dim);
    const std::uint32_t q = static_cast<std::uint32_t>(c.num_attention_heads * c.head_dim);
    const std::uint32_t kv = static_cast<std::uint32_t>(c.num_key_value_heads * c.head_dim);
    const std::uint32_t f = static_cast<std::uint32_t>(c.intermediate_size);

    QH::HybridLoweredPrimitives lowered;
    auto& pg = lowered.graph;
    QH::Emitter em{pg};

    auto weight = [&](const shelf::weights::MatrixWeight& m) {
        lowered.weights.push_back(QH::make_weight_slot(m));
        return static_cast<std::uint32_t>(lowered.weights.size() - 1);
    };
    auto param = [&](const gpu::Tensor& t) {
        RT::StaticParameterSlot p{};
        p.device_ptr = t.data();
        p.elements = t.ndim() == 1 ? static_cast<std::uint32_t>(t.dim(0)) : 0;
        p.dtype = RT::ValueDType::BF16;
        lowered.parameters.push_back(p);
        return static_cast<std::uint32_t>(lowered.parameters.size() - 1);
    };
    auto norm = [&](RT::ValueId input, std::uint32_t width, std::uint32_t group,
                    const gpu::Tensor& tensor, const std::string& name) {
        RT::RmsNormNode node;
        node.shape = RT::RowwiseShapeKey{width};
        node.eps = static_cast<float>(c.rms_norm_eps);
        node.group_size = group;
        node.parameter_index = param(tensor);
        node.weight_mode = group == 128u
            ? RT::RmsNormWeightMode::OFFSET_BF16_SMALL_GROUP
            : RT::RmsNormWeightMode::OFFSET_BF16;
        node.weight_offset = 0.0f;
        RT::ValueId out = em.value(width, RT::ValueDType::BF16, pg.value_row_domain(input));
        em.emit(RT::PrimitiveNode{node}, {input}, {out});
        pg.nodes.back().debug_name = name;
        return out;
    };
    auto residual = [&](RT::ValueId a, RT::ValueId b, const std::string& name) {
        auto out = em.value(h);
        em.emit(RT::PrimitiveNode{RT::ResidualAddNode{RT::RowwiseShapeKey{h}}}, {a, b}, {out});
        pg.nodes.back().debug_name = name;
        return out;
    };

    auto tokens = em.value(0, RT::ValueDType::I32);
    auto x = em.value(h);
    RT::EmbeddingLookupNode emb;
    emb.vocab_size = w.embed_tokens.rows;
    emb.hidden_size = h;
    emb.weight_shape = {w.embed_tokens.rows, w.embed_tokens.cols};
    emb.weight_index = weight(w.embed_tokens);
    em.emit(RT::PrimitiveNode{emb}, {tokens}, {x});
    pg.nodes.back().debug_name = "embedding";
    pg.external_inputs.push_back(tokens);

    const RT::AttentionShapeKey attn{
        static_cast<std::uint32_t>(c.num_attention_heads),
        static_cast<std::uint32_t>(c.num_key_value_heads),
        d,
        d,
    };

    for (std::uint32_t i = 0; i < w.layers.size(); ++i) {
        const auto& l = w.layers[i];
        const std::string p = "L" + std::to_string(i) + ".";
        lowered.layer_input_values.emplace_back(i, x);

        auto n = norm(x, h, h, l.input_layernorm, p + "input_layernorm");
        auto qraw = em.linear(l.q_proj, weight(l.q_proj), n);
        pg.nodes.back().debug_name = p + "q_proj";
        auto kraw = em.linear(l.k_proj, weight(l.k_proj), n);
        pg.nodes.back().debug_name = p + "k_proj";
        auto v = em.linear(l.v_proj, weight(l.v_proj), n);
        pg.nodes.back().debug_name = p + "v_proj";

        RT::RoPENode rope{attn, d, d, static_cast<float>(c.rope_theta), 1.0f, 0u, true};
        auto qr = em.unary(RT::PrimitiveNode{rope}, qraw, q);
        pg.nodes.back().debug_name = p + "q_rope";
        auto kr = em.unary(RT::PrimitiveNode{rope}, kraw, kv);
        pg.nodes.back().debug_name = p + "k_rope";

        auto qn = norm(qr, q, d, l.query_layernorm, p + "query_layernorm");
        auto kn = norm(kr, kv, d, l.key_layernorm, p + "key_layernorm");

        auto state = pg.alloc_state(RT::PrimitiveStateKind::KV_CACHE, i);
        em.emit(RT::PrimitiveNode{RT::KvAppendNode{attn, i}}, {kn, v}, {}, {}, {state});
        pg.nodes.back().debug_name = p + "kv_append";

        auto context = em.value(q);
        RT::PagedAttentionNode pan{attn, 1.0f / std::sqrt(static_cast<float>(d)), i};
        em.emit(RT::PrimitiveNode{pan}, {qn}, {context}, {state});
        pg.nodes.back().debug_name = p + "attention";

        auto out = em.linear(l.o_proj, weight(l.o_proj), context);
        pg.nodes.back().debug_name = p + "o_proj";
        auto ar = residual(x, out, p + "attention_residual");

        n = norm(ar, h, h, l.post_attention_layernorm, p + "post_attention_layernorm");
        auto gate = em.linear(l.gate_proj, weight(l.gate_proj), n);
        pg.nodes.back().debug_name = p + "gate_proj";
        auto up = em.linear(l.up_proj, weight(l.up_proj), n);
        pg.nodes.back().debug_name = p + "up_proj";

        auto product = em.value(f);
        em.emit(RT::PrimitiveNode{RT::SwiGluNode{RT::RowwiseShapeKey{f}, true}},
                {gate, up}, {product});
        pg.nodes.back().debug_name = p + "mlp_product";

        out = em.linear(l.down_proj, weight(l.down_proj), product);
        pg.nodes.back().debug_name = p + "down_proj";
        x = residual(ar, out, p + "output");
        lowered.layer_output_values.emplace_back(i, x);
    }

    auto selected = em.value(h, RT::ValueDType::BF16, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{RT::OutputGatherNode{RT::RowwiseShapeKey{h}}}, {x}, {selected});
    pg.nodes.back().debug_name = "selected_hidden";

    auto normalized = norm(selected, h, h, w.final_norm, "final_norm");
    auto logits = em.linear(w.lm_head, weight(w.lm_head), normalized, RT::ValueDType::F32);
    pg.nodes.back().debug_name = "logits";

    auto sampled = em.value(1, RT::ValueDType::I32, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{RT::SamplingNode{static_cast<std::uint32_t>(c.vocab_size)}},
            {logits}, {sampled});
    pg.nodes.back().debug_name = "sampled";
    pg.external_outputs = {selected, logits, sampled, x};

    for (const auto& node : pg.nodes) {
        if (node.debug_name.rfind("L0.", 0) == 0 && !node.outputs.empty()) {
            lowered.trace_values.emplace_back(node.debug_name, node.outputs[0]);
        }
    }
    return lowered;
}

}
