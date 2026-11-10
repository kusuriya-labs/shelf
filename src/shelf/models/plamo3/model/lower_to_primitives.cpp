#include <shelf/models/plamo3/model/lower_to_primitives.h>
#include <shelf/models/common/qwen_hybrid/emitter.h>
#include <shelf/models/common/qwen_hybrid/head_lowering.h>
#include <shelf/models/common/qwen_hybrid/weight_slot.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace shelf::plamo3 {
namespace RT = shelf::runtime;
namespace QH = shelf::qwen_hybrid;

Result<QH::HybridLoweredPrimitives> lower_to_primitives(const Config& c, const ModelWeights& w) {
    if (w.layers.size() != c.num_hidden_layers || c.sliding_window_pattern == 0 ||
        c.head_dim == 0 || c.num_key_value_heads == 0 ||
        c.num_attention_heads % c.num_key_value_heads != 0 ||
        c.hidden_size > UINT32_MAX || c.intermediate_size > UINT32_MAX / 2 ||
        c.num_attention_heads > UINT32_MAX / c.head_dim ||
        c.num_key_value_heads > UINT32_MAX / c.head_dim)
        return Status::invalid_argument("plamo3: invalid lowering geometry", __FILE__, __LINE__);
    const uint32_t h = static_cast<uint32_t>(c.hidden_size);
    const uint32_t d = static_cast<uint32_t>(c.head_dim);
    const uint32_t q = static_cast<uint32_t>(c.num_attention_heads * c.head_dim);
    const uint32_t kv = static_cast<uint32_t>(c.num_key_value_heads * c.head_dim);
    const uint32_t f = static_cast<uint32_t>(c.intermediate_size);
    QH::HybridLoweredPrimitives lowered;
    auto& pg = lowered.graph;
    QH::Emitter em{pg};
    auto weight = [&](const shelf::weights::MatrixWeight& m) {
        lowered.weights.push_back(QH::make_weight_slot(m));
        return static_cast<uint32_t>(lowered.weights.size() - 1);
    };
    auto param = [&](const gpu::Tensor& t) {
        RT::StaticParameterSlot p{}; p.device_ptr = t.data();
        p.elements = t.ndim() == 1 ? static_cast<uint32_t>(t.dim(0)) : 0;
        p.dtype = RT::ValueDType::BF16;
        lowered.parameters.push_back(p);
        return static_cast<uint32_t>(lowered.parameters.size() - 1);
    };
    auto norm = [&](RT::ValueId input, uint32_t width, uint32_t group,
                    const gpu::Tensor& tensor, float offset, const std::string& name) {
        RT::RmsNormNode node; node.shape = RT::RowwiseShapeKey{width};
        node.eps = static_cast<float>(c.rms_norm_eps); node.group_size = group;
        node.parameter_index = param(tensor); node.weight_mode = RT::RmsNormWeightMode::OFFSET_BF16;
        node.weight_offset = offset;
        auto out = em.unary(RT::PrimitiveNode{node}, input, width);
        pg.nodes.back().debug_name = name; return out;
    };
    auto split = [&](RT::ValueId in, uint32_t a, uint32_t b, const std::string& name) {
        RT::SplitNode n; n.input_shape = RT::RowwiseShapeKey{a + b};
        n.output_shape_a = RT::RowwiseShapeKey{a}; n.output_shape_b = RT::RowwiseShapeKey{b};
        n.split_dim = 1; n.layout = RT::SplitLayout::HALVES;
        auto x = em.value(a), y = em.value(b);
        em.emit(RT::PrimitiveNode{n}, {in}, {x,y}); pg.nodes.back().debug_name = name;
        return std::pair{x,y};
    };
    auto residual = [&](RT::ValueId a, RT::ValueId b, const std::string& name) {
        auto out = em.value(h); em.emit(RT::PrimitiveNode{RT::ResidualAddNode{RT::RowwiseShapeKey{h}}},
                                       {a,b}, {out}); pg.nodes.back().debug_name = name; return out;
    };
    auto tokens = em.value(0, RT::ValueDType::I32);
    auto x = em.value(h); RT::EmbeddingLookupNode emb;
    emb.vocab_size = w.embed_tokens.rows; emb.hidden_size = h;
    emb.weight_shape = {w.embed_tokens.rows, w.embed_tokens.cols}; emb.weight_index = weight(w.embed_tokens);
    em.emit(RT::PrimitiveNode{emb}, {tokens}, {x}); pg.nodes.back().debug_name = "embedding";
    pg.external_inputs.push_back(tokens);
    const RT::AttentionShapeKey attn{static_cast<uint32_t>(c.num_attention_heads),
        static_cast<uint32_t>(c.num_key_value_heads), d, d};
    for (uint32_t i = 0; i < w.layers.size(); ++i) {
        const auto& l = w.layers[i]; const std::string p = "L" + std::to_string(i) + ".";
        lowered.layer_input_values.emplace_back(i, x);
        auto n = norm(x, h, h, l.pre_mixer_norm, 1.0f, p + "pre_mixer_norm");
        auto qkv = em.linear(l.qkv, weight(l.qkv), n); pg.nodes.back().debug_name = p + "qkv";
        auto [qraw, kvraw] = split(qkv, q, 2 * kv, p + "q_kv_split");
        auto [kraw, v] = split(kvraw, kv, kv, p + "k_v_split");
        auto qn = norm(qraw, q, d, l.q_norm, 1.0f, p + "q_norm");
        auto kn = norm(kraw, kv, d, l.k_norm, 1.0f, p + "k_norm");
        const bool full = c.is_full_attention(i);
        RT::RoPENode rope{attn, d, d,
            static_cast<float>(full ? c.full_rope.theta : c.local_rope.theta),
            static_cast<float>(full ? 1.0 + 0.1 * std::log(c.full_rope.factor) : 1.0),
            full ? 1u : 0u, true};
        auto qr = em.unary(RT::PrimitiveNode{rope}, qn, q); pg.nodes.back().debug_name = p + "q_rope";
        auto kr = em.unary(RT::PrimitiveNode{rope}, kn, kv); pg.nodes.back().debug_name = p + "k_rope";
        auto state = pg.alloc_state(RT::PrimitiveStateKind::KV_CACHE, i);
        em.emit(RT::PrimitiveNode{RT::KvAppendNode{attn, i}}, {kr,v}, {}, {}, {state});
        pg.nodes.back().debug_name = p + "kv_append";
        auto context = em.value(q);
        RT::PagedAttentionNode pan{attn, 1.0f / std::sqrt(static_cast<float>(d)), i};
        em.emit(RT::PrimitiveNode{pan}, {qr}, {context}, {state}); pg.nodes.back().debug_name = p + "attention";
        auto out = em.linear(l.o_proj, weight(l.o_proj), context); pg.nodes.back().debug_name = p + "o_proj";
        out = norm(out, h, h, l.post_mixer_norm, 0.2f, p + "post_mixer_norm");
        auto ar = residual(x, out, p + "attention_residual");
        n = norm(ar, h, h, l.pre_mlp_norm, 1.0f, p + "pre_mlp_norm");
        auto gu = em.linear(l.gate_up, weight(l.gate_up), n); pg.nodes.back().debug_name = p + "gate_up";
        auto [gate, up] = split(gu, f, f, p + "gate_up_split");
        auto activated = em.unary(RT::PrimitiveNode{RT::SiLUNode{RT::RowwiseShapeKey{f}}}, gate, f);
        pg.nodes.back().debug_name = p + "silu";
        auto product = em.value(f); em.emit(RT::PrimitiveNode{RT::MulNode{RT::RowwiseShapeKey{f}}},
                                           {activated,up}, {product}); pg.nodes.back().debug_name = p + "mlp_product";
        out = em.linear(l.down, weight(l.down), product); pg.nodes.back().debug_name = p + "down";
        out = norm(out, h, h, l.post_mlp_norm, static_cast<float>(1.0 / std::pow(5.0, 1.5)), p + "post_mlp_norm");
        x = residual(ar, out, p + "output"); lowered.layer_output_values.emplace_back(i, x);
    }
    // The reference returns BF16 lm_head logits. Convert them to F32 only after
    // rounding, since the executor's external logits buffer is F32.
    auto selected = em.value(h, RT::ValueDType::BF16, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{RT::OutputGatherNode{RT::RowwiseShapeKey{h}}}, {x}, {selected});
    pg.nodes.back().debug_name = "selected_hidden";
    RT::RmsNormNode final_norm; final_norm.shape = RT::RowwiseShapeKey{h};
    final_norm.eps = static_cast<float>(c.rms_norm_eps); final_norm.group_size = h;
    final_norm.parameter_index = param(w.final_norm);
    final_norm.weight_mode = RT::RmsNormWeightMode::OFFSET_BF16; final_norm.weight_offset = 1.0f;
    auto normalized = em.value(h, RT::ValueDType::BF16, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{final_norm}, {selected}, {normalized}); pg.nodes.back().debug_name = "final_norm";
    auto logits_bf16 = em.linear(w.lm_head, weight(w.lm_head), normalized);
    pg.nodes.back().debug_name = "logits_bf16";
    auto logits = em.value(w.lm_head.rows, RT::ValueDType::F32, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{RT::ScaleNode{RT::RowwiseShapeKey{w.lm_head.rows}, 1.0f}},
            {logits_bf16}, {logits}); pg.nodes.back().debug_name = "logits";
    auto sampled = em.value(1, RT::ValueDType::I32, RT::ValueRowDomain::OUTPUT_ROWS);
    em.emit(RT::PrimitiveNode{RT::SamplingNode{w.lm_head.rows}}, {logits}, {sampled});
    pg.nodes.back().debug_name = "sampled";
    pg.external_outputs = {selected, logits, sampled, x};
    // The executor only captures these when SHELF_LAYER_DUMP_DIR is enabled.
    // Expose the first block's operations to localize numerical drift.
    for (const auto& node : pg.nodes) {
        if (node.debug_name.rfind("L0.", 0) == 0 && !node.outputs.empty())
            lowered.trace_values.emplace_back(node.debug_name, node.outputs[0]);
    }
    return lowered;
}
}  // namespace shelf::plamo3
