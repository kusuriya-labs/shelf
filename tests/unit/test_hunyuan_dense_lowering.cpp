#include <shelf/models/hunyuan_dense/model/lower_to_primitives.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <shelf/runtime/program/program.h>

#include <cmath>
#include <cstdio>
#include <variant>

int main() {
    shelf::hunyuan_dense::Config c;
    c.num_hidden_layers = 2;
    c.hidden_size = 2048;
    c.num_attention_heads = 16;
    c.num_key_value_heads = 4;
    c.head_dim = 128;
    c.intermediate_size = 6144;
    c.vocab_size = 120818;
    c.max_position_embeddings = 262144;
    c.rms_norm_eps = 1e-5;
    c.rope_theta = 10000.0;
    c.tie_word_embeddings = true;
    c.rope_scaling.type = "dynamic";
    c.rope_scaling.alpha = 1000.0;

    shelf::hunyuan_dense::ModelWeights w;
    w.layers.resize(2);
    auto shape = [](auto& m, std::uint32_t rows, std::uint32_t cols) {
        m.rows = rows;
        m.cols = cols;
        m.k_padded = cols;
    };
    shape(w.embed_tokens, 120818, 2048);
    shape(w.lm_head, 120832, 2048);
    w.lm_head.quant_spec = shelf::quantization::QuantSpecId::EXL3_MUL24_INT8_V1;
    w.lm_head.exl3_ka = 6;
    for (auto& l : w.layers) {
        shape(l.q_proj, 2048, 2048);
        shape(l.k_proj, 512, 2048);
        shape(l.v_proj, 512, 2048);
        shape(l.o_proj, 2048, 2048);
        shape(l.gate_proj, 6144, 2048);
        shape(l.up_proj, 6144, 2048);
        shape(l.down_proj, 2048, 6144);
        for (auto* m : {&l.q_proj, &l.k_proj, &l.v_proj, &l.o_proj,
                        &l.gate_proj, &l.up_proj, &l.down_proj}) {
            m->quant_spec = shelf::quantization::QuantSpecId::EXL3_MUL24_INT8_V1;
            m->exl3_ka = 4;
        }
    }

    auto lowered = shelf::hunyuan_dense::lower_to_primitives(c, w);
    if (!lowered.ok()) {
        std::fprintf(stderr, "lower failed: %s\n", lowered.status().message().c_str());
        return 1;
    }

    const auto& g = lowered.value().graph;
    int q_rope_idx = -1;
    int q_norm_idx = -1;
    int k_rope_idx = -1;
    int k_norm_idx = -1;
    int kv_append_idx = -1;
    bool checked_logits = false;
    bool checked_sampling = false;
    unsigned norms = 0;

    for (int idx = 0; idx < static_cast<int>(g.nodes.size()); ++idx) {
        const auto& n = g.nodes[static_cast<std::size_t>(idx)];
        if (n.debug_name == "L0.q_rope") q_rope_idx = idx;
        if (n.debug_name == "L0.query_layernorm") q_norm_idx = idx;
        if (n.debug_name == "L0.k_rope") k_rope_idx = idx;
        if (n.debug_name == "L0.key_layernorm") k_norm_idx = idx;
        if (n.debug_name == "L0.kv_append") kv_append_idx = idx;
        if (const auto* norm = std::get_if<shelf::runtime::RmsNormNode>(&n.node)) {
            if (norm->weight_offset != 0.0f) return 1;
            if (norm->group_size == 128u &&
                norm->weight_mode != shelf::runtime::RmsNormWeightMode::OFFSET_BF16_SMALL_GROUP) {
                return 1;
            }
            ++norms;
        }
        if (n.debug_name == "logits") {
            const auto* lin = std::get_if<shelf::runtime::LinearNode>(&n.node);
            if (!lin || lin->shape.output_features != 120832u ||
                lin->out_dtype != shelf::runtime::ValueDType::F32) {
                return 1;
            }
            checked_logits = true;
        }
        if (const auto* samp = std::get_if<shelf::runtime::SamplingNode>(&n.node)) {
            if (samp->vocab_size != 120818u) return 1;
            checked_sampling = true;
        }
    }

    if (q_rope_idx < 0 || q_norm_idx <= q_rope_idx ||
        k_rope_idx < 0 || k_norm_idx <= k_rope_idx ||
        kv_append_idx <= k_norm_idx || !checked_logits || !checked_sampling ||
        norms != 9u || g.external_outputs.size() != 4u ||
        g.value_features(g.external_outputs[1]) != 120832u) {
        std::fprintf(stderr, "post-RoPE QK norm or padded lm_head check failed\n");
        return 1;
    }

    for (const auto& ws : lowered.value().weights) {
        if (ws.plamo_exl3_linear) return 1;
    }

    shelf::runtime::ProgramBuildOptions opts;
    opts.max_token_rows = 16;
    opts.max_output_rows = 16;
    auto prefill = shelf::runtime::build_program(
        g, {lowered.value().weights.data(), lowered.value().weights.size()},
        shelf::runtime::RowBucket::R16, shelf::runtime::ExecutionClass::PREFILL, opts);
    if (!prefill.ok()) {
        std::fprintf(stderr, "build_program prefill failed: %s\n",
                     prefill.status().message().c_str());
        return 1;
    }

    bool saw_sampling = false;
    for (const auto& b : prefill.value().dispatches) {
        if (b.kernel_id == shelf::runtime::KernelId::SAMPLING) {
            if (b.rotary_dim != 120818u) return 1;
            saw_sampling = true;
        }
    }
    if (!saw_sampling) return 1;

    std::printf("test_hunyuan_dense_lowering: PASS\n");
    return 0;
}
