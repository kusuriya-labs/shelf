#include <shelf/models/plamo3/model/lower_to_primitives.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <shelf/runtime/program/program.h>
#include <cstdio>
#include <variant>

int main() {
    shelf::qwen_hybrid::QwenHybridConfig hybrid;
    hybrid.declared_model_type = "plamo3"; hybrid.num_hidden_layers = 8;
    hybrid.layer_types.assign(8, 1);
    if (shelf::qwen35::make_qwen35_gdn_state_layout(hybrid).num_gdn_states != 0) return 1;
    hybrid.declared_model_type = "qwen3_5";
    if (shelf::qwen35::make_qwen35_gdn_state_layout(hybrid).num_gdn_states != 1) return 1;
    shelf::plamo3::Config c;
    c.hidden_size=1536; c.head_dim=128; c.num_attention_heads=16; c.num_key_value_heads=1;
    c.intermediate_size=6144; c.vocab_size=107520; c.num_hidden_layers=8;
    c.sliding_window_pattern=8; c.rms_norm_eps=1e-6;
    c.local_rope.theta=10000; c.full_rope.theta=1000000; c.full_rope.factor=256;
    shelf::plamo3::ModelWeights w; w.layers.resize(8);
    auto shape=[](auto& m, uint32_t rows, uint32_t cols) { m.rows=rows; m.cols=cols; m.k_padded=cols; };
    shape(w.embed_tokens,107520,1536); shape(w.lm_head,107520,1536);
    for (auto& l : w.layers) {
        shape(l.qkv,2304,1536); shape(l.o_proj,1536,2048);
        shape(l.gate_up,12288,1536); shape(l.down,1536,6144);
    }
    auto lower = shelf::plamo3::lower_to_primitives(c,w);
    if (!lower.ok()) { std::fprintf(stderr,"%s\n",lower.status().message().c_str()); return 1; }
    auto& v=lower.value(); unsigned ropes=0, kv=0, attention=0, norms=0;
    bool rounded_logits = false;
    for (const auto& n : v.graph.nodes) {
        if (n.debug_name == "logits_bf16") {
            const auto* linear = std::get_if<shelf::runtime::LinearNode>(&n.node);
            rounded_logits = linear && linear->out_dtype == shelf::runtime::ValueDType::BF16;
        }
        if (auto* r=std::get_if<shelf::runtime::RoPENode>(&n.node)) {
            unsigned layer=ropes/2;
            if (r->frequency_table_index != (layer==7 ? 1u : 0u) || !r->bf16_intermediates) return 1;
            ++ropes;
        }
        if (std::holds_alternative<shelf::runtime::KvAppendNode>(n.node)) ++kv;
        if (std::holds_alternative<shelf::runtime::PagedAttentionNode>(n.node)) ++attention;
        if (auto* norm=std::get_if<shelf::runtime::RmsNormNode>(&n.node)) {
            if (norm->weight_mode != shelf::runtime::RmsNormWeightMode::OFFSET_BF16) return 1;
            ++norms;
        }
    }
    if (ropes!=16 || kv!=8 || attention!=8 || norms!=49 || !rounded_logits ||
        v.graph.external_outputs.size()!=4) return 1;
    shelf::runtime::ProgramBuildOptions options; options.max_token_rows=16; options.max_output_rows=16;
    auto program=shelf::runtime::build_program(v.graph, {v.weights.data(),v.weights.size()},
        shelf::runtime::RowBucket::R16, shelf::runtime::ExecutionClass::PREFILL, options);
    if (!program.ok()) { std::fprintf(stderr,"%s\n",program.status().message().c_str()); return 1; }
    unsigned selected=0;
    for (auto& b : program.value().dispatches) if (b.kernel_id==shelf::runtime::KernelId::ROPE) {
        if (b.rope_table_index == 1) ++selected;
    }
    if (selected != 2) return 1;
    std::puts("PLaMo graph and layer-specific RoPE bindings verified"); return 0;
}
