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
    auto& v=lower.value(); unsigned ropes=0, kv=0, attention=0, norms=0, bf16_swiglu=0;
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
        if (std::holds_alternative<shelf::runtime::SwiGluNode>(n.node)) ++bf16_swiglu;
        if (auto* norm=std::get_if<shelf::runtime::RmsNormNode>(&n.node)) {
            if (norm->weight_mode != shelf::runtime::RmsNormWeightMode::OFFSET_BF16) return 1;
            ++norms;
        }
    }
    if (ropes!=16 || kv!=8 || attention!=8 || norms!=49 || bf16_swiglu!=0 || !rounded_logits ||
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
    for (auto& l : w.layers) {
        l.split_projections = true;
        shape(l.q,2048,1536); shape(l.k,128,1536); shape(l.v,128,1536);
        shape(l.gate,6144,1536); shape(l.up,6144,1536);
    }
    auto separate = shelf::plamo3::lower_to_primitives(c,w);
    if (!separate.ok()) return 1;
    unsigned q_proj=0, k_proj=0, v_proj=0, gate_proj=0, up_proj=0, split_nodes=0, rounded_swiglu=0;
    for (const auto& n : separate.value().graph.nodes) {
        if (n.debug_name.find(".q_proj") != std::string::npos) ++q_proj;
        if (n.debug_name.find(".k_proj") != std::string::npos) ++k_proj;
        if (n.debug_name.find(".v_proj") != std::string::npos) ++v_proj;
        if (n.debug_name.find(".gate_proj") != std::string::npos) ++gate_proj;
        if (n.debug_name.find(".up_proj") != std::string::npos) ++up_proj;
        if (std::holds_alternative<shelf::runtime::SplitNode>(n.node)) ++split_nodes;
        if (const auto* swiglu = std::get_if<shelf::runtime::SwiGluNode>(&n.node)) {
            if (!swiglu->bf16_intermediate) return 1;
            ++rounded_swiglu;
        }
    }
    if (q_proj!=8 || k_proj!=8 || v_proj!=8 || gate_proj!=8 || up_proj!=8 || split_nodes!=0 || rounded_swiglu!=8) return 1;
    auto split_program = shelf::runtime::build_program(separate.value().graph,
        {separate.value().weights.data(), separate.value().weights.size()},
        shelf::runtime::RowBucket::R16, shelf::runtime::ExecutionClass::PREFILL, options);
    if (!split_program.ok()) return 1;
    unsigned rounded_bindings = 0;
    for (const auto& binding : split_program.value().dispatches)
        if (binding.kernel_id == shelf::runtime::KernelId::SWIGLU && binding.flags == 1u) ++rounded_bindings;
    if (rounded_bindings != 8) return 1;
    std::puts("PLaMo graph and layer-specific RoPE bindings verified"); return 0;
}
