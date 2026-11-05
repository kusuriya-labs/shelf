#include <shelf/models/qwen4exp/model/layer_lowering.h>
#include <shelf/runtime/graph/primitive_graph.h>

#include <cstdio>
#include <string>

namespace rt = shelf::runtime;

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const char* message) {
    if (condition) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", message);
    }
}

constexpr std::uint32_t kHidden = 64u;
constexpr std::uint32_t kHc = 4u;
constexpr std::uint32_t kLowrank = 64u;
constexpr std::uint32_t kWidth = kHc * kHidden;
constexpr std::uint32_t kExperts = 4u;
constexpr std::uint32_t kTopK = 2u;
constexpr std::uint32_t kIntermediate = 64u;
constexpr std::uint32_t kShared = 64u;
constexpr std::uint32_t kQHeads = 2u;
constexpr std::uint32_t kKvHeads = 1u;
constexpr std::uint32_t kHeadDim = 16u;
constexpr std::uint32_t kRotary = 8u;
constexpr std::uint32_t kQFeatures = kQHeads * kHeadDim;
constexpr std::uint32_t kKvFeatures = kKvHeads * kHeadDim;
constexpr std::uint32_t kIndexHeads = 4u;
constexpr std::uint32_t kIndexDim = 128u;
constexpr std::uint32_t kIndexRotary = 64u;
constexpr std::uint32_t kIndexBlocks = 8u;
constexpr std::uint32_t kIndexTopK = 2u;
constexpr std::uint32_t kMaskLen = 32u;

shelf::weights::MatrixWeight matrix(std::uint32_t rows, std::uint32_t cols) {
    shelf::weights::MatrixWeight w;
    w.rows = rows;
    w.cols = cols;
    w.k_padded = cols;
    return w;
}

void set_site(shelf::qwen4exp::HyperConnectionSiteWeights& out,
              shelf::weights::MatrixWeight& down, shelf::weights::MatrixWeight& up,
              shelf::weights::MatrixWeight& inject) {
    out.down = &down;
    out.up = &up;
    out.inject = &inject;
}

}  // namespace

int main() {
    rt::PrimitiveGraph graph;
    graph.grouped_row_shape = rt::GroupedRowShape{kExperts, kTopK};

    shelf::weights::MatrixWeight attn_down = matrix(kLowrank, kWidth);
    shelf::weights::MatrixWeight attn_up = matrix(kWidth, kLowrank);
    shelf::weights::MatrixWeight attn_inject = matrix(kHc, kWidth);
    shelf::weights::MatrixWeight mlp_down = matrix(kLowrank, kWidth);
    shelf::weights::MatrixWeight mlp_up = matrix(kWidth, kLowrank);
    shelf::weights::MatrixWeight mlp_inject = matrix(kHc, kWidth);

    shelf::weights::MatrixWeight q_proj = matrix(2u * kQFeatures, kHidden);
    shelf::weights::MatrixWeight k_proj = matrix(kKvFeatures, kHidden);
    shelf::weights::MatrixWeight v_proj = matrix(kKvFeatures, kHidden);
    shelf::weights::MatrixWeight o_proj = matrix(kHidden, kQFeatures);
    shelf::weights::MatrixWeight index_qk =
        matrix((kIndexHeads + 1u) * kIndexDim, kHidden);

    shelf::qwen4exp::Qwen4ExpMoeWeights moe;
    moe.router = matrix(kExperts, kHidden);
    moe.shared_gate = matrix(1u, kHidden);
    moe.shared_gate_proj = matrix(kShared, kHidden);
    moe.shared_up_proj = matrix(kShared, kHidden);
    moe.shared_down_proj = matrix(kHidden, kShared);
    moe.gate_experts = matrix(kIntermediate, kHidden);
    moe.gate_experts.experts = kExperts;
    moe.up_experts = moe.gate_experts;
    moe.down_experts = matrix(kHidden, kIntermediate);
    moe.down_experts.experts = kExperts;
    moe.experts = {kExperts, kExperts, 0u};

    const auto add_weight = [&](std::uint32_t rows, std::uint32_t cols) {
        graph.weight_table.push_back(rt::MatrixwiseShapeKey{rows, cols});
    };
    add_weight(kLowrank, kWidth);
    add_weight(kWidth, kLowrank);
    add_weight(kHc, kWidth);
    add_weight(kLowrank, kWidth);
    add_weight(kWidth, kLowrank);
    add_weight(kHc, kWidth);
    add_weight(2u * kQFeatures, kHidden);
    add_weight(kKvFeatures, kHidden);
    add_weight(kKvFeatures, kHidden);
    add_weight(kHidden, kQFeatures);
    add_weight((kIndexHeads + 1u) * kIndexDim, kHidden);
    add_weight(1u, 1u);
    add_weight(1u, 1u);
    add_weight(1u, 1u);
    add_weight(1u, 1u);
    add_weight(kExperts, kHidden);
    add_weight(1u, kHidden);
    add_weight(kShared, kHidden);
    add_weight(kShared, kHidden);
    add_weight(kHidden, kShared);
    add_weight(kIntermediate, kHidden);
    add_weight(kIntermediate, kHidden);
    add_weight(kHidden, kIntermediate);

    const rt::ValueId hyper = graph.alloc_value(kWidth);
    const rt::ValueId embeddings = graph.alloc_value(kHidden);
    graph.external_inputs = {hyper};

    shelf::qwen_hybrid::Emitter em{graph};
    shelf::qwen4exp::Qwen4ExpLayerLowering layer;
    layer.is_gdn = false;
    layer.hyper.hc = kHc;
    layer.hyper.hidden = kHidden;
    layer.hyper.lowrank = kLowrank;
    layer.hyper.eps = 1e-6f;
    layer.hyper.norm_parameter_index = 0u;
    set_site(layer.attn_hyper_weights, attn_down, attn_up, attn_inject);
    set_site(layer.mlp_hyper_weights, mlp_down, mlp_up, mlp_inject);
    layer.attn_hyper_indices = {0u, 1u, 2u};
    layer.mlp_hyper_indices = {3u, 4u, 5u};

    layer.qsa_geometry.attention.query_heads = kQHeads;
    layer.qsa_geometry.attention.kv_heads = kKvHeads;
    layer.qsa_geometry.attention.head_dim = kHeadDim;
    layer.qsa_geometry.attention.rotary_dim = kRotary;
    layer.qsa_geometry.attention.rope_theta = 10000000.0f;
    layer.qsa_geometry.attention.eps = 1e-6f;
    layer.qsa_geometry.attention.tp_combine = false;
    layer.qsa_geometry.indexer.hidden = kHidden;
    layer.qsa_geometry.indexer.index_heads = kIndexHeads;
    layer.qsa_geometry.indexer.index_kv_heads = 1u;
    layer.qsa_geometry.indexer.index_head_dim = kIndexDim;
    layer.qsa_geometry.indexer.index_rotary_dim = kIndexRotary;
    layer.qsa_geometry.indexer.compress_ratio = 4u;
    layer.qsa_geometry.indexer.blocks = kIndexBlocks;
    layer.qsa_geometry.indexer.top_k = kIndexTopK;
    layer.qsa_geometry.indexer.mask_len = kMaskLen;
    layer.qsa_geometry.indexer.state_index = 3u;
    layer.qsa_weights.attention.q = &q_proj;
    layer.qsa_weights.attention.k = &k_proj;
    layer.qsa_weights.attention.v = &v_proj;
    layer.qsa_weights.attention.o = &o_proj;
    layer.qsa_weights.indexer.qk_proj = &index_qk;
    layer.qsa_indices.attention.w_q = 6u;
    layer.qsa_indices.attention.w_k = 7u;
    layer.qsa_indices.attention.w_v = 8u;
    layer.qsa_indices.attention.w_o = 9u;
    layer.qsa_indices.attention.p_q_norm = 1u;
    layer.qsa_indices.attention.p_k_norm = 2u;
    layer.qsa_indices.attention.state_index = 1u;
    layer.qsa_indices.indexer.w_qk_proj = 10u;
    layer.qsa_indices.indexer.p_q_norm = 1u;
    layer.qsa_indices.indexer.p_k_norm = 2u;

    layer.moe_geometry = {kExperts, kExperts, 0u, kTopK, kHidden, kIntermediate, kShared};
    layer.moe_weights = moe;
    layer.moe_indices = {15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u};

    auto emitted = shelf::qwen4exp::emit_qwen4exp_layer(em, layer, hyper, "qsa_layer.");
    if (!emitted.ok()) {
        std::printf("FAIL emit: %s\n", emitted.status().message().c_str());
        return 1;
    }
    graph.external_outputs = {emitted.value()};

    const auto status = rt::validate_primitive_graph(graph);
    if (!status.ok()) {
        std::printf("FAIL validate: %s\n", status.message().c_str());
        return 1;
    }

    std::string sequence;
    for (const auto& node : graph.nodes) {
        if (!sequence.empty()) sequence += ",";
        sequence += rt::to_string(rt::primitive_kind_of(node.node));
    }

    const std::size_t mask_pos = sequence.find("QSA_EXPAND_MASK");
    const std::size_t attention_pos = sequence.find("PAGED_ATTENTION");
    check(mask_pos != std::string::npos, "QSA indexer emits the token mask");
    check(attention_pos != std::string::npos, "QSA layer emits paged attention");
    check(mask_pos < attention_pos, "token mask precedes the paged attention");
    check(sequence.find("PAGED_ATTENTION,SIGMOID,MUL,LINEAR,HYPER_APPLY") !=
              std::string::npos,
          "QSA attention is gated and injected into the hyper stack");
    check(sequence.find("MOE_ROUTER,MOE_GROUPED_LINEAR,MOE_GROUPED_LINEAR,SWIGLU,"
                        "MOE_GROUPED_LINEAR") != std::string::npos,
          "QSA layer feeds the MoE FFN");

    bool masked_attention = false;
    for (const auto& node : graph.nodes) {
        if (rt::primitive_kind_of(node.node) != rt::PrimitiveKind::PAGED_ATTENTION)
            continue;
        const auto& pa = std::get<rt::PagedAttentionNode>(node.node);
        masked_attention = pa.has_token_mask && node.inputs.size() == 2u;
    }
    check(masked_attention, "QSA attention node consumes the token mask");

    {
        std::uint32_t combines = 0;
        for (const auto& node : graph.nodes)
            if (node.tp_combine != 0) ++combines;
        check(combines == 1u,
              "only the MoE combine is a TP reduction for replicated attention/GDN");
    }

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
