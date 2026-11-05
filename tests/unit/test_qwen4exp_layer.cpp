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

constexpr std::uint32_t kHidden = 128u;
constexpr std::uint32_t kHc = 4u;
constexpr std::uint32_t kLowrank = 64u;
constexpr std::uint32_t kWidth = kHc * kHidden;
constexpr std::uint32_t kExperts = 4u;
constexpr std::uint32_t kTopK = 2u;
constexpr std::uint32_t kIntermediate = 128u;
constexpr std::uint32_t kShared = 128u;
constexpr std::uint32_t kKeyHeads = 2u;
constexpr std::uint32_t kValueHeads = 4u;
constexpr std::uint32_t kKeyDim = 16u;
constexpr std::uint32_t kValueDim = 16u;
constexpr std::uint32_t kQk = kKeyHeads * kKeyDim;
constexpr std::uint32_t kValue = kValueHeads * kValueDim;
constexpr std::uint32_t kConv = 2u * kQk + kValue;

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

    shelf::weights::MatrixWeight qkv = matrix(kConv, kHidden);
    shelf::weights::MatrixWeight z = matrix(kValue, kHidden);
    shelf::weights::MatrixWeight b = matrix(kValueHeads, kHidden);
    shelf::weights::MatrixWeight a = matrix(kValueHeads, kHidden);
    shelf::weights::MatrixWeight out = matrix(kHidden, kValue);

    shelf::qwen4exp::Qwen4ExpMoeWeights moe;
    moe.router = matrix(kExperts, kHidden);
    moe.shared_gate = matrix(1u, kHidden);
    moe.shared_gate_proj = matrix(kShared, kHidden);
    moe.shared_up_proj = matrix(kShared, kHidden);
    moe.shared_down_proj = matrix(kHidden, kShared);
    moe.gate_experts.local_experts = kExperts;
    moe.gate_experts.rows = kIntermediate;
    moe.gate_experts.cols = kHidden;
    moe.up_experts = moe.gate_experts;
    moe.down_experts.local_experts = kExperts;
    moe.down_experts.rows = kHidden;
    moe.down_experts.cols = kIntermediate;

    const auto add_weight = [&](std::uint32_t rows, std::uint32_t cols) {
        graph.weight_table.push_back(rt::MatrixwiseShapeKey{rows, cols});
    };
    add_weight(kLowrank, kWidth);
    add_weight(kWidth, kLowrank);
    add_weight(kHc, kWidth);
    add_weight(kLowrank, kWidth);
    add_weight(kWidth, kLowrank);
    add_weight(kHc, kWidth);
    add_weight(kConv, kHidden);
    add_weight(kValue, kHidden);
    add_weight(kValueHeads, kHidden);
    add_weight(kValueHeads, kHidden);
    add_weight(kHidden, kValue);
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
    graph.external_inputs = {hyper};

    shelf::qwen_hybrid::Emitter em{graph};
    shelf::qwen4exp::Qwen4ExpLayerLowering layer;
    layer.is_gdn = true;
    layer.hyper.hc = kHc;
    layer.hyper.hidden = kHidden;
    layer.hyper.lowrank = kLowrank;
    layer.hyper.eps = 1e-6f;
    layer.hyper.norm_parameter_index = 0u;
    set_site(layer.attn_hyper_weights, attn_down, attn_up, attn_inject);
    set_site(layer.mlp_hyper_weights, mlp_down, mlp_up, mlp_inject);
    layer.attn_hyper_indices = {0u, 1u, 2u};
    layer.mlp_hyper_indices = {3u, 4u, 5u};

    layer.gdn_geometry.key_heads = kKeyHeads;
    layer.gdn_geometry.value_heads = kValueHeads;
    layer.gdn_geometry.key_head_dim = kKeyDim;
    layer.gdn_geometry.value_head_dim = kValueDim;
    layer.gdn_geometry.conv_kernel = 4u;
    layer.gdn_geometry.sigmoid_gate = true;
    layer.gdn_weights.qkv = &qkv;
    layer.gdn_weights.z = &z;
    layer.gdn_weights.b = &b;
    layer.gdn_weights.a = &a;
    layer.gdn_weights.out = &out;
    layer.gdn_indices = {6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 0u};

    layer.moe_geometry = {kExperts, kExperts, 0u, kTopK, kHidden, kIntermediate, kShared};
    layer.moe_weights = moe;
    layer.moe_indices = {15u, 16u, 17u, 18u, 19u, 20u, 21u, 22u};

    auto emitted = shelf::qwen4exp::emit_qwen4exp_layer(em, layer, hyper, "test.");
    if (!emitted.ok()) {
        std::printf("FAIL emit: %s\n", emitted.status().message().c_str());
        return 1;
    }
    graph.external_outputs = {emitted.value()};

    auto status = rt::validate_primitive_graph(graph);
    if (!status.ok()) {
        std::printf("FAIL validate: %s\n", status.message().c_str());
        return 1;
    }

    std::string sequence;
    for (const auto& node : graph.nodes) {
        if (!sequence.empty()) sequence += ",";
        sequence += rt::to_string(rt::primitive_kind_of(node.node));
    }
    check(sequence.find("SCALE,LINEAR,LINEAR,LINEAR,LINEAR,STATEFUL_CAUSAL_CONV1D") !=
              std::string::npos,
          "attention HyperConnection feeds GDN");
    check(sequence.find("GDN_RECURRENCE,RMS_NORM,SIGMOID,MUL") != std::string::npos,
          "GDN uses the Qwen4Exp sigmoid gate");
    check(sequence.find("HYPER_APPLY,RMS_NORM,LINEAR") != std::string::npos,
          "attention apply feeds the MLP HyperConnection mix");
    check(sequence.find("MOE_ROUTER,MOE_GROUPED_LINEAR,MOE_GROUPED_LINEAR,SWIGLU,"
                        "MOE_GROUPED_LINEAR") != std::string::npos,
          "MoE uses separate gate/up/down grouped planes");
    check(sequence.find("MOE_COMBINE,HYPER_APPLY") != std::string::npos,
          "the MoE combine is injected back into the hyper stack");

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
