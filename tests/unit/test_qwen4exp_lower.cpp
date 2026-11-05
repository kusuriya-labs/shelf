#include <shelf/models/qwen4exp/model/lower_to_primitives.h>
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
constexpr std::uint32_t kHc = 2u;
constexpr std::uint32_t kLowrank = 16u;
constexpr std::uint32_t kWidth = kHc * kHidden;
constexpr std::uint32_t kVocab = 128u;
constexpr std::uint32_t kExperts = 4u;
constexpr std::uint32_t kTopK = 2u;
constexpr std::uint32_t kIntermediate = 32u;
constexpr std::uint32_t kShared = 32u;
constexpr std::uint32_t kQHeads = 2u;
constexpr std::uint32_t kKvHeads = 1u;
constexpr std::uint32_t kHeadDim = 16u;
constexpr std::uint32_t kQFeatures = kQHeads * kHeadDim;
constexpr std::uint32_t kKvFeatures = kKvHeads * kHeadDim;
constexpr std::uint32_t kGdnKeyHeads = 2u;
constexpr std::uint32_t kGdnValueHeads = 4u;
constexpr std::uint32_t kGdnHeadDim = 8u;
constexpr std::uint32_t kGdnKey = kGdnKeyHeads * kGdnHeadDim;
constexpr std::uint32_t kGdnValue = kGdnValueHeads * kGdnHeadDim;
constexpr std::uint32_t kGdnConv = 2u * kGdnKey + kGdnValue;
constexpr std::uint32_t kIndexHeads = 2u;
constexpr std::uint32_t kIndexDim = 128u;

shelf::weights::MatrixWeight matrix(std::uint32_t rows, std::uint32_t cols) {
    shelf::weights::MatrixWeight w;
    w.rows = rows;
    w.cols = cols;
    w.k_padded = cols;
    return w;
}

shelf::weights::MatrixWeight experts(std::uint32_t rows, std::uint32_t cols) {
    shelf::weights::MatrixWeight w = matrix(rows, cols);
    w.encoding = shelf::weights::MatrixEncoding::Exl3;
    w.experts = kExperts;
    w.exl3_ka = 1u;
    return w;
}

shelf::qwen4exp::Qwen4ExpHyperConnectionWeights hyper() {
    shelf::qwen4exp::Qwen4ExpHyperConnectionWeights w;
    w.input_mix_down = matrix(kLowrank, kWidth);
    w.input_mix_up = matrix(kWidth, kLowrank);
    w.block_inject = matrix(kHc, kWidth);
    return w;
}

}  // namespace

int main() {
    shelf::qwen4exp::Qwen4ExpTextConfig config;
    config.vocab_size = kVocab;
    config.hidden_size = kHidden;
    config.num_hidden_layers = 2u;
    config.full_attention_interval = 4u;
    config.linear_num_key_heads = kGdnKeyHeads;
    config.linear_num_value_heads = kGdnValueHeads;
    config.linear_key_head_dim = kGdnHeadDim;
    config.linear_value_head_dim = kGdnHeadDim;
    config.linear_conv_kernel_dim = 4u;
    config.num_attention_heads = kQHeads;
    config.num_key_value_heads = kKvHeads;
    config.attention_head_dim = kHeadDim;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000000.0f;
    config.partial_rotary_factor = 0.25f;
    config.layer_types = {0, 1};
    config.hc_count = kHc;
    config.hc_lowrank = kLowrank;
    config.ffn.num_experts = kExperts;
    config.ffn.num_experts_per_tok = kTopK;
    config.ffn.moe_intermediate_size = kIntermediate;
    config.ffn.shared_expert_intermediate_size = kShared;
    config.ple.layer_ids = {2u};
    config.ple.embed_dim = kHidden;
    config.ple.ngram_size = 3u;
    config.ple.heads_per_ngram = 8u;
    config.ple.conv_kernel_size = 4u;
    config.indexer.n_heads = kIndexHeads;
    config.indexer.kv_heads = 1u;
    config.indexer.head_dim = kIndexDim;
    config.indexer.budget = 16u;
    config.indexer.compress_ratio = 4u;

    shelf::qwen4exp::Qwen4ExpRankWeights weights;
    weights.embed_tokens = matrix(kVocab, kHidden);
    weights.lm_head = matrix(kVocab, kHidden);
    weights.mixer = hyper();
    weights.layers.resize(config.num_hidden_layers);

    for (std::uint32_t li = 0; li < config.num_hidden_layers; ++li) {
        auto& layer = weights.layers[li];
        layer.attn_hyper_connection = hyper();
        layer.mlp_hyper_connection = hyper();
        layer.attention.is_gdn = li == 0u;
        if (layer.attention.is_gdn) {
            layer.attention.in_proj_qkv = matrix(kGdnConv, kHidden);
            layer.attention.in_proj_z = matrix(kGdnValue, kHidden);
            layer.attention.in_proj_b = matrix(kGdnValueHeads, kHidden);
            layer.attention.in_proj_a = matrix(kGdnValueHeads, kHidden);
            layer.attention.out_proj = matrix(kHidden, kGdnValue);
        } else {
            layer.attention.q_proj = matrix(2u * kQFeatures, kHidden);
            layer.attention.k_proj = matrix(kKvFeatures, kHidden);
            layer.attention.v_proj = matrix(kKvFeatures, kHidden);
            layer.attention.o_proj = matrix(kHidden, kQFeatures);
            layer.attention.index_qk_proj =
                matrix((kIndexHeads + 1u) * kIndexDim, kHidden);
        }
        layer.mlp.router = matrix(kExperts, kHidden);
        layer.mlp.shared_gate = matrix(1u, kHidden);
        layer.mlp.shared_gate_proj = matrix(kShared, kHidden);
        layer.mlp.shared_up_proj = matrix(kShared, kHidden);
        layer.mlp.shared_down_proj = matrix(kHidden, kShared);
        layer.mlp.gate_experts = experts(kIntermediate, kHidden);
        layer.mlp.up_experts = experts(kIntermediate, kHidden);
        layer.mlp.down_experts = experts(kHidden, kIntermediate);
        layer.mlp.experts = {kExperts, kExperts, 0u};
        if (config.layer_has_ple(li)) {
            layer.ple.present = true;
            layer.ple.key_proj = matrix(kWidth, kHidden);
            layer.ple.value_proj = matrix(kHidden, kHidden);
        }
    }

    shelf::qwen4exp::Qwen4ExpLowerOptions options;
    options.max_context_tokens = 64u;
    auto lowered = shelf::qwen4exp::lower_qwen4exp_to_primitives(config, weights, options);
    if (!lowered.ok()) {
        std::printf("FAIL lowering: %s\n", lowered.status().message().c_str());
        return 1;
    }

    const auto& graph = lowered.value().graph;
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
    std::printf("sequence head: %.320s\n", sequence.c_str());

    check(graph.external_inputs.size() == 2u,
          "token ids and PLE rows are the two external inputs");
    check(graph.external_outputs.size() == 4u, "head emits four external outputs");
    check(sequence.find("EMBEDDING_LOOKUP,REPEAT_STREAMS") != std::string::npos,
          "embedding is repeated into the hyper stack");
    check(sequence.find("PLE_STREAM_DOT") != std::string::npos &&
              sequence.find("PLE_CONV") != std::string::npos,
          "PLE injection is lowered");
    check(sequence.find("STATEFUL_CAUSAL_CONV1D") != std::string::npos && sequence.find("GDN_RECURRENCE") != std::string::npos,
          "GDN layer is lowered");
    check(sequence.find("QSA_KEY_CACHE,QSA_SCORES,QSA_TOPK,QSA_EXPAND_MASK") !=
              std::string::npos,
          "QSA indexer is lowered");
    check(sequence.find("PAGED_ATTENTION") != std::string::npos,
          "QSA attention is lowered");
    check(sequence.find("MOE_ROUTER,MOE_GROUPED_LINEAR,MOE_GROUPED_LINEAR,SWIGLU,"
                        "MOE_GROUPED_LINEAR") != std::string::npos,
          "every layer runs the MoE FFN");
    check(sequence.size() > 40u, "all layers are present");

    std::uint32_t combines = 0;
    for (const auto& node : graph.nodes)
        if (node.tp_combine != 0) ++combines;
    check(combines == config.num_hidden_layers,
          "each layer contributes exactly one TP reduction (the MoE combine)");

    bool grouped = false;
    for (const auto& node : graph.nodes) {
        if (rt::primitive_kind_of(node.node) != rt::PrimitiveKind::PAGED_ATTENTION)
            continue;
        grouped = std::get<rt::PagedAttentionNode>(node.node).has_token_mask;
    }
    check(grouped, "QSA attention consumes the expanded mask");

    {
        const std::uint32_t tap_layers[] = {0u};
        shelf::qwen4exp::Qwen4ExpLowerOptions tap_options = options;
        tap_options.hidden_taps = tap_layers;
        auto tapped = shelf::qwen4exp::lower_qwen4exp_to_primitives(config, weights,
                                                                    tap_options);
        check(tapped.ok(), "a hidden tap lowers");
        if (tapped.ok()) {
            const rt::PrimitiveGraph& tap_graph = tapped.value().graph;
            check(tap_graph.external_outputs.size() == 5u,
                  "one target feature is appended after the head outputs");
            std::string tap_sequence;
            for (const auto& node : tap_graph.nodes) {
                if (!tap_sequence.empty()) tap_sequence += ",";
                tap_sequence += rt::to_string(rt::primitive_kind_of(node.node));
            }
            check(tap_sequence.find("MEAN_STREAMS") != std::string::npos,
                  "the target feature is the mean over the hyper streams");
        }
    }

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
