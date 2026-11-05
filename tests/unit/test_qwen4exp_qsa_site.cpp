#include <shelf/models/common/qwen_hybrid/weight_slot.h>
#include <shelf/models/qwen4exp/model/qsa_lowering.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>

#include <cstdio>
#include <string>
#include <vector>

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

std::string join_kinds(const rt::PrimitiveGraph& graph) {
    std::string sequence;
    for (const auto& node : graph.nodes) {
        if (!sequence.empty()) sequence += ",";
        sequence += rt::to_string(rt::primitive_kind_of(node.node));
    }
    return sequence;
}

}  // namespace

int main() {
    constexpr std::uint32_t kHidden = 64u;
    constexpr std::uint32_t kIndexHeads = 4u;
    constexpr std::uint32_t kIndexDim = 128u;
    constexpr std::uint32_t kRotary = 64u;
    constexpr std::uint32_t kBlocks = 8u;
    constexpr std::uint32_t kTopK = 3u;
    constexpr std::uint32_t kMaskLen = 32u;

    shelf::qwen4exp::Qwen4ExpTextConfig config;
    config.hidden_size = kHidden;
    config.attention_head_dim = 256u;
    config.partial_rotary_factor = 0.25f;
    config.rope_theta = 10000000.0f;
    config.rms_norm_eps = 1e-6f;
    config.indexer.n_heads = kIndexHeads;
    config.indexer.kv_heads = 1u;
    config.indexer.head_dim = kIndexDim;
    config.indexer.budget = 8u;
    config.indexer.compress_ratio = 4u;
    const auto from_config = shelf::qwen4exp::make_qwen4exp_qsa_indexer_geometry(
        config, kBlocks, kMaskLen, 2u);
    check(from_config.hidden == kHidden && from_config.index_heads == kIndexHeads &&
              from_config.index_head_dim == kIndexDim &&
              from_config.index_rotary_dim == 64u &&
              from_config.compress_ratio == 4u &&
              from_config.blocks == kBlocks && from_config.top_k == 2u &&
              from_config.mask_len == kMaskLen && from_config.state_index == 2u,
          "config derives the QSA indexer geometry");

    rt::PrimitiveGraph graph;
    const rt::ValueId normed = graph.alloc_value(kHidden);
    graph.external_inputs = {normed};

    shelf::weights::MatrixWeight qk;
    qk.rows = (kIndexHeads + 1u) * kIndexDim;
    qk.cols = kHidden;
    qk.k_padded = kHidden;

    const std::uint32_t weight_index = static_cast<std::uint32_t>(graph.weight_table.size());
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{qk.rows, qk.cols});

    shelf::qwen_hybrid::Emitter em{graph};
    shelf::qwen4exp::Qwen4ExpQsaIndexerGeometry geometry;
    geometry.hidden = kHidden;
    geometry.index_heads = kIndexHeads;
    geometry.index_kv_heads = 1u;
    geometry.index_head_dim = kIndexDim;
    geometry.index_rotary_dim = kRotary;
    geometry.compress_ratio = 4u;
    geometry.blocks = kBlocks;
    geometry.top_k = kTopK;
    geometry.mask_len = kMaskLen;
    geometry.state_index = 2u;
    geometry.theta = 10000000.0f;
    geometry.position_scale = 1.0f;

    shelf::qwen4exp::Qwen4ExpQsaIndexerWeights weights;
    weights.qk_proj = &qk;
    shelf::qwen4exp::Qwen4ExpQsaIndexerIndices indices;
    indices.w_qk_proj = weight_index;
    indices.p_q_norm = 0u;
    indices.p_k_norm = 1u;

    auto emitted = shelf::qwen4exp::emit_qwen4exp_qsa_indexer(
        em, geometry, weights, indices, normed, "qsa_test.");
    if (!emitted.ok()) {
        std::printf("FAIL emit: %s\n", emitted.status().message().c_str());
        return 1;
    }
    graph.external_outputs = {emitted.value().mask};

    const auto status = rt::validate_primitive_graph(graph);
    if (!status.ok()) {
        std::printf("FAIL validate: %s\n", status.message().c_str());
        return 1;
    }

    const std::string sequence = join_kinds(graph);
    check(sequence.find("LINEAR,RMS_NORM,QSA_ROPE,QSA_KEY_CACHE,QSA_SCORES,QSA_TOPK,"
                        "QSA_EXPAND_MASK") != std::string::npos,
          "QSA indexer lowers to key cache, scores, topk and mask");
    check(graph.states.size() == 1u &&
              graph.states[0].kind == rt::PrimitiveStateKind::QSA_KEY_STATE &&
              graph.states[0].state_index == 2u,
          "QSA indexer registers one key state");

    rt::WeightSlot slot = shelf::qwen_hybrid::make_weight_slot(qk);
    const rt::WeightSlot slots[1] = {slot};
    rt::WeightTableView weights_view{slots, 1u};
    rt::StaticParameterSlot parameters[3] = {};
    rt::StaticParameterTableView parameter_view{parameters, 3u};

    rt::ProgramBuildOptions options;
    options.max_token_rows = 16u;
    options.max_output_rows = 16u;
    auto program = rt::build_program(graph, weights_view, rt::RowBucket::R16,
                                      rt::ExecutionClass::DECODE, options);
    if (!program.ok()) {
        std::printf("FAIL program: %s\n", program.status().message().c_str());
        return 1;
    }

    std::string kernels;
    for (const auto& dispatch : program.value().dispatches) {
        if (!kernels.empty()) kernels += ",";
        kernels += rt::kernel_to_name(dispatch.kernel_id);
    }
    std::printf("kernels: %s\n", kernels.c_str());
    check(kernels.find("QSA_ROPE,QSA_KEY_CACHE,QSA_SCORES,QSA_TOPK,QSA_EXPAND_MASK") !=
              std::string::npos,
          "program emits the QSA indexer dispatch sequence");
    check(program.value().states.size() == 1u &&
              program.value().states[0].kind == rt::PrimitiveStateKind::QSA_KEY_STATE &&
              program.value().states[0].state_index == 2u,
          "program binds the QSA key state");
    check(program.value().dispatches.size() >= 3u,
          "program has the q projection and indexer dispatches");

    bool qsa_descriptors_match = true;
    uint32_t qsa_states = 0u;
    for (const auto& dispatch : program.value().dispatches) {
        if (dispatch.kernel_id == rt::KernelId::QSA_KEY_CACHE ||
            dispatch.kernel_id == rt::KernelId::QSA_SCORES) {
            ++qsa_states;
            if (dispatch.descriptor_slot == 0u) qsa_descriptors_match = false;
        }
    }
    check(qsa_states == 2u && qsa_descriptors_match,
          "key cache and scores share the registered state descriptor");

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
