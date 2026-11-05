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

    {
        constexpr std::uint32_t kQHeads = 2u;
        constexpr std::uint32_t kKvHeads = 1u;
        constexpr std::uint32_t kAttnHeadDim = 16u;
        constexpr std::uint32_t kAttnRotary = 8u;
        constexpr std::uint32_t kQFeatures = kQHeads * kAttnHeadDim;
        constexpr std::uint32_t kKvFeatures = kKvHeads * kAttnHeadDim;

        rt::PrimitiveGraph qsa_graph;
        const rt::ValueId qsa_normed = qsa_graph.alloc_value(kHidden);
        qsa_graph.external_inputs = {qsa_normed};

        shelf::weights::MatrixWeight q_proj;
        q_proj.rows = 2u * kQFeatures;
        q_proj.cols = kHidden;
        q_proj.k_padded = kHidden;
        shelf::weights::MatrixWeight k_proj;
        k_proj.rows = kKvFeatures;
        k_proj.cols = kHidden;
        k_proj.k_padded = kHidden;
        shelf::weights::MatrixWeight v_proj = k_proj;
        shelf::weights::MatrixWeight o_proj;
        o_proj.rows = kHidden;
        o_proj.cols = kQFeatures;
        o_proj.k_padded = kQFeatures;

        const std::uint32_t base_weight = static_cast<std::uint32_t>(
            qsa_graph.weight_table.size());
        for (const auto* w : {&q_proj, &k_proj, &v_proj, &o_proj, &qk}) {
            qsa_graph.weight_table.push_back(rt::MatrixwiseShapeKey{w->rows, w->cols});
        }

        shelf::qwen_hybrid::Emitter qsa_em{qsa_graph};
        shelf::qwen4exp::Qwen4ExpQsaLoweringGeometry qsa_geometry;
        qsa_geometry.attention.query_heads = kQHeads;
        qsa_geometry.attention.kv_heads = kKvHeads;
        qsa_geometry.attention.head_dim = kAttnHeadDim;
        qsa_geometry.attention.rotary_dim = kAttnRotary;
        qsa_geometry.attention.rope_theta = 10000000.0f;
        qsa_geometry.attention.eps = 1e-6f;
        qsa_geometry.indexer = geometry;

        shelf::qwen4exp::Qwen4ExpQsaLoweringWeights qsa_weights;
        qsa_weights.attention.q = &q_proj;
        qsa_weights.attention.k = &k_proj;
        qsa_weights.attention.v = &v_proj;
        qsa_weights.attention.o = &o_proj;
        qsa_weights.indexer.qk_proj = &qk;

        shelf::qwen4exp::Qwen4ExpQsaLoweringIndices qsa_indices;
        qsa_indices.attention.w_q = base_weight;
        qsa_indices.attention.w_k = base_weight + 1u;
        qsa_indices.attention.w_v = base_weight + 2u;
        qsa_indices.attention.w_o = base_weight + 3u;
        qsa_indices.attention.p_q_norm = 0u;
        qsa_indices.attention.p_k_norm = 1u;
        qsa_indices.attention.state_index = 0u;
        qsa_indices.indexer.w_qk_proj = base_weight + 4u;
        qsa_indices.indexer.p_q_norm = 2u;
        qsa_indices.indexer.p_k_norm = 3u;

        auto qsa_out = shelf::qwen4exp::emit_qwen4exp_qsa(
            qsa_em, qsa_geometry, qsa_weights, qsa_indices, qsa_normed, "qsa_layer.");
        if (!qsa_out.ok()) {
            std::printf("FAIL qsa site emit: %s\n",
                        qsa_out.status().message().c_str());
            return 1;
        }
        qsa_graph.external_outputs = {qsa_out.value()};

        const auto qsa_status = rt::validate_primitive_graph(qsa_graph);
        if (!qsa_status.ok()) {
            std::printf("FAIL qsa site validate: %s\n",
                        qsa_status.message().c_str());
            return 1;
        }

        bool mask_consumed = false;
        bool mask_after_indexer = false;
        bool indexer_seen = false;
        for (const auto& node : qsa_graph.nodes) {
            const auto kind = rt::primitive_kind_of(node.node);
            if (kind == rt::PrimitiveKind::QSA_EXPAND_MASK) indexer_seen = true;
            if (kind == rt::PrimitiveKind::PAGED_ATTENTION) {
                const auto& pa = std::get<rt::PagedAttentionNode>(node.node);
                mask_consumed = pa.has_token_mask && node.inputs.size() == 2u;
                mask_after_indexer = indexer_seen;
            }
        }
        check(mask_consumed, "QSA mask is the second paged-attention input");
        check(mask_after_indexer, "QSA mask is produced before the attention");

        const rt::WeightSlot qsa_slots[5] = {
            shelf::qwen_hybrid::make_weight_slot(q_proj),
            shelf::qwen_hybrid::make_weight_slot(k_proj),
            shelf::qwen_hybrid::make_weight_slot(v_proj),
            shelf::qwen_hybrid::make_weight_slot(o_proj),
            shelf::qwen_hybrid::make_weight_slot(qk),
        };
        rt::WeightTableView qsa_weights_view{qsa_slots, 5u};
        rt::StaticParameterSlot qsa_parameters[4] = {};
        rt::StaticParameterTableView qsa_parameter_view{qsa_parameters, 4u};
        auto qsa_program = rt::build_program(qsa_graph, qsa_weights_view,
                                             rt::RowBucket::R16,
                                             rt::ExecutionClass::DECODE, options);
        if (!qsa_program.ok()) {
            std::printf("FAIL qsa program: %s\n",
                        qsa_program.status().message().c_str());
            return 1;
        }
        bool dispatch_mask = false;
        bool kv_state = false;
        bool indexer_state = false;
        for (const auto& dispatch : qsa_program.value().dispatches) {
            if (dispatch.kernel_id == rt::KernelId::PAGED_ATTENTION)
                dispatch_mask = dispatch.input_count == 2u;
        }
        for (const auto& state : qsa_program.value().states) {
            if (state.kind == rt::PrimitiveStateKind::KV_CACHE) kv_state = true;
            if (state.kind == rt::PrimitiveStateKind::QSA_KEY_STATE)
                indexer_state = true;
        }
        check(dispatch_mask, "paged-attention dispatch carries the QSA mask");
        check(kv_state && indexer_state,
              "QSA layer registers both KV and indexer states");
    }

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
