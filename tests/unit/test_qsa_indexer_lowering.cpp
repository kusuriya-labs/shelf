#include <shelf/models/common/qwen_hybrid/qsa_indexer_lowering.h>
#include <shelf/runtime/graph/primitive_graph.h>

#include <cstdio>
#include <string>

namespace rt = shelf::runtime;
namespace qh = shelf::qwen_hybrid;

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

const rt::QsaKeyCacheNode* key_cache_node(const rt::PrimitiveGraph& graph) {
    for (const auto& node : graph.nodes) {
        if (rt::primitive_kind_of(node.node) == rt::PrimitiveKind::QSA_KEY_CACHE)
            return &std::get<rt::QsaKeyCacheNode>(node.node);
    }
    return nullptr;
}

constexpr std::uint32_t kHidden = 64u;
constexpr std::uint32_t kHeads = 4u;
constexpr std::uint32_t kHeadDim = 128u;
constexpr std::uint32_t kRotary = 64u;
constexpr std::uint32_t kBlocks = 8u;
constexpr std::uint32_t kTopK = 2u;
constexpr std::uint32_t kMaskBytes = 4u;
constexpr std::uint32_t kCompress = 4u;

qh::QsaIndexerGeometry geometry(bool pre_pool_key_norm) {
    qh::QsaIndexerGeometry out;
    out.hidden = kHidden;
    out.heads = kHeads;
    out.head_dim = kHeadDim;
    out.rotary_dim = kRotary;
    out.compress_ratio = kCompress;
    out.blocks = kBlocks;
    out.top_k = kTopK;
    out.mask_bytes = kMaskBytes;
    out.state_index = 1u;
    out.theta = 10000000.0f;
    out.pre_pool_key_norm = pre_pool_key_norm;
    return out;
}

shelf::weights::MatrixWeight matrix(std::uint32_t rows, std::uint32_t cols) {
    shelf::weights::MatrixWeight w;
    w.rows = rows;
    w.cols = cols;
    w.k_padded = cols;
    return w;
}

void test_fused() {
    rt::PrimitiveGraph graph;
    const rt::ValueId normed = graph.alloc_value(kHidden);
    graph.external_inputs = {normed};

    shelf::weights::MatrixWeight qk = matrix((kHeads + 1u) * kHeadDim, kHidden);
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{qk.rows, qk.cols});

    qh::Emitter em{graph};
    qh::QsaIndexerWeights weights;
    weights.qk_proj = &qk;
    qh::QsaIndexerIndices indices;
    indices.w_qk_proj = 0u;
    indices.p_q_norm = 0u;
    indices.p_k_norm = 1u;

    auto emitted = qh::emit_qsa_indexer(em, geometry(false), weights, indices, normed,
                                        "fused.");
    check(emitted.ok(), "fused indexer emits");
    if (!emitted.ok()) return;
    graph.external_outputs = {emitted.value().mask};

    const auto status = rt::validate_primitive_graph(graph);
    check(status.ok(), "fused indexer graph validates");
    check(join_kinds(graph) == "LINEAR,RMS_NORM,QSA_ROPE,QSA_KEY_CACHE,QSA_SELECT",
          "fused indexer uses one projection");
    const auto* cache = key_cache_node(graph);
    check(cache != nullptr && cache->parameter_index == 1u,
          "fused indexer normalizes the pooled block key");
    check(graph.states.size() == 1u &&
              graph.states[0].kind == rt::PrimitiveStateKind::QSA_KEY_STATE,
          "fused indexer registers one key state");
}

void test_separate_pre_pool_norm() {
    rt::PrimitiveGraph graph;
    const rt::ValueId normed = graph.alloc_value(kHidden);
    graph.external_inputs = {normed};

    shelf::weights::MatrixWeight q_proj = matrix(kHeads * kHeadDim, kHidden);
    shelf::weights::MatrixWeight k_proj = matrix(kHeadDim, kHidden);
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{q_proj.rows, q_proj.cols});
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{k_proj.rows, k_proj.cols});

    qh::Emitter em{graph};
    qh::QsaIndexerWeights weights;
    weights.q_proj = &q_proj;
    weights.k_proj = &k_proj;
    qh::QsaIndexerIndices indices;
    indices.w_q_proj = 0u;
    indices.w_k_proj = 1u;
    indices.p_q_norm = 0u;
    indices.p_k_norm = 1u;
    indices.p_k_block_norm = 2u;

    auto emitted = qh::emit_qsa_indexer(em, geometry(true), weights, indices, normed,
                                        "separate.");
    check(emitted.ok(), "separate indexer emits");
    if (!emitted.ok()) return;
    graph.external_outputs = {emitted.value().mask};

    const auto status = rt::validate_primitive_graph(graph);
    check(status.ok(), "separate indexer graph validates");
    check(join_kinds(graph) ==
              "LINEAR,LINEAR,RMS_NORM,QSA_ROPE,RMS_NORM,QSA_KEY_CACHE,QSA_SELECT",
          "separate indexer normalizes the raw key before the cache");
    const auto* cache = key_cache_node(graph);
    check(cache != nullptr && cache->parameter_index == 2u,
          "separate indexer normalizes the pooled block key with its own parameter");
}

void test_rejects_invalid_geometry() {
    rt::PrimitiveGraph graph;
    const rt::ValueId normed = graph.alloc_value(kHidden);
    graph.external_inputs = {normed};

    shelf::weights::MatrixWeight qk = matrix((kHeads + 1u) * kHeadDim, kHidden);
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{qk.rows, qk.cols});

    qh::Emitter em{graph};
    qh::QsaIndexerWeights weights;
    weights.qk_proj = &qk;
    qh::QsaIndexerIndices indices;

    qh::QsaIndexerGeometry bad_dim = geometry(false);
    bad_dim.head_dim = 64u;
    check(!qh::emit_qsa_indexer(em, bad_dim, weights, indices, normed, "bad_dim.").ok(),
          "indexer rejects a head width the select kernel cannot use");

    qh::QsaIndexerGeometry bad_heads = geometry(false);
    bad_heads.heads = 9u;
    check(!qh::emit_qsa_indexer(em, bad_heads, weights, indices, normed, "bad_heads.").ok(),
          "indexer rejects more heads than the select kernel supports");

    qh::QsaIndexerWeights bad_weights;
    shelf::weights::MatrixWeight wrong = matrix((kHeads + 2u) * kHeadDim, kHidden);
    bad_weights.qk_proj = &wrong;
    check(!qh::emit_qsa_indexer(em, geometry(false), bad_weights, indices, normed,
                                "bad_weights.").ok(),
          "indexer rejects a qk projection of the wrong shape");
}

}  // namespace

int main() {
    test_fused();
    test_separate_pre_pool_norm();
    test_rejects_invalid_geometry();

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
