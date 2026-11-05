#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/graph/shape_spec.h>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace shelf::runtime {
namespace {

struct Arity {
    size_t tensor_inputs;
    size_t tensor_outputs;
};

constexpr Arity arity_of(PrimitiveKind k) noexcept {
    switch (k) {
        case PrimitiveKind::EMBEDDING_LOOKUP: return {1, 1};
        case PrimitiveKind::LINEAR: return {1, 1};
        case PrimitiveKind::RMS_NORM: return {1, 1};
        case PrimitiveKind::RESIDUAL_ADD: return {2, 1};
        case PrimitiveKind::SPLIT: return {1, 2};
        case PrimitiveKind::SILU: return {1, 1};
        case PrimitiveKind::SIGMOID: return {1, 1};
        case PrimitiveKind::MUL: return {2, 1};
        case PrimitiveKind::SWIGLU: return {2, 1};
        case PrimitiveKind::ROPE: return {1, 1};
        case PrimitiveKind::L2_NORMALIZE: return {1, 1};
        case PrimitiveKind::SCALE: return {1, 1};
        case PrimitiveKind::KV_APPEND: return {2, 0};
        case PrimitiveKind::PAGED_ATTENTION: return {1, 1};
        case PrimitiveKind::STATEFUL_CAUSAL_CONV1D: return {1, 1};
        case PrimitiveKind::GDN_RECURRENCE: return {5, 1};
        case PrimitiveKind::OUTPUT_GATHER: return {1, 1};
        case PrimitiveKind::SAMPLING: return {1, 1};
        case PrimitiveKind::CONCAT: return {2, 1};
        case PrimitiveKind::MOE_ROUTER: return {1, 0};
        case PrimitiveKind::MOE_GROUPED_LINEAR: return {1, 1};
        case PrimitiveKind::MOE_COMBINE: return {3, 1};
        case PrimitiveKind::HYPER_MIX: return {2, 1};
        case PrimitiveKind::HYPER_APPLY: return {3, 1};
        case PrimitiveKind::PLE_GATE: return {1, 1};
        case PrimitiveKind::PLE_BROADCAST_MUL: return {2, 1};
        case PrimitiveKind::PLE_CONV: return {1, 1};
        case PrimitiveKind::QSA_KEY_CACHE: return {1, 0};
        case PrimitiveKind::QSA_SCORES: return {1, 1};
        case PrimitiveKind::QSA_TOPK: return {1, 1};
        case PrimitiveKind::QSA_EXPAND_MASK: return {1, 1};
        case PrimitiveKind::QSA_ROPE: return {1, 1};
        case PrimitiveKind::PLE_STREAM_DOT: return {2, 1};
        case PrimitiveKind::REPEAT_STREAMS: return {1, 1};
        case PrimitiveKind::HYPER_INJECT: return {1, 1};
    }
    return {0, 0};
}

Status fail(const char* msg) {
    return Status::invalid_argument(msg, __FILE__, __LINE__);
}

bool allowed_state_kind(PrimitiveStateKind sk, PrimitiveKind nk) noexcept {
    switch (sk) {
        case PrimitiveStateKind::KV_CACHE:
            return nk == PrimitiveKind::KV_APPEND || nk == PrimitiveKind::PAGED_ATTENTION;
        case PrimitiveStateKind::GDN_CONV_STATE:
            return nk == PrimitiveKind::STATEFUL_CAUSAL_CONV1D;
        case PrimitiveStateKind::GDN_RECURRENCE_STATE:
            return nk == PrimitiveKind::GDN_RECURRENCE;
        case PrimitiveStateKind::PLE_CONV_STATE:
            return nk == PrimitiveKind::PLE_CONV;
        case PrimitiveStateKind::QSA_KEY_STATE:
            return nk == PrimitiveKind::QSA_KEY_CACHE || nk == PrimitiveKind::QSA_SCORES;
    }
    return false;
}

uint32_t node_state_index(const PrimitiveGraphNode& n) noexcept {
    return std::visit([](auto&& v) -> uint32_t {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, KvAppendNode>) return v.state_index;
        else if constexpr (std::is_same_v<T, PagedAttentionNode>) return v.state_index;
        else if constexpr (std::is_same_v<T, StatefulCausalConv1DNode>) return v.state_index;
        else if constexpr (std::is_same_v<T, GdnRecurrenceNode>) return v.state_index;
        else if constexpr (std::is_same_v<T, PleConvNode>) return v.state_index;
        else if constexpr (std::is_same_v<T, QsaKeyCacheNode>) return v.state_index;
        else if constexpr (std::is_same_v<T, QsaScoresNode>) return v.state_index;
        else return 0;
    }, n.node);
}

}

Status validate_primitive_graph(const PrimitiveGraph& graph) noexcept {
    if (graph.next_value_id != graph.value_specs.size()) {
        return fail("value id counters desynchronized");
    }

    const bool grouped = graph.grouped_row_shape.experts != 0u;
    for (const auto& spec : graph.value_specs) {
        if (spec.row_domain != ValueRowDomain::GROUPED_ROWS) continue;
        if (!grouped) return fail("grouped row value without a declared grouped row shape");
        if (spec.features == 0u) return fail("grouped row value has no features");
    }
    if (grouped && graph.grouped_row_shape.top_k == 0u) {
        return fail("grouped row shape has an empty selection width");
    }

    std::unordered_map<uint64_t, uint32_t> state_slot_owner;
    for (uint32_t si = 0; si < graph.states.size(); ++si) {
        auto& s = graph.states[si];
        if (s.id.id != si) return fail("state ids are not dense from zero");
        uint64_t slot = (static_cast<uint64_t>(static_cast<uint8_t>(s.kind)) << 32) |
                        s.state_index;
        bool inserted = state_slot_owner.emplace(slot, si).second;
        if (!inserted) return fail("duplicate state registration for same kind and layer");
    }

    std::unordered_map<uint32_t, const PrimitiveValueView*> view_by_value;
    for (auto& v : graph.views) {
        if (v.value.id >= graph.next_value_id || v.base.id >= graph.next_value_id) {
            return fail("view value or base id out of range");
        }
        if (v.value.id == v.base.id) return fail("view aliases its own base");
        if (view_by_value.count(v.value.id)) return fail("duplicate view definition");
        if (v.feature_count == 0) return fail("view with empty feature range");
        uint32_t base_features = graph.value_features(v.base);
        if (base_features == 0) return fail("view base has unknown feature count");
        if (v.feature_offset > base_features ||
            static_cast<uint64_t>(v.feature_offset) + v.feature_count > base_features) {
            return fail("view feature range exceeds base shape");
        }
        view_by_value[v.value.id] = &v;
    }

    constexpr int64_t unproduced = -1;
    std::vector<int64_t> producer(static_cast<size_t>(graph.next_value_id), unproduced);
    for (size_t ni = 0; ni < graph.nodes.size(); ++ni) {
        for (auto& vo : graph.nodes[ni].outputs) {
            uint32_t oid = vo.id;
            if (oid >= graph.next_value_id) return fail("node output id out of range");
            if (producer[oid] != unproduced) return fail("duplicate value producer");
            if (view_by_value.count(oid)) return fail("value used both as node output and view value");
            producer[oid] = static_cast<int64_t>(ni);
        }
    }

    for (size_t ni = 0; ni < graph.nodes.size(); ++ni) {
        auto& n = graph.nodes[ni];
        auto k = primitive_kind_of(n.node);
        bool shapes_ok = true;
        std::visit([&shapes_ok](auto&& v){
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, LinearNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, RmsNormNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, ResidualAddNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, SplitNode>) shapes_ok = valid_shape_key(v.input_shape) && valid_shape_key(v.output_shape_a) && valid_shape_key(v.output_shape_b);
            else if constexpr (std::is_same_v<T, SiLUNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, SigmoidNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, MulNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, SwiGluNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, RoPENode>) shapes_ok = valid_shape_key(v.attention);
            else if constexpr (std::is_same_v<T, L2NormalizeNode>) shapes_ok = valid_shape_key(v.shape) && valid_shape_key(v.gdn);
            else if constexpr (std::is_same_v<T, ScaleNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, KvAppendNode>) shapes_ok = valid_shape_key(v.attention);
            else if constexpr (std::is_same_v<T, PagedAttentionNode>) {
            shapes_ok = valid_shape_key(v.attention) &&
                        (!v.has_token_mask ||
                         (v.mask_features != 0u && v.mask_features <= (1u << 20)));
        }
            else if constexpr (std::is_same_v<T, StatefulCausalConv1DNode>) shapes_ok = valid_shape_key(v.gdn);
            else if constexpr (std::is_same_v<T, GdnRecurrenceNode>) shapes_ok = valid_shape_key(v.gdn);
            else if constexpr (std::is_same_v<T, EmbeddingLookupNode>) shapes_ok = v.vocab_size != 0 && v.hidden_size != 0 && valid_shape_key(v.weight_shape);
            else if constexpr (std::is_same_v<T, OutputGatherNode>) shapes_ok = valid_shape_key(v.shape);
            else if constexpr (std::is_same_v<T, SamplingNode>) shapes_ok = v.vocab_size != 0;
            else if constexpr (std::is_same_v<T, ConcatNode>) {
                shapes_ok = valid_shape_key(v.shape) && valid_shape_key(v.input_shape_a) &&
                            valid_shape_key(v.input_shape_b);
            }
            else if constexpr (std::is_same_v<T, MoeRouterNode>) {
                shapes_ok = v.experts != 0u && v.top_k != 0u && v.top_k <= v.experts;
            }
            else if constexpr (std::is_same_v<T, MoeGroupedLinearNode>) {
                shapes_ok = valid_shape_key(v.shape);
            }
            else if constexpr (std::is_same_v<T, MoeCombineNode>) {
                shapes_ok = valid_shape_key(v.shape) && v.top_k != 0u;
            }
            else if constexpr (std::is_same_v<T, HyperMixNode>) {
                shapes_ok = valid_shape_key(v.shape) && v.hc != 0u && v.hidden != 0u &&
                            v.shape.features == v.hidden;
            }
            else if constexpr (std::is_same_v<T, HyperApplyNode>) {
                shapes_ok = valid_shape_key(v.shape) && v.hc != 0u && v.hidden != 0u &&
                            v.shape.features == static_cast<uint64_t>(v.hc) * v.hidden;
            }
            else if constexpr (std::is_same_v<T, PleGateNode>) {
                shapes_ok = valid_shape_key(v.shape);
            }
            else if constexpr (std::is_same_v<T, PleBroadcastMulNode>) {
                shapes_ok = valid_shape_key(v.shape) && v.hc != 0u && v.hidden != 0u &&
                            v.shape.features == static_cast<uint64_t>(v.hc) * v.hidden;
            }
            else if constexpr (std::is_same_v<T, PleConvNode>) {
                shapes_ok = valid_shape_key(v.shape);
            }
        }, n.node);
        if (!shapes_ok) return fail("invalid shape key");

        auto arity = arity_of(k);
        size_t expected_inputs = arity.tensor_inputs;
        if (k == PrimitiveKind::PAGED_ATTENTION &&
            std::get<PagedAttentionNode>(n.node).has_token_mask) {
            expected_inputs = 2u;
        }
        if (n.inputs.size() != expected_inputs || n.outputs.size() != arity.tensor_outputs) {
            return fail("primitive tensor arity mismatch");
        }

        Status weight_status = Status::make_ok();
        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, LinearNode>) {
                if (v.weight_index >= graph.weight_table.size()) {
                    weight_status = Status::invalid_argument("linear weight index out of bounds", __FILE__, __LINE__);
                } else if (!(graph.weight_table[v.weight_index] == v.shape)) {
                    weight_status = Status::invalid_argument("linear weight geometry mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, MoeGroupedLinearNode>) {
                if (v.weight_index >= graph.weight_table.size()) {
                    weight_status = Status::invalid_argument(
                        "moe grouped linear weight index out of bounds", __FILE__, __LINE__);
                } else if (!(graph.weight_table[v.weight_index] == v.shape)) {
                    weight_status = Status::invalid_argument(
                        "moe grouped linear weight geometry mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, EmbeddingLookupNode>) {
                if (v.weight_index >= graph.weight_table.size()) {
                    weight_status = Status::invalid_argument("embedding weight index out of bounds", __FILE__, __LINE__);
                } else if (!(graph.weight_table[v.weight_index] == v.weight_shape)) {
                    weight_status = Status::invalid_argument("embedding weight geometry mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, SplitNode>) {
                if (v.input_shape.features !=
                    static_cast<uint64_t>(v.output_shape_a.features) + v.output_shape_b.features) {
                    weight_status = Status::invalid_argument("split feature sizes mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, ConcatNode>) {
                if (v.shape.features !=
                    static_cast<uint64_t>(v.input_shape_a.features) + v.input_shape_b.features) {
                    weight_status = Status::invalid_argument("concat feature sizes mismatch", __FILE__, __LINE__);
                }
            }
        }, n.node);
        if (!weight_status.ok()) return weight_status;

        // A MoE node states its geometry and the graph states the row space it belongs to.
        // They have to agree, and the router's logits must be as wide as the expert count.
        Status moe_status = Status::make_ok();
        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            const GroupedRowShape plan = graph.grouped_row_shape;
            if constexpr (std::is_same_v<T, MoeRouterNode>) {
                if (plan.experts == 0u) {
                    moe_status = Status::invalid_argument(
                        "moe router without a grouped row shape", __FILE__, __LINE__);
                } else if (v.owned_experts() != plan.experts || v.top_k != plan.top_k) {
                    moe_status = Status::invalid_argument(
                        "moe router geometry differs from the grouped row shape", __FILE__,
                        __LINE__);
                } else if (v.local_experts != 0u &&
                           v.expert_begin + v.local_experts > v.experts) {
                    moe_status = Status::invalid_argument(
                        "moe router owns experts outside its logits", __FILE__, __LINE__);
                } else if (graph.value_features(n.inputs[0]) != v.experts) {
                    moe_status = Status::invalid_argument(
                        "moe router logits are not one column per expert", __FILE__, __LINE__);
                } else if (graph.value_dtype(n.inputs[0]) != ValueDType::F32) {
                    moe_status = Status::invalid_argument(
                        "moe router logits must be f32", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, MoeGroupedLinearNode>) {
                if (plan.experts == 0u) {
                    moe_status = Status::invalid_argument(
                        "moe grouped linear without a grouped row shape", __FILE__, __LINE__);
                } else if (graph.value_row_domain(n.outputs[0]) != ValueRowDomain::GROUPED_ROWS) {
                    moe_status = Status::invalid_argument(
                        "moe grouped linear output is not in group rows", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, MoeCombineNode>) {
                if (plan.experts == 0u) {
                    moe_status = Status::invalid_argument(
                        "moe combine without a grouped row shape", __FILE__, __LINE__);
                } else if (v.top_k != plan.top_k) {
                    moe_status = Status::invalid_argument(
                        "moe combine selection width differs from the grouped row shape",
                        __FILE__, __LINE__);
                } else if (graph.value_row_domain(n.inputs[0]) != ValueRowDomain::GROUPED_ROWS) {
                    moe_status = Status::invalid_argument(
                        "moe combine expert output is not in group rows", __FILE__, __LINE__);
                } else if (graph.value_row_domain(n.outputs[0]) != ValueRowDomain::TOKEN_ROWS) {
                    moe_status = Status::invalid_argument(
                        "moe combine output is not one row per token", __FILE__, __LINE__);
                } else if (graph.value_features(n.inputs[1]) != v.shape.features ||
                           graph.value_features(n.outputs[0]) != v.shape.features) {
                    moe_status = Status::invalid_argument(
                        "moe combine feature count mismatch", __FILE__, __LINE__);
                }
            }
        }, n.node);
        if (!moe_status.ok()) return moe_status;

        Status hyper_status = Status::make_ok();
        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, HyperMixNode>) {
                const uint64_t width = static_cast<uint64_t>(v.hc) * v.hidden;
                if (graph.value_features(n.inputs[0]) != width ||
                    graph.value_features(n.inputs[1]) != width) {
                    hyper_status = Status::invalid_argument(
                        "hyper mix inputs are not hc * hidden", __FILE__, __LINE__);
                } else if (graph.value_row_domain(n.inputs[0]) !=
                               graph.value_row_domain(n.inputs[1]) ||
                           graph.value_row_domain(n.inputs[0]) !=
                               graph.value_row_domain(n.outputs[0])) {
                    hyper_status = Status::invalid_argument(
                        "hyper mix row domains disagree", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, HyperInjectNode>) {
                if (graph.value_features(n.inputs[0]) != v.shape.features ||
                    graph.value_features(n.outputs[0]) != v.shape.features) {
                    hyper_status = Status::invalid_argument(
                        "hyper inject feature count mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, HyperApplyNode>) {
                const uint64_t width = static_cast<uint64_t>(v.hc) * v.hidden;
                if (graph.value_features(n.inputs[0]) != width ||
                    graph.value_features(n.inputs[1]) != v.hidden ||
                    graph.value_features(n.inputs[2]) != v.hc) {
                    hyper_status = Status::invalid_argument(
                        "hyper apply input feature counts are invalid", __FILE__, __LINE__);
                } else if (graph.value_row_domain(n.inputs[0]) !=
                               graph.value_row_domain(n.inputs[1]) ||
                           graph.value_row_domain(n.inputs[0]) !=
                               graph.value_row_domain(n.inputs[2]) ||
                           graph.value_row_domain(n.inputs[0]) !=
                               graph.value_row_domain(n.outputs[0])) {
                    hyper_status = Status::invalid_argument(
                        "hyper apply row domains disagree", __FILE__, __LINE__);
                }
            }
        }, n.node);
        if (!hyper_status.ok()) return hyper_status;

        Status ple_status = Status::make_ok();
        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, PleGateNode>) {
                if (graph.value_features(n.inputs[0]) != v.shape.features ||
                    graph.value_features(n.outputs[0]) != v.shape.features) {
                    ple_status = Status::invalid_argument(
                        "ple gate feature count mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, PleBroadcastMulNode>) {
                const uint64_t hidden = v.hidden;
                if (graph.value_features(n.inputs[0]) != v.hc ||
                    graph.value_features(n.inputs[1]) != hidden ||
                    graph.value_features(n.outputs[0]) !=
                        static_cast<uint64_t>(v.hc) * hidden) {
                    ple_status = Status::invalid_argument(
                        "ple broadcast feature count mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, PleConvNode>) {
                if (graph.value_features(n.inputs[0]) != v.shape.features ||
                    graph.value_features(n.outputs[0]) != v.shape.features) {
                    ple_status = Status::invalid_argument(
                        "ple conv feature count mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, RepeatStreamsNode>) {
                if (v.hc == 0u || v.hidden == 0u ||
                    v.shape.features != static_cast<uint64_t>(v.hc) * v.hidden ||
                    graph.value_features(n.inputs[0]) != v.hidden ||
                    graph.value_features(n.outputs[0]) != v.shape.features) {
                    ple_status = Status::invalid_argument(
                        "repeat streams shape mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, PleStreamDotNode>) {
                if (v.hc == 0u || v.hidden == 0u || v.shape.features != v.hc ||
                    graph.value_features(n.inputs[0]) !=
                        static_cast<uint64_t>(v.hc) * v.hidden ||
                    graph.value_features(n.inputs[1]) !=
                        static_cast<uint64_t>(v.hc) * v.hidden ||
                    graph.value_features(n.outputs[0]) != v.hc) {
                    ple_status = Status::invalid_argument(
                        "ple stream dot shape mismatch", __FILE__, __LINE__);
                }
            }
        }, n.node);
        if (!ple_status.ok()) return ple_status;

        Status qsa_status = Status::make_ok();
        std::visit([&](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, QsaKeyCacheNode>) {
                if (graph.value_features(n.inputs[0]) != v.shape.features ||
                    v.shape.features != 128u) {
                    qsa_status = Status::invalid_argument(
                        "qsa key cache feature count mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, QsaScoresNode>) {
                if (graph.value_features(n.inputs[0]) !=
                        static_cast<uint64_t>(v.n_heads) * v.head_dim ||
                    graph.value_features(n.outputs[0]) != v.blocks) {
                    qsa_status = Status::invalid_argument(
                        "qsa scores shape mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, QsaTopkNode>) {
                if (graph.value_features(n.inputs[0]) != v.blocks ||
                    graph.value_features(n.outputs[0]) != v.k) {
                    qsa_status = Status::invalid_argument(
                        "qsa topk shape mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, QsaRopeNode>) {
                if (graph.value_features(n.inputs[0]) != v.shape.features ||
                    graph.value_features(n.outputs[0]) != v.shape.features ||
                    v.shape.features !=
                        static_cast<uint64_t>(v.heads) * v.head_dim) {
                    qsa_status = Status::invalid_argument(
                        "qsa rope shape mismatch", __FILE__, __LINE__);
                }
            } else if constexpr (std::is_same_v<T, QsaExpandMaskNode>) {
                if (graph.value_features(n.inputs[0]) != v.shape.features ||
                    graph.value_features(n.outputs[0]) != v.total_len ||
                    graph.value_dtype(n.outputs[0]) != ValueDType::U8) {
                    qsa_status = Status::invalid_argument(
                        "qsa expand mask shape mismatch", __FILE__, __LINE__);
                }
            }
        }, n.node);
        if (!qsa_status.ok()) return qsa_status;

        if (k == PrimitiveKind::PAGED_ATTENTION) {
            const auto& pa = std::get<PagedAttentionNode>(n.node);
            if (pa.has_token_mask) {
                if (graph.value_dtype(n.inputs[1]) != ValueDType::U8 ||
                    graph.value_row_domain(n.inputs[1]) != ValueRowDomain::TOKEN_ROWS ||
                    graph.value_features(n.inputs[1]) != pa.mask_features) {
                    return fail("paged attention token mask shape mismatch");
                }
            } else if (n.inputs.size() != 1u) {
                return fail("paged attention without mask has extra inputs");
            }
        }

        const bool stateful =
            k == PrimitiveKind::KV_APPEND ||
            k == PrimitiveKind::PAGED_ATTENTION ||
            k == PrimitiveKind::STATEFUL_CAUSAL_CONV1D ||
            k == PrimitiveKind::GDN_RECURRENCE ||
            k == PrimitiveKind::PLE_CONV ||
            k == PrimitiveKind::QSA_KEY_CACHE ||
            k == PrimitiveKind::QSA_SCORES;
        if (!stateful && (!n.state_inputs.empty() || !n.state_outputs.empty())) {
            return fail("state reference on primitive without state access");
        }

        for (auto sid : n.state_inputs) {
            if (sid.id >= graph.states.size()) return fail("state input references unregistered state");
            if (!allowed_state_kind(graph.states[sid.id].kind, k)) {
                return fail("state kind incompatible with primitive");
            }
            if (graph.states[sid.id].state_index != node_state_index(n)) {
                return fail("state binding layer mismatch with consuming node");
            }
        }
        for (auto sid : n.state_outputs) {
            if (sid.id >= graph.states.size()) return fail("state output references unregistered state");
            if (!allowed_state_kind(graph.states[sid.id].kind, k)) {
                return fail("state kind incompatible with primitive");
            }
            if (graph.states[sid.id].state_index != node_state_index(n)) {
                return fail("state binding layer mismatch with producing node");
            }
        }

        switch (k) {
            case PrimitiveKind::KV_APPEND: {
                if (!n.state_inputs.empty()) return fail("KV_APPEND must not read state");
                if (n.state_outputs.size() != 1) return fail("KV_APPEND must write exactly one KV state");
                break;
            }
            case PrimitiveKind::PAGED_ATTENTION: {
                if (n.state_inputs.size() != 1) return fail("PAGED_ATTENTION must read exactly one KV state");
                if (!n.state_outputs.empty()) return fail("PAGED_ATTENTION must not write state");
                break;
            }
            case PrimitiveKind::STATEFUL_CAUSAL_CONV1D: {
                if (n.state_inputs.size() != 1 || n.state_outputs.size() != 1 ||
                    !(n.state_inputs[0] == n.state_outputs[0])) {
                    return fail("conv1d must read and write the same GDN conv state");
                }
                break;
            }
            case PrimitiveKind::GDN_RECURRENCE: {
                if (n.state_inputs.size() != 1 || n.state_outputs.size() != 1 ||
                    !(n.state_inputs[0] == n.state_outputs[0])) {
                    return fail("gdn recurrence must read and write the same recurrence state");
                }
                break;
            }
            case PrimitiveKind::QSA_KEY_CACHE: {
                if (!n.state_inputs.empty()) return fail("QSA_KEY_CACHE must not read state");
                if (n.state_outputs.size() != 1)
                    return fail("QSA_KEY_CACHE must write exactly one indexer state");
                break;
            }
            case PrimitiveKind::QSA_SCORES: {
                if (n.state_inputs.size() != 1)
                    return fail("QSA_SCORES must read exactly one indexer state");
                if (!n.state_outputs.empty())
                    return fail("QSA_SCORES must not write state");
                break;
            }
            default: break;
        }
    }

    std::unordered_set<uint32_t> external_in;
    for (auto v : graph.external_inputs) {
        if (v.id >= graph.next_value_id) return fail("external input id out of range");
        if (producer[v.id] != unproduced) return fail("external input shadows a produced value");
        if (!external_in.insert(v.id).second) return fail("duplicate external input entry");
    }

    std::vector<bool> consumed(static_cast<size_t>(graph.next_value_id), false);
    auto resolve_input = [&](ValueId v, size_t consumer_ordinal) -> Status {
        if (external_in.count(v.id)) return Status::make_ok();
        if (v.id >= graph.next_value_id) return fail("node input references unallocated value");
        int64_t p = producer[v.id];
        auto it = view_by_value.find(v.id);
        ValueId sink = v;
        if (it != view_by_value.end()) {
            const PrimitiveValueView* vw = it->second;
            p = producer[vw->base.id];
            if (p == unproduced) return fail("view base is never produced");
            sink = vw->base;
            consumed[v.id] = true;
        }
        if (p == unproduced) return fail("node input has no producer");
        if (p >= static_cast<int64_t>(graph.nodes.size())) return fail("corrupt producer record");
        if (p >= static_cast<int64_t>(consumer_ordinal)) {
            return fail("graph is not in topological order");
        }
        consumed[static_cast<size_t>(sink.id)] = true;
        return Status::make_ok();
    };

    for (size_t ni = 0; ni < graph.nodes.size(); ++ni) {
        for (auto v : graph.nodes[ni].inputs) {
            auto st = resolve_input(v, ni);
            if (!st.ok()) return st;
        }
    }

    std::unordered_set<uint32_t> ext_out;
    for (auto v : graph.external_outputs) {
        if (v.id >= graph.next_value_id) return fail("external output id out of range");
        if (producer[v.id] == unproduced) return fail("external output has no producer");
        if (view_by_value.count(v.id)) return fail("external output may not be a view");
        if (!ext_out.insert(v.id).second) return fail("duplicate external output entry");
        consumed[v.id] = true;
    }

    for (uint32_t vid = 0; vid < graph.next_value_id; ++vid) {
        if (producer[vid] == unproduced) continue;
        if (!consumed[vid]) {
            char detail[64];
            std::snprintf(detail, sizeof(detail), "produced value is never consumed (v%u)", vid);
            return Status::invalid_argument(detail, __FILE__, __LINE__);
        }
    }
    for (auto& v : graph.views) {
        if (!consumed[v.value.id]) return fail("view is never consumed");
    }

    std::unordered_map<uint32_t, uint32_t> kv_writes;
    size_t append_count = 0;
    for (auto& n : graph.nodes) {
        if (primitive_kind_of(n.node) != PrimitiveKind::KV_APPEND) continue;
        ++append_count;
        kv_writes.emplace(n.state_outputs[0].id, n.state_outputs[0].id);
    }
    size_t reads_matched = 0;
    for (auto& n : graph.nodes) {
        if (primitive_kind_of(n.node) != PrimitiveKind::PAGED_ATTENTION) continue;
        uint32_t sid = n.state_inputs[0].id;
        if (!kv_writes.count(sid)) {
            return fail("PAGED_ATTENTION reads KV state that no KV_APPEND writes");
        }
        ++reads_matched;
    }
    if (append_count > reads_matched) {
        return fail("KV_APPEND writes state that no PAGED_ATTENTION reads");
    }

    return Status::make_ok();
}

}
