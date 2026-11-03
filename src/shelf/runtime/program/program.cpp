#include <shelf/runtime/program/program.h>
#include <shelf/runtime/program/activation_workspace_layout.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/runtime/program/linear_quant_contract.h>
#include <shelf/core/memory/alignment.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <limits>
#include <unordered_map>

namespace shelf::runtime {

namespace {

Status validate_workspace_layout(const WorkspaceLayout& layout) {
    for (const auto& range : layout.ranges) {
        if (range.offset > std::numeric_limits<std::size_t>::max() - range.bytes) {
            return Status::overflow("workspace range overflows", __FILE__, __LINE__);
        }
        if (range.offset + range.bytes > layout.total_bytes) {
            return Status::insufficient_memory("workspace range exceeds workspace", __FILE__, __LINE__);
        }
    }
    for (std::size_t i = 0; i < layout.ranges.size(); ++i) {
        const auto left_end = layout.ranges[i].offset + layout.ranges[i].bytes;
        for (std::size_t j = i + 1; j < layout.ranges.size(); ++j) {
            const auto right_end = layout.ranges[j].offset + layout.ranges[j].bytes;
            const bool memory_overlap =
                layout.ranges[i].offset < right_end && layout.ranges[j].offset < left_end;
            if (!memory_overlap) continue;
            const uint32_t live_first =
                std::max(layout.ranges[i].first_command, layout.ranges[j].first_command);
            const uint32_t live_last =
                std::min(layout.ranges[i].last_command, layout.ranges[j].last_command);
            if (live_first <= live_last) {
                return Status::invalid_argument(
                    "workspace ranges overlap with overlapping lifetime", __FILE__, __LINE__);
            }
        }
    }
    return Status::make_ok();
}

}

Status WorkspaceLayout::validate() const {
    return validate_workspace_layout(*this);
}

namespace {

constexpr uint64_t kWorkspaceAlignment = 256;

Status fail(const char* msg) {
    return Status::invalid_argument(msg, __FILE__, __LINE__);
}

struct ValueLifetime {
    uint32_t first_command = 0;
    uint32_t last_command = 0;
    bool defined = false;
};

struct ActivationWorkspace {
    uint32_t input_value = 0;
    uint32_t k = 0;
    uint32_t k_padded = 0;
    uint32_t max_rows = 0;
    uint64_t total_bytes = 0;
    uint32_t first_command = 0;
    uint32_t last_command = 0;
    uint32_t range_slot = kNoWorkspace;
    quantization::ActivationQuantSpec activation{};
    KernelId activation_kernel = KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED;
    std::vector<uint32_t> binding_refs;
};

struct Ctx {
    const PrimitiveGraph* graph = nullptr;
    WeightTableView weights;
    Program* plan = nullptr;
    ProgramBuildOptions options;
    uint32_t current_source_node = 0;
    std::unordered_map<uint32_t, uint32_t> external_indices;
    std::map<uint32_t, ValueLifetime> lifetimes;
    std::map<uint64_t, uint32_t> activation_ws_by_input;
    std::vector<ActivationWorkspace> activation_workspaces;

    const ValueBinding* find(ValueId v) const {
        for (const auto& b : plan->values) {
            if (b.logical_value.id == v.id) return &b;
        }
        return nullptr;
    }

    uint32_t rows_for(ValueRowDomain domain) const {
        const uint32_t limit = row_bucket_limit(plan->row_bucket);
        if (domain == ValueRowDomain::OUTPUT_ROWS) {
            return options.max_output_rows != 0 ? options.max_output_rows : limit;
        }
        if (options.max_token_rows != 0) return std::min(limit, options.max_token_rows);
        return limit;
    }

    void note_use(ValueId v, uint32_t dispatch_index) {
        auto& lt = lifetimes[v.id];
        if (!lt.defined) {
            lt.defined = true;
            lt.first_command = dispatch_index;
            lt.last_command = dispatch_index;
        } else if (dispatch_index > lt.last_command) {
            lt.last_command = dispatch_index;
        }
    }

    void note_def(ValueId v, uint32_t dispatch_index) {
        auto& lt = lifetimes[v.id];
        lt.defined = true;
        lt.first_command = dispatch_index;
        if (dispatch_index > lt.last_command) lt.last_command = dispatch_index;
    }

    ValueBinding make_binding(ValueId v, ValueStorage storage,
                                       uint32_t slot, uint32_t row_stride) const {
        const auto* spec = graph->value_spec(v);
        PrimitiveValueSpec fallback{};
        const PrimitiveValueSpec& sp = spec != nullptr ? *spec : fallback;
        const uint32_t rows = rows_for(sp.row_domain);
        ValueBinding b{};
        b.logical_value = v;
        b.storage = storage;
        b.dtype = sp.dtype;
        b.row_domain = sp.row_domain;
        b.slot = slot;
        b.feature_count = sp.features;
        if (row_stride != 0) {
            b.row_stride = row_stride;
        } else if (storage == ValueStorage::WORKSPACE) {
            b.row_stride = ::shelf::gpu::aligned_row_stride(sp.features, value_dtype_bytes(sp.dtype));
        } else {
            b.row_stride = sp.features;
        }
        b.offset = 0;
        b.bytes = static_cast<uint64_t>(rows) * b.row_stride * value_dtype_bytes(sp.dtype);
        return b;
    }

    Status require_input(ValueId v) {
        if (find(v) != nullptr) return Status::make_ok();
        bool external = false;
        uint32_t ext_slot = 0;
        for (uint32_t i = 0; i < graph->external_inputs.size(); ++i) {
            if (graph->external_inputs[i].id == v.id) {
                external = true;
                ext_slot = i;
            }
        }
        if (!external) {
            ValueId base_source{v.id};
            for (const auto& vw : graph->views) {
                if (vw.value.id == v.id) {
                    base_source = vw.base;
                    break;
                }
            }
            if (find(base_source) != nullptr) return Status::make_ok();
            return fail("input value has no producer or external source");
        }
        external_indices[v.id] = ext_slot;
        plan->values.push_back(make_binding(v, ValueStorage::EXTERNAL_INPUT, ext_slot, 0));
        return Status::make_ok();
    }

    Status produce(ValueId v) {
        if (find(v) != nullptr) return fail("value physically bound twice");
        plan->values.push_back(make_binding(v, ValueStorage::WORKSPACE, 0, 0));
        note_def(v, static_cast<uint32_t>(plan->dispatches.size()));
        return Status::make_ok();
    }

    void push(const DispatchBinding& base) {
        plan->dispatches.push_back(base);
        plan->dispatch_source_node.push_back(current_source_node);
    }
};

DispatchBinding base_binding(KernelId id) {
    DispatchBinding b{};
    b.kernel_id = id;
    return b;
}

uint32_t register_state(Ctx& c, StateId sid, PrimitiveStateKind kind, uint32_t state_index) {
    for (const auto& s : c.plan->states) {
        if (s.state.id == sid.id) return s.slot;
    }
    StateBinding sb;
    sb.state = sid;
    sb.kind = kind;
    sb.state_index = state_index;
    sb.slot = static_cast<uint32_t>(c.plan->states.size());
    c.plan->states.push_back(sb);
    return sb.slot;
}

template <typename T>
const T& node_as(const PrimitiveGraphNode& n) {
    return std::get<T>(n.node);
}

Status emit_node(Ctx& c, const PrimitiveGraphNode& node) {
    using namespace shelf::runtime;
    auto kind = primitive_kind_of(node.node);
    for (auto in : node.inputs) {
        auto st = c.require_input(in);
        if (!st.ok()) return st;
    }
    switch (kind) {
        case PrimitiveKind::EMBEDDING_LOOKUP: {
            const auto& n = node_as<EmbeddingLookupNode>(node);
            if (n.weight_index >= c.weights.count) return fail("embedding weight index exceeds view");
            DispatchBinding b = base_binding(KernelId::EMBEDDING_LOOKUP);
            b.weight_index = n.weight_index;
            b.input_slots[0] = node.inputs[0].id;
            b.input_count = 1;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::LINEAR: {
            const auto& n = node_as<LinearNode>(node);
            if (n.weight_index >= c.weights.count) return fail("linear weight index exceeds view");
            auto w = c.weights.at(n.weight_index);
            if (node.outputs.empty()) return fail("linear node has no output value");
            {
                const auto* out_spec = c.graph->value_spec(node.outputs[0]);
                const uint64_t declared_out_features =
                    out_spec != nullptr ? static_cast<uint64_t>(out_spec->features) : 0u;
                const uint64_t expected_out_features =
                    w.output_col_total != 0u ? static_cast<uint64_t>(w.output_col_total)
                                            : static_cast<uint64_t>(w.rows);
                if (declared_out_features != expected_out_features) {
                    return fail(
                        "linear output features do not match the weight column width");
                }
                if (w.output_col_total != 0u &&
                    static_cast<uint64_t>(w.output_col_offset) + w.rows >
                        static_cast<uint64_t>(w.output_col_total)) {
                    return fail("linear column offset exceeds the weight column width");
                }
            }
            const Result<ResolvedLinearQuantContract> resolved =
                resolve_linear_quant_contract(w.encoding, w.compute_spec);
            if (!resolved.ok()) return resolved.status();
            const ResolvedLinearQuantContract& contract = resolved.value();
            if (!contract.requires_activation_quant) {
                uint64_t expected_bf16_bytes =
                    static_cast<uint64_t>(w.rows) * static_cast<uint64_t>(w.cols) * value_dtype_bytes(ValueDType::BF16);
                if (w.bytes != 0 && w.bytes != expected_bf16_bytes) {
                    return Status::unsupported(
                        "non bf16 linear weights are outside baseline physical coverage",
                        __FILE__, __LINE__);
                }
            }
            const KernelId variant = contract.linear_kernel;
            const bool need_act_ws = contract.requires_activation_quant;
            const uint64_t act_ws_key = static_cast<uint64_t>(node.inputs[0].id);
            const uint32_t cmd_idx = static_cast<uint32_t>(c.plan->dispatches.size());
            DispatchBinding b = base_binding(variant);
            b.weight_index = n.weight_index;
            b.compute_spec = w.compute_spec;
            b.flags = c.graph->value_dtype(node.outputs[0]) == ValueDType::F32 ? 1u : 0u;
            if (contract.activation_kernel == KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF)
                b.flags |= kActFlagRowRefFolded;
            if (need_act_ws) {
                const uint32_t kdim = w.cols;
                const uint32_t kp =
                    activation_k_padded_for(contract.activation_k_alignment, kdim);
                auto it = c.activation_ws_by_input.find(act_ws_key);
                if (it == c.activation_ws_by_input.end()) {
                    const uint32_t ws_idx =
                        static_cast<uint32_t>(c.activation_workspaces.size());
                    ActivationWorkspace aw{};
                    aw.input_value = node.inputs[0].id;
                    aw.k = kdim;
                    aw.k_padded = kp;
                    const auto* in_spec = c.graph->value_spec(node.inputs[0]);
                    aw.max_rows = c.rows_for(
                        in_spec != nullptr ? in_spec->row_domain : ValueRowDomain::TOKEN_ROWS);
                    if (aw.max_rows == 0) return fail("activation workspace has no rows");
                    Int8ActivationWorkspaceLayout a8l;
                    const Status layout_status = make_activation_workspace_layout(
                        contract.activation, kp, aw.max_rows, a8l);
                    if (!layout_status.ok()) return layout_status;
                    aw.activation = contract.activation;
                    aw.activation_kernel = contract.activation_kernel;
                    aw.total_bytes = a8l.total_bytes;
                    aw.first_command = cmd_idx;
                    aw.last_command = cmd_idx;
                    c.activation_workspaces.push_back(aw);
                    c.activation_ws_by_input[act_ws_key] = ws_idx;
                    DispatchBinding qb = base_binding(contract.activation_kernel);
                    qb.group_size = 32;
                    qb.activation_kp = kp;
                    qb.activation_max_rows = aw.max_rows;
                    qb.activation_code_stride = a8l.code_row_stride_bytes;
                    qb.activation_scale_stride = a8l.scale_row_stride_bytes;
                    qb.input_slots[0] = node.inputs[0].id;
                    qb.input_count = 1;
                    c.push(qb);
                    c.activation_workspaces[ws_idx].binding_refs.push_back(
                        static_cast<uint32_t>(c.plan->dispatches.size()) - 1u);
                    c.activation_workspaces[ws_idx].last_command =
                        static_cast<uint32_t>(c.plan->dispatches.size()) - 1u;
                } else {
                    auto& aw = c.activation_workspaces[it->second];
                    if (aw.k != kdim || aw.k_padded != kp)
                        return fail("activation workspace K mismatch for shared input");
                    if (!activation_quant_matches(aw.activation, contract.activation))
                        return fail(
                            "activation workspace quant spec mismatch for shared input");
                    if (aw.activation_kernel != contract.activation_kernel)
                        return fail(
                            "activation workspace plane form mismatch for shared input");
                    aw.last_command = static_cast<uint32_t>(c.plan->dispatches.size());
                }
                auto& aw = c.activation_workspaces[c.activation_ws_by_input[act_ws_key]];
                b.activation_kp = aw.k_padded;
                b.activation_max_rows = aw.max_rows;
                {
                    Int8ActivationWorkspaceLayout a8l;
                    const Status layout_status = make_activation_workspace_layout(
                        aw.activation, aw.k_padded, aw.max_rows, a8l);
                    if (!layout_status.ok()) return layout_status;
                    b.activation_code_stride = a8l.code_row_stride_bytes;
                    b.activation_scale_stride = a8l.scale_row_stride_bytes;
                }
                b.input_count = 0;
            } else {
                b.input_slots[0] = node.inputs[0].id;
                b.input_count = 1;
            }
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            const uint32_t binding_idx = static_cast<uint32_t>(c.plan->dispatches.size());
            c.push(b);
            if (need_act_ws) {
                auto& aw = c.activation_workspaces[c.activation_ws_by_input[act_ws_key]];
                aw.binding_refs.push_back(binding_idx);
                aw.last_command = static_cast<uint32_t>(c.plan->dispatches.size()) - 1u;
            }
            return Status::make_ok();
        }
        case PrimitiveKind::RMS_NORM: {
            const auto& n = node_as<RmsNormNode>(node);
            DispatchBinding b = base_binding(KernelId::RMS_NORM);
            b.scalar_a = n.eps;
            b.group_size = n.group_size;
            b.parameter_index0 = n.parameter_index;
            b.flags = static_cast<uint32_t>(n.weight_mode);
            b.input_slots[0] = node.inputs[0].id;
            b.input_count = 1;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::RESIDUAL_ADD: {
            DispatchBinding b = base_binding(KernelId::RESIDUAL_ADD);
            for (uint32_t i = 0; i < node.inputs.size() && i < kMaxIoSlots; ++i)
                b.input_slots[i] = node.inputs[i].id;
            b.input_count = static_cast<uint32_t>(node.inputs.size());
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::SPLIT: {
            const auto& sn = node_as<SplitNode>(node);
            DispatchBinding b = base_binding(KernelId::SPLIT);
            b.flags = static_cast<uint32_t>(sn.layout);
            b.group_size = sn.head_dim;
            b.input_slots[0] = node.inputs[0].id;
            b.input_count = 1;
            auto p0 = c.produce(node.outputs[0]);
            if (!p0.ok()) return p0;
            auto p1 = c.produce(node.outputs[1]);
            if (!p1.ok()) return p1;
            b.output_slots[0] = node.outputs[0].id;
            b.output_slots[1] = node.outputs[1].id;
            b.output_count = 2;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::SILU:
        case PrimitiveKind::SIGMOID:
        case PrimitiveKind::L2_NORMALIZE:
        case PrimitiveKind::ROPE:
        case PrimitiveKind::SCALE: {
            KernelId kid = KernelId::SILU;
            float sa = 1.0f;
            float sb = 0.0f;
            uint32_t rot = 0;
            uint32_t group = 0;
            if (kind == PrimitiveKind::SIGMOID) kid = KernelId::SIGMOID;
            else if (kind == PrimitiveKind::L2_NORMALIZE) {
                kid = KernelId::L2_NORMALIZE;
                sa = node_as<L2NormalizeNode>(node).eps;
                group = node_as<L2NormalizeNode>(node).group_size;
            } else if (kind == PrimitiveKind::ROPE) {
                kid = KernelId::ROPE;
                rot = node_as<RoPENode>(node).rotary_dim;
                sb = node_as<RoPENode>(node).theta;
                group = node_as<RoPENode>(node).head_dim;
                sa = node_as<RoPENode>(node).attention_factor;
            } else if (kind == PrimitiveKind::SCALE) {
                kid = KernelId::SCALE;
                sa = node_as<ScaleNode>(node).scale;
            }
            DispatchBinding b = base_binding(kid);
            b.scalar_a = sa;
            b.scalar_b = sb;
            b.rotary_dim = rot;
            b.group_size = group;
            b.input_slots[0] = node.inputs[0].id;
            b.input_count = 1;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::MUL:
        case PrimitiveKind::SWIGLU: {
            DispatchBinding b = base_binding(kind == PrimitiveKind::MUL ? KernelId::MUL
                                                                          : KernelId::SWIGLU);
            b.input_slots[0] = node.inputs[0].id;
            b.input_slots[1] = node.inputs[1].id;
            b.input_count = 2;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::CONCAT: {
            const auto& cn = node_as<ConcatNode>(node);
            DispatchBinding b = base_binding(KernelId::CONCAT);
            b.group_size = static_cast<uint32_t>(cn.input_shape_a.features);
            b.input_slots[0] = node.inputs[0].id;
            b.input_slots[1] = node.inputs[1].id;
            b.input_count = 2;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::KV_APPEND: {
            const auto& n = node_as<KvAppendNode>(node);
            DispatchBinding b = base_binding(KernelId::KV_APPEND);
            b.descriptor_slot =
                register_state(c, node.state_outputs.front(),
                               PrimitiveStateKind::KV_CACHE, n.state_index) +
                1;
            b.group_size = n.attention.head_dim;
            b.input_slots[0] = node.inputs[0].id;
            b.input_slots[1] = node.inputs[1].id;
            b.input_count = 2;
            b.output_count = 0;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::PAGED_ATTENTION: {
            const auto& n = node_as<PagedAttentionNode>(node);
            DispatchBinding b = base_binding(KernelId::PAGED_ATTENTION);
            b.scalar_a = n.scale;
            b.descriptor_slot =
                register_state(c, node.state_inputs.front(), PrimitiveStateKind::KV_CACHE,
                               n.state_index) +
                1;
            b.group_size = n.attention.head_dim;
            b.input_slots[0] = node.inputs[0].id;
            b.input_count = 1;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::STATEFUL_CAUSAL_CONV1D: {
            const auto& n = node_as<StatefulCausalConv1DNode>(node);
            DispatchBinding b = base_binding(KernelId::STATEFUL_CAUSAL_CONV1D);
            b.descriptor_slot =
                register_state(c, node.state_inputs.front(), PrimitiveStateKind::GDN_CONV_STATE,
                               n.state_index) +
                1;
            b.parameter_index0 = n.parameter_index;
            b.input_slots[0] = node.inputs[0].id;
            b.input_count = 1;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::GDN_RECURRENCE: {
            const auto& n = node_as<GdnRecurrenceNode>(node);
            DispatchBinding b = base_binding(KernelId::GDN_RECURRENCE);
            b.descriptor_slot =
                register_state(c, node.state_inputs.front(), PrimitiveStateKind::GDN_RECURRENCE_STATE,
                               n.state_index) +
                1;
            b.parameter_index0 = n.dt_bias_parameter_index;
            b.parameter_index1 = n.a_log_parameter_index;
            for (uint32_t i = 0; i < node.inputs.size() && i < kMaxIoSlots; ++i)
                b.input_slots[i] = node.inputs[i].id;
            b.input_count = static_cast<uint32_t>(node.inputs.size());
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::OUTPUT_GATHER: {
            DispatchBinding b = base_binding(KernelId::OUTPUT_GATHER);
            b.input_slots[0] = node.inputs[0].id;
            b.input_count = 1;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
        case PrimitiveKind::SAMPLING: {
            const auto& n = node_as<SamplingNode>(node);
            DispatchBinding b = base_binding(KernelId::SAMPLING);
            b.rotary_dim = n.vocab_size;
            b.input_slots[0] = node.inputs[0].id;
            b.input_count = 1;
            auto po = c.produce(node.outputs[0]);
            if (!po.ok()) return po;
            b.output_slots[0] = node.outputs[0].id;
            b.output_count = 1;
            c.push(b);
            return Status::make_ok();
        }
    }
    return fail("unknown primitive kind");
}

Status finalize_layout(Ctx& c) {
    const auto& options = c.options;
    for (const auto& view : c.graph->views) {
        auto vit = c.lifetimes.find(view.value.id);
        if (vit == c.lifetimes.end()) continue;
        auto bit = c.lifetimes.find(view.base.id);
        if (bit == c.lifetimes.end()) continue;
        bit->second.first_command =
            std::min(bit->second.first_command, vit->second.first_command);
        bit->second.last_command =
            std::max(bit->second.last_command, vit->second.last_command);
    }

    struct AllocItem {
        uint64_t bytes = 0;
        uint32_t first_command = 0;
        uint32_t last_command = 0;
        ValueBinding* value = nullptr;
        int32_t aw_index = -1;
    };
    std::vector<AllocItem> items;
    for (auto& vb : c.plan->values) {
        if (vb.storage != ValueStorage::WORKSPACE) continue;
        const auto& lt = c.lifetimes[vb.logical_value.id];
        items.push_back(AllocItem{vb.bytes, lt.first_command, lt.last_command, &vb, -1});
    }
    for (uint32_t i = 0; i < c.activation_workspaces.size(); ++i) {
        auto& aw = c.activation_workspaces[i];
        items.push_back(AllocItem{aw.total_bytes, aw.first_command, aw.last_command,
                                  nullptr, static_cast<int32_t>(i)});
    }
    std::sort(items.begin(), items.end(), [](const AllocItem& a, const AllocItem& b) {
        if (a.first_command != b.first_command) return a.first_command < b.first_command;
        return a.bytes > b.bytes;
    });

    struct FreeBlock {
        uint64_t offset = 0;
        uint64_t bytes = 0;
        uint32_t free_after = 0;
    };
    struct LiveBlock {
        uint64_t offset = 0;
        uint64_t bytes = 0;
        uint32_t last_command = 0;
    };
    std::vector<FreeBlock> free_list;
    std::vector<LiveBlock> live;
    uint64_t cursor = 0;
    c.plan->workspace.ranges.clear();

    for (const auto& item : items) {
        for (auto it = live.begin(); it != live.end();) {
            if (it->last_command < item.first_command) {
                free_list.push_back(FreeBlock{it->offset, it->bytes, it->last_command});
                it = live.erase(it);
            } else {
                ++it;
            }
        }
        auto best = free_list.end();
        if (!options.no_workspace_reuse) {
            for (auto it = free_list.begin(); it != free_list.end(); ++it) {
                if (it->free_after < item.first_command && it->bytes >= item.bytes &&
                    (best == free_list.end() || it->bytes < best->bytes)) {
                    best = it;
                }
            }
        }
        uint64_t offset = 0;
        if (best != free_list.end()) {
            offset = best->offset;
            const uint64_t rem = best->bytes - item.bytes;
            const uint32_t free_after = best->free_after;
            free_list.erase(best);
            if (rem >= kWorkspaceAlignment) {
                free_list.push_back(FreeBlock{offset + item.bytes, rem, free_after});
            }
        } else {
            offset = (cursor + kWorkspaceAlignment - 1) / kWorkspaceAlignment *
                     kWorkspaceAlignment;
            cursor = offset + item.bytes;
        }
        const uint32_t slot = static_cast<uint32_t>(c.plan->workspace.ranges.size());
        WorkspaceRange r{};
        r.offset = offset;
        r.bytes = item.bytes;
        r.first_command = item.first_command;
        r.last_command = item.last_command;
        c.plan->workspace.ranges.push_back(r);
        if (item.value != nullptr) {
            item.value->offset = offset;
            item.value->slot = slot;
        } else {
            auto& aw = c.activation_workspaces[item.aw_index];
            aw.range_slot = slot;
            for (uint32_t ref : aw.binding_refs)
                c.plan->dispatches[ref].workspace_slot = slot;
        }
        live.push_back(LiveBlock{offset, item.bytes, item.last_command});
    }
    c.plan->workspace.total_bytes = cursor;
    if (const char* wsdbg = std::getenv("SHELF_WS_DEBUG")) {
        if (wsdbg[0] != '\0') {
            uint64_t act_ws_bytes = 0;
            for (auto& aw : c.activation_workspaces) act_ws_bytes += aw.total_bytes;
            uint64_t value_ws_bytes = cursor - act_ws_bytes;
            uint64_t top = 0;
            for (auto& r : c.plan->workspace.ranges) top = std::max(top, r.bytes);
            std::fprintf(stderr,
                         "WSDBG total=%.2fGB value_ws=%.2fGB act_ws=%.2fGB ranges=%zu max_range=%.3fGB\n",
                         cursor / 1e9, value_ws_bytes / 1e9, act_ws_bytes / 1e9,
                         c.plan->workspace.ranges.size(), (double)top / 1e9);
        }
    }
    if (!options.validate_workspace) return Status::make_ok();
    return c.plan->workspace.validate();
}

Status validate_views(Ctx& c) {
    for (const auto& view : c.graph->views) {
        auto* base = c.find(view.base);
        if (base == nullptr) return fail("view base lacks physical binding");
        const auto* spec = c.graph->value_spec(view.value);
        if (spec == nullptr) return fail("view value lacks spec");
        if (spec->features != view.feature_count)
            return fail("view feature count mismatch with spec");
        const uint32_t elem = value_dtype_bytes(base->dtype);
        ValueBinding alias{};
        alias.logical_value = view.value;
        alias.storage = ValueStorage::WORKSPACE_VIEW;
        alias.dtype = base->dtype;
        alias.row_domain = base->row_domain;
        alias.slot = base->slot;
        alias.feature_count = view.feature_count;
        alias.row_stride = base->row_stride;
        alias.offset = base->offset + static_cast<uint64_t>(view.feature_offset) * elem;
        alias.bytes = static_cast<uint64_t>(view.feature_count) * elem;
        if ((alias.offset & 15ull) != 0ull) {
            return fail("production workspace view is not 16-byte aligned");
        }
        if ((static_cast<uint64_t>(alias.row_stride) * elem & 15ull) != 0ull) {
            return fail("production workspace view row is not 16-byte aligned");
        }
        c.plan->values.push_back(alias);
    }
    return Status::make_ok();
}


Status finalize_externals(Ctx& c) {
    for (auto eo : c.graph->external_outputs) {
        ValueBinding* found = nullptr;
        for (auto& vb : c.plan->values) {
            if (vb.logical_value.id == eo.id) {
                if (vb.storage != ValueStorage::WORKSPACE) {
                    return fail("external output bound twice");
                }
                found = &vb;
                break;
            }
        }
        if (found == nullptr) return fail("external output lacks producing node");
        found->storage = ValueStorage::EXTERNAL_OUTPUT;
        found->slot = c.plan->external_output_count++;
    }
    return Status::make_ok();
}

}

Status Program::validate() const {
    if (!valid_row_bucket(row_bucket)) return fail("invalid RowBucket");
    if (!is_target_execution_class(execution_class)) {
        return Status::unsupported("program requires target execution class", __FILE__, __LINE__);
    }
    if (dispatches.empty()) return Status::invalid_state("program has no dispatches", __FILE__, __LINE__);
    if (!dispatch_source_node.empty() &&
        dispatch_source_node.size() != dispatches.size())
        return fail("dispatch source metadata size mismatch");
    for (const auto& b : dispatches) {
        if (static_cast<uint32_t>(b.kernel_id) >= kernel_id_count())
            return fail("binding kernel id invalid");
        if (b.input_count > kMaxIoSlots || b.output_count > kMaxIoSlots)
            return fail("binding io counts exceed capacity");
        for (uint32_t s = 0; s < b.input_count; ++s) {
            if (find_value(ValueId{b.input_slots[s]}) == nullptr)
                return fail("dispatch input value unbound");
        }
        for (uint32_t s = 0; s < b.output_count; ++s) {
            if (find_value(ValueId{b.output_slots[s]}) == nullptr)
                return fail("dispatch output value unbound");
        }
        if (b.weight_index != kNoWeight && b.weight_index >= weight_slot_count)
            return fail("binding weight index exceeds table");
        if (b.descriptor_slot != 0) {
            bool found_state = false;
            for (const auto& sb : states)
                found_state |= sb.slot + 1 == b.descriptor_slot;
            if (!found_state) return fail("descriptor slot unknown");
        }
        if (b.workspace_slot != kNoWorkspace &&
            b.workspace_slot >= workspace.ranges.size())
            return fail("binding workspace slot out of range");
    }
    auto ws = workspace.validate();
    if (!ws.ok()) return ws;
    return Status::make_ok();
}

Result<Program> build_program(
    const PrimitiveGraph& graph,
    const WeightTableView& weights,
    RowBucket bucket,
    ExecutionClass execution_class,
    const ProgramBuildOptions& options) {
    if (!valid_row_bucket(bucket)) return fail("invalid RowBucket");
    if (!is_target_execution_class(execution_class)) {
        return Status::unsupported(
            "program requires target execution class", __FILE__, __LINE__);
    }
    if (graph.nodes.empty()) return fail("empty primitive graph");

    Program plan;
    plan.row_bucket = bucket;
    plan.execution_class = execution_class;
    plan.weight_slot_count = static_cast<uint32_t>(weights.count);
    plan.external_input_count = static_cast<uint32_t>(graph.external_inputs.size());

    Ctx c;
    c.graph = &graph;
    c.weights = weights;
    c.plan = &plan;
    c.options = options;

    for (size_t ni = 0; ni < graph.nodes.size(); ++ni) {
        for (auto in : graph.nodes[ni].inputs) {
            c.note_use(in, static_cast<uint32_t>(c.plan->dispatches.size()));
        }
        for (auto keep : graph.nodes[ni].keep_alive) {
            c.note_use(keep, static_cast<uint32_t>(c.plan->dispatches.size()));
        }
        c.current_source_node = static_cast<uint32_t>(ni);
        auto st = emit_node(c, graph.nodes[ni]);
        if (!st.ok()) return st;
    }
    auto st_ext = finalize_externals(c);
    if (!st_ext.ok()) return st_ext;
    auto st_ws = finalize_layout(c);
    if (!st_ws.ok()) return st_ws;
    auto st_views = validate_views(c);
    if (!st_views.ok()) return st_views;

    auto v = plan.validate();
    if (!v.ok()) return v;
    return plan;
}

Program* ProgramSet::resolve(RowBucket bucket) {
    for (auto& program : programs) {
        if (program.row_bucket == bucket) return &program;
    }
    return nullptr;
}

const Program* ProgramSet::resolve(RowBucket bucket) const {
    for (const auto& program : programs) {
        if (program.row_bucket == bucket) return &program;
    }
    return nullptr;
}

Result<ProgramSet> build_program_set(
    const PrimitiveGraph& graph,
    const WeightTableView& weights,
    const StaticParameterTableView& parameters,
    ExecutionClass execution_class,
    const ProgramBuildOptions& options) {
    const RowBucket buckets[] = {RowBucket::R16, RowBucket::R32, RowBucket::R64, RowBucket::R128,
                                  RowBucket::R256, RowBucket::R512, RowBucket::R1024, RowBucket::R2048,
                                  RowBucket::R4096, RowBucket::R8192};
    ProgramSet set;
    for (size_t i = 0; i < ProgramSet::kRowBucketCount; ++i) {
        auto program_res = build_program(graph, weights, buckets[i], execution_class, options);
        if (!program_res.ok()) return program_res.status();
        set.programs[i] = program_res.release();
        set.programs[i].parameter_slot_count = static_cast<uint32_t>(parameters.count);
    }
    return set;
}

}
