#pragma once
#include <shelf/runtime/graph/grouped_row_shape.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <shelf/runtime/graph/value_type.h>
#include <shelf/core/status.h>
#include <cstdint>
#include <vector>
#include <string>

namespace shelf::runtime {

struct ValueId {
    uint32_t id = 0;
    bool operator==(const ValueId&) const = default;
};

struct StateId {
    uint32_t id = 0;
    bool operator==(const StateId&) const = default;
};

enum class PrimitiveStateKind : uint8_t {
    KV_CACHE = 0,
    GDN_CONV_STATE = 1,
    GDN_RECURRENCE_STATE = 2,
    PLE_CONV_STATE = 3,
};

inline const char* to_string(PrimitiveStateKind k) noexcept {
    switch (k) {
        case PrimitiveStateKind::KV_CACHE: return "KV_CACHE";
        case PrimitiveStateKind::GDN_CONV_STATE: return "GDN_CONV_STATE";
        case PrimitiveStateKind::GDN_RECURRENCE_STATE: return "GDN_RECURRENCE_STATE";
        case PrimitiveStateKind::PLE_CONV_STATE: return "PLE_CONV_STATE";
    }
    return "UNKNOWN";
}

struct PrimitiveStateBinding {
    StateId id;
    PrimitiveStateKind kind = PrimitiveStateKind::KV_CACHE;
    uint32_t state_index = 0;
};

struct PrimitiveValueView {
    ValueId value;
    ValueId base;
    uint32_t feature_offset = 0;
    uint32_t feature_count = 0;
};

struct PrimitiveEdge {
    ValueId from;
    ValueId to;
};

struct PrimitiveGraphNode {
    PrimitiveNode node;
    std::vector<ValueId> inputs;
    std::vector<ValueId> outputs;
    std::vector<StateId> state_inputs;
    std::vector<StateId> state_outputs;
    std::vector<ValueId> keep_alive;
    std::string debug_name;
    uint8_t tp_combine = 0;
    uint32_t tp_combine_local_cols = 0;
};

struct PrimitiveGraph {
    std::vector<PrimitiveGraphNode> nodes;
    std::vector<ValueId> external_inputs;
    std::vector<ValueId> external_outputs;
    std::vector<PrimitiveStateBinding> states;
    std::vector<PrimitiveValueView> views;
    std::vector<MatrixwiseShapeKey> weight_table;
    uint32_t next_value_id = 0;
    std::vector<PrimitiveValueSpec> value_specs;
    // The plan geometry of the GROUPED_ROWS row space, or experts == 0 when the graph has
    // no grouped rows. Every grouped value and every grouped node reads it from here, so a
    // lowering states it once instead of repeating it per value.
    GroupedRowShape grouped_row_shape;

    ValueId alloc_value(
        uint32_t features = 0,
        ValueDType dtype = ValueDType::BF16,
        ValueRowDomain rows = ValueRowDomain::TOKEN_ROWS) {
        value_specs.push_back(PrimitiveValueSpec{
            .features = features,
            .dtype = dtype,
            .row_domain = rows,
        });
        return ValueId{next_value_id++};
    }

    ValueId alloc_grouped_value(uint32_t features,
                                ValueDType dtype = ValueDType::BF16) {
        return alloc_value(features, dtype, ValueRowDomain::GROUPED_ROWS);
    }

    bool has_grouped_rows() const noexcept {
        return grouped_row_shape.experts != 0u;
    }

    const PrimitiveValueSpec* value_spec(ValueId v) const noexcept {
        if (v.id >= value_specs.size()) return nullptr;
        return &value_specs[v.id];
    }

    uint32_t value_features(ValueId v) const noexcept {
        const auto* spec = value_spec(v);
        return spec != nullptr ? spec->features : 0u;
    }

    ValueDType value_dtype(ValueId v) const noexcept {
        const auto* spec = value_spec(v);
        return spec != nullptr ? spec->dtype : ValueDType::BF16;
    }

    ValueRowDomain value_row_domain(ValueId v) const noexcept {
        const auto* spec = value_spec(v);
        return spec != nullptr ? spec->row_domain : ValueRowDomain::TOKEN_ROWS;
    }

    StateId alloc_state(PrimitiveStateKind kind, uint32_t state_index) {
        states.push_back(PrimitiveStateBinding{StateId{static_cast<uint32_t>(states.size())}, kind, state_index});
        return states.back().id;
    }

    uint32_t register_weight(MatrixwiseShapeKey shape) {
        weight_table.push_back(shape);
        return static_cast<uint32_t>(weight_table.size() - 1);
    }

    PrimitiveGraphNode& add_node(PrimitiveNode n) {
        PrimitiveGraphNode g;
        g.node = std::move(n);
        nodes.push_back(std::move(g));
        return nodes.back();
    }

    std::vector<PrimitiveKind> kind_sequence() const {
        std::vector<PrimitiveKind> out;
        out.reserve(nodes.size());
        for (auto& gn : nodes) out.push_back(primitive_kind_of(gn.node));
        return out;
    }
};

Status validate_primitive_graph(const PrimitiveGraph& graph) noexcept;


}
