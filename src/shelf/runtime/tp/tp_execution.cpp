#include <shelf/runtime/tp/tp_execution.h>

namespace shelf::runtime {

namespace {

bool value_used_elsewhere(const PrimitiveGraph& graph, ValueId value, uint32_t node_index) {
    for (uint32_t n = 0; n < graph.nodes.size(); ++n) {
        if (n == node_index) continue;
        for (const ValueId input : graph.nodes[n].inputs) {
            if (input.id == value.id) return true;
        }
    }
    for (const ValueId output : graph.external_outputs) {
        if (output.id == value.id) return true;
    }
    for (const PrimitiveValueView& view : graph.views) {
        if (view.base.id == value.id) return true;
    }
    return false;
}

void try_fuse_combine_consumer(const PrimitiveGraph& graph, const Program& program,
                               std::size_t consumer_dispatch, ValueId combine,
                               TpProgramSegment& segment) {
    if (consumer_dispatch >= program.dispatches.size()) return;
    const uint32_t consumer_index = program.dispatch_source_node[consumer_dispatch];
    if (consumer_index >= graph.nodes.size()) return;
    if (consumer_dispatch + 1 < program.dispatches.size() &&
        program.dispatch_source_node[consumer_dispatch + 1] == consumer_index) {
        return;
    }
    const ValueBinding* binding = program.find_value(combine);
    if (binding == nullptr || binding->dtype != ValueDType::BF16) return;
    if (value_used_elsewhere(graph, combine, consumer_index)) return;

    const PrimitiveGraphNode& consumer = graph.nodes[consumer_index];
    if (consumer.outputs.size() != 1u) return;
    const PrimitiveKind kind = primitive_kind_of(consumer.node);
    if (kind == PrimitiveKind::RESIDUAL_ADD && consumer.inputs.size() == 2u) {
        ValueId residual{};
        bool found = false;
        for (const ValueId input : consumer.inputs) {
            if (input.id != combine.id) {
                residual = input;
                found = true;
            }
        }
        if (!found) return;
        segment.fusion = TpCombineFusion::ResidualAdd;
        segment.fusion_residual = residual;
        segment.fusion_output = consumer.outputs[0];
    } else if (kind == PrimitiveKind::HYPER_APPLY && consumer.inputs.size() == 3u &&
               consumer.inputs[1].id == combine.id) {
        segment.fusion = TpCombineFusion::HyperApply;
        segment.fusion_residual = consumer.inputs[0];
        segment.fusion_injection = consumer.inputs[2];
        segment.fusion_output = consumer.outputs[0];
    } else {
        return;
    }
    segment.dispatch_end = static_cast<uint32_t>(consumer_dispatch + 1u);
}

}  // namespace

Result<TpExecutionSchedule> build_tp_execution_schedule(
    const PrimitiveGraph& graph,
    const Program& program) {
    if (program.dispatch_source_node.size() != program.dispatches.size()) {
        return Status::invalid_argument(
            "dispatch source node map does not match dispatches", __FILE__, __LINE__);
    }

    TpExecutionSchedule schedule;
    TpProgramSegment current;
    current.dispatch_begin = 0;

    std::size_t i = 0;
    while (i < program.dispatches.size()) {
        const uint32_t node_index = program.dispatch_source_node[i];
        std::size_t group_end = i + 1;
        while (group_end < program.dispatches.size() &&
               program.dispatch_source_node[group_end] == node_index) {
            ++group_end;
        }
        if (node_index >= graph.nodes.size()) {
            return Status::invalid_argument(
                "dispatch source node index out of range", __FILE__, __LINE__);
        }
        const PrimitiveGraphNode& node = graph.nodes[node_index];
        if (node.tp_combine != 0) {
            if (node.outputs.empty()) {
                return Status::invalid_argument(
                    "tp_combine node has no output value", __FILE__, __LINE__);
            }
            const ValueId combine = node.outputs[0];
            if (program.find_value(combine) == nullptr) {
                return Status::invalid_argument(
                    "tp_combine output value is not bound in the program",
                    __FILE__, __LINE__);
            }
            current.dispatch_end = static_cast<uint32_t>(group_end);
            current.producer_end = static_cast<uint32_t>(group_end);
            current.barrier_after = TpBarrierKind::SumHidden;
            current.combine_value = combine;
            current.combine_local_cols = node.tp_combine_local_cols;
            try_fuse_combine_consumer(graph, program, group_end, combine, current);
            const uint32_t next_begin = current.dispatch_end;
            schedule.segments.push_back(current);
            current = TpProgramSegment{};
            current.dispatch_begin = next_begin;
        }
        i = group_end;
    }

    if (current.dispatch_begin < program.dispatches.size()) {
        current.dispatch_end = static_cast<uint32_t>(program.dispatches.size());
        schedule.segments.push_back(current);
    }
    return schedule;
}

}  // namespace shelf::runtime
