#include <phaseshift/runtime/tp/tp_execution.h>

namespace ps::runtime {

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
            current.barrier_after = TpBarrierKind::SumHidden;
            current.combine_value = combine;
            current.combine_local_cols = node.tp_combine_local_cols;
            schedule.segments.push_back(current);
            current = TpProgramSegment{};
            current.dispatch_begin = static_cast<uint32_t>(group_end);
        }
        i = group_end;
    }

    if (current.dispatch_begin < program.dispatches.size()) {
        current.dispatch_end = static_cast<uint32_t>(program.dispatches.size());
        schedule.segments.push_back(current);
    }
    return schedule;
}

}  // namespace ps::runtime
