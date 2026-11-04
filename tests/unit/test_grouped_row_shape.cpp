#include <shelf/runtime/graph/grouped_row_shape.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>

#include <array>
#include <cstdio>

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

// Values from the plan contract of docs/developer/qwen35moe.md §6, computed by hand.
void check_capacity_contract() {
    check(rt::grouped_slot_count(1, {256, 8}) == 8u, "one token selects 8 slots");
    check(rt::grouped_row_tile_count(1, {256, 8}) == 8u,
          "8 slots spread over 8 groups when every expert may win");
    check(rt::grouped_row_capacity(1, {256, 8}) == 128u,
          "one token needs 128 group rows: 16 rows per group");
    check(rt::grouped_slot_count(512, {256, 8}) == 4096u, "512 tokens select 4096 slots");
    check(rt::grouped_row_tile_count(512, {256, 8}) == 512u,
          "4096 slots need at most 512 groups");
    check(rt::grouped_row_capacity(512, {256, 8}) == 8192u,
          "512 groups hold at most 8192 group rows");
    check(rt::grouped_slot_tile_count(1) == 1u, "one slot tile for one token");
    check(rt::grouped_slot_tile_count(64) == 1u, "64 tokens fit one slot tile");
    check(rt::grouped_slot_tile_count(65) == 2u, "65 tokens need two slot tiles");

    const std::array<uint32_t, 5> rows_values = {1u, 3u, 16u, 64u, 512u};
    const std::array<rt::GroupedRowShape, 4> shapes = {
        rt::GroupedRowShape{1u, 1u}, rt::GroupedRowShape{4u, 2u},
        rt::GroupedRowShape{256u, 8u}, rt::GroupedRowShape{512u, 16u},
    };
    bool invariants = true;
    for (const uint32_t rows : rows_values) {
        for (const auto& shape : shapes) {
            const uint32_t slots = rt::grouped_slot_count(rows, shape);
            const uint32_t tiles = rt::grouped_row_tile_count(rows, shape);
            const uint32_t capacity = rt::grouped_row_capacity(rows, shape);
            if (tiles > slots) invariants = false;
            if (capacity != tiles * rt::kGroupedRowTile) invariants = false;
            if (capacity < slots) invariants = false;
            if ((capacity % rt::kGroupedRowTile) != 0u) invariants = false;
            if (rt::grouped_slot_count(rows, shape) != rows * shape.top_k) invariants = false;
        }
    }
    check(invariants, "group count never exceeds slots and capacity always covers them");
}

rt::PrimitiveGraph grouped_graph(rt::GroupedRowShape shape, bool declare_shape) {
    rt::PrimitiveGraph graph;
    if (declare_shape) graph.grouped_row_shape = shape;

    const auto rows_in =
        graph.alloc_value(8, rt::ValueDType::BF16, rt::ValueRowDomain::GROUPED_ROWS);
    const auto rows_out =
        graph.alloc_value(8, rt::ValueDType::BF16, rt::ValueRowDomain::GROUPED_ROWS);
    graph.external_inputs.push_back(rows_in);
    graph.external_outputs.push_back(rows_out);

    rt::SiLUNode node{};
    node.shape = rt::RowwiseShapeKey{8};
    auto& graph_node = graph.add_node(node);
    graph_node.inputs = {rows_in};
    graph_node.outputs = {rows_out};
    return graph;
}

void check_program_row_space() {
    const rt::GroupedRowShape shape{4u, 2u};
    auto graph = grouped_graph(shape, true);
    check(rt::validate_primitive_graph(graph).ok(), "a declared grouped shape validates");

    auto program = rt::build_program(graph, rt::WeightTableView{}, rt::RowBucket::R16,
                                     rt::ExecutionClass::DECODE, rt::ProgramBuildOptions{});
    if (!program.ok()) {
        std::printf("      %s\n", program.status().message().c_str());
        check(false, "program build for grouped rows");
        return;
    }
    // bucket R16, experts 4, top_k 2: 32 slots over at most 6 groups, so 96 group rows.
    const uint32_t capacity = rt::grouped_row_capacity(16u, shape);
    check(capacity == 96u, "16 tokens with 4 experts and top_k 2 hold 96 group rows");

    const auto& values = program.value().values;
    uint32_t grouped_values = 0;
    uint32_t at_capacity = 0;
    for (const auto& b : values) {
        if (b.row_domain != rt::ValueRowDomain::GROUPED_ROWS) continue;
        ++grouped_values;
        if (b.feature_count != 8u || b.dtype != rt::ValueDType::BF16) {
            check(false, "grouped row value keeps its features");
            continue;
        }
        const uint64_t rows = b.bytes / (static_cast<uint64_t>(b.row_stride) * 2u);
        if (rows != capacity) {
            std::printf("      value %u rows %llu expected %u\n", b.logical_value.id,
                        static_cast<unsigned long long>(rows), capacity);
            continue;
        }
        if (b.storage != rt::ValueStorage::EXTERNAL_INPUT &&
            b.storage != rt::ValueStorage::EXTERNAL_OUTPUT) {
            check(false, "grouped row test graph binds an external input and output");
            continue;
        }
        ++at_capacity;
    }
    check(grouped_values == 2u && at_capacity == 2u,
          "both grouped row values are bound at capacity");

    auto undeclared = grouped_graph(shape, false);
    check(!rt::validate_primitive_graph(undeclared).ok(),
          "grouped row values without a declared shape are rejected");
    check(!rt::build_program(undeclared, rt::WeightTableView{}, rt::RowBucket::R16,
                             rt::ExecutionClass::DECODE, rt::ProgramBuildOptions{})
               .ok(),
          "program build refuses grouped rows without a shape");

    auto token_only = rt::PrimitiveGraph{};
    const auto tin = token_only.alloc_value(8, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const auto tout =
        token_only.alloc_value(8, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    token_only.external_inputs.push_back(tin);
    token_only.external_outputs.push_back(tout);
    rt::SiLUNode n{};
    n.shape = rt::RowwiseShapeKey{8};
    auto& gn = token_only.add_node(n);
    gn.inputs = {tin};
    gn.outputs = {tout};
    auto plain = rt::build_program(token_only, rt::WeightTableView{}, rt::RowBucket::R16,
                                   rt::ExecutionClass::DECODE, rt::ProgramBuildOptions{});
    if (!plain.ok()) {
        check(false, "a token-rows program still builds without a grouped shape");
        return;
    }
    bool any_grouped = false;
    for (const auto& b : plain.value().values) {
        if (b.row_domain == rt::ValueRowDomain::GROUPED_ROWS) any_grouped = true;
    }
    check(!any_grouped, "a dense graph gains no grouped row space");
}

}  // namespace

int main() {
    check_capacity_contract();
    check_program_row_space();
    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed, failed);
    return failed == 0 ? 0 : 1;
}
