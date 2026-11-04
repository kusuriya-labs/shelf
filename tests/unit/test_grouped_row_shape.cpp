#include <shelf/runtime/graph/grouped_row_shape.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>

#include <algorithm>
#include <array>
#include <cstdio>
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

// Values from the plan contract of docs/developer/qwen35moe.md §6, computed by hand.
void check_capacity_contract() {
    check(rt::grouped_slot_count(1, {256, 8}) == 8u, "one token selects 8 slots");
    check(rt::grouped_row_tile_count(1, {256, 8}) == 8u,
          "8 slots spread over 8 groups when every expert may win");
    check(rt::grouped_row_capacity(1, {256, 8}) == 128u,
          "one token needs 128 group rows: 16 rows per group");
    check(rt::grouped_slot_count(512, {256, 8}) == 4096u, "512 tokens select 4096 slots");
    check(rt::grouped_row_tile_count(512, {256, 8}) == 496u,
          "4096 slots need at most 496 groups (256 experts waste at most 15 slots each)");
    check(rt::grouped_row_capacity(512, {256, 8}) == 7936u,
          "496 groups hold at most 7936 group rows");
    check(rt::grouped_slot_tile_count(1) == 1u, "one slot tile for one token");
    check(rt::grouped_slot_tile_count(64) == 1u, "64 tokens fit one slot tile");
    check(rt::grouped_slot_tile_count(65) == 2u, "65 tokens need two slot tiles");

    check(rt::grouped_row_tile_count(4096, {256, 8}) == 2288u,
          "the 4096 token bucket has 2288 groups of capacity");
    check(rt::grouped_row_capacity(4096, {256, 8}) == 36608u,
          "the 4096 token bucket has 36608 group rows of capacity");

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

// The bound must be both sound (it covers every routing outcome) and tight (it is the exact
// maximum over routings). The exact maximum is computed by dynamic programming over small
// shapes: distribute `slots` slots over `experts` experts, at most `rows` per expert because a
// token never selects the same expert twice, counting ceil(c_e / tile) per expert.
uint32_t exact_max_tiles(uint32_t experts, uint32_t slots, uint32_t rows) {
    const uint32_t B = rt::kGroupedRowTile;
    std::vector<int32_t> cur(slots + 1u, -1), nxt(slots + 1u, -1);
    cur[0] = 0;
    for (uint32_t e = 0; e < experts; ++e) {
        std::fill(nxt.begin(), nxt.end(), -1);
        for (uint32_t used = 0; used <= slots; ++used) {
            if (cur[used] < 0) continue;
            const uint32_t room = rows < slots - used ? rows : slots - used;
            for (uint32_t take = 0; take <= room; ++take) {
                const uint32_t to = used + take;
                const int32_t value =
                    cur[used] + static_cast<int32_t>((take + B - 1u) / B);
                if (value > nxt[to]) nxt[to] = value;
            }
        }
        cur.swap(nxt);
    }
    return cur[slots] < 0 ? 0u : static_cast<uint32_t>(cur[slots]);
}

void check_bound_is_exact() {
    const std::array<std::array<uint32_t, 3>, 7> cases = {{
        {{4u, 1u, 4u}},     {{6u, 3u, 8u}},    {{8u, 1u, 16u}},
        {{8u, 2u, 16u}},    {{16u, 4u, 32u}},  {{32u, 8u, 32u}},
        {{256u, 8u, 64u}},
    }};
    for (const auto& c : cases) {
        const uint32_t experts = c[0];
        const uint32_t top_k = c[1];
        const uint32_t rows = c[2];
        const uint32_t slots = rows * top_k;
        const uint32_t bound = rt::grouped_row_tile_count(rows, {experts, top_k});
        const uint32_t exact = exact_max_tiles(experts, slots, rows);
        char msg[96];
        std::snprintf(msg, sizeof(msg),
                      "group bound is the exact maximum (experts %u top_k %u rows %u: %u vs %u)",
                      experts, top_k, rows, bound, exact);
        check(bound == exact, msg);
    }
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
    // bucket R16, experts 4, top_k 2: 32 slots over 4 experts, and a token never repeats an
    // expert, so each expert holds at most 16 slots = 1 group: 4 groups, 64 group rows.
    const uint32_t capacity = rt::grouped_row_capacity(16u, shape);
    check(capacity == 64u, "16 tokens with 4 experts and top_k 2 hold 64 group rows");

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
    check_bound_is_exact();
    check_program_row_space();
    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed, failed);
    return failed == 0 ? 0 : 1;
}
