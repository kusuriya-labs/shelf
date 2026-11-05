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

constexpr uint32_t kHc = 4u;
constexpr uint32_t kHidden = 128u;
constexpr uint32_t kWidth = kHc * kHidden;
constexpr uint32_t kBucketTokens = 16u;

struct Graph {
    rt::PrimitiveGraph g;
};

Graph build_graph(bool bad_features = false) {
    Graph out;
    out.g.grouped_row_shape = rt::GroupedRowShape{0u, 0u};

    const rt::ValueId gate = out.g.alloc_value(kWidth);
    const rt::ValueId normed = out.g.alloc_value(kWidth);
    const rt::ValueId raw = out.g.alloc_value(kWidth);
    const rt::ValueId y = out.g.alloc_value(kHidden);
    const rt::ValueId injection = out.g.alloc_value(kHc);
    out.g.external_inputs = {gate, normed, raw, y, injection};

    const rt::ValueId mixed = out.g.alloc_value(kHidden);
    {
        rt::HyperMixNode n;
        n.shape = rt::RowwiseShapeKey{kHidden};
        n.hc = kHc;
        n.hidden = bad_features ? kHidden + 1u : kHidden;
        auto& node = out.g.add_node(rt::PrimitiveNode{n});
        node.inputs = {gate, normed};
        node.outputs = {mixed};
    }

    const rt::ValueId injected = out.g.alloc_value(kWidth);
    {
        rt::HyperApplyNode n;
        n.shape = rt::RowwiseShapeKey{kWidth};
        n.hc = kHc;
        n.hidden = kHidden;
        auto& node = out.g.add_node(rt::PrimitiveNode{n});
        node.inputs = {raw, y, injection};
        node.outputs = {injected};
    }

    out.g.external_outputs = {mixed, injected};
    return out;
}

std::string kernel_sequence(const rt::Program& program) {
    std::string out;
    for (std::size_t i = 0; i < program.dispatches.size(); ++i) {
        out += rt::kernel_to_name(program.dispatches[i].kernel_id);
        if (i + 1 != program.dispatches.size()) out += ",";
    }
    return out;
}

}  // namespace

int main() {
    {
        auto graph = build_graph();
        auto status = rt::validate_primitive_graph(graph.g);
        check(status.ok(), "the hyper-connection graph validates");
        const std::vector<rt::WeightSlot> weights;
        const rt::WeightTableView view{weights.data(), weights.size()};
        auto built = rt::build_program(graph.g, view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE,
                                       rt::ProgramBuildOptions{});
        check(built.ok(), "the hyper-connection graph lowers");
        if (built.ok()) {
            const std::string got = kernel_sequence(built.value());
            check(got == "HYPER_MIX,HYPER_APPLY",
                  "the hyper-connection graph lowers to mix then apply");
            if (got != "HYPER_MIX,HYPER_APPLY") std::printf("      got %s\n", got.c_str());
        }
    }
    {
        auto graph = build_graph(true);
        check(!rt::validate_primitive_graph(graph.g).ok(),
              "hyper mix with the wrong hc * hidden width is rejected");
    }

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
