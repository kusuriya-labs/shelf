#include <shelf/runtime/program/program.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/quantization/quantization_types.h>

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

const rt::WorkspaceRange* workspace_range_for(
    const rt::Program& program,
    rt::ValueId value) {
    const auto* binding = program.find_value(value);
    if (binding == nullptr) {
        return nullptr;
    }
    if (binding->storage != rt::ValueStorage::WORKSPACE) {
        return nullptr;
    }
    if (binding->slot >= program.workspace.ranges.size()) {
        return nullptr;
    }
    return &program.workspace.ranges[binding->slot];
}

}

int main() {
    rt::PrimitiveGraph graph;

    const auto input =
        graph.alloc_value(64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const auto linear0_out =
        graph.alloc_value(64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const auto silu_out =
        graph.alloc_value(64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const auto linear1_out =
        graph.alloc_value(64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);

    graph.external_inputs.push_back(input);
    graph.external_outputs.push_back(linear1_out);

    const rt::MatrixwiseShapeKey linear_shape{
        .output_features = 64,
        .input_features = 64,
    };

    const uint32_t weight0 = graph.register_weight(linear_shape);
    const uint32_t weight1 = graph.register_weight(linear_shape);

    {
        rt::LinearNode node{};
        node.shape = linear_shape;
        node.weight_index = weight0;
        node.out_dtype = rt::ValueDType::BF16;
        auto& graph_node = graph.add_node(node);
        graph_node.inputs = {input};
        graph_node.outputs = {linear0_out};
    }
    {
        rt::SiLUNode node{};
        node.shape = rt::RowwiseShapeKey{64};
        auto& graph_node = graph.add_node(node);
        graph_node.inputs = {linear0_out};
        graph_node.outputs = {silu_out};
    }
    {
        rt::LinearNode node{};
        node.shape = linear_shape;
        node.weight_index = weight1;
        node.out_dtype = rt::ValueDType::BF16;
        auto& graph_node = graph.add_node(node);
        graph_node.inputs = {silu_out};
        graph_node.outputs = {linear1_out};
    }

    auto graph_status = rt::validate_primitive_graph(graph);
    if (!graph_status.ok()) {
        std::printf("FAIL: graph validation: %s\n", graph_status.message().c_str());
        return 1;
    }

    std::array<rt::WeightSlot, 2> weights{};
    for (auto& weight : weights) {
        weight.rows = 64;
        weight.cols = 64;
        weight.k_padded = 64;
        weight.encoding = static_cast<uint8_t>(shelf::weights::MatrixEncoding::Fp8Finegrained);
        weight.compute_spec = static_cast<uint8_t>(
            shelf::quantization::ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16);
    }

    const rt::WeightTableView weight_view{weights.data(), weights.size()};

    rt::ProgramBuildOptions options{};
    options.validate_workspace = true;
    options.no_workspace_reuse = false;
    options.max_token_rows = 16;
    options.max_output_rows = 16;

    auto program_result =
        rt::build_program(graph, weight_view, rt::RowBucket::R16,
                          rt::ExecutionClass::DECODE, options);
    if (!program_result.ok()) {
        std::printf("FAIL: build_program: %s\n",
                    program_result.status().message().c_str());
        return 1;
    }

    auto program = program_result.release();

    check(program.dispatches.size() == 5, "expected five physical dispatches");

    if (program.dispatches.size() == 5) {
        check(program.dispatches[0].kernel_id == rt::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED,
              "dispatch 0 activation quantize");
        check(program.dispatches[1].kernel_id == rt::KernelId::LINEAR_FP8_FINEGRAINED,
              "dispatch 1 first linear");
        check(program.dispatches[2].kernel_id == rt::KernelId::SILU,
              "dispatch 2 silu");
        check(program.dispatches[3].kernel_id == rt::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED,
              "dispatch 3 activation quantize");
        check(program.dispatches[4].kernel_id == rt::KernelId::LINEAR_FP8_FINEGRAINED,
              "dispatch 4 second linear");
    }

    const auto* linear0_range = workspace_range_for(program, linear0_out);
    const auto* silu_range = workspace_range_for(program, silu_out);

    check(linear0_range != nullptr, "first linear output has workspace range");
    check(silu_range != nullptr, "silu output has workspace range");

    if (linear0_range != nullptr) {
        check(linear0_range->first_command == 1,
              "first linear output definition is dispatch 1");
        check(linear0_range->last_command == 2,
              "first linear output last use is dispatch 2");
    }
    if (silu_range != nullptr) {
        check(silu_range->first_command == 2,
              "silu output definition is dispatch 2");
        check(silu_range->last_command == 3,
              "silu output last use is dispatch 3");
    }

    auto workspace_status = program.workspace.validate();
    check(workspace_status.ok(), "workspace layout validates");

    const auto* final_binding = program.find_value(linear1_out);
    check(final_binding != nullptr, "external output binding exists");
    if (final_binding != nullptr) {
        check(final_binding->storage == rt::ValueStorage::EXTERNAL_OUTPUT,
              "final value is external output");
    }

    {
        rt::PrimitiveGraph norm_graph;
        const auto x = norm_graph.alloc_value(128, rt::ValueDType::BF16,
                                              rt::ValueRowDomain::TOKEN_ROWS);
        const auto y = norm_graph.alloc_value(128, rt::ValueDType::BF16,
                                              rt::ValueRowDomain::TOKEN_ROWS);
        norm_graph.external_inputs.push_back(x);
        norm_graph.external_outputs.push_back(y);
        rt::RmsNormNode norm{};
        norm.shape = rt::RowwiseShapeKey{128};
        norm.group_size = 128;
        norm.parameter_index = 0;
        norm.weight_mode = rt::RmsNormWeightMode::OFFSET_BF16;
        norm.weight_offset = 0.2f;
        auto& node = norm_graph.add_node(norm);
        node.inputs = {x};
        node.outputs = {y};
        auto result = rt::build_program(norm_graph, {}, rt::RowBucket::R16,
                                        rt::ExecutionClass::DECODE, options);
        check(result.ok(), "offset RMSNorm program builds");
        if (result.ok() && result.value().dispatches.size() == 1u) {
            const auto& binding = result.value().dispatches[0];
            check(binding.flags == static_cast<uint32_t>(rt::RmsNormWeightMode::OFFSET_BF16) &&
                      binding.scalar_b == 0.2f,
                  "offset RMSNorm mode and offset reach dispatch");
        } else {
            check(false, "offset RMSNorm emits one dispatch");
        }
    }

    {
        rt::PrimitiveGraph swiglu_graph;
        const auto input = swiglu_graph.alloc_value(
            64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
        const auto gate = swiglu_graph.alloc_value(
            64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
        const auto up = swiglu_graph.alloc_value(
            64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
        const auto swiglu_out = swiglu_graph.alloc_value(
            64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
        const auto out = swiglu_graph.alloc_value(
            64, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
        swiglu_graph.external_inputs.push_back(input);
        swiglu_graph.external_outputs.push_back(out);

        const rt::MatrixwiseShapeKey shape{64, 64};
        const uint32_t gate_weight = swiglu_graph.register_weight(shape);
        const uint32_t up_weight = swiglu_graph.register_weight(shape);
        const uint32_t ffn_weight = swiglu_graph.register_weight(shape);
        for (uint32_t weight : {gate_weight, up_weight}) {
            rt::LinearNode node{};
            node.shape = shape;
            node.weight_index = weight;
            node.out_dtype = rt::ValueDType::BF16;
            auto& graph_node = swiglu_graph.add_node(node);
            graph_node.inputs = {input};
            graph_node.outputs = {weight == gate_weight ? gate : up};
        }
        {
            rt::SwiGluNode node{};
            node.shape = rt::RowwiseShapeKey{64};
            auto& graph_node = swiglu_graph.add_node(node);
            graph_node.inputs = {gate, up};
            graph_node.outputs = {swiglu_out};
        }
        {
            rt::LinearNode node{};
            node.shape = shape;
            node.weight_index = ffn_weight;
            node.out_dtype = rt::ValueDType::BF16;
            auto& graph_node = swiglu_graph.add_node(node);
            graph_node.inputs = {swiglu_out};
            graph_node.outputs = {out};
        }

        auto graph_status = rt::validate_primitive_graph(swiglu_graph);
        check(graph_status.ok(), "swiglu graph validation");
        if (!graph_status.ok()) {
            std::printf("FAIL: swiglu graph validation: %s\n",
                        graph_status.message().c_str());
        } else {
            std::array<rt::WeightSlot, 3> weights{};
            weights[gate_weight].rows = 64;
            weights[gate_weight].cols = 64;
            weights[gate_weight].k_padded = 64;
            weights[up_weight] = weights[gate_weight];
            weights[ffn_weight] = weights[gate_weight];
            weights[ffn_weight].encoding = static_cast<uint8_t>(
                shelf::weights::MatrixEncoding::Fp8Finegrained);
            weights[ffn_weight].compute_spec = static_cast<uint8_t>(
                shelf::quantization::ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16);

            const rt::WeightTableView view{weights.data(), weights.size()};
            auto result = rt::build_program(swiglu_graph, view, rt::RowBucket::R16,
                                            rt::ExecutionClass::DECODE, options);
            check(result.ok(), "swiglu program builds");
            if (result.ok()) {
                auto program = result.release();
                check(program.dispatches.size() == 5,
                      "swiglu emits five physical dispatches");
                if (program.dispatches.size() == 5) {
                    check(program.dispatches[2].kernel_id == rt::KernelId::SWIGLU,
                          "swiglu is dispatch 2");
                    const uint32_t slot = program.dispatches[3].workspace_slot;
                    check(slot != rt::kNoWorkspace, "swiglu activation has workspace slot");
                    if (slot != rt::kNoWorkspace && slot < program.workspace.ranges.size()) {
                        const auto& range = program.workspace.ranges[slot];
                        check(range.first_command <= 2u,
                              "activation workspace starts no later than swiglu");
                        check(range.last_command >= 4u,
                              "activation workspace lives through fp8 linear");
                    }
                }
            }
        }
    }

    std::printf("test_program_workspace_lifetime: passed=%d failed=%d\n",
                passed, failed);
    return failed == 0 ? 0 : 1;
}
