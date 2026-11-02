#include <shelf/quantization/quantization_types.h>
#include <shelf/runtime/execution/execution_types.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <shelf/runtime/graph/shape_spec.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/runtime/program/kernel_id.h>
#include <shelf/runtime/program/program.h>
#include <shelf/weights/matrix_weight.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace rt = ::ps::runtime;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", message.c_str());
    }
}

const char* kernel_name(rt::KernelId id) {
    return rt::kernel_to_name(id);
}

rt::WeightSlot make_slot(
    uint8_t encoding, uint8_t compute_spec, uint32_t rows, uint32_t k, uint32_t k_padded) {
    rt::WeightSlot slot;
    slot.rows = rows;
    slot.cols = k;
    slot.k_padded = k_padded;
    slot.encoding = encoding;
    slot.compute_spec = compute_spec;
    slot.codes = reinterpret_cast<const void*>(0x1000);
    slot.scales = reinterpret_cast<const void*>(0x2000);
    slot.preshuffled = true;
    return slot;
}

struct Case {
    const char* tag = "";
    uint8_t encoding = 0;
    uint8_t compute_spec = 0;
    rt::KernelId linear_kernel = rt::KernelId::LINEAR_BF16;
    bool activation_quant = false;
    uint32_t k = 0;
    uint32_t k_padded = 0;
};

void run_case(const Case& c) {
    const uint32_t k_out = 1024;
    const uint32_t max_rows = 128;

    rt::PrimitiveGraph graph;
    const rt::ValueId input =
        graph.alloc_value(c.k, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const rt::ValueId output =
        graph.alloc_value(k_out, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    rt::LinearNode linear;
    linear.shape = rt::MatrixwiseShapeKey{k_out, c.k};
    linear.weight_index = 0;
    linear.out_dtype = rt::ValueDType::BF16;
    auto& node = graph.add_node(rt::PrimitiveNode{linear});
    node.inputs = {input};
    node.outputs = {output};
    node.debug_name = "linear";
    graph.external_inputs = {input};
    graph.external_outputs = {output};
    graph.register_weight(linear.shape);

    rt::WeightSlot slot =
        make_slot(c.encoding, c.compute_spec, k_out, c.k, c.k_padded);
    rt::WeightTableView table;
    table.slots = &slot;
    table.count = 1;

    rt::ProgramBuildOptions options;
    options.max_token_rows = max_rows;

    auto built =
        rt::build_program(graph, table, rt::RowBucket::R128, rt::ExecutionClass::DECODE, options);
    check(built.ok(), std::string(c.tag) + ": build_program");
    if (!built.ok()) return;
    const rt::Program& program = built.value();

    if (!c.activation_quant) {
        check(program.dispatches.size() == 1u, std::string(c.tag) + ": single dispatch");
        if (program.dispatches.size() != 1u) return;
        const rt::DispatchBinding& linear_dispatch = program.dispatches[0];
        check(linear_dispatch.kernel_id == rt::KernelId::LINEAR_BF16,
              std::string(c.tag) + ": LINEAR_BF16");
        check(linear_dispatch.input_count == 1u, std::string(c.tag) + ": reads its input");
        check(linear_dispatch.activation_kp == 0u, std::string(c.tag) + ": no activation k_padded");
        std::printf(
            "fingerprint %s: dispatch=[%s] act_ws=none\n",
            c.tag, kernel_name(linear_dispatch.kernel_id));
        return;
    }

    check(program.dispatches.size() == 2u, std::string(c.tag) + ": quantize + linear");
    if (program.dispatches.size() != 2u) return;

    const rt::DispatchBinding& quantize = program.dispatches[0];
    const rt::DispatchBinding& linear_dispatch = program.dispatches[1];

    check(quantize.kernel_id == rt::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED,
          std::string(c.tag) + ": ACTIVATION_QUANTIZE_FP8_FINEGRAINED");
    check(quantize.group_size == 32u, std::string(c.tag) + ": quantize group 32");
    check(quantize.activation_kp == c.k_padded, std::string(c.tag) + ": quantize k_padded");
    check(quantize.activation_max_rows == max_rows, std::string(c.tag) + ": quantize rows");
    check(quantize.input_count == 1u, std::string(c.tag) + ": quantize reads its input");

    check(linear_dispatch.kernel_id == c.linear_kernel, std::string(c.tag) + ": linear kernel");
    check(linear_dispatch.compute_spec == c.compute_spec, std::string(c.tag) + ": compute spec");
    check(linear_dispatch.input_count == 0u, std::string(c.tag) + ": linear takes no input slot");
    check(linear_dispatch.activation_kp == c.k_padded, std::string(c.tag) + ": linear k_padded");
    check(linear_dispatch.activation_max_rows == max_rows, std::string(c.tag) + ": linear rows");

    const rt::Int8ActivationWorkspaceLayout layout =
        rt::Int8ActivationWorkspaceLayout::make(c.k_padded, max_rows);
    check(layout.k_padded == c.k_padded, std::string(c.tag) + ": layout k_padded");
    check(quantize.activation_code_stride == layout.code_row_stride_bytes,
          std::string(c.tag) + ": quantize code stride");
    check(quantize.activation_scale_stride == layout.scale_row_stride_bytes,
          std::string(c.tag) + ": quantize scale stride");
    check(linear_dispatch.activation_code_stride == layout.code_row_stride_bytes,
          std::string(c.tag) + ": linear code stride");
    check(linear_dispatch.activation_scale_stride == layout.scale_row_stride_bytes,
          std::string(c.tag) + ": linear scale stride");

    check(quantize.workspace_slot != rt::kNoWorkspace,
          std::string(c.tag) + ": quantize has a workspace slot");
    check(linear_dispatch.workspace_slot == quantize.workspace_slot,
          std::string(c.tag) + ": linear shares the activation workspace");
    if (quantize.workspace_slot >= program.workspace.ranges.size()) return;
    const rt::WorkspaceRange& range = program.workspace.ranges[quantize.workspace_slot];
    check(range.bytes == layout.total_bytes, std::string(c.tag) + ": workspace bytes");
    check(range.offset % 16u == 0u, std::string(c.tag) + ": workspace offset alignment");
    check(program.workspace.validate().ok(), std::string(c.tag) + ": workspace validates");

    std::printf(
        "fingerprint %s: dispatch=[%s,%s] kp=%u code_stride=%u scale_stride=%u "
        "ws_bytes=%zu ws_offset=%zu\n",
        c.tag, kernel_name(quantize.kernel_id), kernel_name(linear_dispatch.kernel_id),
        linear_dispatch.activation_kp, linear_dispatch.activation_code_stride,
        linear_dispatch.activation_scale_stride, range.bytes, range.offset);
}

void run_shared_input_case() {
    const uint32_t k = 2560;
    const uint32_t k_out = 1024;
    const uint32_t max_rows = 128;

    rt::PrimitiveGraph graph;
    const rt::ValueId input =
        graph.alloc_value(k, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const rt::ValueId out0 =
        graph.alloc_value(k_out, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const rt::ValueId out1 =
        graph.alloc_value(k_out, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);

    rt::LinearNode linear;
    linear.shape = rt::MatrixwiseShapeKey{k_out, k};
    linear.out_dtype = rt::ValueDType::BF16;

    linear.weight_index = 0;
    auto& first = graph.add_node(rt::PrimitiveNode{linear});
    first.inputs = {input};
    first.outputs = {out0};
    first.debug_name = "linear0";

    linear.weight_index = 1;
    auto& second = graph.add_node(rt::PrimitiveNode{linear});
    second.inputs = {input};
    second.outputs = {out1};
    second.debug_name = "linear1";

    graph.external_inputs = {input};
    graph.external_outputs = {out0, out1};
    graph.register_weight(linear.shape);
    graph.register_weight(linear.shape);

    std::array<rt::WeightSlot, 2> slots{};
    for (rt::WeightSlot& slot : slots) {
        slot = make_slot(
            static_cast<uint8_t>(ps::weights::MatrixEncoding::Fp8Finegrained),
            static_cast<uint8_t>(ps::quantization::ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16),
            k_out, k, k);
    }
    rt::WeightTableView table;
    table.slots = slots.data();
    table.count = slots.size();

    rt::ProgramBuildOptions options;
    options.max_token_rows = max_rows;

    auto built =
        rt::build_program(graph, table, rt::RowBucket::R128, rt::ExecutionClass::DECODE, options);
    check(built.ok(), "shared input: build_program");
    if (!built.ok()) return;
    const rt::Program& program = built.value();

    check(program.dispatches.size() == 3u, "shared input: one quantize for two linears");
    if (program.dispatches.size() != 3u) return;

    uint32_t quantize_count = 0;
    uint32_t quantize_slot = rt::kNoWorkspace;
    for (const rt::DispatchBinding& dispatch : program.dispatches) {
        if (dispatch.kernel_id == rt::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED) {
            ++quantize_count;
            quantize_slot = dispatch.workspace_slot;
        }
    }
    check(quantize_count == 1u, "shared input: exactly one activation quantize dispatch");

    const rt::DispatchBinding* first_linear = nullptr;
    const rt::DispatchBinding* second_linear = nullptr;
    for (const rt::DispatchBinding& dispatch : program.dispatches) {
        if (dispatch.kernel_id != rt::KernelId::LINEAR_FP8_FINEGRAINED) continue;
        if (first_linear == nullptr) {
            first_linear = &dispatch;
        } else {
            second_linear = &dispatch;
        }
    }
    check(first_linear != nullptr && second_linear != nullptr,
          "shared input: both linears lowered");
    if (first_linear == nullptr || second_linear == nullptr) return;
    check(first_linear->workspace_slot == second_linear->workspace_slot,
          "shared input: linears share one workspace range");
    check(first_linear->workspace_slot == quantize_slot,
          "shared input: quantize writes the shared range");

    const rt::Int8ActivationWorkspaceLayout layout =
        rt::Int8ActivationWorkspaceLayout::make(k, max_rows);
    uint32_t matching_ranges = 0;
    for (const rt::WorkspaceRange& range : program.workspace.ranges) {
        if (range.bytes == layout.total_bytes) ++matching_ranges;
    }
    check(matching_ranges == 1u, "shared input: a single activation workspace range");
    check(program.workspace.validate().ok(), "shared input: workspace validates");

    std::printf(
        "fingerprint shared: dispatch=[%s,%s,%s] ws_bytes=%zu\n",
        kernel_name(program.dispatches[0].kernel_id),
        kernel_name(program.dispatches[1].kernel_id),
        kernel_name(program.dispatches[2].kernel_id), layout.total_bytes);
}

void run_shared_input_mismatch_case() {
    const uint32_t k = 2560u;
    const uint32_t k_other = 1280u;
    const uint32_t k_out = 1024u;

    rt::PrimitiveGraph graph;
    const rt::ValueId input =
        graph.alloc_value(k, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const rt::ValueId out0 =
        graph.alloc_value(k_out, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const rt::ValueId out1 =
        graph.alloc_value(k_out, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);

    rt::LinearNode linear;
    linear.out_dtype = rt::ValueDType::BF16;

    linear.shape = rt::MatrixwiseShapeKey{k_out, k};
    linear.weight_index = 0;
    auto& first = graph.add_node(rt::PrimitiveNode{linear});
    first.inputs = {input};
    first.outputs = {out0};
    first.debug_name = "linear0";

    linear.shape = rt::MatrixwiseShapeKey{k_out, k_other};
    linear.weight_index = 1;
    auto& second = graph.add_node(rt::PrimitiveNode{linear});
    second.inputs = {input};
    second.outputs = {out1};
    second.debug_name = "linear1";

    graph.external_inputs = {input};
    graph.external_outputs = {out0, out1};
    graph.register_weight(rt::MatrixwiseShapeKey{k_out, k});
    graph.register_weight(rt::MatrixwiseShapeKey{k_out, k_other});

    std::array<rt::WeightSlot, 2> slots{};
    slots[0] = make_slot(5u, 1u, k_out, k, k);
    slots[1] = make_slot(5u, 1u, k_out, k_other, k_other);
    rt::WeightTableView table;
    table.slots = slots.data();
    table.count = slots.size();

    rt::ProgramBuildOptions options;
    options.max_token_rows = 128;

    auto built =
        rt::build_program(graph, table, rt::RowBucket::R128, rt::ExecutionClass::DECODE, options);
    check(!built.ok(), "shared input mismatch: build fails");
    if (built.ok()) return;
    check(built.status().message().find("activation workspace K mismatch for shared input") !=
              std::string::npos,
          "shared input mismatch: message keeps the K mismatch contract");
}

ps::Result<rt::Program> build_single_linear(
    uint8_t encoding, uint8_t compute_spec, uint32_t k, uint64_t bytes) {
    const uint32_t k_out = 1024;

    rt::PrimitiveGraph graph;
    const rt::ValueId input =
        graph.alloc_value(k, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const rt::ValueId output =
        graph.alloc_value(k_out, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    rt::LinearNode linear;
    linear.shape = rt::MatrixwiseShapeKey{k_out, k};
    linear.weight_index = 0;
    linear.out_dtype = rt::ValueDType::BF16;
    auto& node = graph.add_node(rt::PrimitiveNode{linear});
    node.inputs = {input};
    node.outputs = {output};
    node.debug_name = "linear";
    graph.external_inputs = {input};
    graph.external_outputs = {output};
    graph.register_weight(linear.shape);

    rt::WeightSlot slot = make_slot(encoding, compute_spec, k_out, k, k);
    slot.bytes = bytes;
    rt::WeightTableView table;
    table.slots = &slot;
    table.count = 1;

    rt::ProgramBuildOptions options;
    options.max_token_rows = 128;

    return rt::build_program(
        graph, table, rt::RowBucket::R128, rt::ExecutionClass::DECODE, options);
}

void expect_build_failure(
    uint8_t encoding,
    uint8_t compute_spec,
    uint64_t bytes,
    const std::string& text,
    const std::string& tag) {
    auto built = build_single_linear(encoding, compute_spec, 2560u, bytes);
    check(!built.ok(), tag + ": build fails");
    if (built.ok()) return;
    check(built.status().message().find(text) != std::string::npos,
          tag + ": message keeps '" + text + "'");
}

void run_error_cases() {
    expect_build_failure(
        5u, 0u, 0u, "fp8 linear weight requires fp8 compute spec", "fp8 bf16 spec");
    expect_build_failure(
        0u, 1u, 0u, "bf16 linear weight requires bf16 compute spec", "bf16 fp8 spec");
    expect_build_failure(
        0u, 0u, 1024u * 2560u * 2u + 1u,
        "non bf16 linear weights are outside baseline physical coverage", "bf16 byte mismatch");
}

}

int main() {
    run_case(Case{"bf16", 0u, 0u, rt::KernelId::LINEAR_BF16, false, 2560u, 2560u});
    run_case(Case{"fp8", 5u, 1u, rt::KernelId::LINEAR_FP8_FINEGRAINED, true, 2560u, 2560u});
    run_case(Case{"fp8-k2500", 5u, 1u, rt::KernelId::LINEAR_FP8_FINEGRAINED, true, 2500u, 2560u});
    run_shared_input_case();
    run_shared_input_mismatch_case();
    run_error_cases();

    std::printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
