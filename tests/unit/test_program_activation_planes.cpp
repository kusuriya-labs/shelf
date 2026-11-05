#include <shelf/quantization/quantization_types.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>
#include <shelf/weights/matrix_weight.h>

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

constexpr std::uint32_t kFeatures = 128u;
constexpr std::uint32_t kExperts = 4u;
constexpr std::uint32_t kTopK = 2u;

rt::WeightSlot mxfp4_matrix(std::uint32_t rows, std::uint32_t cols, std::uint32_t experts) {
    rt::WeightSlot s{};
    s.rows = rows;
    s.cols = cols;
    s.k_padded = cols;
    s.encoding = static_cast<std::uint8_t>(shelf::weights::MatrixEncoding::Mxfp4);
    s.compute_spec = static_cast<std::uint8_t>(
        shelf::quantization::ComputeSpecId::MXFP4_W4A8_F32_BF16);
    s.codes = reinterpret_cast<const void*>(0x1000);
    s.scales = reinterpret_cast<const void*>(0x2000);
    s.row_ref = reinterpret_cast<const void*>(0x3000);
    s.codes_row_stride_bytes = (cols / 32u) * 256u;
    s.scale_row_stride_bytes = cols / 32u;
    s.weight_scale_group = 32u;
    s.preshuffled = true;
    s.experts = experts;
    if (experts != 0u) {
        s.codes_batch_stride_bytes =
            (static_cast<std::uint64_t>(rows) / 16u) * (cols / 32u) * 256u;
        s.scales_batch_stride_bytes = static_cast<std::uint64_t>(rows) * (cols / 32u);
    }
    return s;
}

int run() {
    rt::PrimitiveGraph graph;
    graph.grouped_row_shape = rt::GroupedRowShape{kExperts, kTopK};

    const rt::ValueId input =
        graph.alloc_value(kFeatures, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const rt::ValueId dense =
        graph.alloc_value(kFeatures, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    {
        rt::LinearNode n;
        n.shape = rt::MatrixwiseShapeKey{kFeatures, kFeatures};
        n.weight_index = 0u;
        n.out_dtype = rt::ValueDType::BF16;
        auto& node = graph.add_node(rt::PrimitiveNode{n});
        node.inputs = {input};
        node.outputs = {dense};
    }
    const rt::ValueId grouped =
        graph.alloc_value(kFeatures, rt::ValueDType::BF16, rt::ValueRowDomain::GROUPED_ROWS);
    {
        rt::MoeGroupedLinearNode n;
        n.shape = rt::MatrixwiseShapeKey{kFeatures, kFeatures};
        n.weight_index = 1u;
        n.out_dtype = rt::ValueDType::BF16;
        auto& node = graph.add_node(rt::PrimitiveNode{n});
        node.inputs = {input};
        node.outputs = {grouped};
    }
    graph.external_inputs = {input};
    graph.external_outputs = {dense, grouped};

    std::vector<rt::WeightSlot> weights = {
        mxfp4_matrix(kFeatures, kFeatures, 0u),
        mxfp4_matrix(kFeatures, kFeatures, kExperts)};
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{kFeatures, kFeatures});
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{kFeatures, kFeatures});

    const rt::WeightTableView view{weights.data(), weights.size()};
    auto built = rt::build_program(graph, view, rt::RowBucket::R16,
                                   rt::ExecutionClass::DECODE,
                                   rt::ProgramBuildOptions{});
    if (!built.ok()) {
        std::printf("FAIL build_program: %s\n", built.status().message().c_str());
        return 1;
    }
    const rt::Program& program = built.value();

    std::vector<std::uint32_t> quant_rows;
    std::vector<std::uint32_t> quant_slots;
    for (const auto& b : program.dispatches) {
        if (b.kernel_id != rt::KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF) continue;
        if (b.input_slots[0] != input.id) continue;
        quant_rows.push_back(b.activation_max_rows);
        quant_slots.push_back(b.workspace_slot);
    }
    check(quant_rows.size() == 2u,
          "the token and grouped consumers each quantize the shared input");
    if (quant_rows.size() == 2u) {
        check(quant_rows[0] != quant_rows[1],
              "the token plane and the grouped plane have different row capacities");
        check(quant_slots[0] != quant_slots[1],
              "the token plane and the grouped plane have different workspaces");
    }
    return failed == 0 ? 0 : 1;
}

}  // namespace

int main() {
    const int rc = run();
    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return rc;
}
