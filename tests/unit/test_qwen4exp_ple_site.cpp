#include <shelf/models/qwen4exp/model/ple_lowering.h>
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

constexpr std::uint32_t kHc = 4u;
constexpr std::uint32_t kHidden = 128u;
constexpr std::uint32_t kEmbed = 160u;
constexpr std::uint32_t kWidth = kHc * kHidden;

rt::WeightSlot bf16_slot(std::uint32_t rows, std::uint32_t cols) {
    rt::WeightSlot slot{};
    slot.rows = rows;
    slot.cols = cols;
    slot.k_padded = cols;
    slot.encoding = static_cast<std::uint8_t>(shelf::weights::MatrixEncoding::Bf16);
    slot.compute_spec =
        static_cast<std::uint8_t>(shelf::quantization::ComputeSpecId::BF16_F32_BF16);
    slot.device_ptr = reinterpret_cast<const void*>(0x1000);
    slot.bytes = static_cast<std::uint64_t>(rows) * cols * 2u;
    return slot;
}

}  // namespace

int main() {
    rt::PrimitiveGraph graph;
    graph.grouped_row_shape = rt::GroupedRowShape{0u, 0u};
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{kWidth, kEmbed});
    graph.weight_table.push_back(rt::MatrixwiseShapeKey{kHidden, kEmbed});
    std::vector<rt::WeightSlot> weights = {bf16_slot(kWidth, kEmbed),
                                           bf16_slot(kHidden, kEmbed)};

    shelf::weights::MatrixWeight key;
    key.rows = kWidth;
    key.cols = kEmbed;
    key.k_padded = kEmbed;
    shelf::weights::MatrixWeight value;
    value.rows = kHidden;
    value.cols = kEmbed;
    value.k_padded = kEmbed;

    const rt::ValueId hyper = graph.alloc_value(kWidth);
    const rt::ValueId embeddings = graph.alloc_value(kEmbed);
    graph.external_inputs = {hyper, embeddings};

    shelf::qwen_hybrid::Emitter em{graph};
    shelf::qwen4exp::Qwen4ExpPleLoweringGeometry geometry;
    geometry.hc = kHc;
    geometry.hidden = kHidden;
    geometry.embed_dim = kEmbed;
    geometry.eps = 1e-6f;
    geometry.norm_key_parameter_index = 0u;
    geometry.norm_query_parameter_index = 1u;
    geometry.norm_conv_parameter_index = 2u;
    geometry.conv_weight_parameter_index = 3u;
    geometry.state_index = 0u;
    shelf::qwen4exp::Qwen4ExpPleLoweringWeights weights_in;
    weights_in.key_proj = &key;
    weights_in.value_proj = &value;
    shelf::qwen4exp::Qwen4ExpPleLoweringIndices indices{0u, 1u};

    auto emitted = shelf::qwen4exp::emit_qwen4exp_ple(
        em, geometry, weights_in, indices, hyper, embeddings, "test.");
    if (!emitted.ok()) {
        std::printf("FAIL emit: %s\n", emitted.status().message().c_str());
        return 1;
    }
    graph.external_outputs = {emitted.value()};

    auto status = rt::validate_primitive_graph(graph);
    check(status.ok(), "the PLE graph validates");
    if (!status.ok()) {
        std::printf("      %s\n", status.message().c_str());
        return 1;
    }

    const rt::WeightTableView view{weights.data(), weights.size()};
    auto built = rt::build_program(graph, view, rt::RowBucket::R16,
                                   rt::ExecutionClass::DECODE,
                                   rt::ProgramBuildOptions{});
    check(built.ok(), "the PLE graph lowers");
    if (!built.ok()) {
        std::printf("      %s\n", built.status().message().c_str());
        return 1;
    }
    std::string sequence;
    for (std::size_t i = 0; i < built.value().dispatches.size(); ++i) {
        sequence += rt::kernel_to_name(built.value().dispatches[i].kernel_id);
        if (i + 1 != built.value().dispatches.size()) sequence += ",";
    }
    const std::string want =
        "LINEAR_BF16,RMS_NORM,RMS_NORM,PLE_STREAM_DOT,SCALE,PLE_GATE,LINEAR_BF16,"
        "PLE_BROADCAST_MUL,RMS_NORM,PLE_CONV,RESIDUAL_ADD,RESIDUAL_ADD";
    check(sequence == want, "the PLE injection lowers in the expected order");
    if (sequence != want) std::printf("      got %s\n", sequence.c_str());

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
