#include <shelf/models/common/qwen_hybrid/gdn_lowering.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>

#include <cstdio>
#include <string>
#include <vector>

namespace rt = shelf::runtime;
using shelf::qwen_hybrid::Emitter;
using shelf::qwen_hybrid::GdnLoweringGeometry;
using shelf::qwen_hybrid::GdnLoweringIndices;
using shelf::qwen_hybrid::GdnLoweringWeights;

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

constexpr std::uint32_t kHidden = 64u;
constexpr std::uint32_t kKeyHeads = 2u;
constexpr std::uint32_t kValueHeads = 4u;
constexpr std::uint32_t kKeyDim = 16u;
constexpr std::uint32_t kValueDim = 16u;
constexpr std::uint32_t kConvKernel = 4u;
constexpr std::uint32_t kQk = kKeyHeads * kKeyDim;
constexpr std::uint32_t kValue = kValueHeads * kValueDim;
constexpr std::uint32_t kConv = 2u * kQk + kValue;

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

std::string build(bool sigmoid_gate) {
    rt::PrimitiveGraph graph;
    graph.grouped_row_shape = rt::GroupedRowShape{0u, 0u};

    shelf::weights::MatrixWeight qkv;
    qkv.rows = kConv;
    qkv.cols = kHidden;
    qkv.k_padded = kHidden;
    shelf::weights::MatrixWeight z;
    z.rows = kValue;
    z.cols = kHidden;
    z.k_padded = kHidden;
    shelf::weights::MatrixWeight b;
    b.rows = kValueHeads;
    b.cols = kHidden;
    b.k_padded = kHidden;
    shelf::weights::MatrixWeight a = b;
    shelf::weights::MatrixWeight out;
    out.rows = kHidden;
    out.cols = kValue;
    out.k_padded = kValue;

    const std::uint32_t w_qkv = graph.register_weight({kConv, kHidden});
    const std::uint32_t w_z = graph.register_weight({kValue, kHidden});
    const std::uint32_t w_b = graph.register_weight({kValueHeads, kHidden});
    const std::uint32_t w_a = graph.register_weight({kValueHeads, kHidden});
    const std::uint32_t w_out = graph.register_weight({kHidden, kValue});
    std::vector<rt::WeightSlot> weights = {
        bf16_slot(kConv, kHidden), bf16_slot(kValue, kHidden),
        bf16_slot(kValueHeads, kHidden), bf16_slot(kValueHeads, kHidden),
        bf16_slot(kHidden, kValue)};

    const rt::ValueId normed = graph.alloc_value(kHidden);
    graph.external_inputs = {normed};

    Emitter em{graph};
    GdnLoweringGeometry geometry;
    geometry.key_heads = kKeyHeads;
    geometry.value_heads = kValueHeads;
    geometry.key_head_dim = kKeyDim;
    geometry.value_head_dim = kValueDim;
    geometry.conv_kernel = kConvKernel;
    geometry.sigmoid_gate = sigmoid_gate;
    GdnLoweringWeights gdn_weights;
    gdn_weights.qkv = &qkv;
    gdn_weights.z = &z;
    gdn_weights.b = &b;
    gdn_weights.a = &a;
    gdn_weights.out = &out;
    GdnLoweringIndices indices{w_qkv, w_z, w_b, w_a, w_out, 0u, 1u, 2u, 3u, 0u};
    const rt::ValueId block_out = shelf::qwen_hybrid::emit_gdn(
        em, geometry, gdn_weights, indices, normed, "test.");
    graph.external_outputs = {block_out};

    auto status = rt::validate_primitive_graph(graph);
    if (!status.ok()) {
        std::printf("FAIL validate: %s\n", status.message().c_str());
        ++failed;
        return {};
    }
    const rt::WeightTableView view{weights.data(), weights.size()};
    auto built = rt::build_program(graph, view, rt::RowBucket::R16,
                                   rt::ExecutionClass::DECODE,
                                   rt::ProgramBuildOptions{});
    if (!built.ok()) {
        std::printf("FAIL build: %s\n", built.status().message().c_str());
        ++failed;
        return {};
    }
    std::string sequence;
    for (std::size_t i = 0; i < built.value().dispatches.size(); ++i) {
        sequence += rt::kernel_to_name(built.value().dispatches[i].kernel_id);
        if (i + 1 != built.value().dispatches.size()) sequence += ",";
    }
    return sequence;
}

}  // namespace

int main() {
    const std::string sigmoid = build(true);
    check(sigmoid.find("GDN_RECURRENCE,RMS_NORM,SIGMOID,MUL") != std::string::npos,
          "Qwen4Exp GDN uses a sigmoid output gate");
    const std::string silu = build(false);
    check(silu.find("GDN_RECURRENCE,RMS_NORM,SILU,MUL") != std::string::npos,
          "Qwen3.5 GDN keeps the silu output gate");

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
