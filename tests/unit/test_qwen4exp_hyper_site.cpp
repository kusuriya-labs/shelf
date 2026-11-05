#include <shelf/models/qwen4exp/model/hyper_connection_lowering.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/program.h>

#include <cstdio>
#include <string>
#include <vector>

namespace rt = shelf::runtime;
using shelf::qwen4exp::HyperConnectionSiteGeometry;
using shelf::qwen4exp::HyperConnectionSiteIndices;
using shelf::qwen4exp::HyperConnectionSiteWeights;
using shelf::qwen_hybrid::Emitter;

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
constexpr std::uint32_t kLowrank = 64u;
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

std::string kernel_sequence(const rt::Program& program) {
    std::string out;
    for (std::size_t i = 0; i < program.dispatches.size(); ++i) {
        out += rt::kernel_to_name(program.dispatches[i].kernel_id);
        if (i + 1 != program.dispatches.size()) out += ",";
    }
    return out;
}

struct Site {
    rt::PrimitiveGraph graph;
    std::vector<rt::WeightSlot> weights;
    shelf::weights::MatrixWeight down;
    shelf::weights::MatrixWeight up;
    shelf::weights::MatrixWeight inject;
};

Site make_site(bool with_inject) {
    Site site;
    site.down.rows = kLowrank;
    site.down.cols = kWidth;
    site.down.k_padded = kWidth;
    site.up.rows = kWidth;
    site.up.cols = kLowrank;
    site.up.k_padded = kLowrank;
    site.inject.rows = kHc;
    site.inject.cols = kWidth;
    site.inject.k_padded = kWidth;

    const std::uint32_t w_down = site.graph.register_weight({kLowrank, kWidth});
    const std::uint32_t w_up = site.graph.register_weight({kWidth, kLowrank});
    const std::uint32_t w_inject = site.graph.register_weight({kHc, kWidth});
    site.weights = {bf16_slot(kLowrank, kWidth), bf16_slot(kWidth, kLowrank),
                    bf16_slot(kHc, kWidth)};

    const rt::ValueId hyper = site.graph.alloc_value(kWidth);
    const rt::ValueId sublayer = site.graph.alloc_value(kHidden);
    site.graph.external_inputs = {hyper};
    if (with_inject) site.graph.external_inputs.push_back(sublayer);

    Emitter em{site.graph};
    HyperConnectionSiteGeometry geometry;
    geometry.hc = kHc;
    geometry.hidden = kHidden;
    geometry.lowrank = kLowrank;
    geometry.eps = 1e-6f;
    geometry.norm_parameter_index = 0u;
    HyperConnectionSiteWeights weights;
    weights.down = &site.down;
    weights.up = &site.up;
    weights.inject = with_inject ? &site.inject : nullptr;
    HyperConnectionSiteIndices indices{w_down, w_up, w_inject};
    auto emitted = shelf::qwen4exp::emit_hyper_connection_site(
        em, geometry, weights, indices, hyper, sublayer, with_inject, "test.");
    if (!emitted.ok()) {
        std::printf("FAIL emit: %s\n", emitted.status().message().c_str());
        ++failed;
        return site;
    }
    site.graph.external_outputs = {emitted.value().mixed};
    if (with_inject) site.graph.external_outputs.push_back(emitted.value().hyper);
    return site;
}

std::string build_and_sequence(Site& site) {
    auto status = rt::validate_primitive_graph(site.graph);
    if (!status.ok()) {
        std::printf("FAIL validate: %s\n", status.message().c_str());
        ++failed;
        return {};
    }
    const rt::WeightTableView view{site.weights.data(), site.weights.size()};
    auto built = rt::build_program(site.graph, view, rt::RowBucket::R16,
                                   rt::ExecutionClass::DECODE,
                                   rt::ProgramBuildOptions{});
    if (!built.ok()) {
        std::printf("FAIL build: %s\n", built.status().message().c_str());
        ++failed;
        return {};
    }
    return kernel_sequence(built.value());
}

}  // namespace

int main() {
    {
        Site site = make_site(true);
        const std::string got = build_and_sequence(site);
        const std::string want =
            "RMS_NORM,LINEAR_BF16,SCALE,SILU,LINEAR_BF16,SIGMOID,HYPER_MIX,"
            "LINEAR_BF16,SCALE,SIGMOID,SCALE,HYPER_APPLY";
        check(got == want, "the gated-residual site lowers to the expected sequence");
        if (got != want) std::printf("      got %s\n", got.c_str());
    }
    {
        Site site = make_site(false);
        const std::string got = build_and_sequence(site);
        const std::string want =
            "RMS_NORM,LINEAR_BF16,SCALE,SILU,LINEAR_BF16,SIGMOID,HYPER_MIX";
        check(got == want, "the final mixer lowers without an injection branch");
        if (got != want) std::printf("      got %s\n", got.c_str());
    }

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed,
                failed);
    return failed == 0 ? 0 : 1;
}
