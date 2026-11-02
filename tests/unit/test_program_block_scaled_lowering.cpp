#include <shelf/runtime/program/program.h>
#include <shelf/runtime/program/kernel_id.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/graph/primitive_node.h>
#include <shelf/runtime/graph/shape_spec.h>
#include <shelf/runtime/execution/execution_types.h>
#include <shelf/runtime/execution/row_bucket.h>

#include <cstdint>
#include <cstdio>

namespace rt = ::shelf::runtime;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

rt::PrimitiveGraph make_linear_graph(uint32_t out_features, uint32_t k) {
    rt::PrimitiveGraph g;
    const rt::ValueId in = g.alloc_value(k, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    const rt::ValueId out =
        g.alloc_value(out_features, rt::ValueDType::BF16, rt::ValueRowDomain::TOKEN_ROWS);
    rt::LinearNode ln;
    ln.shape = rt::MatrixwiseShapeKey{out_features, k};
    ln.weight_index = 0;
    ln.out_dtype = rt::ValueDType::BF16;
    auto& node = g.add_node(rt::PrimitiveNode{ln});
    node.inputs = {in};
    node.outputs = {out};
    node.debug_name = "linear";
    g.external_inputs = {in};
    g.external_outputs = {out};
    g.register_weight(ln.shape);
    return g;
}

rt::WeightSlot make_slot(uint8_t encoding, uint8_t compute_spec, uint32_t rows, uint32_t k,
                          uint32_t k_padded, uint32_t scale_group, uint32_t scale_stride,
                          uint32_t codes_stride) {
    rt::WeightSlot s;
    s.rows = rows;
    s.cols = k;
    s.k_padded = k_padded;
    s.encoding = encoding;
    s.compute_spec = compute_spec;
    s.codes = reinterpret_cast<const void*>(0x1000);
    s.scales = reinterpret_cast<const void*>(0x2000);
    s.storage_scale_stride_bytes = scale_stride;
    s.weight_scale_group = scale_group;
    s.codes_row_stride_bytes = codes_stride;
    s.preshuffled = true;
    return s;
}

void run_case(uint8_t encoding, uint8_t compute_spec, uint32_t scale_group, uint32_t scale_stride,
              uint32_t codes_stride, rt::KernelId expect_kernel, uint32_t expect_act_kp,
              const char* tag) {
    constexpr uint32_t kOut = 1024;
    constexpr uint32_t kK = 2560;
    const uint32_t kp = kK;

    rt::PrimitiveGraph g = make_linear_graph(kOut, kK);
    rt::WeightSlot slot = make_slot(encoding, compute_spec, kOut, kK, kp, scale_group,
                                     scale_stride, codes_stride);
    rt::WeightTableView table;
    table.slots = &slot;
    table.count = 1;

    rt::ProgramBuildOptions opts;
    opts.max_token_rows = 128u;

    auto prog = rt::build_program(g, table, rt::RowBucket::R128, rt::ExecutionClass::DECODE, opts);
    if (!prog.ok()) {
        check(false, std::string(tag) + ": build_program: " + prog.status().message());
        return;
    }
    const rt::Program& p = prog.value();
    check(p.dispatches.size() == 2u, std::string(tag) + ": two dispatches");
    if (p.dispatches.size() != 2u) return;
    const auto& lin = p.dispatches[1];
    check(lin.kernel_id == expect_kernel, std::string(tag) + ": linear kernel id");
    check(lin.compute_spec == compute_spec, std::string(tag) + ": compute spec");
    check(lin.activation_kp == expect_act_kp, std::string(tag) + ": activation k_padded");
    check(p.dispatches[0].kernel_id == rt::KernelId::ACTIVATION_QUANTIZE_FP8_FINEGRAINED,
          std::string(tag) + ": activation quantize kind");
}

}  // namespace

int main() {
    run_case(1u, static_cast<uint8_t>(::shelf::quantization::ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16), 128u, (2560u / 128u) * 4u, (2560u / 32u) * 512u,
             rt::KernelId::LINEAR_FP8_FINEGRAINED, 2560u, "fp8");
    // K that is not a multiple of 128 must still pad the FP8 activation to 128.
    {
        rt::PrimitiveGraph g = make_linear_graph(1024u, 2500u);
        rt::WeightSlot slot = make_slot(1u, static_cast<uint8_t>(::shelf::quantization::ComputeSpecId::FP8_FINEGRAINED_W8A8_F32_BF16), 1024u, 2500u, 2560u, 128u,
                                         (2560u / 128u) * 4u, (2560u / 32u) * 512u);
        rt::WeightTableView table;
        table.slots = &slot;
        table.count = 1;
        rt::ProgramBuildOptions opts;
        opts.max_token_rows = 128u;
        auto prog = rt::build_program(g, table, rt::RowBucket::R128,
                                      rt::ExecutionClass::DECODE, opts);
        check(prog.ok(), "fp8 k=2500 build_program");
        if (prog.ok()) {
            const auto& p = prog.value();
            check(p.dispatches.size() == 2u, "fp8 k=2500 two dispatches");
            if (p.dispatches.size() == 2u)
                check(p.dispatches[1].activation_kp == 2560u,
                      "fp8 k=2500 activation k_padded == 2560");
        }
    }

    std::printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
