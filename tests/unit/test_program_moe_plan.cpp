#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/moe_plan_scratch.h>
#include <shelf/runtime/program/program.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/weights/matrix_weight.h>

#include <cstdio>
#include <cstdlib>
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

constexpr uint32_t kExperts = 4u;
constexpr uint32_t kTopK = 2u;
constexpr uint32_t kHidden = 128u;
constexpr uint32_t kInter = 128u;
constexpr uint32_t kFused = 2u * kInter;
constexpr uint32_t kBucketTokens = 16u;

rt::WeightSlot expert_plane(uint32_t rows, uint32_t cols, bool exl3 = false) {
    rt::WeightSlot s{};
    s.rows = rows;
    s.cols = cols;
    s.k_padded = cols;
    s.encoding = static_cast<uint8_t>(shelf::weights::MatrixEncoding::Mxfp4);
    s.compute_spec = static_cast<uint8_t>(
        shelf::quantization::ComputeSpecId::MXFP4_W4A8_F32_BF16);
    s.codes = reinterpret_cast<const void*>(0x1000);
    s.scales = reinterpret_cast<const void*>(0x2000);
    s.row_ref = reinterpret_cast<const void*>(0x3000);
    s.codes_row_stride_bytes = (cols / 32u) * 256u;
    s.scale_row_stride_bytes = cols / 32u;
    s.weight_scale_group = 32u;
    s.preshuffled = true;
    s.experts = kExperts;
    s.codes_batch_stride_bytes = (static_cast<uint64_t>(rows) / 16u) * (cols / 32u) * 256u;
    s.scales_batch_stride_bytes = static_cast<uint64_t>(rows) * (cols / 32u);
    if (exl3) {
        s.encoding = static_cast<uint8_t>(shelf::weights::MatrixEncoding::Exl3);
        s.compute_spec = static_cast<uint8_t>(
            shelf::quantization::ComputeSpecId::EXL3_W16A16_F32_BF16);
        s.codes = reinterpret_cast<const void*>(0x5000);
        s.scales = nullptr;
        s.row_ref = nullptr;
        s.exl3_suh = reinterpret_cast<const void*>(0x6000);
        s.exl3_svh = reinterpret_cast<const void*>(0x7000);
        s.exl3_marker = reinterpret_cast<const void*>(0x8000);
        s.exl3_ka = 3u;
        s.exl3_half = 1u;
        const uint64_t words = 16u * 3u + 8u;
        s.codes_row_stride_bytes =
            static_cast<uint32_t>((static_cast<uint64_t>(rows) / 16u) * words * 2u);
        s.codes_batch_stride_bytes = s.codes_row_stride_bytes;
        s.scales_batch_stride_bytes = 0u;
        s.exl3_suh_batch_stride_bytes = static_cast<uint64_t>(cols) * 2u;
        s.exl3_svh_batch_stride_bytes = static_cast<uint64_t>(rows) * 2u;
        s.exl3_marker_batch_stride_bytes = 4u;
    }
    return s;
}

rt::WeightSlot bf16_matrix(uint32_t rows, uint32_t cols) {
    rt::WeightSlot s{};
    s.rows = rows;
    s.cols = cols;
    s.k_padded = cols;
    s.encoding = static_cast<uint8_t>(shelf::weights::MatrixEncoding::Bf16);
    s.compute_spec = static_cast<uint8_t>(shelf::quantization::ComputeSpecId::BF16_F32_BF16);
    s.device_ptr = reinterpret_cast<const void*>(0x4000);
    s.bytes = static_cast<uint64_t>(rows) * cols * 2ull;
    return s;
}

// One MoE FFN between an attention residual and the layer output: routing, the routed
// experts as two grouped linears, a shared expert as ordinary linears, and the combine.
struct MoeGraph {
    rt::PrimitiveGraph g;
    std::vector<rt::WeightSlot> weights;

    uint32_t add_weight(const rt::WeightSlot& slot) {
        weights.push_back(slot);
        g.weight_table.push_back(rt::MatrixwiseShapeKey{slot.rows, slot.cols});
        return static_cast<uint32_t>(weights.size() - 1);
    }

    rt::WeightSlot& slot_at(uint32_t i) { return weights.at(i); }

        rt::ValueId linear(const rt::MatrixwiseShapeKey& shape, uint32_t weight_index,
                       rt::ValueId in, rt::ValueRowDomain out_rows,
                       rt::ValueDType dtype = rt::ValueDType::BF16) {
        rt::LinearNode n;
        n.shape = shape;
        n.weight_index = weight_index;
        n.out_dtype = dtype;
        rt::ValueId out = g.alloc_value(shape.output_features, dtype, out_rows);
        auto& gn = g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {in};
        gn.outputs = {out};
        return out;
    }
};

MoeGraph build_moe_graph(bool router_matches_shape = true,
                        bool combine_in_group_rows = true, bool exl3 = false) {
    MoeGraph m;
    m.g.grouped_row_shape = rt::GroupedRowShape{kExperts, kTopK};

    const uint32_t w_router = m.add_weight(bf16_matrix(kExperts, kHidden));
    const uint32_t w13 = m.add_weight(expert_plane(kFused, kHidden, exl3));
    const uint32_t w2 = m.add_weight(expert_plane(kHidden, kInter, exl3));
    const uint32_t w_shared_gate = m.add_weight(bf16_matrix(kInter, kHidden));
    const uint32_t w_shared_up = m.add_weight(bf16_matrix(kInter, kHidden));
    const uint32_t w_shared_down = m.add_weight(bf16_matrix(kHidden, kInter));
    const uint32_t w_shared_scale = m.add_weight(bf16_matrix(1u, kHidden));

    rt::ValueId post_norm = m.g.alloc_value(kHidden);
    rt::ValueId residual_in = m.g.alloc_value(kHidden);
    m.g.external_inputs.push_back(post_norm);
    m.g.external_inputs.push_back(residual_in);

    rt::ValueId logits = m.linear(rt::MatrixwiseShapeKey{kExperts, kHidden}, w_router,
                                  post_norm, rt::ValueRowDomain::TOKEN_ROWS,
                                  rt::ValueDType::F32);

    {
        rt::MoeRouterNode n;
        n.experts = router_matches_shape ? kExperts : kExperts + 1u;
        n.top_k = kTopK;
        auto& gn = m.g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {logits};
    }

    rt::ValueId y13 = m.g.alloc_value(kFused, rt::ValueDType::BF16,
                                      rt::ValueRowDomain::GROUPED_ROWS);
    {
        rt::MoeGroupedLinearNode n;
        n.shape = rt::MatrixwiseShapeKey{kFused, kHidden};
        n.weight_index = w13;
        n.out_dtype = rt::ValueDType::BF16;
        auto& gn = m.g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {post_norm};
        gn.outputs = {y13};
    }

    const rt::ValueId gate = m.g.alloc_value(kInter, rt::ValueDType::BF16,
                                            rt::ValueRowDomain::GROUPED_ROWS);
    const rt::ValueId up = m.g.alloc_value(kInter, rt::ValueDType::BF16,
                                          rt::ValueRowDomain::GROUPED_ROWS);
    m.g.views.push_back(rt::PrimitiveValueView{gate, y13, 0u, kInter});
    m.g.views.push_back(rt::PrimitiveValueView{up, y13, kInter, kInter});

    rt::ValueId act = m.g.alloc_value(kInter, rt::ValueDType::BF16,
                                      rt::ValueRowDomain::GROUPED_ROWS);
    {
        rt::SwiGluNode n{rt::RowwiseShapeKey{kInter}};
        auto& gn = m.g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {gate, up};
        gn.outputs = {act};
    }

    const rt::ValueRowDomain expert_rows =
        combine_in_group_rows ? rt::ValueRowDomain::GROUPED_ROWS
                              : rt::ValueRowDomain::TOKEN_ROWS;
    rt::ValueId expert_out = m.g.alloc_value(kHidden, rt::ValueDType::BF16, expert_rows);
    {
        rt::MoeGroupedLinearNode n;
        n.shape = rt::MatrixwiseShapeKey{kHidden, kInter};
        n.weight_index = w2;
        n.out_dtype = rt::ValueDType::BF16;
        auto& gn = m.g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {act};
        gn.outputs = {expert_out};
    }

    rt::ValueId shared_gate = m.linear(rt::MatrixwiseShapeKey{kInter, kHidden},
                                       w_shared_gate, post_norm,
                                       rt::ValueRowDomain::TOKEN_ROWS);
    rt::ValueId shared_up = m.linear(rt::MatrixwiseShapeKey{kInter, kHidden}, w_shared_up,
                                     post_norm, rt::ValueRowDomain::TOKEN_ROWS);
    rt::ValueId shared_act = m.g.alloc_value(kInter);
    {
        rt::SwiGluNode n{rt::RowwiseShapeKey{kInter}};
        auto& gn = m.g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {shared_gate, shared_up};
        gn.outputs = {shared_act};
    }
    rt::ValueId shared_out = m.linear(rt::MatrixwiseShapeKey{kHidden, kInter},
                                      w_shared_down, shared_act,
                                      rt::ValueRowDomain::TOKEN_ROWS);
    rt::ValueId scale_logit = m.linear(rt::MatrixwiseShapeKey{1u, kHidden}, w_shared_scale,
                                       post_norm, rt::ValueRowDomain::TOKEN_ROWS,
                                       rt::ValueDType::F32);
    rt::ValueId scale = m.g.alloc_value(1u, rt::ValueDType::F32,
                                        rt::ValueRowDomain::TOKEN_ROWS);
    {
        rt::SigmoidNode n{rt::RowwiseShapeKey{1u}};
        auto& gn = m.g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {scale_logit};
        gn.outputs = {scale};
    }

    rt::ValueId ffn_out = m.g.alloc_value(kHidden);
    {
        rt::MoeCombineNode n;
        n.shape = rt::RowwiseShapeKey{kHidden};
        n.top_k = kTopK;
        auto& gn = m.g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {expert_out, shared_out, scale};
        gn.outputs = {ffn_out};
    }

    rt::ValueId layer_out = m.g.alloc_value(kHidden);
    {
        rt::ResidualAddNode n{rt::RowwiseShapeKey{kHidden}};
        auto& gn = m.g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {ffn_out, residual_in};
        gn.outputs = {layer_out};
    }
    m.g.external_outputs.push_back(layer_out);
    return m;
}

std::string kernel_sequence(const rt::Program& program, size_t from) {
    std::string out;
    for (size_t i = from; i < program.dispatches.size(); ++i) {
        out += rt::kernel_to_name(program.dispatches[i].kernel_id);
        if (i + 1 != program.dispatches.size()) out += ",";
    }
    return out;
}

void check_plan_scratch(const rt::Program& program) {
    const rt::MoePlanScratch plan = rt::MoePlanScratch::make(kBucketTokens, kExperts, kTopK);
    check(plan.slots == kBucketTokens * kTopK, "scratch holds one entry per slot");
    check(plan.group_capacity == 4u && plan.row_capacity == 64u,
          "scratch sizes match the grouped row capacity");
    check(plan.row_token_offset > plan.group_count_offset &&
              plan.slot_row_of_offset > plan.row_token_offset &&
              plan.capacity_units == plan.slot_row_of_offset + plan.slots,
          "scratch regions are laid out end to end");

    size_t scratch_ranges = 0;
    size_t scratch_bindings = 0;
    uint32_t scratch_slot = rt::kNoWorkspace;
    for (const auto& range : program.workspace.ranges) {
        if (range.bytes == plan.bytes()) {
            ++scratch_ranges;
            scratch_slot = static_cast<uint32_t>(&range - program.workspace.ranges.data());
        }
    }
    check(scratch_ranges == 1u, "the program allocates exactly one plan scratch range");
    if (scratch_ranges != 1u) return;

    for (const auto& b : program.dispatches) {
        if (b.scratch_slot == rt::kNoWorkspace) continue;
        ++scratch_bindings;
        check(b.scratch_slot == scratch_slot, "every scratch binding uses the one range");
        check(b.scratch_tokens == kBucketTokens && b.scratch_experts == kExperts &&
                  b.scratch_top_k == kTopK,
              "scratch bindings carry the layout geometry");
    }
    // router, scan, scatter, both plane producers, both grouped linears, the group-row
    // swiglu, combine.
    check(scratch_bindings == 9u, "nine dispatches read or size themselves by the plan scratch");

    const rt::ValueBinding* y13 = nullptr;
    for (const auto& v : program.values) {
        if (v.row_domain == rt::ValueRowDomain::GROUPED_ROWS && v.feature_count == kFused)
            y13 = &v;
    }
    check(y13 != nullptr, "the fused expert output is bound");
    if (y13 == nullptr) return;
    const uint64_t rows = y13->bytes / (static_cast<uint64_t>(y13->row_stride) * 2u);
    check(rows == plan.row_capacity, "the fused expert output is sized in group rows");

    uint32_t plane_rows = 0;
    for (const auto& b : program.dispatches) {
        if (b.kernel_id != rt::KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF) continue;
        if (b.activation_kp != kHidden) continue;
        plane_rows = b.activation_max_rows;
    }
    check(plane_rows == plan.row_capacity, "the routed activation plane holds group rows");
}

}  // namespace

int main() {
    {
        auto m = build_moe_graph();
        auto st = rt::validate_primitive_graph(m.g);
        if (!st.ok()) {
            std::printf("FAIL graph validation: %s\n", st.message().c_str());
            return 1;
        }
        check(true, "the MoE graph validates");

        std::vector<rt::WeightSlot> weights = m.weights;
        const rt::WeightTableView view{weights.data(), weights.size()};
        auto built = rt::build_program(m.g, view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE,
                                       rt::ProgramBuildOptions{});
        if (!built.ok()) {
            std::printf("FAIL build_program: %s\n", built.status().message().c_str());
            return 1;
        }
        const rt::Program& program = built.value();

        size_t ffn_start = 0;
        for (size_t i = 0; i < program.dispatches.size(); ++i) {
            if (program.dispatches[i].kernel_id == rt::KernelId::MOE_ROUTER) {
                ffn_start = i;
                break;
            }
        }
        check(ffn_start != 0, "a router dispatch exists");
        const std::string want =
            "MOE_ROUTER,MOE_PLAN_SCAN,MOE_PLAN_SCATTER,"
            "ACTIVATION_QUANTIZE_MXFP4_ROWREF,LINEAR_MXFP4_W4A8_GROUPED,SWIGLU,"
            "ACTIVATION_QUANTIZE_MXFP4_ROWREF,LINEAR_MXFP4_W4A8_GROUPED,"
            "LINEAR_BF16,LINEAR_BF16,SWIGLU,LINEAR_BF16,LINEAR_BF16,SIGMOID,"
            "MOE_COMBINE,RESIDUAL_ADD";
        const std::string got = kernel_sequence(program, ffn_start);
        if (got != want) std::printf("      got %s\n", got.c_str());
        check(got == want, "the MoE FFN lowers to the planned dispatch sequence");

        check_plan_scratch(program);
    }

    {
        auto m = build_moe_graph(true, true, true);
        auto st = rt::validate_primitive_graph(m.g);
        if (!st.ok()) {
            std::printf("FAIL exl3 graph validation: %s\n", st.message().c_str());
            return 1;
        }
        std::vector<rt::WeightSlot> weights = m.weights;
        const rt::WeightTableView view{weights.data(), weights.size()};
        auto built = rt::build_program(m.g, view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE,
                                       rt::ProgramBuildOptions{});
        if (!built.ok()) {
            std::printf("FAIL exl3 build_program: %s\n", built.status().message().c_str());
            return 1;
        }
        const rt::Program& program = built.value();
        size_t ffn_start = 0;
        for (size_t i = 0; i < program.dispatches.size(); ++i) {
            if (program.dispatches[i].kernel_id == rt::KernelId::MOE_ROUTER) {
                ffn_start = i;
                break;
            }
        }
        const std::string want =
            "MOE_ROUTER,MOE_PLAN_SCAN,MOE_PLAN_SCATTER,"
            "LINEAR_EXL3_W16A16_GROUPED,SWIGLU,LINEAR_EXL3_W16A16_GROUPED,"
            "LINEAR_BF16,LINEAR_BF16,SWIGLU,LINEAR_BF16,LINEAR_BF16,SIGMOID,"
            "MOE_COMBINE,RESIDUAL_ADD";
        const std::string got = kernel_sequence(program, ffn_start);
        if (got != want) std::printf("      got %s\n", got.c_str());
        check(got == want, "the EXL3 MoE FFN skips activation quantization");
    }

    for (const bool half : {false, true}) {
        // mul24 experts run through the same grouped dispatch; only integer bit rates exist for them.
        auto m = build_moe_graph(true, true, true);
        for (const uint32_t i : {1u, 2u}) {
            m.slot_at(i).compute_spec = static_cast<uint8_t>(
                shelf::quantization::ComputeSpecId::EXL3_MUL24_INT8_W8A8_F32_BF16);
            m.slot_at(i).exl3_half = half ? 1u : 0u;
            const uint64_t words = 16u * 3u + (half ? 8u : 0u);
            const uint32_t rows = m.slot_at(i).rows;
            m.slot_at(i).codes_row_stride_bytes =
                static_cast<uint32_t>((static_cast<uint64_t>(rows) / 16u) * words * 2u);
            m.slot_at(i).codes_batch_stride_bytes = m.slot_at(i).codes_row_stride_bytes;
        }
        std::vector<rt::WeightSlot> weights = m.weights;
        const rt::WeightTableView view{weights.data(), weights.size()};
        auto built = rt::build_program(m.g, view, rt::RowBucket::R16, rt::ExecutionClass::DECODE,
                                       rt::ProgramBuildOptions{});
        check(built.ok() == !half,
              half ? "a mul24 expert plane with a half bit rate is rejected"
                   : "a mul24 expert plane builds a grouped EXL3 program");
        if (built.ok()) {
            bool grouped = false;
            for (const auto& d : built.value().dispatches)
                grouped = grouped || d.kernel_id == rt::KernelId::LINEAR_EXL3_W16A16_GROUPED;
            check(grouped, "the mul24 expert plane dispatches the grouped EXL3 kernel");
            for (const char* env_name : {"SHELF_EXL3_MUL24_PRE", "SHELF_EXL3_MUL24_AUTO"}) {
                setenv(env_name, "1", 1);
                auto pre = rt::build_program(m.g, view, rt::RowBucket::R16, rt::ExecutionClass::DECODE,
                                             rt::ProgramBuildOptions{});
                unsetenv(env_name);
                check(pre.ok(), "mul24 の活性 workspace を持つ program を作成する");
                if (pre.ok()) {
                    const auto scratch = rt::MoePlanScratch::make(kBucketTokens, kExperts, kTopK);
                    const auto slot = pre.value().plan_scratch_slot;
                    check(slot < pre.value().workspace.ranges.size() &&
                          pre.value().workspace.ranges[slot].bytes >= scratch.mul24_offset() +
                              rt::MoePlanScratch::mul24_bytes(scratch.row_capacity, kHidden),
                          "mul24 の活性 workspace は group row の容量を含む");
                }
            }
        }
    }

    {
        auto m = build_moe_graph(false);
        check(!rt::validate_primitive_graph(m.g).ok(),
              "a router that disagrees with the grouped row shape is rejected");
    }
    {
        auto m = build_moe_graph(true, false);
        check(!rt::validate_primitive_graph(m.g).ok(),
              "a combine reading token rows as expert output is rejected");
    }
    {
        auto m = build_moe_graph();
        m.slot_at(1).experts = 1u;
        m.slot_at(1).codes_batch_stride_bytes = 0u;
        m.slot_at(1).scales_batch_stride_bytes = 0u;
        std::vector<rt::WeightSlot> weights = m.weights;
        const rt::WeightTableView view{weights.data(), weights.size()};
        check(!rt::build_program(m.g, view, rt::RowBucket::R16, rt::ExecutionClass::DECODE,
                                 rt::ProgramBuildOptions{})
                   .ok(),
              "a grouped linear over a rank-2 weight is rejected");
    }
    {
        auto m = build_moe_graph();
        m.slot_at(1).k_padded = kHidden + 64u;
        std::vector<rt::WeightSlot> weights = m.weights;
        const rt::WeightTableView view{weights.data(), weights.size()};
        check(!rt::build_program(m.g, view, rt::RowBucket::R16, rt::ExecutionClass::DECODE,
                                 rt::ProgramBuildOptions{})
                   .ok(),
              "an expert plane whose k_padded is not the activation plane width is rejected");
    }

    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed, failed);
    return failed == 0 ? 0 : 1;
}
