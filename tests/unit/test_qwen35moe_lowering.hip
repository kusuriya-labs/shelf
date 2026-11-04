#include <shelf/models/qwen35/runtime/executor_footprint.h>
#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/qwen35moe/runtime/executor_footprint.h>
#include <shelf/models/qwen35moe/weights/moe_weight_plan.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/runtime/program/moe_plan_scratch.h>
#include <shelf/runtime/program/program.h>

#include <hip/hip_runtime.h>

#include "moe_expert_fixture.h"

#include <cstdint>
#include <cstdio>
#include <span>
#include <filesystem>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;
namespace moe = shelf::qwen35moe;
namespace q35 = shelf::qwen35;
namespace rt = shelf::runtime;
namespace qmx = shelf::quantization::mxfp4;
using namespace moe_expert_fixture;

namespace {

int passed = 0;
int failed = 0;

void check(bool cond, const std::string& msg) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

const char* kind_name(rt::PrimitiveKind k) { return rt::to_string(k); }

std::string node_kind_sequence(const rt::PrimitiveGraph& g, std::size_t begin,
                               std::size_t end) {
    std::string out;
    for (std::size_t i = begin; i < end && i < g.nodes.size(); ++i) {
        out += kind_name(rt::primitive_kind_of(g.nodes[i].node));
        if (i + 1 != end && i + 1 != g.nodes.size()) out += ",";
    }
    return out;
}

std::vector<std::size_t> positions_of(const rt::PrimitiveGraph& g, rt::PrimitiveKind kind) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < g.nodes.size(); ++i) {
        if (rt::primitive_kind_of(g.nodes[i].node) == kind) out.push_back(i);
    }
    return out;
}

std::size_t dispatch_count(const rt::Program& p, rt::KernelId id) {
    std::size_t n = 0;
    for (const auto& b : p.dispatches)
        if (b.kernel_id == id) ++n;
    return n;
}

uint32_t value_binding_rows(const rt::Program& p, rt::ValueId v) {
    const rt::ValueBinding* b = p.find_value(v);
    if (b == nullptr || b->row_stride == 0u) return 0u;
    const uint32_t elem = rt::value_dtype_bytes(b->dtype);
    return elem == 0u ? 0u
                      : static_cast<uint32_t>(b->bytes / (uint64_t{b->row_stride} * elem));
}

// The fixture's FFN constants that the fixture header does not name.
constexpr uint32_t kTopK = 2u;
constexpr uint32_t kFused = 2u * kInter;
constexpr uint32_t kBucketTokens = 16u;

}  // namespace

int main() {
    std::printf("=== qwen35moe lowering to primitives (GPU) ===\n");
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    std::error_code ec;
    const fs::path root =
        fs::temp_directory_path(ec) / ("shelf_moe_lowering_" + std::to_string(::getpid()));
    fs::remove_all(root, ec);

    Fixture f;
    if (!write_fixture(f, root / "moe", false, true)) {
        std::printf("FAIL moe model fixture write\n");
        return 1;
    }

    auto arena = shelf::gpu::GpuArena::create(0, 2ull * 1024ull * 1024ull * 1024ull);
    check(arena.ok(), "arena create");
    shelf::gpu::GpuArena gpu_arena = arena.release();

    auto loaded = moe::load_qwen35moe_weights((root / "moe").string(), gpu_arena, stream);
    if (!loaded.ok()) {
        std::printf("FAIL load_qwen35moe_weights: %s\n", loaded.status().message().c_str());
        return 1;
    }
    moe::Qwen35MoeModelWeights& weights = loaded.value();
    const moe::Qwen35MoeTextConfig& config = f.config;

    auto plan0 = moe::build_moe_expert_group_plan(config, 0);
    check(plan0.ok(), "plan for layer 0");
    const moe::MoeExpertGroupPlan& p0 = plan0.value();
    check(p0.w13_rows == kFused && p0.w13_cols == kHidden && p0.w2_rows == kHidden &&
              p0.w2_cols == kInter,
          "the fixture plans a fused gate / up plane and a down plane");

    auto lowered = moe::lower_qwen35_moe_to_primitives(config, weights);
    if (!lowered.ok()) {
        std::printf("FAIL lower_qwen35_moe_to_primitives: %s\n",
                    lowered.status().message().c_str());
        return 1;
    }
    const rt::PrimitiveGraph& g = lowered.value().graph;

    check(g.grouped_row_shape.experts == kExperts && g.grouped_row_shape.top_k == kTopK,
          "the graph declares the grouped row shape from the config");
    check(g.has_grouped_rows(), "the graph reports grouped rows");

    const std::vector<std::size_t> routers = positions_of(g, rt::PrimitiveKind::MOE_ROUTER);
    const std::vector<std::size_t> grouped =
        positions_of(g, rt::PrimitiveKind::MOE_GROUPED_LINEAR);
    const std::vector<std::size_t> combines = positions_of(g, rt::PrimitiveKind::MOE_COMBINE);
    check(routers.size() == kLayers && grouped.size() == 2u * kLayers &&
              combines.size() == kLayers,
          "every layer routes once and runs two grouped projections and one combine");

    // One layer's FFN, from its router linear through the residual add.
    const std::string want_ffn =
        "LINEAR,MOE_ROUTER,MOE_GROUPED_LINEAR,SWIGLU,MOE_GROUPED_LINEAR,"
        "LINEAR,LINEAR,SWIGLU,LINEAR,LINEAR,SIGMOID,MOE_COMBINE,RESIDUAL_ADD";
    bool ffn_sequences_match = routers.size() == kLayers;
    for (std::size_t l = 0; l < routers.size(); ++l) {
        const std::string got = node_kind_sequence(g, routers[l] - 1u, routers[l] + 12u);
        if (got != want_ffn) {
            ffn_sequences_match = false;
            std::printf("      layer %zu FFN: %s\n", l, got.c_str());
        }
    }
    check(ffn_sequences_match, "both layers lower their FFN to the planned node sequence");

    const std::vector<std::size_t> gathers = positions_of(g, rt::PrimitiveKind::OUTPUT_GATHER);
    const std::vector<std::size_t> samplings = positions_of(g, rt::PrimitiveKind::SAMPLING);
    check(gathers.size() == 1u && samplings.size() == 1u &&
              samplings[0] + 1u == g.nodes.size(),
          "the model head gathers once, samples once, and ends the graph");
    check(node_kind_sequence(g, gathers[0], g.nodes.size()) ==
              "OUTPUT_GATHER,RMS_NORM,LINEAR,SAMPLING",
          "the head is gather, final norm, lm projection, sampling");

    // The fused plane is read as two views, never split or copied. The GDN layers add their
    // own q / k / v views, so the check looks at the grouped ones.
    std::vector<const rt::PrimitiveValueView*> fused_views;
    for (const auto& v : g.views) {
        if (g.value_row_domain(v.base) == rt::ValueRowDomain::GROUPED_ROWS) fused_views.push_back(&v);
    }
    check(fused_views.size() == 2u * kLayers, "each layer reads its fused plane as two views");
    bool views_ok = fused_views.size() == 2u * kLayers;
    for (std::size_t i = 0; i + 1u < fused_views.size(); i += 2u) {
        const auto& a = *fused_views[i];
        const auto& b = *fused_views[i + 1u];
        if (a.base != b.base) views_ok = false;
        if (a.feature_offset != 0u || b.feature_offset != kInter) views_ok = false;
        if (a.feature_count != kInter || b.feature_count != kInter) views_ok = false;
        if (g.value_row_domain(a.base) != rt::ValueRowDomain::GROUPED_ROWS) views_ok = false;
        if (g.value_features(a.base) != kFused) views_ok = false;
    }
    check(views_ok, "the views name the gate half and the up half of one grouped plane");

    std::size_t grouped_value_specs = 0;
    bool grouped_widths_ok = true;
    for (const auto& spec : g.value_specs) {
        if (spec.row_domain != rt::ValueRowDomain::GROUPED_ROWS) continue;
        ++grouped_value_specs;
        if (spec.features != kFused && spec.features != kInter && spec.features != kHidden)
            grouped_widths_ok = false;
    }
    check(grouped_value_specs == 5u * kLayers,
          "every layer binds five group-row values: fused, gate, up, act, expert out");
    check(grouped_widths_ok, "group-row values keep the expert widths");

    // Weight table: the expert planes are rank-3 with the native layout strides.
    bool planes_named = false;
    for (uint32_t wi = 0; wi < g.weight_table.size(); ++wi) {
        const rt::WeightSlot& s = lowered.value().weights[wi];
        if (s.experts < 2u) continue;
        planes_named = true;
        const bool is_w13 = g.weight_table[wi] == rt::MatrixwiseShapeKey{kFused, kHidden};
        const bool is_w2 = g.weight_table[wi] == rt::MatrixwiseShapeKey{kHidden, kInter};
        if (!is_w13 && !is_w2) {
            std::printf("      weight %u has shape (%u,%u)\n", wi,
                        g.weight_table[wi].output_features, g.weight_table[wi].input_features);
            planes_named = false;
            continue;
        }
        const uint32_t rows = is_w13 ? p0.w13_rows : p0.w2_rows;
        const uint32_t cols = is_w13 ? p0.w13_cols : p0.w2_cols;
        const uint32_t kp = is_w13 ? p0.w13_k_padded : p0.w2_k_padded;
        if (s.rows != rows || s.cols != cols || s.k_padded != kp) planes_named = false;
        if (s.codes_row_stride_bytes != (cols / 32u) * qmx::mxfp4_native_block_bytes ||
            s.scale_row_stride_bytes != cols / 32u)
            planes_named = false;
        if (s.codes_batch_stride_bytes != qmx::mxfp4_native_codes_batch_stride_bytes(rows, kp))
            planes_named = false;
        if (s.scales_batch_stride_bytes != qmx::mxfp4_native_scale_batch_stride_bytes(rows, kp))
            planes_named = false;
        if (s.codes == nullptr || s.scales == nullptr || s.row_ref == nullptr)
            planes_named = false;
    }
    check(planes_named, "both expert planes reach the weight table as rank-3 native planes");
    check(lowered.value().weights.size() == g.weight_table.size(),
          "the weight table and the weight slots stay the same length");

    // The router reads the projection's f32 logits, one column per expert.
    bool routers_read_logits = true;
    for (std::size_t l = 0; l < routers.size(); ++l) {
        const rt::PrimitiveGraphNode& rn = g.nodes[routers[l]];
        if (rn.inputs.size() != 1u) routers_read_logits = false;
        const rt::ValueId logits = rn.inputs[0];
        if (g.value_dtype(logits) != rt::ValueDType::F32) routers_read_logits = false;
        if (g.value_features(logits) != kExperts) routers_read_logits = false;
        if (g.value_row_domain(logits) != rt::ValueRowDomain::TOKEN_ROWS) routers_read_logits = false;
        const rt::PrimitiveGraphNode& ln = g.nodes[routers[l] - 1u];
        if (rt::primitive_kind_of(ln.node) != rt::PrimitiveKind::LINEAR) routers_read_logits = false;
    }
    check(routers_read_logits, "the router consumes f32 logits of one column per expert");

    // Combine crosses the row spaces: group rows in, one row per token out.
    bool combines_ok = combines.size() == kLayers;
    for (std::size_t c : combines) {
        const rt::PrimitiveGraphNode& n = g.nodes[c];
        if (n.inputs.size() != 3u) combines_ok = false;
        if (g.value_row_domain(n.inputs[0]) != rt::ValueRowDomain::GROUPED_ROWS)
            combines_ok = false;
        if (g.value_row_domain(n.inputs[1]) != rt::ValueRowDomain::TOKEN_ROWS)
            combines_ok = false;
        if (g.value_dtype(n.inputs[2]) != rt::ValueDType::F32) combines_ok = false;
        if (g.value_features(n.inputs[2]) != 1u) combines_ok = false;
        if (g.value_row_domain(n.outputs[0]) != rt::ValueRowDomain::TOKEN_ROWS)
            combines_ok = false;
        if (g.value_features(n.outputs[0]) != kHidden) combines_ok = false;
    }
    check(combines_ok, "the combine folds group rows and a gated shared expert into tokens");

    check(g.external_outputs.size() == 4u,
          "the head exposes hidden, logits, sampled and the last block output");
    bool head_outputs_ok = g.external_outputs.size() == 4u;
    if (head_outputs_ok) {
        const std::vector<rt::ValueRowDomain> want_domains = {
            rt::ValueRowDomain::OUTPUT_ROWS, rt::ValueRowDomain::OUTPUT_ROWS,
            rt::ValueRowDomain::OUTPUT_ROWS, rt::ValueRowDomain::TOKEN_ROWS};
        const std::vector<uint32_t> want_features = {kHidden, static_cast<uint32_t>(kVocab), 1u, kHidden};
        for (std::size_t i = 0; i < 4u; ++i) {
            if (g.value_row_domain(g.external_outputs[i]) != want_domains[i])
                head_outputs_ok = false;
            if (g.value_features(g.external_outputs[i]) != want_features[i])
                head_outputs_ok = false;
        }
    }
    check(head_outputs_ok, "the head's external outputs are the same shape and order as dense");

    const std::vector<rt::WeightSlot>& slots = lowered.value().weights;
    const rt::WeightTableView view{slots.data(), slots.size()};
    auto built = rt::build_program(g, view, rt::RowBucket::R16,
                                   rt::ExecutionClass::DECODE, rt::ProgramBuildOptions{});
    if (!built.ok()) {
        std::printf("FAIL build_program: %s\n", built.status().message().c_str());
        return 1;
    }
    const rt::Program& program = built.value();
    check(dispatch_count(program, rt::KernelId::MOE_ROUTER) == kLayers &&
              dispatch_count(program, rt::KernelId::MOE_PLAN_SCAN) == kLayers &&
              dispatch_count(program, rt::KernelId::MOE_PLAN_SCATTER) == kLayers &&
              dispatch_count(program, rt::KernelId::LINEAR_MXFP4_W4A8_GROUPED) == 2u * kLayers &&
              dispatch_count(program, rt::KernelId::MOE_COMBINE) == kLayers,
          "the program carries one routing round and two grouped projections per layer");
    check(dispatch_count(program, rt::KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF) ==
              2u * kLayers,
          "both grouped projections quantize their own activation plane");

    const rt::MoePlanScratch bucket =
        rt::MoePlanScratch::make(kBucketTokens, kExperts, kTopK);
    check(bucket.row_capacity == 64u && bucket.group_capacity == 4u,
          "bucket R16 with 4 experts and top_k 2 holds 64 group rows");
    check(program.plan_scratch_slot != rt::kNoWorkspace &&
              program.workspace.ranges[program.plan_scratch_slot].bytes >= bucket.bytes(),
          "the program carries one plan scratch range at the bucket capacity");
    bool scratch_consistent = true;
    for (const auto& b : program.dispatches) {
        if (b.scratch_slot == rt::kNoWorkspace) continue;
        if (b.scratch_slot != program.plan_scratch_slot) scratch_consistent = false;
        if (b.scratch_tokens != kBucketTokens || b.scratch_experts != kExperts ||
            b.scratch_top_k != kTopK)
            scratch_consistent = false;
    }
    check(scratch_consistent, "every plan-scratch binding names the one range and its geometry");

    // A group-row plane the program owns holds the plan capacity of rows. A view of one
    // keeps the base's row stride and slot and only names a column range, so the two are
    // checked apart.
    bool grouped_at_capacity = true;
    std::size_t grouped_planes = 0;
    std::size_t grouped_views = 0;
    for (const auto& v : program.values) {
        if (v.row_domain != rt::ValueRowDomain::GROUPED_ROWS) continue;
        if (v.storage == rt::ValueStorage::WORKSPACE_VIEW) {
            ++grouped_views;
            const rt::ValueBinding* base = nullptr;
            for (const auto& o : program.values) {
                if (o.storage != rt::ValueStorage::WORKSPACE_VIEW && o.slot == v.slot &&
                    o.feature_count == kFused)
                    base = &o;
            }
            if (base == nullptr || v.feature_count != kInter || v.row_stride != base->row_stride)
                grouped_at_capacity = false;
            else if (v.offset != base->offset && v.offset != base->offset + kInter * 2ull)
                grouped_at_capacity = false;
            continue;
        }
        ++grouped_planes;
        if (value_binding_rows(program, v.logical_value) != bucket.row_capacity) {
            grouped_at_capacity = false;
            std::printf("      value %u storage %d feats %u stride %u rows %u\n",
                        v.logical_value.id, static_cast<int>(v.storage), v.feature_count,
                        v.row_stride, value_binding_rows(program, v.logical_value));
        }
    }
    check(grouped_planes == 3u * kLayers && grouped_at_capacity,
          "each layer's three group-row planes are sized by the plan capacity");
    check(grouped_views == 2u * kLayers, "each layer's fused plane is read through two views");

    // The MoE footprint term has to cover what the program actually allocated for the
    // grouped row space, and the whole reserve has to cover the whole workspace.
    std::size_t grouped_values = 0;
    for (const auto& v : program.values) {
        if (v.row_domain != rt::ValueRowDomain::GROUPED_ROWS) continue;
        if (v.storage == rt::ValueStorage::WORKSPACE_VIEW) continue;
        grouped_values += v.bytes;
    }
    std::size_t grouped_plane_bytes = 0;
    for (const auto& b : program.dispatches) {
        if (b.kernel_id != rt::KernelId::ACTIVATION_QUANTIZE_MXFP4_ROWREF) continue;
        grouped_plane_bytes += rt::Int8ActivationWorkspaceLayout::make(b.activation_kp,
                                                                  b.activation_max_rows)
                              .total_bytes;
    }
    const std::size_t plan_range = program.workspace.ranges[program.plan_scratch_slot].bytes;
    // The layers are identical, so one layer's grouped demand is the sum divided by the
    // layer count. The workspace itself reuses ranges across layers, so its total is the
    // weaker check below.
    const std::size_t grouped_actual =
        (grouped_values + grouped_plane_bytes) / kLayers + plan_range;
    const std::size_t grouped_reserve = moe::moe_executor_grouped_bytes(config, kBucketTokens);
    std::printf("fixture bucket %u: grouped reserve %zu B, one layer grouped %zu B "
                "(values %zu planes %zu plan %zu), workspace total %zu B\n",
                kBucketTokens, grouped_reserve, grouped_actual, grouped_values, grouped_plane_bytes,
                plan_range, program.workspace.total_bytes);
    check(grouped_reserve >= grouped_actual,
          "the MoE grouped footprint term covers the group-row allocations");

    q35::ExecutorConfig reserve_config;
    reserve_config.max_batch_rows = kBucketTokens;
    reserve_config.max_scheduled_requests = 4u;
    reserve_config.target_hidden_tap_count = 0u;
    const std::size_t reserve =
        q35::runtime::qwen35_executor_reserve_bytes(config,
            static_cast<std::size_t>(config.ffn.shared_expert_intermediate_size),
            reserve_config, 1u) + grouped_reserve;
    std::printf("fixture reserve %zu B vs program workspace %zu B\n", reserve,
                program.workspace.total_bytes);
    check(reserve >= program.workspace.total_bytes,
          "the MoE executor reserve covers the program workspace");

    {
        const std::uint32_t taps[] = {0u};
        moe::Qwen35MoeLowerOptions options;
        options.hidden_taps = std::span<const uint32_t>(taps, 1);
        auto tapped = moe::lower_qwen35_moe_to_primitives(config, weights, options);
        check(tapped.ok(), "a hidden tap on an inner layer lowers");
        if (tapped.ok()) {
            const rt::PrimitiveGraph& tg = tapped.value().graph;
            check(tg.external_outputs.size() == 5u, "the tap adds one external output");
            if (tg.external_outputs.size() == 5u) {
                const rt::ValueId tap = tg.external_outputs[4u];
                check(g.value_features(tap) == kHidden &&
                          g.value_row_domain(tap) == rt::ValueRowDomain::TOKEN_ROWS,
                      "the tap names one layer's output rows");
            }
        }
        const std::uint32_t last[] = {static_cast<std::uint32_t>(kLayers - 1u)};
        options.hidden_taps = std::span<const uint32_t>(last, 1);
        check(!moe::lower_qwen35_moe_to_primitives(config, weights, options).ok(),
              "a tap on the last layer is rejected as a duplicate output");
        const std::uint32_t wide[] = {static_cast<std::uint32_t>(kLayers)};
        options.hidden_taps = std::span<const uint32_t>(wide, 1);
        check(!moe::lower_qwen35_moe_to_primitives(config, weights, options).ok(),
              "a tap outside the layer range is rejected");
    }

    {
        const uint32_t keep = weights.layers[0].ffn.expert_w13.experts;
        weights.layers[0].ffn.expert_w13.experts = 1u;
        check(!moe::lower_qwen35_moe_to_primitives(config, weights).ok(),
              "an expert plane without a rank-3 expert count is rejected");
        weights.layers[0].ffn.expert_w13.experts = keep;
    }
    {
        const uint32_t keep = weights.layers[1].ffn.shared_gate_proj.rows;
        weights.layers[1].ffn.shared_gate_proj.rows = 8u;
        check(!moe::lower_qwen35_moe_to_primitives(config, weights).ok(),
              "a shared expert that does not match the plan is rejected");
        weights.layers[1].ffn.shared_gate_proj.rows = keep;
    }
    {
        moe::Qwen35MoeTextConfig bad = config;
        bad.ffn.num_experts_per_tok = kExperts + 2u;
        check(!moe::lower_qwen35_moe_to_primitives(bad, weights).ok(),
              "selecting more experts than exist is rejected");
    }
    {
        const uint64_t keep = weights.layers[0].ffn.expert_w2.codes_batch_stride_bytes;
        weights.layers[0].ffn.expert_w2.codes_batch_stride_bytes = keep / 2u;
        check(!moe::lower_qwen35_moe_to_primitives(config, weights).ok(),
              "an expert plane with a wrong batch stride is rejected");
        weights.layers[0].ffn.expert_w2.codes_batch_stride_bytes = keep;
    }
    check(moe::lower_qwen35_moe_to_primitives(config, weights).ok(),
          "the untouched weights lower again after the rejections");

    fs::remove_all(root, ec);
    std::printf("%s (%d passed, %d failed)\n", failed == 0 ? "PASS" : "FAIL", passed, failed);
    return failed == 0 ? 0 : 1;
}
