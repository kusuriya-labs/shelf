#include <shelf/inference/execution/program_executor.h>
#include <shelf/inference/execution/moe_dispatch.h>
#include <shelf/kernels/optimized/linear/exl3_grouped.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/linear/mxfp4.h>
#include <shelf/kernels/optimized/moe/combine.h>
#include <shelf/kernels/optimized/moe/router.h>
#include <shelf/quantization/mxfp4/mxfp4_native.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/runtime/program/moe_plan_scratch.h>
#include <shelf/runtime/program/program.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace rt = shelf::runtime;
namespace kn = shelf::kernel;
namespace qmx = shelf::quantization::mxfp4;
namespace ex = shelf::execution;

namespace {

int failed = 0;

void check(bool cond, const char* msg) {
    if (!cond) {
        ++failed;
        std::printf("FAIL: %s\n", msg);
    }
}

// Measured geometry of models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4.
constexpr uint32_t kExperts = 256u;
constexpr uint32_t kTopK = 8u;
constexpr uint32_t kHidden = 2048u;
constexpr uint32_t kInter = 512u;
constexpr uint32_t kFused = 2u * kInter;
constexpr uint32_t kTokens = 3u;
constexpr uint32_t kBucketRows = 16u;

template <typename T>
T* dalloc(std::size_t count) {
    T* p = nullptr;
    if (hipMalloc(&p, count * sizeof(T)) != hipSuccess) {
        std::printf("FAIL hipMalloc %zu\n", count);
        std::exit(1);
    }
    return p;
}

std::vector<uint8_t> host_bytes(std::size_t n, uint32_t seed) {
    std::vector<uint8_t> out(n);
    uint32_t s = seed != 0u ? seed : 1u;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        out[i] = static_cast<uint8_t>((s >> 16) & 0xffu);
    }
    return out;
}

rt::WeightSlot expert_plane(uint32_t rows, uint32_t cols, uint32_t experts) {
    rt::WeightSlot s{};
    s.rows = rows;
    s.cols = cols;
    s.k_padded = cols;
    s.encoding = static_cast<uint8_t>(shelf::weights::MatrixEncoding::Mxfp4);
    s.compute_spec =
        static_cast<uint8_t>(shelf::quantization::ComputeSpecId::MXFP4_W4A8_F32_BF16);
    s.codes_row_stride_bytes = (cols / 32u) * qmx::mxfp4_native_block_bytes;
    s.scale_row_stride_bytes = cols / 32u;
    s.storage_scale_stride_bytes = cols / 32u;
    s.weight_scale_group = 32u;
    s.preshuffled = true;
    s.experts = experts;
    s.codes_batch_stride_bytes = qmx::mxfp4_native_codes_batch_stride_bytes(rows, cols);
    s.scales_batch_stride_bytes = qmx::mxfp4_native_scale_batch_stride_bytes(rows, cols);
    return s;
}

struct MoeGraphValues {
    rt::ValueId post_norm;
    rt::ValueId logits;
    rt::ValueId shared_out;
    rt::ValueId gate;
    rt::ValueId ffn_out;
    rt::ValueId y13;
    rt::ValueId act;
    rt::ValueId expert_out;
    uint32_t w13_index = 0;
    uint32_t w2_index = 0;
};

// The routed experts and the routing round, with what the experts do not produce (router
// logits, shared expert output, activated gate) given as program inputs. The reference runs
// the same kernels over its own buffers, so the comparison tests the program's bindings:
// plan offsets, group rows, activation planes and weight strides. The kernels' arithmetic
// is Gate C's contract.
rt::PrimitiveGraph moe_program_graph(std::vector<rt::WeightSlot>& weights, MoeGraphValues& v) {
    rt::PrimitiveGraph g;
    g.grouped_row_shape = rt::GroupedRowShape{kExperts, kTopK};

    auto add_plane = [&](uint32_t rows, uint32_t cols) {
        weights.push_back(expert_plane(rows, cols, kExperts));
        g.weight_table.push_back(rt::MatrixwiseShapeKey{rows, cols});
        return static_cast<uint32_t>(weights.size() - 1);
    };
    v.w13_index = add_plane(kFused, kHidden);
    v.w2_index = add_plane(kHidden, kInter);

    v.post_norm = g.alloc_value(kHidden);
    v.logits = g.alloc_value(kExperts, rt::ValueDType::F32);
    v.shared_out = g.alloc_value(kHidden);
    v.gate = g.alloc_value(1u, rt::ValueDType::F32);
    for (rt::ValueId id : {v.post_norm, v.logits, v.shared_out, v.gate})
        g.external_inputs.push_back(id);

    {
        rt::MoeRouterNode n;
        n.experts = kExperts;
        n.top_k = kTopK;
        auto& gn = g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {v.logits};
    }

    v.y13 = g.alloc_value(kFused, rt::ValueDType::BF16, rt::ValueRowDomain::GROUPED_ROWS);
    {
        rt::MoeGroupedLinearNode n;
        n.shape = rt::MatrixwiseShapeKey{kFused, kHidden};
        n.weight_index = v.w13_index;
        auto& gn = g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {v.post_norm};
        gn.outputs = {v.y13};
    }
    const rt::ValueId gate_view =
        g.alloc_value(kInter, rt::ValueDType::BF16, rt::ValueRowDomain::GROUPED_ROWS);
    const rt::ValueId up_view =
        g.alloc_value(kInter, rt::ValueDType::BF16, rt::ValueRowDomain::GROUPED_ROWS);
    g.views.push_back(rt::PrimitiveValueView{gate_view, v.y13, 0u, kInter});
    g.views.push_back(rt::PrimitiveValueView{up_view, v.y13, kInter, kInter});

    v.act = g.alloc_value(kInter, rt::ValueDType::BF16, rt::ValueRowDomain::GROUPED_ROWS);
    {
        rt::SwiGluNode n{rt::RowwiseShapeKey{kInter}};
        auto& gn = g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {gate_view, up_view};
        gn.outputs = {v.act};
    }
    v.expert_out =
        g.alloc_value(kHidden, rt::ValueDType::BF16, rt::ValueRowDomain::GROUPED_ROWS);
    {
        rt::MoeGroupedLinearNode n;
        n.shape = rt::MatrixwiseShapeKey{kHidden, kInter};
        n.weight_index = v.w2_index;
        auto& gn = g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {v.act};
        gn.outputs = {v.expert_out};
    }
    v.ffn_out = g.alloc_value(kHidden);
    {
        rt::MoeCombineNode n;
        n.shape = rt::RowwiseShapeKey{kHidden};
        n.top_k = kTopK;
        auto& gn = g.add_node(rt::PrimitiveNode{n});
        gn.inputs = {v.expert_out, v.shared_out, v.gate};
        gn.outputs = {v.ffn_out};
    }
    g.external_outputs.push_back(v.ffn_out);
    return g;
}

uint32_t binding_stride(const rt::Program& program, rt::ValueId v) {
    const rt::ValueBinding* b = program.find_value(v);
    return b != nullptr ? b->row_stride : 0u;
}

uint64_t binding_rows(const rt::Program& program, rt::ValueId v) {
    const rt::ValueBinding* b = program.find_value(v);
    if (b == nullptr || b->row_stride == 0u) return 0ull;
    return b->bytes / (static_cast<uint64_t>(b->row_stride) *
                       rt::value_dtype_bytes(b->dtype));
}

void test_mul24_auto_dispatch(hipStream_t stream) {
    constexpr uint32_t tokens = 256u, experts = 2u, topk = 1u, k = 2560u, n = 640u, bits = 3u;
    const auto layout = rt::MoePlanScratch::make(tokens, experts, topk);
    const uint64_t act_bytes = rt::MoePlanScratch::mul24_bytes(layout.row_capacity, k);
    const uint64_t scratch_bytes = layout.mul24_offset() + act_bytes;
    auto* workspace = dalloc<uint8_t>(scratch_bytes);
    auto* words = reinterpret_cast<uint32_t*>(workspace);
    auto* input = dalloc<uint16_t>(static_cast<size_t>(tokens) * k);
    auto* output = dalloc<uint16_t>(static_cast<size_t>(layout.row_capacity) * n);
    auto* reference = dalloc<uint16_t>(static_cast<size_t>(layout.row_capacity) * n);
    const uint64_t plane_bytes = static_cast<uint64_t>(k) * n * bits / 8u;
    auto* codes = dalloc<uint8_t>(plane_bytes * experts);
    auto* suh = dalloc<uint16_t>(static_cast<size_t>(experts) * k);
    auto* svh = dalloc<uint16_t>(static_cast<size_t>(experts) * n);
    std::vector<uint16_t> in(static_cast<size_t>(tokens) * k);
    for (size_t i = 0; i < in.size(); ++i) in[i] = (i & 1u) ? 0x3e80u : 0xbe80u;
    std::vector<uint16_t> hs(static_cast<size_t>(experts) * k, 0x3c00u);
    std::vector<uint16_t> hv(static_cast<size_t>(experts) * n, 0x3c00u);
    std::vector<uint32_t> ge(16u, 0u), gr(16u, 16u), begin{0u, 16u, 16u};
    std::vector<int32_t> rows(layout.row_capacity, -1);
    for (uint32_t i = 0; i < tokens; ++i) rows[i] = static_cast<int32_t>(i);
    const uint32_t group_count = 16u;
    const size_t out_bytes = static_cast<size_t>(layout.row_capacity) * n * 2u;
    bool ok = hipMemset(workspace, 0, scratch_bytes) == hipSuccess &&
        hipMemset(codes, 0, plane_bytes * experts) == hipSuccess &&
        hipMemset(output, 0xff, out_bytes) == hipSuccess &&
        hipMemset(reference, 0xff, out_bytes) == hipSuccess &&
        hipMemcpy(input, in.data(), in.size() * 2u, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(suh, hs.data(), hs.size() * 2u, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(svh, hv.data(), hv.size() * 2u, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(words + layout.group_expert_offset, ge.data(), ge.size() * 4u, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(words + layout.group_rows_offset, gr.data(), gr.size() * 4u, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(words + layout.group_count_offset, &group_count, 4u, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(words + layout.expert_group_begin_offset, begin.data(), begin.size() * 4u, hipMemcpyHostToDevice) == hipSuccess &&
        hipMemcpy(words + layout.row_token_offset, rows.data(), rows.size() * 4u, hipMemcpyHostToDevice) == hipSuccess;
    rt::WeightSlot weight{};
    weight.rows = n; weight.cols = k; weight.k_padded = k; weight.experts = experts;
    weight.encoding = 4u;
    weight.compute_spec = static_cast<uint8_t>(shelf::quantization::ComputeSpecId::EXL3_MUL24_INT8_W8A8_F32_BF16);
    weight.codes = codes; weight.exl3_suh = suh; weight.exl3_svh = svh; weight.exl3_ka = bits;
    weight.codes_batch_stride_bytes = plane_bytes;
    weight.exl3_suh_batch_stride_bytes = static_cast<uint64_t>(k) * 2u;
    weight.exl3_svh_batch_stride_bytes = static_cast<uint64_t>(n) * 2u;
    rt::Program p;
    p.workspace.ranges.push_back(rt::WorkspaceRange{0u, static_cast<size_t>(scratch_bytes), 0u, 0u});
    rt::ValueBinding vin{}; vin.logical_value = rt::ValueId{0u}; vin.storage = rt::ValueStorage::EXTERNAL_INPUT;
    vin.feature_count = k; vin.row_stride = k;
    rt::ValueBinding vout{}; vout.logical_value = rt::ValueId{1u}; vout.storage = rt::ValueStorage::EXTERNAL_OUTPUT;
    vout.feature_count = n; vout.row_stride = n; vout.row_domain = rt::ValueRowDomain::GROUPED_ROWS;
    p.values = {vin, vout};
    rt::DispatchBinding b{}; b.input_count = 1u; b.output_count = 1u; b.input_slots[0] = 0u;
    b.output_slots[0] = 1u; b.weight_index = 0u; b.scratch_slot = 0u;
    b.scratch_tokens = tokens; b.scratch_experts = experts; b.scratch_top_k = topk;
    const void* inputs[] = {input}; void* outputs[] = {output};
    ex::HostExecutionContext ctx{}; ctx.weights = &weight; ctx.weight_count = 1u;
    ctx.workspace = workspace; ctx.actual_rows = tokens;
    ctx.external_inputs = inputs; ctx.external_input_count = 1u;
    ctx.external_outputs = outputs; ctx.external_output_count = 1u;
    setenv("SHELF_EXL3_MUL24_AUTO", "1", 1);
    auto launched = ex::try_launch_moe_grouped_exl3(p, b, ctx, stream);
    unsetenv("SHELF_EXL3_MUL24_AUTO");
    ok = ok && launched.ok();
    kn::Exl3GroupedArgs a{}; a.trellis = codes; a.suh = suh; a.svh = svh; a.input = input; a.output = reference;
    a.out_features = n; a.k = k; a.input_row_stride = k; a.output_row_stride = n; a.ka = bits; a.mul24 = 1u;
    a.int8_act = 1u; a.trellis_batch_stride = plane_bytes; a.suh_batch_stride = weight.exl3_suh_batch_stride_bytes;
    a.svh_batch_stride = weight.exl3_svh_batch_stride_bytes; a.group_expert = words + layout.group_expert_offset;
    a.group_rows = words + layout.group_rows_offset; a.group_count = words + layout.group_count_offset;
    a.expert_group_begin = words + layout.expert_group_begin_offset;
    a.row_token = reinterpret_cast<int32_t*>(words + layout.row_token_offset); a.group_capacity = layout.group_capacity;
    a.mul24_activation_workspace = workspace + layout.mul24_offset(); a.mul24_activation_workspace_bytes = act_bytes;
    ok = ok && kn::launch_gemm_exl3_w16a16_grouped(a, stream) == hipSuccess && hipStreamSynchronize(stream) == hipSuccess;
    std::vector<uint16_t> got(static_cast<size_t>(layout.row_capacity) * n), expected(got.size());
    ok = ok && hipMemcpy(got.data(), output, out_bytes, hipMemcpyDeviceToHost) == hipSuccess &&
         hipMemcpy(expected.data(), reference, out_bytes, hipMemcpyDeviceToHost) == hipSuccess;
    check(ok && got == expected, "mul24 の自動選択と scratch の実行結果が直接呼び出しと一致する");
    (void)hipFree(workspace); (void)hipFree(input); (void)hipFree(output); (void)hipFree(reference);
    (void)hipFree(codes); (void)hipFree(suh); (void)hipFree(svh);
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    hipStream_t stream = nullptr;
    if (hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) != hipSuccess) {
        std::printf("FAIL stream create\n");
        return 1;
    }

    std::vector<rt::WeightSlot> weights;
    MoeGraphValues v;
    const rt::PrimitiveGraph graph = moe_program_graph(weights, v);
    {
        const shelf::Status st = rt::validate_primitive_graph(graph);
        if (!st.ok()) {
            std::printf("FAIL graph: %s\n", st.message().c_str());
            return 1;
        }
    }
    const rt::WeightTableView weight_view{weights.data(), weights.size()};
    auto built = rt::build_program(graph, weight_view, rt::RowBucket::R16,
                                   rt::ExecutionClass::DECODE, rt::ProgramBuildOptions{});
    if (!built.ok()) {
        std::printf("FAIL build_program: %s\n", built.status().message().c_str());
        return 1;
    }
    const rt::Program program = built.release();

    const uint32_t slot_tiles = kn::moe_slot_tile_count(kTokens);
    const uint32_t slots = kTokens * kTopK;
    const uint32_t group_capacity = kn::moe_group_count_upper_bound(kTokens, kExperts, kTopK);
    const uint32_t row_capacity = kn::moe_group_row_capacity(kTokens, kExperts, kTopK);
    const rt::MoePlanScratch bucket_layout =
        rt::MoePlanScratch::make(kBucketRows, kExperts, kTopK);

    check(program.plan_scratch_slot != rt::kNoWorkspace, "the program carries a plan scratch");
    const auto& scratch_range = program.workspace.ranges[program.plan_scratch_slot];
    check(scratch_range.bytes >= bucket_layout.bytes(),
          "the plan scratch range fits the bucket capacity");
    check(binding_rows(program, v.y13) == bucket_layout.row_capacity &&
              binding_rows(program, v.expert_out) == bucket_layout.row_capacity,
          "group row values are bound at the bucket's group row capacity");

    for (auto& w : weights) {
        const std::size_t codes_bytes =
            static_cast<std::size_t>(w.codes_batch_stride_bytes * w.experts);
        const std::size_t scales_bytes =
            static_cast<std::size_t>(w.scales_batch_stride_bytes * w.experts);
        const std::size_t ref_bytes = static_cast<std::size_t>(w.rows) * w.experts;
        const std::vector<uint8_t> codes = host_bytes(codes_bytes, 0x9e3779b9u);
        const std::vector<uint8_t> scales = host_bytes(scales_bytes, 0x85ebca6bu);
        const std::vector<uint8_t> row_ref(ref_bytes, 127u);
        void* dc = dalloc<uint8_t>(codes_bytes);
        void* ds = dalloc<uint8_t>(scales_bytes);
        void* dr = dalloc<uint8_t>(ref_bytes);
        if (hipMemcpy(dc, codes.data(), codes_bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(ds, scales.data(), scales_bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(dr, row_ref.data(), ref_bytes, hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL plane upload\n");
            return 1;
        }
        w.codes = dc;
        w.scales = ds;
        w.row_ref = dr;
        w.row_ref_exact = true;
    }
    rt::WeightSlot* device_weights = nullptr;
    {
        const rt::WeightTableView view{weights.data(), weights.size()};
        const shelf::Status up = ex::upload_host_weight_table(view, &device_weights, stream);
        if (!up.ok()) {
            std::printf("FAIL weight table upload: %s\n", up.message().c_str());
            return 1;
        }
    }

    const uint32_t post_stride = binding_stride(program, v.post_norm);
    const uint32_t logits_stride = binding_stride(program, v.logits);
    const uint32_t shared_stride = binding_stride(program, v.shared_out);
    const uint32_t gate_stride = binding_stride(program, v.gate);
    const uint32_t y13_stride = binding_stride(program, v.y13);
    const uint32_t act_stride = binding_stride(program, v.act);
    const uint32_t expert_stride = binding_stride(program, v.expert_out);
    check(y13_stride >= kFused && act_stride >= kInter && expert_stride >= kHidden,
          "group row bindings carry usable row strides");

    auto* d_post = dalloc<shelf::bf16_t>(static_cast<std::size_t>(kBucketRows) * post_stride);
    auto* d_logits = dalloc<float>(static_cast<std::size_t>(kBucketRows) * logits_stride);
    auto* d_shared = dalloc<shelf::bf16_t>(static_cast<std::size_t>(kBucketRows) * shared_stride);
    auto* d_gate = dalloc<float>(kBucketRows * gate_stride);
    const std::vector<uint8_t> post_raw =
        host_bytes(static_cast<std::size_t>(kBucketRows) * post_stride * 2u, 0x1234abcdU);
    std::vector<float> logits_raw(static_cast<std::size_t>(kBucketRows) * logits_stride);
    for (std::size_t i = 0; i < logits_raw.size(); ++i)
        logits_raw[i] = static_cast<float>(static_cast<int>((i * 37u) % 200u) - 100) * 0.05f;
    const std::vector<uint8_t> shared_raw =
        host_bytes(static_cast<std::size_t>(kBucketRows) * shared_stride * 2u, 0x51ed270bU);
    std::vector<float> gate_raw(kBucketRows * gate_stride, 0.0f);
    for (uint32_t r = 0; r < kBucketRows; ++r)
        gate_raw[r * gate_stride] = 0.25f + 0.0625f * static_cast<float>(r);

    if (hipMemcpy(d_post, post_raw.data(), post_raw.size(), hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_logits, logits_raw.data(), logits_raw.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_shared, shared_raw.data(), shared_raw.size(),
                  hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(d_gate, gate_raw.data(), gate_raw.size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
        std::printf("FAIL input upload\n");
        return 1;
    }

    uint8_t* d_workspace = dalloc<uint8_t>(program.workspace.total_bytes);
    if (hipMemset(d_workspace, 0, program.workspace.total_bytes) != hipSuccess) {
        std::printf("FAIL workspace memset\n");
        return 1;
    }
    uint32_t* d_error_word = dalloc<uint32_t>(1);

    ex::HostExecutionContext ctx;
    ctx.weights = device_weights;
    ctx.weight_count = static_cast<uint32_t>(weights.size());
    ctx.workspace = d_workspace;
    ctx.workspace_bytes = program.workspace.total_bytes;
    ctx.error_word = d_error_word;
    ctx.actual_rows = kTokens;
    ctx.actual_outputs = 0u;
    ctx.bucket_m = kBucketRows;
    const void* inputs[4] = {d_post, d_logits, d_shared, d_gate};
    void* program_out = nullptr;
    const uint32_t out_stride = binding_stride(program, v.ffn_out);
    auto* d_prog_out =
        dalloc<shelf::bf16_t>(static_cast<std::size_t>(kBucketRows) * out_stride);
    program_out = d_prog_out;
    void* outs[1] = {program_out};
    ctx.external_inputs = inputs;
    ctx.external_input_count = 4u;
    ctx.external_outputs = outs;
    ctx.external_output_count = 1u;

    const shelf::Status run = ex::execute_program(program, ctx, stream);
    if (!run.ok()) {
        std::printf("FAIL execute_program: %s\n", run.message().c_str());
        return 1;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL program sync\n");
        return 1;
    }
    std::vector<uint16_t> prog_out(static_cast<std::size_t>(kTokens) * kHidden);
    if (hipMemcpy(prog_out.data(), d_prog_out, prog_out.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL program output readback\n");
        return 1;
    }

    // A second run over the same workspace: the router accumulates into the plan scratch, so
    // the program run has to start from a zeroed scratch again. Nothing else is re-zeroed.
    if (hipMemset(d_prog_out, 0, prog_out.size() * sizeof(uint16_t)) != hipSuccess) {
        std::printf("FAIL program output clear\n");
        return 1;
    }
    const shelf::Status second = ex::execute_program(program, ctx, stream);
    if (!second.ok()) {
        std::printf("FAIL execute_program (second run): %s\n", second.message().c_str());
        return 1;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL second run sync\n");
        return 1;
    }
    std::vector<uint16_t> second_out(prog_out.size(), 0u);
    if (hipMemcpy(second_out.data(), d_prog_out, second_out.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL second run readback\n");
        return 1;
    }
    check(second_out == prog_out, "a second program run over the same workspace is identical");

    // A clean run leaves the error word alone, and one non-finite logit reports the run as a
    // numeric error without faulting it: the caller stops on the word instead of the plan
    // arrays being walked past their capacity.
    uint32_t clean_word = 0xFFFFFFFFu;
    if (hipMemcpy(&clean_word, d_error_word, sizeof(clean_word), hipMemcpyDeviceToHost) !=
        hipSuccess) {
        std::printf("FAIL clean error word readback\n");
        return 1;
    }
    check(clean_word == static_cast<uint32_t>(rt::ProgramStatus::COMPLETE),
          "a clean program run leaves the error word at COMPLETE");
    {
        std::vector<float> bad(static_cast<std::size_t>(kTokens) * logits_stride, 0.0f);
        bad[0] = std::numeric_limits<float>::quiet_NaN();
        if (hipMemcpy(d_logits, bad.data(), bad.size() * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess) {
            std::printf("FAIL NaN logits upload\n");
            return 1;
        }
        const shelf::Status numeric = ex::execute_program(program, ctx, stream);
        check(numeric.ok(), "a non-finite logit does not fault the program run");
        check(hipStreamSynchronize(stream) == hipSuccess, "non-finite logit run sync");
        uint32_t word = 0u;
        if (hipMemcpy(&word, d_error_word, sizeof(word), hipMemcpyDeviceToHost) != hipSuccess) {
            std::printf("FAIL numeric error word readback\n");
            return 1;
        }
        check(word == static_cast<uint32_t>(rt::ProgramStatus::NUMERIC_ERROR),
              "one non-finite logit reports the run as NUMERIC_ERROR");
    }

    // Reference: the same kernels, separate arrays, arguments derived from the plan contract.
    int32_t* r_sel_expert = dalloc<int32_t>(slots);
    float* r_sel_weight = dalloc<float>(slots);
    uint32_t* r_tile_counts = dalloc<uint32_t>(static_cast<std::size_t>(kExperts) * slot_tiles);
    uint32_t* r_row_begin = dalloc<uint32_t>(kExperts + 1u);
    uint32_t* r_tile_base = dalloc<uint32_t>(static_cast<std::size_t>(kExperts) * slot_tiles);
    uint32_t* r_group_begin = dalloc<uint32_t>(kExperts + 1u);
    uint32_t* r_group_expert = dalloc<uint32_t>(group_capacity);
    uint32_t* r_group_rows = dalloc<uint32_t>(group_capacity);
    uint32_t* r_group_count = dalloc<uint32_t>(1);
    int32_t* r_row_token = dalloc<int32_t>(row_capacity);
    int32_t* r_slot_row_of = dalloc<int32_t>(slots);
    const auto layout13 = rt::Int8ActivationWorkspaceLayout::make(kHidden, row_capacity);
    const auto layout2 = rt::Int8ActivationWorkspaceLayout::make(kInter, row_capacity);
    uint8_t* r_plane13 = dalloc<uint8_t>(layout13.total_bytes);
    float* r_plane13_scales =
        reinterpret_cast<float*>(r_plane13 + layout13.scales_offset);
    uint8_t* r_plane2 = dalloc<uint8_t>(layout2.total_bytes);
    float* r_plane2_scales = reinterpret_cast<float*>(r_plane2 + layout2.scales_offset);
    shelf::bf16_t* r_y13 =
        dalloc<shelf::bf16_t>(static_cast<std::size_t>(row_capacity) * y13_stride);
    shelf::bf16_t* r_act =
        dalloc<shelf::bf16_t>(static_cast<std::size_t>(row_capacity) * act_stride);
    shelf::bf16_t* r_expert =
        dalloc<shelf::bf16_t>(static_cast<std::size_t>(row_capacity) * expert_stride);
    shelf::bf16_t* r_out = dalloc<shelf::bf16_t>(static_cast<std::size_t>(kTokens) * kHidden);

    const std::size_t counts_bytes =
        static_cast<std::size_t>(kExperts) * slot_tiles * sizeof(uint32_t);
    if (hipMemset(r_tile_counts, 0, counts_bytes) != hipSuccess) {
        std::printf("FAIL reference counts memset\n");
        return 1;
    }
    if (kn::launch_moe_router(d_logits, logits_stride, r_sel_expert, r_sel_weight,
                              r_tile_counts, kTokens, kExperts, kTopK, 0u, kExperts, nullptr,
                              stream) != hipSuccess)
        check(false, "reference router launch");
    if (kn::launch_moe_slot_scan(r_tile_counts, r_row_begin, r_tile_base, r_group_begin,
                                 r_group_expert, r_group_rows, r_group_count, r_row_token,
                                 kTokens, kExperts, kTopK, stream) != hipSuccess)
        check(false, "reference scan launch");
    if (kn::launch_moe_slot_scatter(r_sel_expert, r_tile_base, r_row_begin, r_group_begin,
                                    r_row_token, r_slot_row_of, kTokens, kExperts, kTopK,
                                    stream) != hipSuccess)
        check(false, "reference scatter launch");

    if (kn::launch_activation_quantize_mxfp4_rowref(
            d_post, r_plane13, r_plane13_scales, row_capacity, kHidden, kHidden, post_stride,
            layout13.code_row_stride_bytes, layout13.scale_row_stride_bytes, r_row_token,
            stream) != hipSuccess)
        check(false, "reference w13 plane launch");
    if (kn::launch_gemm_mxfp4_w4a8_grouped(
            static_cast<const uint8_t*>(weights[v.w13_index].codes),
            static_cast<const uint8_t*>(weights[v.w13_index].scales),
            static_cast<const uint8_t*>(weights[v.w13_index].row_ref),
            reinterpret_cast<const int8_t*>(r_plane13), r_plane13_scales, r_y13,
            kn::Mxfp4GemmOutputDType::BF16, kFused, kHidden, kFused,
            weights[v.w13_index].codes_batch_stride_bytes,
            weights[v.w13_index].scales_batch_stride_bytes,
            weights[v.w13_index].scale_row_stride_bytes, layout13.code_row_stride_bytes,
            layout13.scale_row_stride_bytes, y13_stride, r_group_expert, r_group_rows,
            r_group_count, group_capacity, stream) != hipSuccess)
        check(false, "reference grouped w13 launch");
    if (kn::launch_swiglu_bf16(r_y13, y13_stride, r_y13 + kInter, y13_stride, r_act, act_stride,
                               row_capacity, kInter, stream) != hipSuccess)
        check(false, "reference swiglu launch");
    if (kn::launch_activation_quantize_mxfp4_rowref(
            r_act, r_plane2, r_plane2_scales, row_capacity, kInter, kInter, act_stride,
            layout2.code_row_stride_bytes, layout2.scale_row_stride_bytes, nullptr,
            stream) != hipSuccess)
        check(false, "reference w2 plane launch");
    if (kn::launch_gemm_mxfp4_w4a8_grouped(
            static_cast<const uint8_t*>(weights[v.w2_index].codes),
            static_cast<const uint8_t*>(weights[v.w2_index].scales),
            static_cast<const uint8_t*>(weights[v.w2_index].row_ref),
            reinterpret_cast<const int8_t*>(r_plane2), r_plane2_scales, r_expert,
            kn::Mxfp4GemmOutputDType::BF16, kHidden, kInter, kHidden,
            weights[v.w2_index].codes_batch_stride_bytes,
            weights[v.w2_index].scales_batch_stride_bytes,
            weights[v.w2_index].scale_row_stride_bytes, layout2.code_row_stride_bytes,
            layout2.scale_row_stride_bytes, expert_stride, r_group_expert, r_group_rows,
            r_group_count, group_capacity, stream) != hipSuccess)
        check(false, "reference grouped w2 launch");
    if (kn::launch_moe_combine(r_out, r_expert, r_sel_weight, r_slot_row_of, d_shared, d_gate,
                               kTokens, kTopK, kHidden, expert_stride, shared_stride,
                               gate_stride, stream) != hipSuccess)
        check(false, "reference combine launch");
    if (hipStreamSynchronize(stream) != hipSuccess) {
        std::printf("FAIL reference sync\n");
        return 1;
    }

    std::vector<uint16_t> ref_out(static_cast<std::size_t>(kTokens) * kHidden);
    if (hipMemcpy(ref_out.data(), r_out, ref_out.size() * sizeof(uint16_t),
                  hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("FAIL reference readback\n");
        return 1;
    }

    size_t first_mismatch = ref_out.size();
    for (size_t i = 0; i < ref_out.size(); ++i) {
        if (prog_out[i] != ref_out[i]) {
            first_mismatch = i;
            break;
        }
    }
    if (first_mismatch != ref_out.size()) {
        std::printf("      first mismatch at %zu program=%d reference=%d\n", first_mismatch,
                    static_cast<int>(prog_out[first_mismatch]),
                    static_cast<int>(ref_out[first_mismatch]));
    }
    check(first_mismatch == ref_out.size(),
          "the program path and the direct kernel path give the same MoE FFN");

    test_mul24_auto_dispatch(stream);
    std::printf("%s (%d failed)\n", failed == 0 ? "PASS" : "FAIL", failed);
    return failed == 0 ? 0 : 1;
}
