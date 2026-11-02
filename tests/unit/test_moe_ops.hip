#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/models/common/qwen_hybrid/moe_math.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace rt = ps::runtime;
namespace qh = ps::qwen_hybrid;

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

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

float uniform01(uint64_t x) {
    return static_cast<float>(static_cast<int32_t>(h32(x) % 2000u) - 1000) /
           1000.0f;
}

float widen_bf16(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::uint16_t bf16_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (std::uint16_t)(u >> 16);
}

float bf16_round(float f) { return widen_bf16(bf16_bits(f)); }

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    ~Buf() { free_dev(); }
    void free_dev() {
        if (dev) {
            (void)hipFree(dev);
            dev = nullptr;
        }
    }
    bool alloc(size_t b) {
        free_dev();
        bytes = b;
        return hipMalloc(&dev, b) == hipSuccess;
    }
    bool fill(const void* src, size_t b) {
        return hipMemcpy(dev, src, b, hipMemcpyHostToDevice) == hipSuccess;
    }
    bool read(void* dst, size_t b) const {
        return hipMemcpy(dst, dev, b, hipMemcpyDeviceToHost) == hipSuccess;
    }
};

struct Metrics {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rmse = 0.0;
    bool non_finite = false;
};

Metrics compare(const float* got, const float* ref, size_t count) {
    Metrics m;
    double sum2 = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double g = got[i];
        if (!std::isfinite(g)) {
            m.non_finite = true;
            return m;
        }
        const double r = ref[i];
        const double da = std::fabs(g - r);
        const double dr = da / std::max(1e-6, std::fabs(r));
        m.max_abs = std::max(m.max_abs, da);
        m.max_rel = std::max(m.max_rel, dr);
        sum2 += da * da;
    }
    m.rmse = count != 0 ? std::sqrt(sum2 / static_cast<double>(count)) : 0.0;
    return m;
}

void print_metrics(const char* tag, const Metrics& m) {
    std::printf("  %-28s max_abs=%.6e max_rel=%.6e rmse=%.6e%s\n", tag, m.max_abs,
                m.max_rel, m.rmse, m.non_finite ? " NON-FINITE" : "");
}

struct DeviceCtx {
    rt::DeviceBatchContext h{};
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    void* values_dev = nullptr;
    void* weights_dev = nullptr;
    void* workspace = nullptr;
    ~DeviceCtx() {
        void* bufs[] = {ctx_dev, scratch, err, values_dev, weights_dev, workspace};
        for (void* q : bufs)
            if (q) (void)hipFree(q);
    }
};

int run_plan(const rt::Program& plan,
             const std::vector<rt::WeightSlot>& host_weights, DeviceCtx& dc,
             uint32_t rows, hipStream_t stream) {
    dc.h = {};
    dc.h.actual_rows = rows;
    if (hipMalloc(&dc.ctx_dev, sizeof(dc.h)) != hipSuccess) return -1;
    if (hipMemcpy(dc.ctx_dev, &dc.h, sizeof(dc.h), hipMemcpyHostToDevice) !=
        hipSuccess)
        return -1;
    if (hipMalloc(&dc.scratch, (size_t)256 * 4096 * sizeof(float)) != hipSuccess)
        return -1;
    if (hipMalloc(&dc.err, sizeof(uint32_t)) != hipSuccess) return -1;

    std::vector<rt::DeviceValueBinding> vals(plan.values.size());
    for (size_t i = 0; i < plan.values.size(); ++i) {
        const auto& s = plan.values[i];
        auto& d = vals[i];
        d.logical_value = s.logical_value.id;
        d.storage = s.storage;
        d.dtype = s.dtype;
        d.row_domain = s.row_domain;
        d.slot = s.slot;
        d.feature_count = s.feature_count;
        d.row_stride = s.row_stride;
        d.offset = s.offset;
        d.bytes = s.bytes;
    }
    for (size_t i = 0; i < plan.values.size(); ++i) {
        const auto& s = plan.values[i];
        if ((s.storage == rt::ValueStorage::EXTERNAL_INPUT ||
             s.storage == rt::ValueStorage::EXTERNAL_OUTPUT) &&
            s.offset == 0) {
            std::printf("FAIL: external value %u lacks patched device pointer\n",
                        s.logical_value.id);
            return -1;
        }
    }
    if (hipMalloc(&dc.values_dev, vals.size() * sizeof(rt::DeviceValueBinding)) !=
        hipSuccess)
        return -1;
    if (hipMemcpy(dc.values_dev, vals.data(), vals.size() * sizeof(rt::DeviceValueBinding),
                  hipMemcpyHostToDevice) != hipSuccess)
        return -1;

    if (!host_weights.empty()) {
        if (hipMalloc(&dc.weights_dev, host_weights.size() * sizeof(rt::WeightSlot)) !=
            hipSuccess)
            return -1;
        if (hipMemcpy(dc.weights_dev, host_weights.data(),
                      host_weights.size() * sizeof(rt::WeightSlot),
                      hipMemcpyHostToDevice) != hipSuccess)
            return -1;
    }
    if (plan.workspace.total_bytes != 0) {
        if (hipMalloc(&dc.workspace, plan.workspace.total_bytes) != hipSuccess)
            return -1;
    }

    rt::DeviceProgramView view{};
    view.values = static_cast<const rt::DeviceValueBinding*>(dc.values_dev);
    view.value_count = static_cast<uint32_t>(vals.size());
    view.weight_table = dc.weights_dev;
    view.weight_slot_count = static_cast<uint32_t>(host_weights.size());
    view.workspace_base = reinterpret_cast<uint64_t>(dc.workspace);

    for (size_t di = 0; di < plan.dispatches.size(); ++di) {
        ps::gpu::discard_cleanup_result(hipMemsetAsync(dc.err, 0, sizeof(uint32_t), stream));
        ::ps::kernel::ModelDispatchStateView sv{};
        const ps::Status st = ::ps::kernel::launch_model_dispatch_correctness(
            view, plan.dispatches[di],
            reinterpret_cast<rt::DeviceBatchContext*>(dc.ctx_dev), sv,
            static_cast<float*>(dc.scratch), static_cast<uint32_t*>(dc.err), stream);
        if (!st.ok()) {
            std::printf("FAIL: dispatch %zu launch: %s\n", di, st.message().c_str());
            return -1;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) return -1;
        uint32_t e = 0;
        if (hipMemcpy(&e, dc.err, 4, hipMemcpyDeviceToHost) != hipSuccess) return -1;
        if (e != 0u) return static_cast<int>(e);
    }
    return 0;
}

rt::WeightSlot make_bf16_weight(const void* dev_ptr, uint32_t rows, uint32_t cols) {
    rt::WeightSlot w;
    w.rows = rows;
    w.cols = cols;
    w.device_ptr = dev_ptr;
    w.bytes = static_cast<uint64_t>(rows) * cols * 2ull;
    w.encoding = 0;
    return w;
}

void patch_external(rt::Program& plan, rt::ValueId id, void* dev_ptr) {
    for (auto& v : plan.values) {
        if (v.logical_value.id == id.id) {
            v.offset = reinterpret_cast<uint64_t>(dev_ptr);
        }
    }
}

constexpr uint32_t kHidden = 64;
constexpr uint32_t kInter = 32;
constexpr uint32_t kExperts = 8;
constexpr uint32_t kTopK = 3;
constexpr uint32_t kRouterSlot = 0;
constexpr uint32_t kGateUpSlot = 1;
constexpr uint32_t kDownSlot = kGateUpSlot + kExperts;
constexpr uint32_t kWeightSlots = kDownSlot + kExperts;

enum class CaseKind { NORMAL, ZERO_X, TIE_WEIGHTS, EXTREME_X, SAME_X };

struct Case {
    const char* name;
    uint32_t tokens;
    bool renorm;
    CaseKind kind;
};

const Case kCases[] = {
    {"random_t3_renorm", 3, true, CaseKind::NORMAL},
    {"zero_t2_full_tie", 2, true, CaseKind::ZERO_X},
    {"tie_t3_boundary", 3, true, CaseKind::TIE_WEIGHTS},
    {"extreme_t2_renorm", 2, true, CaseKind::EXTREME_X},
    {"same_t4_renorm", 4, true, CaseKind::SAME_X},
    {"random_t3_raw", 3, false, CaseKind::NORMAL},
};

struct HostWeights {
    std::vector<float> router;
    std::vector<float> gate_up;
    std::vector<float> down;
};

void fill_router(HostWeights& w, CaseKind kind) {
    w.router.assign(static_cast<size_t>(kExperts) * kHidden, 0.0f);
    const float wscale = 0.05f / std::sqrt(static_cast<float>(kHidden));
    for (uint32_t e = 0; e < kExperts; ++e)
        for (uint32_t h = 1; h < kHidden; ++h)
            w.router[static_cast<size_t>(e) * kHidden + h] =
                bf16_round(wscale * uniform01(1000u + e * kHidden + h));
    if (kind == CaseKind::TIE_WEIGHTS) {
        const float offs[kExperts] = {5.0f, 4.0f, 3.0f, 3.0f,
                                      2.0f, 2.0f, 2.0f, 1.0f};
        for (uint32_t e = 0; e < kExperts; ++e)
            w.router[static_cast<size_t>(e) * kHidden] = offs[e];
        std::memcpy(w.router.data() + 3ull * kHidden,
                    w.router.data() + 2ull * kHidden, kHidden * sizeof(float));
        std::memcpy(w.router.data() + 5ull * kHidden,
                    w.router.data() + 4ull * kHidden, 2ull * kHidden * sizeof(float));
    } else {
        const float offs[kExperts] = {5.0f, 4.0f, 3.0f, 2.0f,
                                      1.0f, 0.0f, -1.0f, -2.0f};
        for (uint32_t e = 0; e < kExperts; ++e)
            w.router[static_cast<size_t>(e) * kHidden] = offs[e];
    }
}

void fill_experts(HostWeights& w, uint64_t seed) {
    w.gate_up.assign(static_cast<size_t>(kExperts) * 2ull * kInter * kHidden, 0.0f);
    w.down.assign(static_cast<size_t>(kExperts) * kHidden * kInter, 0.0f);
    const float gscale = 1.0f / std::sqrt(static_cast<float>(kHidden));
    const float dscale = 1.0f / std::sqrt(static_cast<float>(kInter));
    for (size_t i = 0; i < w.gate_up.size(); ++i)
        w.gate_up[i] = bf16_round(gscale * uniform01(seed + i));
    for (size_t i = 0; i < w.down.size(); ++i)
        w.down[i] = bf16_round(dscale * uniform01(seed + 900000u + i));
}

void fill_input(std::vector<float>& raw, std::vector<float>& bf, uint32_t tokens,
                CaseKind kind) {
    raw.assign(static_cast<size_t>(tokens) * kHidden, 0.0f);
    for (uint32_t t = 0; t < tokens; ++t) {
        const bool same = kind == CaseKind::SAME_X;
        for (uint32_t h = 0; h < kHidden; ++h) {
            const uint64_t key = same ? (700000u + h)
                                      : (500000u + static_cast<uint64_t>(t) * kHidden + h);
            float v = 0.1f * uniform01(key);
            if (h == 0) v = 1.0f;
            if (kind == CaseKind::ZERO_X) v = 0.0f;
            if (kind == CaseKind::EXTREME_X) v = 80.0f * uniform01(key);
            raw[static_cast<size_t>(t) * kHidden + h] = v;
        }
    }
    bf.resize(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) bf[i] = bf16_round(raw[i]);
}

void oracle_moe(const std::vector<float>& x_bf, const HostWeights& w,
                uint32_t tokens, bool renorm, std::vector<std::uint32_t>& top_i,
                std::vector<float>& top_w, std::vector<float>& routed) {
    const size_t te = static_cast<size_t>(tokens) * kExperts;
    const size_t tk = static_cast<size_t>(tokens) * kTopK;
    const size_t th = static_cast<size_t>(tokens) * kHidden;
    std::vector<float> probs(te, 0.0f);
    qh::moe_router_logits_f32(x_bf.data(), w.router.data(), tokens, kHidden, kExperts,
                              probs.data());
    qh::moe_softmax_f32(probs.data(), tokens, kExperts);
    top_i.assign(tk, 0u);
    top_w.assign(tk, 0.0f);
    qh::moe_topk_stable_f32(probs.data(), tokens, kExperts, kTopK, top_i.data(),
                            top_w.data());
    if (renorm) qh::moe_normalize_topk_f32(top_w.data(), tokens, kTopK);

    routed.assign(th, 0.0f);
    std::vector<float> xrow(kHidden);
    std::vector<float> gu(static_cast<size_t>(2) * kInter * kHidden);
    std::vector<float> dn(static_cast<size_t>(kHidden) * kInter);
    std::vector<float> out_e(kHidden);
    for (uint32_t t = 0; t < tokens; ++t) {
        std::memcpy(xrow.data(), x_bf.data() + static_cast<size_t>(t) * kHidden,
                    kHidden * sizeof(float));
        float* acc = routed.data() + static_cast<size_t>(t) * kHidden;
        for (uint32_t s = 0; s < kTopK; ++s) {
            const std::uint32_t e = top_i[static_cast<size_t>(t) * kTopK + s];
            const float wgt = top_w[static_cast<size_t>(t) * kTopK + s];
            std::memcpy(gu.data(),
                        w.gate_up.data() +
                            static_cast<size_t>(e) * 2ull * kInter * kHidden,
                        gu.size() * sizeof(float));
            std::memcpy(dn.data(),
                        w.down.data() +
                            static_cast<size_t>(e) * kHidden * kInter,
                        dn.size() * sizeof(float));
            qh::moe_expert_forward_f32(xrow.data(), gu.data(), dn.data(), kInter,
                                       kHidden, out_e.data());
            for (uint32_t h = 0; h < kHidden; ++h) acc[h] += wgt * out_e[h];
        }
    }
}

void register_moe_weights(rt::PrimitiveGraph& g) {
    g.register_weight({kExperts, kHidden});
    for (uint32_t e = 0; e < kExperts; ++e) g.register_weight({2ull * kInter, kHidden});
    for (uint32_t e = 0; e < kExperts; ++e) g.register_weight({kHidden, kInter});
}

rt::ValueId add_router(rt::PrimitiveGraph& graph, rt::ValueId x_in, bool renorm) {
    rt::MoERouterNode n;
    n.weight_shape = {kExperts, kHidden};
    n.weight_index = kRouterSlot;
    n.experts = kExperts;
    n.top_k = kTopK;
    n.renorm = renorm;
    auto& gn = graph.add_node(n);
    gn.inputs = {x_in};
    const rt::ValueId top_i = graph.alloc_value(kTopK, rt::ValueDType::I32);
    const rt::ValueId top_w = graph.alloc_value(kTopK, rt::ValueDType::F32);
    gn.outputs = {top_i, top_w};
    gn.debug_name = "moe_router";
    return top_i;
}

rt::ValueId add_experts(rt::PrimitiveGraph& graph, rt::ValueId x_in, rt::ValueId top_i,
                        rt::ValueId top_w) {
    rt::MoEExpertsNode n;
    n.gate_up_shape = {2ull * kInter, kHidden};
    n.down_shape = {kHidden, kInter};
    n.weight_index = kGateUpSlot;
    n.experts = kExperts;
    n.top_k = kTopK;
    auto& gn = graph.add_node(n);
    gn.inputs = {x_in, top_i, top_w};
    const rt::ValueId out = graph.alloc_value(kHidden, rt::ValueDType::F32);
    gn.outputs = {out};
    gn.debug_name = "moe_experts";
    return out;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;
    hipStream_t stream = nullptr;
    if (hipStreamCreate(&stream) != hipSuccess) return 1;

    HostWeights hw;
    fill_router(hw, CaseKind::NORMAL);
    fill_experts(hw, 42u);

    std::vector<std::uint16_t> router_bits(hw.router.size());
    std::vector<std::uint16_t> gu_bits(hw.gate_up.size());
    std::vector<std::uint16_t> dn_bits(hw.down.size());
    for (size_t i = 0; i < hw.router.size(); ++i)
        router_bits[i] = bf16_bits(hw.router[i]);
    for (size_t i = 0; i < hw.gate_up.size(); ++i) gu_bits[i] = bf16_bits(hw.gate_up[i]);
    for (size_t i = 0; i < hw.down.size(); ++i) dn_bits[i] = bf16_bits(hw.down[i]);

    Buf router_dev, gu_dev, dn_dev;
    if (!router_dev.alloc(router_bits.size() * 2) ||
        !router_dev.fill(router_bits.data(), router_bits.size() * 2) ||
        !gu_dev.alloc(gu_bits.size() * 2) ||
        !gu_dev.fill(gu_bits.data(), gu_bits.size() * 2) ||
        !dn_dev.alloc(dn_bits.size() * 2) ||
        !dn_dev.fill(dn_bits.data(), dn_bits.size() * 2)) {
        std::printf("FAIL: weight device allocation\n");
        return 1;
    }

    std::vector<rt::WeightSlot> weight_slots(kWeightSlots);
    weight_slots[kRouterSlot] = make_bf16_weight(router_dev.dev, kExperts, kHidden);
    for (uint32_t e = 0; e < kExperts; ++e) {
        weight_slots[kGateUpSlot + e] = make_bf16_weight(
            static_cast<std::uint8_t*>(gu_dev.dev) +
                static_cast<size_t>(e) * 2ull * kInter * kHidden * 2ull,
            2ull * kInter, kHidden);
        weight_slots[kDownSlot + e] = make_bf16_weight(
            static_cast<std::uint8_t*>(dn_dev.dev) +
                static_cast<size_t>(e) * kHidden * kInter * 2ull,
            kHidden, kInter);
    }
    const rt::WeightTableView weight_view{weight_slots.data(), weight_slots.size()};

    rt::ProgramBuildOptions options;
    options.validate_workspace = true;
    options.max_token_rows = 8;
    options.max_output_rows = 8;

    {
        rt::PrimitiveGraph g;
        register_moe_weights(g);
        const rt::ValueId x = g.alloc_value(kHidden, rt::ValueDType::BF16);
        const rt::ValueId top_i = add_router(g, x, true);
        g.external_inputs.push_back(x);
        g.external_outputs.push_back(top_i);
        g.external_outputs.push_back(g.nodes[0].outputs[1]);
        const ps::Status vst = rt::validate_primitive_graph(g);
        check(vst.ok(), std::string("router-only graph validates")
                            + (vst.ok() ? "" : " (" + vst.message() + ")"));
    }

    {
        rt::PrimitiveGraph g;
        register_moe_weights(g);
        const rt::ValueId x = g.alloc_value(kHidden, rt::ValueDType::BF16);
        add_router(g, x, true);
        add_experts(g, x, g.nodes[0].outputs[0], g.nodes[0].outputs[1]);
        g.external_inputs.push_back(x);
        g.external_outputs.push_back(g.nodes[1].outputs[0]);
        const ps::Status vst = rt::validate_primitive_graph(g);
        check(vst.ok(), std::string("router+experts graph validates")
                            + (vst.ok() ? "" : " (" + vst.message() + ")"));
    }

    {
        rt::PrimitiveGraph g;
        register_moe_weights(g);
        const rt::ValueId x = g.alloc_value(kHidden, rt::ValueDType::BF16);
        rt::MoEExpertsNode n;
        n.gate_up_shape = {2ull * kInter, kHidden};
        n.down_shape = {kHidden, kInter};
        n.weight_index = 0;
        n.experts = kExperts;
        n.top_k = kTopK;
        auto& gn = g.add_node(n);
        const rt::ValueId top_i = g.alloc_value(kTopK, rt::ValueDType::I32);
        const rt::ValueId top_w = g.alloc_value(kTopK, rt::ValueDType::F32);
        gn.inputs = {x, top_i, top_w};
        const rt::ValueId out = g.alloc_value(kHidden, rt::ValueDType::F32);
        gn.outputs = {out};
        (void)out;
        const ps::Status vst = rt::validate_primitive_graph(g);
        check(!vst.ok(), "experts overlapping router slot rejected");
    }

    {
        rt::PrimitiveGraph g;
        register_moe_weights(g);
        const rt::ValueId x = g.alloc_value(kHidden, rt::ValueDType::BF16);
        rt::MoERouterNode n;
        n.weight_shape = {kExperts, kHidden};
        n.weight_index = kRouterSlot;
        n.experts = kExperts;
        n.top_k = kExperts + 1u;
        auto& gn = g.add_node(n);
        gn.inputs = {x};
        const rt::ValueId top_i = g.alloc_value(kTopK, rt::ValueDType::I32);
        const rt::ValueId top_w = g.alloc_value(kTopK, rt::ValueDType::F32);
        gn.outputs = {top_i, top_w};
        const ps::Status vst = rt::validate_primitive_graph(g);
        check(!vst.ok(), "router top_k larger than experts rejected");
    }

    {
        rt::PrimitiveGraph g;
        register_moe_weights(g);
        const rt::ValueId x = g.alloc_value(kHidden, rt::ValueDType::BF16);
        add_router(g, x, true);
        add_experts(g, x, g.nodes[0].outputs[0], g.nodes[0].outputs[1]);
        g.external_inputs.push_back(x);
        g.external_outputs.push_back(g.nodes[1].outputs[0]);
        const rt::WeightTableView short_view{weight_slots.data(), 4};
        auto built = rt::build_program(g, short_view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE, options);
        check(!built.ok(), "experts weight range beyond table rejected at build");
    }

    {
        rt::PrimitiveGraph g;
        register_moe_weights(g);
        const rt::ValueId x = g.alloc_value(kHidden, rt::ValueDType::BF16);
        add_router(g, x, true);
        add_experts(g, x, g.nodes[0].outputs[0], g.nodes[0].outputs[1]);
        g.external_inputs.push_back(x);
        g.external_outputs.push_back(g.nodes[1].outputs[0]);
        auto built = rt::build_program(g, weight_view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE, options);
        check(built.ok(), std::string("full graph builds")
                              + (built.ok() ? "" : " (" + built.status().message() + ")"));
        if (built.ok()) {
            rt::Program plan = built.release();
            const bool seq_ok =
                plan.dispatches.size() == 2u &&
                plan.dispatches[0].kernel_id == rt::KernelId::MOE_ROUTER &&
                plan.dispatches[1].kernel_id == rt::KernelId::MOE_EXPERTS;
            check(seq_ok, "dispatch sequence is MOE_ROUTER then MOE_EXPERTS");
            const auto& rb = plan.dispatches[0];
            check(rb.group_size == kExperts && rb.rotary_dim == kTopK &&
                      rb.weight_index == kRouterSlot && rb.flags == 1u,
                  "router binding geometry packing");
            const auto& eb = plan.dispatches[1];
            check(eb.group_size == kExperts && eb.rotary_dim == kTopK &&
                      eb.weight_index == kGateUpSlot,
                  "experts binding geometry packing");
        }
    }

    for (const Case& c : kCases) {
        HostWeights caze_hw = hw;
        fill_router(caze_hw, c.kind);
        std::vector<std::uint16_t> caze_router_bits(caze_hw.router.size());
        for (size_t i = 0; i < caze_hw.router.size(); ++i)
            caze_router_bits[i] = bf16_bits(caze_hw.router[i]);
        Buf caze_router_dev;
        if (!caze_router_dev.alloc(caze_router_bits.size() * 2) ||
            !caze_router_dev.fill(caze_router_bits.data(),
                                  caze_router_bits.size() * 2)) {
            std::printf("FAIL: %s: router weight upload\n", c.name);
            return 1;
        }
        std::vector<rt::WeightSlot> slots = weight_slots;
        slots[kRouterSlot] =
            make_bf16_weight(caze_router_dev.dev, kExperts, kHidden);
        const rt::WeightTableView cview{slots.data(), slots.size()};

        std::vector<float> raw;
        std::vector<float> x_f32;
        fill_input(raw, x_f32, c.tokens, c.kind);

        Buf x_dev;
        std::vector<std::uint16_t> x_bits(x_f32.size());
        for (size_t i = 0; i < x_f32.size(); ++i) x_bits[i] = bf16_bits(x_f32[i]);
        if (!x_dev.alloc(x_bits.size() * 2) ||
            !x_dev.fill(x_bits.data(), x_bits.size() * 2)) {
            std::printf("FAIL: %s: input upload\n", c.name);
            return 1;
        }
        const uint32_t tokens = c.tokens;

        rt::PrimitiveGraph graph;
        register_moe_weights(graph);
        const rt::ValueId x = graph.alloc_value(kHidden, rt::ValueDType::BF16);
        add_router(graph, x, c.renorm);
        const rt::ValueId routed =
            add_experts(graph, x, graph.nodes[0].outputs[0], graph.nodes[0].outputs[1]);
        const rt::ValueId top_idx_val = graph.nodes[0].outputs[0];
        const rt::ValueId top_w = graph.nodes[0].outputs[1];
        graph.external_inputs.push_back(x);
        graph.external_outputs.push_back(top_idx_val);
        graph.external_outputs.push_back(top_w);
        graph.external_outputs.push_back(routed);

        const ps::Status vst = rt::validate_primitive_graph(graph);
        check(vst.ok(), std::string(c.name) + ": graph validates");

        auto built = rt::build_program(graph, cview, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE, options);
        check(built.ok(), std::string(c.name) + ": build_program");
        if (!built.ok()) continue;
        rt::Program plan = built.release();
        const rt::ValueBinding* vb_idx = plan.find_value(top_idx_val);
        const rt::ValueBinding* vb_w = plan.find_value(top_w);
        const rt::ValueBinding* vb_o = plan.find_value(routed);
        if (vb_idx == nullptr || vb_w == nullptr || vb_o == nullptr) {
            std::printf("FAIL: %s: output binding lookup\n", c.name);
            return 1;
        }
        const uint32_t stride_idx =
            vb_idx->row_stride != 0 ? vb_idx->row_stride : kTopK;
        const uint32_t stride_w = vb_w->row_stride != 0 ? vb_w->row_stride : kTopK;
        const uint32_t stride_o = vb_o->row_stride != 0 ? vb_o->row_stride : kHidden;
        Buf top_i_dev, top_w_dev, out_dev;
        if (!top_i_dev.alloc(static_cast<size_t>(tokens) * stride_idx * 4) ||
            !top_w_dev.alloc(static_cast<size_t>(tokens) * stride_w * 4) ||
            !out_dev.alloc(static_cast<size_t>(tokens) * stride_o * 4)) {
            std::printf("FAIL: %s: output allocation\n", c.name);
            return 1;
        }
        patch_external(plan, x, x_dev.dev);
        patch_external(plan, top_idx_val, top_i_dev.dev);
        patch_external(plan, top_w, top_w_dev.dev);
        patch_external(plan, routed, out_dev.dev);

        DeviceCtx dc;
        const int status = run_plan(plan, slots, dc, tokens, stream);
        check(status == 0, std::string(c.name) + ": dispatches execute (status=" +
                               std::to_string(status) + ")");
        if (status != 0) continue;

        std::vector<std::uint32_t> raw_idx(static_cast<size_t>(tokens) * stride_idx);
        std::vector<float> raw_w(static_cast<size_t>(tokens) * stride_w);
        std::vector<float> raw_out(static_cast<size_t>(tokens) * stride_o);
        check(top_i_dev.read(raw_idx.data(), raw_idx.size() * 4),
              std::string(c.name) + ": top index readback");
        check(top_w_dev.read(raw_w.data(), raw_w.size() * 4),
              std::string(c.name) + ": top weight readback");
        check(out_dev.read(raw_out.data(), raw_out.size() * 4),
              std::string(c.name) + ": routed output readback");
        std::vector<std::uint32_t> got_idx(static_cast<size_t>(tokens) * kTopK);
        std::vector<float> got_w(static_cast<size_t>(tokens) * kTopK);
        std::vector<float> got_out(static_cast<size_t>(tokens) * kHidden);
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t j = 0; j < kTopK; ++j) {
                got_idx[static_cast<size_t>(t) * kTopK + j] =
                    raw_idx[static_cast<size_t>(t) * stride_idx + j];
                got_w[static_cast<size_t>(t) * kTopK + j] =
                    raw_w[static_cast<size_t>(t) * stride_w + j];
            }
            for (uint32_t h = 0; h < kHidden; ++h) {
                got_out[static_cast<size_t>(t) * kHidden + h] =
                    raw_out[static_cast<size_t>(t) * stride_o + h];
            }
        }

        std::vector<std::uint32_t> ref_idx;
        std::vector<float> ref_w;
        std::vector<float> ref_routed;
        oracle_moe(x_f32, caze_hw, tokens, c.renorm, ref_idx, ref_w, ref_routed);

        check(std::memcmp(got_idx.data(), ref_idx.data(),
                          got_idx.size() * sizeof(std::uint32_t)) == 0,
              std::string(c.name) + ": top-k index bitwise vs moe_math oracle");
        if (std::memcmp(got_idx.data(), ref_idx.data(),
                        got_idx.size() * sizeof(std::uint32_t)) != 0) {
            std::printf("  got :");
            for (auto v : got_idx) std::printf(" %u", v);
            std::printf("\n  ref :");
            for (auto v : ref_idx) std::printf(" %u", v);
            std::printf("\n");
        }

        const Metrics m_w = compare(got_w.data(), ref_w.data(), got_w.size());
        print_metrics((std::string(c.name) + " top_w").c_str(), m_w);
        check(!m_w.non_finite && m_w.max_abs <= 1.0e-5,
              std::string(c.name) + ": top_w max_abs <= 1e-5");

        const Metrics m_out = compare(got_out.data(), ref_routed.data(), got_out.size());
        print_metrics((std::string(c.name) + " routed").c_str(), m_out);
        check(!m_out.non_finite && m_out.max_abs <= 5.0e-3 && m_out.rmse <= 5.0e-4,
              std::string(c.name) + ": routed matches moe_math oracle");

        if (c.renorm) {
            bool sums_one = true;
            for (uint32_t t = 0; t < tokens; ++t) {
                float s = 0.0f;
                for (uint32_t j = 0; j < kTopK; ++j)
                    s += got_w[static_cast<size_t>(t) * kTopK + j];
                sums_one &= std::fabs(s - 1.0f) <= 1.0e-5f;
            }
            check(sums_one, std::string(c.name) + ": renormalized weights sum to 1");
        } else {
            float s = 0.0f;
            for (uint32_t j = 0; j < kTopK; ++j) s += got_w[j];
            check(s < 1.0f, std::string(c.name) + ": raw top-k weights sum below 1");
        }

        if (c.kind == CaseKind::SAME_X) {
            bool same_sel = true;
            for (uint32_t t = 1; t < tokens; ++t)
                for (uint32_t j = 0; j < kTopK; ++j)
                    same_sel &=
                        got_idx[static_cast<size_t>(t) * kTopK + j] ==
                        got_idx[static_cast<size_t>(j)];
            check(same_sel, std::string(c.name) +
                                ": identical tokens select identical experts");
        }
        if (c.kind == CaseKind::ZERO_X) {
            bool index_order = true;
            for (uint32_t t = 0; t < tokens; ++t)
                for (uint32_t j = 0; j < kTopK; ++j)
                    index_order &=
                        got_idx[static_cast<size_t>(t) * kTopK + j] == j;
            check(index_order, std::string(c.name) +
                                   ": full tie selects experts 0..k-1 in index order");
        }
        if (c.kind == CaseKind::TIE_WEIGHTS) {
            bool tie_broken = true;
            for (uint32_t t = 0; t < tokens; ++t)
                tie_broken &= got_idx[static_cast<size_t>(t) * kTopK + 2u] == 2u;
            check(tie_broken, std::string(c.name) +
                                  ": boundary tie resolves to smaller expert index");
        }
    }

    {
        std::vector<float> raw;
        std::vector<float> x_f32;
        fill_input(raw, x_f32, 2, CaseKind::NORMAL);
        Buf x_dev;
        std::vector<std::uint16_t> x_bits(x_f32.size());
        for (size_t i = 0; i < x_f32.size(); ++i) x_bits[i] = bf16_bits(x_f32[i]);
        if (!x_dev.alloc(x_bits.size() * 2) ||
            !x_dev.fill(x_bits.data(), x_bits.size() * 2)) {
            std::printf("FAIL: negative-case input upload\n");
            return 1;
        }
        Buf top_i_dev, top_w_dev, out_dev;
        const size_t padded_topk_bytes =
            ((static_cast<size_t>(kTopK) * 4u + 15u) / 16u) * 16u;
        (void)top_i_dev.alloc(2ull * padded_topk_bytes);
        (void)top_w_dev.alloc(2ull * padded_topk_bytes);
        (void)out_dev.alloc(2ull * kHidden * 4);

        std::vector<rt::WeightSlot> bad = weight_slots;
        bad[kRouterSlot].encoding = 5u;

        rt::PrimitiveGraph graph;
        register_moe_weights(graph);
        const rt::ValueId x = graph.alloc_value(kHidden, rt::ValueDType::BF16);
        add_router(graph, x, true);
        const rt::ValueId out =
            add_experts(graph, x, graph.nodes[0].outputs[0], graph.nodes[0].outputs[1]);
        graph.external_inputs.push_back(x);
        graph.external_outputs.push_back(graph.nodes[0].outputs[0]);
        graph.external_outputs.push_back(graph.nodes[0].outputs[1]);
        graph.external_outputs.push_back(out);

        auto built = rt::build_program(graph, weight_view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE, options);
        check(built.ok(), "negative encoding case builds");
        if (built.ok()) {
            rt::Program plan = built.release();
            patch_external(plan, x, x_dev.dev);
            patch_external(plan, graph.nodes[0].outputs[0], top_i_dev.dev);
            patch_external(plan, graph.nodes[0].outputs[1], top_w_dev.dev);
            patch_external(plan, out, out_dev.dev);
            DeviceCtx dc;
            const int status = run_plan(plan, bad, dc, 2, stream);
            check(status != 0, "non-BF16 router weight rejected by kernel");
        }
    }

    std::printf("test_moe_ops: passed=%d failed=%d\n", passed, failed);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failed == 0 ? 0 : 1;
}
