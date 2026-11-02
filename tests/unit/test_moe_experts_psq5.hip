#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/models/common/qwen_hybrid/moe_math.h>
#include <shelf/quantization/manifest/quantized_manifest.h>
#include <shelf/quantization/psq5/psq5.h>
#include <shelf/quantization/quantization_types.h>
#include <shelf/runtime/graph/primitive_graph.h>
#include <shelf/runtime/program/device_program.h>
#include <shelf/runtime/program/program.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace rt = ps::runtime;
namespace qh = ps::qwen_hybrid;
namespace pq = ps::quantization::psq5;

namespace {

constexpr uint32_t kHidden = 64;
constexpr uint32_t kInter = 48;
constexpr uint32_t kExperts = 8;
constexpr uint32_t kTopK = 3;
constexpr uint32_t kRouterSlot = 0;
constexpr uint32_t kGateUpSlot = 1;
constexpr uint32_t kDownSlot = kGateUpSlot + kExperts;
constexpr uint32_t kWeightSlots = kDownSlot + kExperts;
constexpr uint32_t kGuRows = 2u * kInter;
constexpr uint32_t kGuK = kHidden;
constexpr uint32_t kDnRows = kHidden;
constexpr uint32_t kDnK = kInter;
constexpr uint32_t kGuKp = 64;
constexpr uint32_t kDnKp = 64;

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
    return static_cast<uint32_t>(x);
}

float uniform01(uint64_t x) {
    return static_cast<float>(static_cast<int32_t>(h32(x) % 2000u) - 1000) /
           1000.0f;
}

std::uint16_t bf16_bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return static_cast<std::uint16_t>(u >> 16);
}

float bf16_round(float f) {
    const std::uint16_t b = bf16_bits(f);
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float g;
    std::memcpy(&g, &u, sizeof(g));
    return g;
}

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
    if (hipMalloc(&dc.scratch, static_cast<size_t>(256) * 4096 * sizeof(float)) !=
        hipSuccess)
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

void patch_external(rt::Program& plan, rt::ValueId id, void* dev_ptr) {
    for (auto& v : plan.values) {
        if (v.logical_value.id == id.id) {
            v.offset = reinterpret_cast<uint64_t>(dev_ptr);
        }
    }
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

struct Psq5Tensor {
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t kp = 0;
    uint32_t nb = 0;
    std::vector<uint8_t> mag;
    std::vector<uint8_t> sign;
    std::vector<uint8_t> scale;
};

void fill_psq5(Psq5Tensor& t, uint32_t rows, uint32_t cols, uint32_t kp,
               uint64_t seed) {
    t.rows = rows;
    t.cols = cols;
    t.kp = kp;
    t.nb = kp / pq::kBlockK;
    t.mag.assign(static_cast<size_t>(rows) * t.nb * pq::kMagnitudeBytesPerBlock, 0u);
    t.sign.assign(static_cast<size_t>(rows) * t.nb * pq::kSignBytesPerBlock, 0u);
    t.scale.assign(static_cast<size_t>(rows) * t.nb * pq::kScaleBytesPerBlock, 0u);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t ib = 0; ib < t.nb; ++ib) {
            const size_t key = static_cast<size_t>(r) * 131u + ib * 17u;
            const size_t mb =
                (static_cast<size_t>(r) * t.nb + ib) * pq::kMagnitudeBytesPerBlock;
            for (uint32_t j = 0; j < pq::kMagnitudeBytesPerBlock; ++j) {
                const uint32_t a = h32(seed + key * 5u + j);
                const uint32_t b = h32(seed + 7777u + key * 5u + j);
                t.mag[mb + j] =
                    static_cast<uint8_t>((a & 0x0Fu) | ((b & 0x0Fu) << 4));
            }
            const size_t sb =
                (static_cast<size_t>(r) * t.nb + ib) * pq::kSignBytesPerBlock;
            const uint32_t sg = h32(seed + 31337u + key);
            for (uint32_t j = 0; j < pq::kSignBytesPerBlock; ++j)
                t.sign[sb + j] = static_cast<uint8_t>((sg >> (8u * j)) & 0xFFu);
            const uint32_t sc = h32(seed + 99991u + key);
            const uint32_t expb = 8u + (sc & 0x03u);
            const uint32_t mant = (sc >> 8) & 0x0Fu;
            t.scale[static_cast<size_t>(r) * t.nb + ib] =
                static_cast<uint8_t>((expb << 4) | mant);
        }
    }
}

pq::Psq5CanonicalView canonical_view(const Psq5Tensor& t) {
    pq::Psq5CanonicalView cv;
    cv.n = t.rows;
    cv.logical_k = t.cols;
    cv.padded_k = t.kp;
    cv.magnitude = t.mag.data();
    cv.magnitude_bytes = t.mag.size();
    cv.sign = t.sign.data();
    cv.sign_bytes = t.sign.size();
    cv.scale = t.scale.data();
    cv.scale_bytes = t.scale.size();
    return cv;
}

struct Psq5Group {
    std::vector<uint8_t> mag;
    std::vector<uint8_t> sign;
    std::vector<uint8_t> scale;
    std::vector<size_t> mag_off;
    std::vector<size_t> sign_off;
    std::vector<size_t> scale_off;
    uint32_t nb = 0;
};

bool stage_expert(const Psq5Tensor& t, Psq5Group& g) {
    const pq::Psq5CanonicalView cv = canonical_view(t);
    if (!cv.validate()) return false;
    pq::Psq5NativeHost native;
    if (!pq::preshuffle_psq5_native(cv, native)) return false;
    g.nb = t.nb;
    g.mag_off.push_back(g.mag.size());
    g.sign_off.push_back(g.sign.size());
    g.scale_off.push_back(g.scale.size());
    const std::span<const uint8_t> ms = native.magnitude_span();
    const std::span<const uint8_t> ss = native.sign_span();
    const std::span<const uint8_t> cs = native.scale_span();
    g.mag.insert(g.mag.end(), ms.begin(), ms.end());
    g.sign.insert(g.sign.end(), ss.begin(), ss.end());
    g.scale.insert(g.scale.end(), cs.begin(), cs.end());
    return true;
}

rt::WeightSlot make_psq5_weight(const Psq5Group& g, size_t expert, uint32_t rows,
                                uint32_t cols, uint32_t kp, const void* mag_dev,
                                const void* sign_dev, const void* scale_dev) {
    rt::WeightSlot w;
    w.rows = rows;
    w.cols = cols;
    w.k_padded = kp;
    w.encoding =
        static_cast<uint8_t>(ps::quantization::manifest::QuantizedEncoding::Psq5);
    w.compute_spec =
        static_cast<uint8_t>(ps::quantization::ComputeSpecId::PSQ5_W8A8_F32_BF16);
    w.codes = static_cast<const uint8_t*>(mag_dev) + g.mag_off[expert];
    w.sign = static_cast<const uint8_t*>(sign_dev) + g.sign_off[expert];
    w.scales = static_cast<const uint8_t*>(scale_dev) + g.scale_off[expert];
    w.codes_row_stride_bytes = pq::kTileRows * pq::kMagnitudeBytesPerBlock;
    w.scale_row_stride_bytes =
        g.nb * pq::kTileRows * pq::kScaleBytesPerBlock;
    w.storage_scale_stride_bytes = w.scale_row_stride_bytes;
    w.weight_scale_group = pq::kBlockK;
    w.preshuffled = true;
    return w;
}

void fill_router(std::vector<float>& router) {
    router.assign(static_cast<size_t>(kExperts) * kHidden, 0.0f);
    const float wscale = 0.05f / std::sqrt(static_cast<float>(kHidden));
    for (uint32_t e = 0; e < kExperts; ++e)
        for (uint32_t h = 1; h < kHidden; ++h)
            router[static_cast<size_t>(e) * kHidden + h] =
                bf16_round(wscale * uniform01(1000u + e * kHidden + h));
    const float offs[kExperts] = {5.0f, 4.0f, 3.0f, 2.0f,
                                  1.0f, 0.0f, -1.0f, -2.0f};
    for (uint32_t e = 0; e < kExperts; ++e)
        router[static_cast<size_t>(e) * kHidden] = offs[e];
}

void fill_input(std::vector<float>& raw, uint32_t tokens) {
    raw.assign(static_cast<size_t>(tokens) * kHidden, 0.0f);
    for (uint32_t t = 0; t < tokens; ++t)
        for (uint32_t h = 0; h < kHidden; ++h) {
            const uint64_t key = 500000u + static_cast<uint64_t>(t) * kHidden + h;
            float v = 0.1f * uniform01(key);
            if (h == 0) v = 1.0f;
            raw[static_cast<size_t>(t) * kHidden + h] = v;
        }
}

void oracle_moe(const std::vector<float>& x_bf, const std::vector<float>& router,
                const std::vector<float>& gate_up, const std::vector<float>& down,
                uint32_t tokens, std::vector<std::uint32_t>& top_i,
                std::vector<float>& top_w, std::vector<float>& routed) {
    const size_t te = static_cast<size_t>(tokens) * kExperts;
    const size_t tk = static_cast<size_t>(tokens) * kTopK;
    const size_t th = static_cast<size_t>(tokens) * kHidden;
    std::vector<float> probs(te, 0.0f);
    qh::moe_router_logits_f32(x_bf.data(), router.data(), tokens, kHidden, kExperts,
                              probs.data());
    qh::moe_softmax_f32(probs.data(), tokens, kExperts);
    top_i.assign(tk, 0u);
    top_w.assign(tk, 0.0f);
    qh::moe_topk_stable_f32(probs.data(), tokens, kExperts, kTopK, top_i.data(),
                            top_w.data());
    qh::moe_normalize_topk_f32(top_w.data(), tokens, kTopK);

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
                        gate_up.data() +
                            static_cast<size_t>(e) * 2ull * kInter * kHidden,
                        gu.size() * sizeof(float));
            std::memcpy(dn.data(),
                        down.data() + static_cast<size_t>(e) * kHidden * kInter,
                        dn.size() * sizeof(float));
            qh::moe_expert_forward_f32(xrow.data(), gu.data(), dn.data(), kInter,
                                       kHidden, out_e.data());
            for (uint32_t h = 0; h < kHidden; ++h) acc[h] += wgt * out_e[h];
        }
    }
}

void register_moe_weights(rt::PrimitiveGraph& g) {
    g.register_weight({kExperts, kHidden});
    for (uint32_t e = 0; e < kExperts; ++e) g.register_weight({kGuRows, kHidden});
    for (uint32_t e = 0; e < kExperts; ++e) g.register_weight({kHidden, kDnK});
}

rt::ValueId add_router(rt::PrimitiveGraph& graph, rt::ValueId x_in) {
    rt::MoERouterNode n;
    n.weight_shape = {kExperts, kHidden};
    n.weight_index = kRouterSlot;
    n.experts = kExperts;
    n.top_k = kTopK;
    n.renorm = true;
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
    n.gate_up_shape = {kGuRows, kHidden};
    n.down_shape = {kHidden, kDnK};
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

    std::vector<Psq5Tensor> gu_canon(kExperts);
    std::vector<Psq5Tensor> dn_canon(kExperts);
    Psq5Group gu_group;
    Psq5Group dn_group;
    std::vector<float> hw_gate_up(static_cast<size_t>(kExperts) * kGuRows * kHidden);
    std::vector<float> hw_down(static_cast<size_t>(kExperts) * kDnRows * kDnK);
    for (uint32_t e = 0; e < kExperts; ++e) {
        fill_psq5(gu_canon[e], kGuRows, kGuK, kGuKp, 100000u + e * 977u);
        fill_psq5(dn_canon[e], kDnRows, kDnK, kDnKp, 500000u + e * 977u);
        check(stage_expert(gu_canon[e], gu_group),
              "gate_up expert preshuffles to native layout");
        check(stage_expert(dn_canon[e], dn_group),
              "down expert preshuffles to native layout");

        std::vector<float> gu_f(static_cast<size_t>(kGuRows) * kGuKp, 0.0f);
        pq::dequantize_psq5_canonical(canonical_view(gu_canon[e]), gu_f.data());
        std::memcpy(hw_gate_up.data() + static_cast<size_t>(e) * kGuRows * kHidden,
                    gu_f.data(), static_cast<size_t>(kGuRows) * kHidden * sizeof(float));

        std::vector<float> dn_f(static_cast<size_t>(kDnRows) * kDnKp, 0.0f);
        pq::dequantize_psq5_canonical(canonical_view(dn_canon[e]), dn_f.data());
        for (uint32_t r = 0; r < kDnRows; ++r)
            std::memcpy(hw_down.data() + static_cast<size_t>(e) * kDnRows * kDnK +
                            static_cast<size_t>(r) * kDnK,
                        dn_f.data() + static_cast<size_t>(r) * kDnKp,
                        kDnK * sizeof(float));
    }
    check(gu_group.mag_off.size() == kExperts && dn_group.mag_off.size() == kExperts,
          "psq5 native staging produced one region per expert");

    Buf router_dev, gu_dev, dn_dev;
    std::vector<float> router;
    fill_router(router);
    std::vector<std::uint16_t> router_bits(router.size());
    for (size_t i = 0; i < router.size(); ++i) router_bits[i] = bf16_bits(router[i]);
    if (!router_dev.alloc(router_bits.size() * 2) ||
        !router_dev.fill(router_bits.data(), router_bits.size() * 2) ||
        !gu_dev.alloc(gu_group.mag.size() + gu_group.sign.size() + gu_group.scale.size()) ||
        !dn_dev.alloc(dn_group.mag.size() + dn_group.sign.size() + dn_group.scale.size())) {
        std::printf("FAIL: device allocation\n");
        return 1;
    }

    std::vector<uint8_t> gu_stage(gu_group.mag.size() + gu_group.sign.size() +
                                  gu_group.scale.size());
    std::memcpy(gu_stage.data(), gu_group.mag.data(), gu_group.mag.size());
    std::memcpy(gu_stage.data() + gu_group.mag.size(), gu_group.sign.data(),
                gu_group.sign.size());
    std::memcpy(gu_stage.data() + gu_group.mag.size() + gu_group.sign.size(),
                gu_group.scale.data(), gu_group.scale.size());
    if (!gu_dev.fill(gu_stage.data(), gu_stage.size())) {
        std::printf("FAIL: gate_up psq5 upload\n");
        return 1;
    }
    std::vector<uint8_t> dn_stage(dn_group.mag.size() + dn_group.sign.size() +
                                  dn_group.scale.size());
    std::memcpy(dn_stage.data(), dn_group.mag.data(), dn_group.mag.size());
    std::memcpy(dn_stage.data() + dn_group.mag.size(), dn_group.sign.data(),
                dn_group.sign.size());
    std::memcpy(dn_stage.data() + dn_group.mag.size() + dn_group.sign.size(),
                dn_group.scale.data(), dn_group.scale.size());
    if (!dn_dev.fill(dn_stage.data(), dn_stage.size())) {
        std::printf("FAIL: down psq5 upload\n");
        return 1;
    }

    std::vector<rt::WeightSlot> weight_slots(kWeightSlots);
    weight_slots[kRouterSlot] = make_bf16_weight(router_dev.dev, kExperts, kHidden);
    const uint8_t* gu_mag = static_cast<const uint8_t*>(gu_dev.dev);
    const uint8_t* gu_sign = gu_mag + gu_group.mag.size();
    const uint8_t* gu_scale = gu_sign + gu_group.sign.size();
    const uint8_t* dn_mag = static_cast<const uint8_t*>(dn_dev.dev);
    const uint8_t* dn_sign = dn_mag + dn_group.mag.size();
    const uint8_t* dn_scale = dn_sign + dn_group.sign.size();
    for (uint32_t e = 0; e < kExperts; ++e) {
        weight_slots[kGateUpSlot + e] =
            make_psq5_weight(gu_group, e, kGuRows, kGuK, kGuKp, gu_mag, gu_sign,
                             gu_scale);
        weight_slots[kDownSlot + e] =
            make_psq5_weight(dn_group, e, kDnRows, kDnK, kDnKp, dn_mag, dn_sign,
                             dn_scale);
    }
    const rt::WeightTableView weight_view{weight_slots.data(), weight_slots.size()};

    const uint32_t tokens = 3;
    std::vector<float> raw;
    fill_input(raw, tokens);
    std::vector<float> x_f32(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) x_f32[i] = bf16_round(raw[i]);

    Buf x_dev;
    std::vector<std::uint16_t> x_bits(x_f32.size());
    for (size_t i = 0; i < x_f32.size(); ++i) x_bits[i] = bf16_bits(x_f32[i]);
    if (!x_dev.alloc(x_bits.size() * 2) ||
        !x_dev.fill(x_bits.data(), x_bits.size() * 2)) {
        std::printf("FAIL: input upload\n");
        return 1;
    }

    rt::PrimitiveGraph graph;
    register_moe_weights(graph);
    const rt::ValueId x = graph.alloc_value(kHidden, rt::ValueDType::BF16);
    add_router(graph, x);
    const rt::ValueId routed =
        add_experts(graph, x, graph.nodes[0].outputs[0], graph.nodes[0].outputs[1]);
    const rt::ValueId top_idx_val = graph.nodes[0].outputs[0];
    const rt::ValueId top_w = graph.nodes[0].outputs[1];
    graph.external_inputs.push_back(x);
    graph.external_outputs.push_back(top_idx_val);
    graph.external_outputs.push_back(top_w);
    graph.external_outputs.push_back(routed);

    const ps::Status vst = rt::validate_primitive_graph(graph);
    check(vst.ok(), std::string("graph validates") +
                        (vst.ok() ? "" : " (" + vst.message() + ")"));

    rt::ProgramBuildOptions options;
    options.validate_workspace = true;
    options.max_token_rows = 8;
    options.max_output_rows = 8;
    auto built = rt::build_program(graph, weight_view, rt::RowBucket::R16,
                                   rt::ExecutionClass::DECODE, options);
    check(built.ok(), std::string("psq5 expert program builds") +
                          (built.ok() ? "" : " (" + built.status().message() + ")"));
    if (!built.ok()) {
        ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
        return failed == 0 ? 0 : 1;
    }
    rt::Program plan = built.release();

    const rt::ValueBinding* vb_idx = plan.find_value(top_idx_val);
    const rt::ValueBinding* vb_w = plan.find_value(top_w);
    const rt::ValueBinding* vb_o = plan.find_value(routed);
    if (vb_idx == nullptr || vb_w == nullptr || vb_o == nullptr) {
        std::printf("FAIL: output binding lookup\n");
        return 1;
    }
    const uint32_t stride_idx = vb_idx->row_stride != 0 ? vb_idx->row_stride : kTopK;
    const uint32_t stride_w = vb_w->row_stride != 0 ? vb_w->row_stride : kTopK;
    const uint32_t stride_o = vb_o->row_stride != 0 ? vb_o->row_stride : kHidden;
    Buf top_i_dev, top_w_dev, out_dev;
    if (!top_i_dev.alloc(static_cast<size_t>(tokens) * stride_idx * 4) ||
        !top_w_dev.alloc(static_cast<size_t>(tokens) * stride_w * 4) ||
        !out_dev.alloc(static_cast<size_t>(tokens) * stride_o * 4)) {
        std::printf("FAIL: output allocation\n");
        return 1;
    }
    patch_external(plan, x, x_dev.dev);
    patch_external(plan, top_idx_val, top_i_dev.dev);
    patch_external(plan, top_w, top_w_dev.dev);
    patch_external(plan, routed, out_dev.dev);

    DeviceCtx dc;
    const int status = run_plan(plan, weight_slots, dc, tokens, stream);
    check(status == 0, "psq5 experts dispatch executes (status=" +
                           std::to_string(status) + ")");
    if (status == 0) {
        std::vector<std::uint32_t> raw_idx(static_cast<size_t>(tokens) * stride_idx);
        std::vector<float> raw_w(static_cast<size_t>(tokens) * stride_w);
        std::vector<float> raw_out(static_cast<size_t>(tokens) * stride_o);
        check(top_i_dev.read(raw_idx.data(), raw_idx.size() * 4),
              "top index readback");
        check(top_w_dev.read(raw_w.data(), raw_w.size() * 4), "top weight readback");
        check(out_dev.read(raw_out.data(), raw_out.size() * 4),
              "routed output readback");
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
            for (uint32_t h = 0; h < kHidden; ++h)
                got_out[static_cast<size_t>(t) * kHidden + h] =
                    raw_out[static_cast<size_t>(t) * stride_o + h];
        }

        std::vector<std::uint32_t> ref_idx;
        std::vector<float> ref_w;
        std::vector<float> ref_routed;
        oracle_moe(x_f32, router, hw_gate_up, hw_down, tokens, ref_idx, ref_w,
                   ref_routed);

        check(std::memcmp(got_idx.data(), ref_idx.data(),
                          got_idx.size() * sizeof(std::uint32_t)) == 0,
              "top-k index bitwise vs moe_math oracle");
        const Metrics m_w = compare(got_w.data(), ref_w.data(), got_w.size());
        print_metrics("psq5 top_w", m_w);
        check(!m_w.non_finite && m_w.max_abs <= 1.0e-5,
              "top_w max_abs <= 1e-5");

        const Metrics m_out = compare(got_out.data(), ref_routed.data(), got_out.size());
        print_metrics("psq5 routed", m_out);
        check(!m_out.non_finite && m_out.max_abs <= 1.0e-5 && m_out.rmse <= 1.0e-6,
              "routed matches dequantized host oracle");
    }

    std::vector<rt::WeightSlot> bad = weight_slots;
    bad[kGateUpSlot].codes = nullptr;
    DeviceCtx dc_bad;
    const int bad_status = run_plan(plan, bad, dc_bad, tokens, stream);
    check(bad_status != 0, "psq5 expert slot without codes is rejected");

    std::printf("test_moe_experts_psq5: passed=%d failed=%d\n", passed, failed);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failed == 0 ? 0 : 1;
}
