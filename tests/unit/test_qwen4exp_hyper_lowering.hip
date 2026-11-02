#include <phaseshift/models/common/qwen_hybrid/hyper_lowering.h>
#include <phaseshift/models/qwen4exp/hyper/gated_residual.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/runtime/program/device_program.h>
#include <phaseshift/runtime/program/program.h>
#include <phaseshift/quantization/quantization_types.h>
#include <phaseshift/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace hy = ps::qwen4exp::hyper;
namespace rt = ps::runtime;
namespace kh = ps::qwen_hybrid;

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
    return static_cast<float>(static_cast<int32_t>(h32(x) % 2000u) - 1000) / 1000.0f;
}

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

float bf16_round(float f) {
    const uint16_t b = bf16_bits(f);
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float out;
    std::memcpy(&out, &u, 4);
    return out;
}

struct Buf {
    void* dev = nullptr;
    size_t bytes = 0;
    ~Buf() { if (dev) free_dev(); }
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
    std::printf("  %s: max_abs=%.6g max_rel=%.6g rmse=%.6g%s\n", tag, m.max_abs,
                m.max_rel, m.rmse, m.non_finite ? " NON-FINITE" : "");
}

struct DeviceCtx {
    rt::DeviceBatchContext h{};
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* err = nullptr;
    void* values_dev = nullptr;
    void* weights_dev = nullptr;
    void* params_dev = nullptr;
    void* workspace = nullptr;
    ~DeviceCtx() {
        void* bufs[] = {ctx_dev, scratch, err, values_dev, weights_dev, params_dev, workspace};
        for (void* q : bufs)
            if (q) (void)hipFree(q);
    }
};

bool run_program(const rt::Program& plan,
                 const std::vector<rt::WeightSlot>& host_weights,
                 const std::vector<rt::StaticParameterSlot>& host_params,
                 DeviceCtx& dc, uint32_t rows, hipStream_t stream) {
    dc.h = {};
    dc.h.actual_rows = rows;
    if (hipMalloc(&dc.ctx_dev, sizeof(dc.h)) != hipSuccess) return false;
    if (hipMemcpy(dc.ctx_dev, &dc.h, sizeof(dc.h), hipMemcpyHostToDevice) != hipSuccess)
        return false;
    if (hipMalloc(&dc.scratch, (size_t)256 * 4096 * sizeof(float)) != hipSuccess)
        return false;
    if (hipMalloc(&dc.err, sizeof(uint32_t)) != hipSuccess) return false;

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
            return false;
        }
    }
    if (hipMalloc(&dc.values_dev, vals.size() * sizeof(rt::DeviceValueBinding)) != hipSuccess)
        return false;
    if (hipMemcpy(dc.values_dev, vals.data(), vals.size() * sizeof(rt::DeviceValueBinding),
                  hipMemcpyHostToDevice) != hipSuccess) {
        return false;
    }

    if (!host_weights.empty()) {
        if (hipMalloc(&dc.weights_dev, host_weights.size() * sizeof(rt::WeightSlot)) !=
            hipSuccess) {
            return false;
        }
        if (hipMemcpy(dc.weights_dev, host_weights.data(),
                      host_weights.size() * sizeof(rt::WeightSlot),
                      hipMemcpyHostToDevice) != hipSuccess) {
            return false;
        }
    }
    if (!host_params.empty()) {
        if (hipMalloc(&dc.params_dev, host_params.size() * sizeof(rt::StaticParameterSlot)) !=
            hipSuccess) {
            return false;
        }
        if (hipMemcpy(dc.params_dev, host_params.data(),
                      host_params.size() * sizeof(rt::StaticParameterSlot),
                      hipMemcpyHostToDevice) != hipSuccess) {
            return false;
        }
    }
    if (plan.workspace.total_bytes != 0) {
        if (hipMalloc(&dc.workspace, plan.workspace.total_bytes) != hipSuccess)
            return false;
    }

    rt::DeviceProgramView view{};
    view.values = static_cast<const rt::DeviceValueBinding*>(dc.values_dev);
    view.value_count = static_cast<uint32_t>(vals.size());
    view.weight_table = dc.weights_dev;
    view.weight_slot_count = static_cast<uint32_t>(host_weights.size());
    view.parameter_table = dc.params_dev;
    view.parameter_slot_count = static_cast<uint32_t>(host_params.size());
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
            return false;
        }
        if (hipStreamSynchronize(stream) != hipSuccess) return false;
        uint32_t e = 0;
        if (hipMemcpy(&e, dc.err, 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
        if (e != 0u) {
            std::printf("FAIL: dispatch %zu (%s) status=%u\n", di,
                        rt::kernel_to_name(plan.dispatches[di].kernel_id), e);
            return false;
        }
    }
    return true;
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

bool read_workspace_value(const rt::Program& plan, const DeviceCtx& dc, rt::ValueId id,
                          uint32_t rows, std::vector<float>& out) {
    const rt::ValueBinding* b = plan.find_value(id);
    if (b == nullptr || b->storage != rt::ValueStorage::WORKSPACE || dc.workspace == nullptr)
        return false;
    std::vector<float> strided(static_cast<size_t>(rows) * b->row_stride);
    if (hipMemcpy(strided.data(),
                  reinterpret_cast<const uint8_t*>(dc.workspace) + b->offset,
                  strided.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }
    out.resize(static_cast<size_t>(rows) * b->feature_count);
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < b->feature_count; ++c) {
            out[static_cast<size_t>(r) * b->feature_count + c] =
                strided[static_cast<size_t>(r) * b->row_stride + c];
        }
    }
    return true;
}

rt::ValueId node_output_by_debug_name(const rt::PrimitiveGraph& graph, const char* name) {
    for (const auto& gn : graph.nodes) {
        if (gn.debug_name == name && !gn.outputs.empty()) return gn.outputs[0];
    }
    return rt::ValueId{UINT32_MAX};
}

void patch_external(rt::Program& plan, rt::ValueId id, void* dev_ptr) {
    for (auto& v : plan.values) {
        if (v.logical_value.id == id.id) {
            v.offset = reinterpret_cast<uint64_t>(dev_ptr);
        }
    }
}

constexpr uint32_t kStreams = 4;
constexpr uint32_t kHidden = 2560;
constexpr uint32_t kLowrank = 320;
constexpr uint32_t kHcHidden = kStreams * kHidden;
constexpr uint32_t kTokens = 5;
constexpr float kEps = 1e-6f;

struct HostWeights {
    std::vector<float> hc_norm;
    std::vector<float> mix_down;
    std::vector<float> mix_up;
    std::vector<float> block_inject;
};

void fill_host_weights(HostWeights& w, uint64_t seed) {
    w.hc_norm.resize(kHcHidden);
    w.mix_down.resize(static_cast<size_t>(kLowrank) * kHcHidden);
    w.mix_up.resize(static_cast<size_t>(kHcHidden) * kLowrank);
    w.block_inject.resize(static_cast<size_t>(kStreams) * kHcHidden);
    for (uint32_t i = 0; i < kHcHidden; ++i)
        w.hc_norm[i] = 0.02f * uniform01(seed + i);
    const float down_scale = 1.0f / std::sqrt(static_cast<float>(kHcHidden));
    for (size_t i = 0; i < w.mix_down.size(); ++i)
        w.mix_down[i] = down_scale * uniform01(seed + 100000u + i);
    const float up_scale = 1.0f / std::sqrt(static_cast<float>(kLowrank));
    for (size_t i = 0; i < w.mix_up.size(); ++i)
        w.mix_up[i] = up_scale * uniform01(seed + 200000u + i);
    for (size_t i = 0; i < w.block_inject.size(); ++i)
        w.block_inject[i] = down_scale * uniform01(seed + 300000u + i);
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
    fill_host_weights(hw, 7u);

    std::vector<float> hw_down_bf16(hw.mix_down.size());
    std::vector<float> hw_up_bf16(hw.mix_up.size());
    std::vector<float> hw_inject_bf16(hw.block_inject.size());
    for (size_t i = 0; i < hw.mix_down.size(); ++i)
        hw_down_bf16[i] = bf16_round(hw.mix_down[i]);
    for (size_t i = 0; i < hw.mix_up.size(); ++i)
        hw_up_bf16[i] = bf16_round(hw.mix_up[i]);
    for (size_t i = 0; i < hw.block_inject.size(); ++i)
        hw_inject_bf16[i] = bf16_round(hw.block_inject[i]);

    std::vector<float> x4(static_cast<size_t>(kTokens) * kHcHidden);
    for (size_t i = 0; i < x4.size(); ++i) x4[i] = uniform01(900000u + i);
    std::vector<float> sub_out(static_cast<size_t>(kTokens) * kHidden);
    for (size_t i = 0; i < sub_out.size(); ++i) sub_out[i] = uniform01(800000u + i);

    Buf x4_dev, sub_dev, mixed_dev, inject_dev, ref_inject_dev, out_dev;
    if (!x4_dev.alloc(x4.size() * 4) || !x4_dev.fill(x4.data(), x4.size() * 4) ||
        !sub_dev.alloc(sub_out.size() * 4) || !sub_dev.fill(sub_out.data(), sub_out.size() * 4) ||
        !mixed_dev.alloc(static_cast<size_t>(kTokens) * kHidden * 4) ||
        !inject_dev.alloc(static_cast<size_t>(kTokens) * kStreams * 4) ||
        !ref_inject_dev.alloc(static_cast<size_t>(kTokens) * kStreams * 4) ||
        !out_dev.alloc(static_cast<size_t>(kTokens) * kHcHidden * 4)) {
        std::printf("FAIL: device buffer allocation\n");
        return 1;
    }

    Buf down_dev, up_dev, inj_dev, norm_dev;
    if (!down_dev.alloc(hw_down_bf16.size() * 2) ||
        !up_dev.alloc(hw_up_bf16.size() * 2) ||
        !inj_dev.alloc(hw_inject_bf16.size() * 2) ||
        !norm_dev.alloc(hw.hc_norm.size() * 4)) {
        std::printf("FAIL: weight buffer allocation\n");
        return 1;
    }
    {
        std::vector<uint16_t> packed(hw_down_bf16.size());
        for (size_t i = 0; i < packed.size(); ++i) packed[i] = bf16_bits(hw_down_bf16[i]);
        if (!down_dev.fill(packed.data(), packed.size() * 2)) return 1;
        packed.resize(hw_up_bf16.size());
        for (size_t i = 0; i < packed.size(); ++i) packed[i] = bf16_bits(hw_up_bf16[i]);
        if (!up_dev.fill(packed.data(), packed.size() * 2)) return 1;
        packed.resize(hw_inject_bf16.size());
        for (size_t i = 0; i < packed.size(); ++i) packed[i] = bf16_bits(hw_inject_bf16[i]);
        if (!inj_dev.fill(packed.data(), packed.size() * 2)) return 1;
    }
    if (!norm_dev.fill(hw.hc_norm.data(), hw.hc_norm.size() * 4)) return 1;

    const std::vector<rt::WeightSlot> weight_slots = {
        make_bf16_weight(down_dev.dev, kLowrank, kHcHidden),
        make_bf16_weight(up_dev.dev, kHcHidden, kLowrank),
        make_bf16_weight(inj_dev.dev, kStreams, kHcHidden),
    };
    std::vector<rt::StaticParameterSlot> param_slots(1);
    param_slots[0].device_ptr = norm_dev.dev;
    param_slots[0].elements = kHcHidden;
    param_slots[0].dtype = rt::ValueDType::F32;

    const rt::WeightTableView weight_view{weight_slots.data(), weight_slots.size()};
    const rt::StaticParameterTableView param_view{param_slots.data(), param_slots.size()};

    kh::HyperLoweringGeometry geo;
    geo.streams = kStreams;
    geo.hidden = kHidden;
    geo.lowrank = kLowrank;
    geo.eps = kEps;

    rt::ProgramBuildOptions options;
    options.validate_workspace = true;
    options.max_token_rows = kTokens;
    options.max_output_rows = kTokens;

    const auto expect_kinds = [](const rt::Program& p,
                                 const std::vector<rt::KernelId>& want,
                                 const char* tag) {
        bool ok = p.dispatches.size() == want.size();
        if (ok) {
            for (size_t i = 0; i < want.size(); ++i)
                ok = ok && p.dispatches[i].kernel_id == want[i];
        }
        check(ok, std::string(tag) + ": dispatch kernel sequence");
    };

    hy::GatedResidualDims dims;
    dims.streams = kStreams;
    dims.hidden = kHidden;
    dims.lowrank = kLowrank;
    dims.eps = kEps;

    std::vector<float> ref_mixed(static_cast<size_t>(kTokens) * kHidden);
    std::vector<float> ref_inject(static_cast<size_t>(kTokens) * kStreams);
    hy::GatedResidualReadTrace trace;
    {
        hy::GatedResidualWeights gw;
        gw.hc_norm = hw.hc_norm.data();
        gw.mix_down = hw_down_bf16.data();
        gw.mix_up = hw_up_bf16.data();
        gw.block_inject = hw_inject_bf16.data();
        const ps::Status st = hy::gated_residual_read(
            dims, gw, x4.data(), kTokens, ref_mixed.data(), ref_inject.data(), &trace);
        check(st.ok(), "reference: gated_residual_read runs");
    }
    if (!ref_inject_dev.fill(ref_inject.data(), ref_inject.size() * 4)) {
        std::printf("FAIL: reference inject upload\n");
        return 1;
    }

    {
        rt::PrimitiveGraph graph;
        const rt::ValueId x4_in = graph.alloc_value(kHcHidden, rt::ValueDType::F32);
        kh::Emitter em{graph};
        kh::HyperLoweringIndices idx;
        idx.w_mix_down = graph.register_weight({kLowrank, kHcHidden});
        idx.w_mix_up = graph.register_weight({kHcHidden, kLowrank});
        idx.w_block_inject = graph.register_weight({kStreams, kHcHidden});
        idx.p_hc_norm = 0;
        kh::HyperLoweringWeights wl;
        ps::weights::MatrixWeight w_down;
        w_down.rows = kLowrank;
        w_down.cols = kHcHidden;
        ps::weights::MatrixWeight w_up;
        w_up.rows = kHcHidden;
        w_up.cols = kLowrank;
        ps::weights::MatrixWeight w_inj;
        w_inj.rows = kStreams;
        w_inj.cols = kHcHidden;
        wl.mix_down = &w_down;
        wl.mix_up = &w_up;
        wl.block_inject = &w_inj;
        const kh::HyperReadOutputs out =
            kh::emit_hyper_read(em, geo, wl, idx, x4_in, "gr_read_");
        graph.external_inputs.push_back(x4_in);
        graph.external_outputs.push_back(out.mixed);
        graph.external_outputs.push_back(out.inject);

        const ps::Status vst = rt::validate_primitive_graph(graph);
        check(vst.ok(), std::string("read: graph validates")
                            + (vst.ok() ? "" : " (" + vst.message() + ")"));

        rt::ProgramBuildOptions read_options = options;
        read_options.no_workspace_reuse = true;
        auto built = rt::build_program(graph, weight_view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE, read_options);
        check(built.ok(), std::string("read: build_program")
                              + (built.ok() ? "" : " (" + built.status().message() + ")"));
        if (!built.ok()) return 1;
        rt::Program plan = built.release();
        plan.parameter_slot_count = static_cast<uint32_t>(param_slots.size());

        expect_kinds(plan,
                     {rt::KernelId::RMS_NORM, rt::KernelId::LINEAR_BF16, rt::KernelId::SCALE,
                      rt::KernelId::SILU, rt::KernelId::LINEAR_BF16, rt::KernelId::SIGMOID,
                      rt::KernelId::MUL, rt::KernelId::GROUP_MEAN, rt::KernelId::LINEAR_BF16,
                      rt::KernelId::SCALE, rt::KernelId::SIGMOID, rt::KernelId::SCALE},
                     "read");

        patch_external(plan, x4_in, x4_dev.dev);
        patch_external(plan, out.mixed, mixed_dev.dev);
        patch_external(plan, out.inject, inject_dev.dev);

        DeviceCtx dc;
        const bool ran = run_program(plan, weight_slots, param_slots, dc, kTokens, stream);
        check(ran, "read: dispatches execute");

        std::vector<float> got_mixed(static_cast<size_t>(kTokens) * kHidden);
        std::vector<float> got_inject(static_cast<size_t>(kTokens) * kStreams);
        if (ran) {
            check(mixed_dev.read(got_mixed.data(), got_mixed.size() * 4),
                  "read: mixed copied back");
            check(inject_dev.read(got_inject.data(), got_inject.size() * 4),
                  "read: inject copied back");
            const Metrics mm = compare(got_mixed.data(), ref_mixed.data(), got_mixed.size());
            const Metrics mi = compare(got_inject.data(), ref_inject.data(), got_inject.size());
            print_metrics("read mixed", mm);
            print_metrics("read inject", mi);
            check(!mm.non_finite && mm.max_abs <= 5e-3 && mm.rmse <= 1e-3,
                  "read: mixed matches reference");
            check(!mi.non_finite && mi.max_abs <= 1e-3 && mi.rmse <= 5e-4,
                  "read: inject matches reference");

            std::vector<float> got_normed;
            std::vector<float> got_up;
            const rt::ValueId normed_id =
                node_output_by_debug_name(graph, "gr_read_hyper_hc_norm");
            const rt::ValueId up_id = node_output_by_debug_name(graph, "gr_read_hyper_up");
            check(normed_id.id != UINT32_MAX && up_id.id != UINT32_MAX,
                  "read: intermediate nodes present");
            if (normed_id.id != UINT32_MAX && up_id.id != UINT32_MAX &&
                read_workspace_value(plan, dc, normed_id, kTokens, got_normed) &&
                read_workspace_value(plan, dc, up_id, kTokens, got_up)) {
                const Metrics mn = compare(got_normed.data(), trace.normed.data(),
                                           got_normed.size());
                const Metrics mu = compare(got_up.data(), trace.up.data(), got_up.size());
                print_metrics("read normed", mn);
                print_metrics("read up", mu);
                check(!mn.non_finite && mn.max_abs <= 2e-2 && mn.rmse <= 3e-3,
                      "read: normed matches reference");
                check(!mu.non_finite && mu.max_abs <= 2e-2 && mu.rmse <= 3e-3,
                      "read: up matches reference");
            } else {
                check(false, "read: intermediate values readable");
            }
        }
    }

    {
        std::vector<float> ref_out(static_cast<size_t>(kTokens) * kHcHidden);
        const ps::Status st = hy::gated_residual_write(dims, x4.data(), sub_out.data(),
                                                       ref_inject.data(), kTokens,
                                                       ref_out.data());
        check(st.ok(), "reference: gated_residual_write runs");

        rt::PrimitiveGraph graph;
        const rt::ValueId x4_in = graph.alloc_value(kHcHidden, rt::ValueDType::F32);
        const rt::ValueId sub_in = graph.alloc_value(kHidden, rt::ValueDType::F32);
        const rt::ValueId inj_in = graph.alloc_value(kStreams, rt::ValueDType::F32);
        kh::Emitter em{graph};
        const rt::ValueId out =
            kh::emit_hyper_write(em, geo, x4_in, sub_in, inj_in, "gr_write_");
        graph.external_inputs.push_back(x4_in);
        graph.external_inputs.push_back(sub_in);
        graph.external_inputs.push_back(inj_in);
        graph.external_outputs.push_back(out);

        const ps::Status vst = rt::validate_primitive_graph(graph);
        check(vst.ok(), std::string("write: graph validates")
                            + (vst.ok() ? "" : " (" + vst.message() + ")"));

        auto built = rt::build_program(graph, weight_view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE, options);
        check(built.ok(), std::string("write: build_program")
                              + (built.ok() ? "" : " (" + built.status().message() + ")"));
        if (!built.ok()) return 1;
        rt::Program plan = built.release();

        expect_kinds(plan, {rt::KernelId::GROUP_BROADCAST_MUL, rt::KernelId::RESIDUAL_ADD},
                     "write");

        patch_external(plan, x4_in, x4_dev.dev);
        patch_external(plan, sub_in, sub_dev.dev);
        patch_external(plan, inj_in, ref_inject_dev.dev);
        patch_external(plan, out, out_dev.dev);

        DeviceCtx dc;
        const bool ran = run_program(plan, weight_slots, param_slots, dc, kTokens, stream);
        check(ran, "write: dispatches execute");

        std::vector<float> got_out(static_cast<size_t>(kTokens) * kHcHidden);
        if (ran) {
            check(out_dev.read(got_out.data(), got_out.size() * 4),
                  "write: output copied back");
            const Metrics mo = compare(got_out.data(), ref_out.data(), got_out.size());
            print_metrics("write out", mo);
            check(!mo.non_finite && mo.max_abs <= 1e-5 && mo.rmse <= 1e-6,
                  "write: output matches reference");
        }
    }

    {
        std::vector<float> ref_collapse(static_cast<size_t>(kTokens) * kHidden);
        hy::GatedResidualCollapseWeights cw;
        cw.hc_norm = hw.hc_norm.data();
        cw.mix_down = hw_down_bf16.data();
        cw.mix_up = hw_up_bf16.data();
        const ps::Status st = hy::gated_residual_collapse(dims, cw, x4.data(), kTokens,
                                                          ref_collapse.data());
        check(st.ok(), "reference: gated_residual_collapse runs");

        rt::PrimitiveGraph graph;
        const rt::ValueId x4_in = graph.alloc_value(kHcHidden, rt::ValueDType::F32);
        kh::Emitter em{graph};
        kh::HyperLoweringIndices idx;
        idx.w_mix_down = graph.register_weight({kLowrank, kHcHidden});
        idx.w_mix_up = graph.register_weight({kHcHidden, kLowrank});
        idx.w_block_inject = graph.register_weight({kStreams, kHcHidden});
        idx.p_hc_norm = 0;
        kh::HyperLoweringWeights wl;
        ps::weights::MatrixWeight w_down;
        w_down.rows = kLowrank;
        w_down.cols = kHcHidden;
        ps::weights::MatrixWeight w_up;
        w_up.rows = kHcHidden;
        w_up.cols = kLowrank;
        ps::weights::MatrixWeight w_inj;
        w_inj.rows = kStreams;
        w_inj.cols = kHcHidden;
        wl.mix_down = &w_down;
        wl.mix_up = &w_up;
        wl.block_inject = &w_inj;
        const rt::ValueId mixed =
            kh::emit_hyper_collapse(em, geo, wl, idx, x4_in, "gr_collapse_");
        graph.external_inputs.push_back(x4_in);
        graph.external_outputs.push_back(mixed);

        const ps::Status vst = rt::validate_primitive_graph(graph);
        check(vst.ok(), std::string("collapse: graph validates")
                            + (vst.ok() ? "" : " (" + vst.message() + ")"));

        auto built = rt::build_program(graph, weight_view, rt::RowBucket::R16,
                                       rt::ExecutionClass::DECODE, options);
        check(built.ok(), std::string("collapse: build_program")
                              + (built.ok() ? "" : " (" + built.status().message() + ")"));
        if (!built.ok()) return 1;
        rt::Program plan = built.release();
        plan.parameter_slot_count = static_cast<uint32_t>(param_slots.size());

        expect_kinds(plan,
                     {rt::KernelId::RMS_NORM, rt::KernelId::LINEAR_BF16, rt::KernelId::SCALE,
                      rt::KernelId::SILU, rt::KernelId::LINEAR_BF16, rt::KernelId::SIGMOID,
                      rt::KernelId::MUL, rt::KernelId::GROUP_MEAN},
                     "collapse");

        patch_external(plan, x4_in, x4_dev.dev);
        patch_external(plan, mixed, mixed_dev.dev);

        DeviceCtx dc;
        const bool ran = run_program(plan, weight_slots, param_slots, dc, kTokens, stream);
        check(ran, "collapse: dispatches execute");

        std::vector<float> got(static_cast<size_t>(kTokens) * kHidden);
        if (ran) {
            check(mixed_dev.read(got.data(), got.size() * 4),
                  "collapse: output copied back");
            const Metrics m = compare(got.data(), ref_collapse.data(), got.size());
            print_metrics("collapse out", m);
            check(!m.non_finite && m.max_abs <= 5e-3 && m.rmse <= 1e-3,
                  "collapse: output matches reference");
        }
    }

    std::printf("test_qwen4exp_hyper_lowering: passed=%d failed=%d\n", passed, failed);
    ps::gpu::discard_cleanup_result(hipStreamDestroy(stream));
    return failed == 0 ? 0 : 1;
}
