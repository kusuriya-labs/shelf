#include <phaseshift/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/kernels/correctness/model_dispatch_correctness.h>
#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/runtime/program/device_program.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using bf16t = ps::bf16_t;
using V = ::ps::runtime::DeviceValueBinding;
using S = ::ps::runtime::DeviceStateBinding;
using P = ::ps::runtime::DeviceProgramView;
using B = ::ps::runtime::DispatchBinding;

uint32_t h32(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x;
}

int32_t rng1k(uint64_t x) { return (int32_t)(h32(x) % 2001u) - 1000; }

uint16_t bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

struct Req {
    uint32_t slot;
    uint32_t row_begin;
    uint32_t row_count;
};

struct Case {
    const char* name;
    uint32_t conv_dim;
    uint32_t state_index;
    uint32_t num_gdn_states;
    uint32_t max_sequences;
    std::vector<Req> reqs;
    uint32_t rows;
    uint32_t in_stride;
    uint32_t out_stride;
    uint32_t seed;
};

struct Bufs {
    void* weight = nullptr;
    void* input = nullptr;
    void* out_correct = nullptr;
    void* out_opt = nullptr;
    void* state_correct = nullptr;
    void* state_opt = nullptr;
    void* requests = nullptr;
    void* vals_dev = nullptr;
    void* state_dev = nullptr;
    void* param_dev = nullptr;
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* error_word = nullptr;
    ~Bufs() {
        void* all[] = {weight, input, out_correct, out_opt, state_correct, state_opt, requests,
                       vals_dev, state_dev, param_dev, ctx_dev, scratch, error_word};
        for (auto* p : all)
            if (p) ps::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool run_case(const Case& c) {
    const uint32_t conv_dim = c.conv_dim;
    const uint32_t history = 3u;
    const uint32_t hist_stride = (conv_dim + 7u) & ~7u;
    const uint64_t layer_stride = (uint64_t)history * hist_stride;
    const uint64_t slot_stride = (uint64_t)c.num_gdn_states * layer_stride;
    const uint64_t state_total = (uint64_t)c.max_sequences * slot_stride;
    const uint32_t in_stride = c.in_stride != 0u ? c.in_stride : conv_dim;
    const uint32_t out_stride = c.out_stride != 0u ? c.out_stride : conv_dim;
    const uint32_t rows = c.rows;
    const uint32_t nreq = (uint32_t)c.reqs.size();

    Bufs buf;
    auto m = [&](void** dst, size_t bytes) { return hipMalloc(dst, bytes) == hipSuccess; };
    if (!m(&buf.weight, (size_t)conv_dim * 4u * 2)) return false;
    if (!m(&buf.input, (size_t)rows * in_stride * 2)) return false;
    if (!m(&buf.out_correct, (size_t)rows * out_stride * 4)) return false;
    if (!m(&buf.out_opt, (size_t)rows * out_stride * 4)) return false;
    if (!m(&buf.state_correct, state_total * 2)) return false;
    if (!m(&buf.state_opt, state_total * 2)) return false;
    if (!m(&buf.requests, (size_t)nreq * sizeof(::ps::runtime::DeviceRequestDescriptor))) return false;

    std::vector<uint16_t> h_w((size_t)conv_dim * 4u);
    std::vector<uint16_t> h_in((size_t)rows * in_stride);
    std::vector<uint16_t> h_state(state_total);
    for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = bf16_bits(rng1k(c.seed * 3u + i) / 1000.0f);
    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t ch = 0; ch < conv_dim; ++ch)
            h_in[(size_t)r * in_stride + ch] = bf16_bits(rng1k(c.seed * 7u + (uint64_t)r * 131u + ch) / 1000.0f);
    for (size_t i = 0; i < h_state.size(); ++i) h_state[i] = bf16_bits(rng1k(c.seed * 11u + i) / 512.0f);

    if (hipMemcpy(buf.weight, h_w.data(), h_w.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.input, h_in.data(), h_in.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.state_correct, h_state.data(), h_state.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.state_opt, h_state.data(), h_state.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemset(buf.out_correct, 0, (size_t)rows * out_stride * 4) != hipSuccess) return false;
    if (hipMemset(buf.out_opt, 0, (size_t)rows * out_stride * 4) != hipSuccess) return false;

    std::vector<::ps::runtime::DeviceRequestDescriptor> h_req(nreq);
    for (uint32_t r = 0; r < nreq; ++r) {
        std::memset(&h_req[r], 0, sizeof(h_req[r]));
        h_req[r].request_handle.slot = c.reqs[r].slot;
        h_req[r].row_begin = c.reqs[r].row_begin;
        h_req[r].row_count = c.reqs[r].row_count;
    }
    if (hipMemcpy(buf.requests, h_req.data(), h_req.size() * sizeof(h_req[0]), hipMemcpyHostToDevice) != hipSuccess) return false;

    V vals[2];
    auto setv = [&](V& v, uint32_t id, uint32_t feat, uint32_t stride, void* ptr,
                    ::ps::runtime::ValueDType dt) {
        v.logical_value = id;
        v.storage = ::ps::runtime::ValueStorage::WORKSPACE;
        v.dtype = dt;
        v.row_domain = ::ps::runtime::ValueRowDomain::TOKEN_ROWS;
        v.slot = 0;
        v.feature_count = feat;
        v.row_stride = stride;
        v.offset = (uint64_t)(uintptr_t)ptr;
        v.bytes = 0;
    };
    setv(vals[0], 0, conv_dim, in_stride, buf.input, ::ps::runtime::ValueDType::BF16);
    setv(vals[1], 1, conv_dim, out_stride, buf.out_correct, ::ps::runtime::ValueDType::F32);
    if (hipMalloc(&buf.vals_dev, 2 * sizeof(V)) != hipSuccess) return false;
    std::memcpy(buf.vals_dev, vals, 2 * sizeof(V));

    S st;
    st.slot = 0;
    st.kind = (uint8_t)::ps::runtime::PrimitiveStateKind::GDN_CONV_STATE;
    st.state_index = c.state_index;
    if (hipMalloc(&buf.state_dev, sizeof(S)) != hipSuccess) return false;
    std::memcpy(buf.state_dev, &st, sizeof(S));

    ::ps::runtime::StaticParameterSlot param;
    param.device_ptr = buf.weight;
    param.elements = conv_dim * 4u;
    param.dtype = ::ps::runtime::ValueDType::BF16;
    if (hipMalloc(&buf.param_dev, sizeof(param)) != hipSuccess) return false;
    std::memcpy(buf.param_dev, &param, sizeof(param));

    ::ps::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_requests = nreq;
    ctx.requests = (const ::ps::runtime::DeviceRequestDescriptor*)buf.requests;
    if (hipMalloc(&buf.ctx_dev, sizeof(ctx)) != hipSuccess) return false;
    if (hipMemcpy(buf.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMalloc(&buf.scratch, (size_t)1024 * 4096 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&buf.error_word, sizeof(uint32_t)) != hipSuccess) return false;

    P prog{};
    prog.values = (const V*)buf.vals_dev;
    prog.value_count = 2;
    prog.states = (const S*)buf.state_dev;
    prog.state_count = 1;
    prog.workspace_base = 0;
    prog.parameter_table = (const ::ps::runtime::StaticParameterSlot*)buf.param_dev;
    prog.parameter_slot_count = 1;

    B binding{};
    binding.kernel_id = ::ps::runtime::KernelId::STATEFUL_CAUSAL_CONV1D;
    binding.descriptor_slot = 1;
    binding.input_slots[0] = 0;
    binding.output_slots[0] = 1;
    binding.input_count = 1;
    binding.output_count = 1;
    binding.parameter_index0 = 0;

    ::ps::kernel::ModelDispatchStateView state_view{};
    state_view.gdn.conv_base = (bf16t*)buf.state_correct;
    state_view.gdn.num_gdn_states = c.num_gdn_states;
    state_view.gdn.max_sequences = c.max_sequences;
    state_view.gdn.conv_slot_stride = slot_stride;
    state_view.gdn.conv_layer_stride = layer_stride;
    state_view.gdn.conv_history_stride = hist_stride;
    state_view.gdn.conv_dim = conv_dim;
    state_view.gdn.conv_history = history;

    ps::gpu::discard_cleanup_result(hipMemset(buf.error_word, 0, sizeof(uint32_t)));
    auto stc = ps::kernel::launch_model_dispatch_correctness(
        prog, binding, (::ps::runtime::DeviceBatchContext*)buf.ctx_dev, state_view,
        (float*)buf.scratch, (uint32_t*)buf.error_word, nullptr);
    if (!stc.ok()) {
        std::printf("  [%s] correctness launch failed\n", c.name);
        return false;
    }
    ps::gpu::discard_cleanup_result(hipGetLastError());

    ::ps::kernel::GdnConv1dArgs ka{};
    ka.input = (const bf16t*)buf.input;
    ka.output = (float*)buf.out_opt;
    ka.conv_weight = (const bf16t*)buf.weight;
    ka.conv_state = (bf16t*)buf.state_opt;
    ka.requests = (const ::ps::runtime::DeviceRequestDescriptor*)buf.requests;
    ka.num_requests = nreq;
    ka.actual_rows = rows;
    ka.input_row_stride = in_stride;
    ka.output_row_stride = out_stride;
    ka.conv_dim = conv_dim;
    ka.conv_history = history;
    ka.state_index = c.state_index;
    ka.max_sequences = c.max_sequences;
    ka.conv_slot_stride = slot_stride;
    ka.conv_layer_stride = layer_stride;
    ka.conv_history_stride = hist_stride;
    hipError_t he = ::ps::kernel::launch_gdn_conv1d_bf16_f32(ka, nullptr);
    if (he != hipSuccess) {
        std::printf("  [%s] optimized launch failed: %s\n", c.name, hipGetErrorString(he));
        return false;
    }
    ps::gpu::discard_cleanup_result(hipGetLastError());

    uint32_t ew = 0;
    ps::gpu::discard_cleanup_result(hipMemcpy(&ew, buf.error_word, sizeof(ew), hipMemcpyDeviceToHost));
    std::vector<float> oc((size_t)rows * out_stride), oo((size_t)rows * out_stride);
    std::vector<uint16_t> sc(state_total), so(state_total);
    if (hipMemcpy(oc.data(), buf.out_correct, oc.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(oo.data(), buf.out_opt, oo.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(sc.data(), buf.state_correct, sc.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(so.data(), buf.state_opt, so.size() * 2, hipMemcpyDeviceToHost) != hipSuccess) return false;

    double out_max_abs = 0.0, out_max_rel = 0.0;
    size_t out_bad = 0, out_nan = 0;
    for (size_t i = 0; i < oc.size(); ++i) {
        const double ref = (double)oc[i];
        const double got = (double)oo[i];
        if (!std::isfinite(got)) { ++out_nan; }
        if (!std::isfinite(ref) && !std::isfinite(got)) continue;
        const double diff = std::fabs(got - ref);
        const double rel = diff / std::fmax(1.0, std::fabs(ref));
        if (diff > out_max_abs) out_max_abs = diff;
        if (rel > out_max_rel) out_max_rel = rel;
        if (rel > 2e-4) ++out_bad;
    }
    size_t state_diff = 0;
    for (size_t i = 0; i < sc.size(); ++i)
        if (sc[i] != so[i]) ++state_diff;

    const bool pass = (ew == 0u) && (out_bad == 0u) && (out_nan == 0u) && (state_diff == 0u);
    std::printf("  [%s] dim=%u rows=%u nreq=%u si=%u | ew=%u out(maxabs=%.3e maxrel=%.3e bad=%zu nan=%zu) state_diff=%zu %s\n",
                c.name, conv_dim, rows, nreq, c.state_index, ew, out_max_abs, out_max_rel,
                out_bad, out_nan, state_diff, pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    if (hipSetDevice(0) != hipSuccess) return 1;

    int failures = 0;
    auto run = [&](const Case& c) { if (!run_case(c)) ++failures; };

    run({"27b-decode", 10240, 0, 1, 4, {{0, 0, 1}}, 1, 0, 0, 101});
    run({"27b-decode-multi", 10240, 0, 2, 4,
         {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}}, 4, 0, 0, 102});
    run({"27b-prefill", 10240, 0, 1, 1, {{0, 0, 128}}, 128, 0, 0, 103});
    run({"4b-prefill", 8192, 0, 1, 1, {{0, 0, 64}}, 64, 0, 0, 104});
    run({"4b-decode-multi", 8192, 0, 2, 4,
         {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}}, 4, 0, 0, 105});
    run({"state-index", 8192, 1, 3, 2, {{1, 0, 2}}, 2, 0, 0, 106});
    run({"row-begin", 8192, 0, 1, 2, {{0, 0, 2}, {1, 4, 2}}, 6, 0, 0, 107});
    run({"mixed-requests", 8192, 0, 1, 3,
         {{0, 0, 1}, {1, 1, 5}, {2, 6, 1}}, 7, 0, 0, 108});
    run({"padded-stride", 8192, 0, 1, 2, {{0, 0, 4}}, 4, 8192 + 16, 8192 + 32, 109});
    run({"conv-tail", 300, 0, 1, 2, {{0, 0, 4}}, 4, 0, 0, 110});
    run({"conv-tail-255", 255, 0, 1, 1, {{0, 0, 3}}, 3, 0, 0, 111});
    run({"slot-ne-reqidx", 8192, 0, 1, 4, {{2, 0, 1}}, 1, 0, 0, 112});
    run({"27b-tp2-decode", 5120, 0, 1, 4, {{0, 0, 1}}, 1, 0, 0, 201});
    run({"27b-tp2-prefill", 5120, 0, 1, 1, {{0, 0, 128}}, 128, 0, 0, 202});
    run({"27b-tp2-decode-multi", 5120, 0, 2, 4,
         {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}}, 4, 0, 0, 203});

    std::printf("%s: %d failures\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
