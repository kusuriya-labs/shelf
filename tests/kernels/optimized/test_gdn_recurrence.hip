#include <shelf/kernels/optimized/gdn/recurrence.h>
#include <shelf/kernels/correctness/model_dispatch_correctness.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/core/gpu/cleanup.h>
#include <shelf/core/memory/types.h>
#include <shelf/runtime/program/device_program.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using bf16t = shelf::bf16_t;
using fp16t = shelf::fp16_t;
using V = ::shelf::runtime::DeviceValueBinding;
using S = ::shelf::runtime::DeviceStateBinding;
using P = ::shelf::runtime::DeviceProgramView;
using B = ::shelf::runtime::DispatchBinding;

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

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

struct Req {
    uint32_t slot;
    uint32_t row_begin;
    uint32_t row_count;
};

struct Case {
    const char* name;
    uint32_t key_heads;
    uint32_t num_v_heads;
    uint32_t head_k;
    uint32_t head_v;
    uint32_t state_index;
    uint32_t num_gdn_states;
    uint32_t max_sequences;
    std::vector<Req> reqs;
    uint32_t rows;
    uint32_t seed;
    bool poison_v = false;
    bool verify = false;
    // Verify with skip_state_store: outputs match the reference, the state is untouched.
    bool skip_store = false;
};

struct Bufs {
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    void* dt_bias = nullptr;
    void* a_log = nullptr;
    void* state_correct = nullptr;
    void* state_opt = nullptr;
    void* out_correct = nullptr;
    void* out_opt = nullptr;
    void* requests = nullptr;
    void* vals_dev = nullptr;
    void* state_dev = nullptr;
    void* param_dev = nullptr;
    void* ctx_dev = nullptr;
    void* scratch = nullptr;
    void* error_word = nullptr;
    ~Bufs() {
        void* all[] = {q, k, v, a, b, dt_bias, a_log, state_correct, state_opt, out_correct,
                       out_opt, requests, vals_dev, state_dev, param_dev, ctx_dev, scratch,
                       error_word};
        for (auto* p : all)
            if (p) shelf::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool run_case(const Case& c) {
    const uint32_t qk = c.key_heads * c.head_k;
    const uint32_t vdim = c.num_v_heads * c.head_v;
    const uint32_t nvh = c.num_v_heads;
    const uint32_t rows = c.rows;
    const uint32_t nreq = (uint32_t)c.reqs.size();
    const uint32_t repeat = nvh / c.key_heads;
    const uint64_t layer_stride = (uint64_t)nvh * c.head_k * c.head_v;
    const uint64_t slot_stride = (uint64_t)c.num_gdn_states * layer_stride;
    const uint64_t state_total = (uint64_t)c.max_sequences * slot_stride;

    Bufs buf;
    auto m = [&](void** dst, size_t bytes) {
        if (hipMalloc(dst, bytes) != hipSuccess) return false;
        return true;
    };
    if (!m(&buf.q, (size_t)rows * qk * 4)) return false;
    if (!m(&buf.k, (size_t)rows * qk * 4)) return false;
    if (!m(&buf.v, (size_t)rows * vdim * 2)) return false;
    if (!m(&buf.a, (size_t)rows * nvh * 2)) return false;
    if (!m(&buf.b, (size_t)rows * nvh * 2)) return false;
    if (!m(&buf.dt_bias, (size_t)nvh * 2)) return false;
    if (!m(&buf.a_log, (size_t)nvh * 2)) return false;
    if (!m(&buf.state_correct, state_total * sizeof(fp16t))) return false;
    if (!m(&buf.state_opt, state_total * sizeof(fp16t))) return false;
    if (!m(&buf.out_correct, (size_t)rows * vdim * 4)) return false;
    if (!m(&buf.out_opt, (size_t)rows * vdim * 4)) return false;
    if (!m(&buf.requests, (size_t)nreq * sizeof(::shelf::runtime::DeviceRequestDescriptor))) return false;

    std::vector<float> h_q((size_t)rows * qk), h_k((size_t)rows * qk);
    std::vector<uint16_t> h_v((size_t)rows * vdim), h_a((size_t)rows * nvh), h_b((size_t)rows * nvh);
    std::vector<uint16_t> h_dt(nvh), h_al(nvh);
    std::vector<float> h_state(state_total);
    for (size_t i = 0; i < h_q.size(); ++i) h_q[i] = rng1k(c.seed * 3u + i) / 1000.0f;
    for (size_t i = 0; i < h_k.size(); ++i) h_k[i] = rng1k(c.seed * 5u + i) / 1000.0f;
    for (size_t i = 0; i < h_v.size(); ++i) h_v[i] = bf16_bits(rng1k(c.seed * 7u + i) / 1000.0f);
    if (c.poison_v) h_v[(size_t)(rows - 1u) * vdim + 5 * c.head_v + 7] = 0x7F80u;
    for (size_t i = 0; i < h_a.size(); ++i) h_a[i] = bf16_bits(rng1k(c.seed * 11u + i) / 1000.0f);
    for (size_t i = 0; i < h_b.size(); ++i) h_b[i] = bf16_bits(rng1k(c.seed * 13u + i) / 1000.0f);
    for (size_t i = 0; i < h_dt.size(); ++i) h_dt[i] = bf16_bits(rng1k(c.seed * 17u + i) / 1000.0f);
    for (size_t i = 0; i < h_al.size(); ++i) h_al[i] = bf16_bits(rng1k(c.seed * 19u + i) / 1000.0f);
    for (size_t i = 0; i < h_state.size(); ++i) h_state[i] = rng1k(c.seed * 23u + i) / 512.0f;

    for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t kh = 0; kh < c.key_heads; ++kh) {
            const size_t off = (size_t)r * qk + (size_t)kh * c.head_k;
            double sq = 0.0, sk = 0.0;
            for (uint32_t d = 0; d < c.head_k; ++d) {
                sq += (double)h_q[off + d] * (double)h_q[off + d];
                sk += (double)h_k[off + d] * (double)h_k[off + d];
            }
            const float nq = sq > 0.0 ? (float)(1.0 / std::sqrt(sq)) : 0.0f;
            const float nk = sk > 0.0 ? (float)(1.0 / std::sqrt(sk)) : 0.0f;
            for (uint32_t d = 0; d < c.head_k; ++d) {
                h_q[off + d] *= nq;
                h_k[off + d] *= nk;
            }
        }

    if (hipMemcpy(buf.q, h_q.data(), h_q.size() * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.k, h_k.data(), h_k.size() * 4, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.v, h_v.data(), h_v.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.a, h_a.data(), h_a.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.b, h_b.data(), h_b.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.dt_bias, h_dt.data(), h_dt.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.a_log, h_al.data(), h_al.size() * 2, hipMemcpyHostToDevice) != hipSuccess) return false;
    std::vector<fp16t> h_state16(h_state.size());
    for (size_t i = 0; i < h_state.size(); ++i) h_state16[i] = shelf::f32_to_fp16(h_state[i]);
    if (hipMemcpy(buf.state_correct, h_state16.data(), h_state16.size() * sizeof(fp16t), hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemcpy(buf.state_opt, h_state16.data(), h_state16.size() * sizeof(fp16t), hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMemset(buf.out_correct, 0, (size_t)rows * vdim * 4) != hipSuccess) return false;
    if (hipMemset(buf.out_opt, 0, (size_t)rows * vdim * 4) != hipSuccess) return false;

    std::vector<::shelf::runtime::DeviceRequestDescriptor> h_req(nreq);
    for (uint32_t r = 0; r < nreq; ++r) {
        std::memset(&h_req[r], 0, sizeof(h_req[r]));
        h_req[r].request_handle.slot = c.reqs[r].slot;
        h_req[r].row_begin = c.reqs[r].row_begin;
        h_req[r].row_count = c.reqs[r].row_count;
    }
    if (hipMemcpy(buf.requests, h_req.data(), h_req.size() * sizeof(h_req[0]), hipMemcpyHostToDevice) != hipSuccess) return false;

    V vals[6];
    auto setv = [&](V& v, uint32_t id, uint32_t feat, uint32_t stride, void* ptr,
                    ::shelf::runtime::ValueDType dt) {
        v.logical_value = id;
        v.storage = ::shelf::runtime::ValueStorage::WORKSPACE;
        v.dtype = dt;
        v.row_domain = ::shelf::runtime::ValueRowDomain::TOKEN_ROWS;
        v.slot = 0;
        v.feature_count = feat;
        v.row_stride = stride;
        v.offset = (uint64_t)(uintptr_t)ptr;
        v.bytes = 0;
    };
    setv(vals[0], 0, qk, qk, buf.q, ::shelf::runtime::ValueDType::F32);
    setv(vals[1], 1, qk, qk, buf.k, ::shelf::runtime::ValueDType::F32);
    setv(vals[2], 2, vdim, vdim, buf.v, ::shelf::runtime::ValueDType::BF16);
    setv(vals[3], 3, nvh, nvh, buf.a, ::shelf::runtime::ValueDType::BF16);
    setv(vals[4], 4, nvh, nvh, buf.b, ::shelf::runtime::ValueDType::BF16);
    setv(vals[5], 5, vdim, vdim, buf.out_correct, ::shelf::runtime::ValueDType::F32);
    if (hipMalloc(&buf.vals_dev, 6 * sizeof(V)) != hipSuccess) return false;
    std::memcpy(buf.vals_dev, vals, 6 * sizeof(V));

    S st;
    st.slot = 0;
    st.kind = (uint8_t)::shelf::runtime::PrimitiveStateKind::GDN_RECURRENCE_STATE;
    st.state_index = c.state_index;
    if (hipMalloc(&buf.state_dev, sizeof(S)) != hipSuccess) return false;
    std::memcpy(buf.state_dev, &st, sizeof(S));

    ::shelf::runtime::StaticParameterSlot params[2];
    params[0].device_ptr = buf.dt_bias;
    params[0].elements = nvh;
    params[0].dtype = ::shelf::runtime::ValueDType::BF16;
    params[1].device_ptr = buf.a_log;
    params[1].elements = nvh;
    params[1].dtype = ::shelf::runtime::ValueDType::BF16;
    if (hipMalloc(&buf.param_dev, 2 * sizeof(::shelf::runtime::StaticParameterSlot)) != hipSuccess) return false;
    std::memcpy(buf.param_dev, params, 2 * sizeof(::shelf::runtime::StaticParameterSlot));

    ::shelf::runtime::DeviceBatchContext ctx{};
    ctx.actual_rows = rows;
    ctx.num_requests = nreq;
    ctx.requests = (const ::shelf::runtime::DeviceRequestDescriptor*)buf.requests;
    if (hipMalloc(&buf.ctx_dev, sizeof(ctx)) != hipSuccess) return false;
    if (hipMemcpy(buf.ctx_dev, &ctx, sizeof(ctx), hipMemcpyHostToDevice) != hipSuccess) return false;
    if (hipMalloc(&buf.scratch, (size_t)1024 * 4096 * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&buf.error_word, sizeof(uint32_t)) != hipSuccess) return false;

    P prog{};
    prog.values = (const V*)buf.vals_dev;
    prog.value_count = 6;
    prog.states = (const S*)buf.state_dev;
    prog.state_count = 1;
    prog.workspace_base = 0;
    prog.parameter_table = (const ::shelf::runtime::StaticParameterSlot*)buf.param_dev;
    prog.parameter_slot_count = 2;

    B binding{};
    binding.kernel_id = ::shelf::runtime::KernelId::GDN_RECURRENCE;
    binding.descriptor_slot = 1;
    binding.input_slots[0] = 0;
    binding.input_slots[1] = 1;
    binding.input_slots[2] = 2;
    binding.input_slots[3] = 3;
    binding.input_slots[4] = 4;
    binding.output_slots[0] = 5;
    binding.input_count = 5;
    binding.output_count = 1;
    binding.parameter_index0 = 0;
    binding.parameter_index1 = 1;

    ::shelf::kernel::ModelDispatchStateView state_view{};
    state_view.gdn.recurrent_base = (fp16t*)buf.state_correct;
    state_view.gdn.num_gdn_states = c.num_gdn_states;
    state_view.gdn.max_sequences = c.max_sequences;
    state_view.gdn.recurrent_slot_stride = slot_stride;
    state_view.gdn.recurrent_layer_stride = layer_stride;
    state_view.gdn.num_v_heads = nvh;
    state_view.gdn.head_k = c.head_k;
    state_view.gdn.head_v = c.head_v;

    shelf::gpu::discard_cleanup_result(hipMemset(buf.error_word, 0, sizeof(uint32_t)));
    auto stc = shelf::kernel::launch_model_dispatch_correctness(
        prog, binding, (::shelf::runtime::DeviceBatchContext*)buf.ctx_dev, state_view,
        (float*)buf.scratch, (uint32_t*)buf.error_word, nullptr);
    if (!stc.ok()) {
        std::printf("  [%s] correctness launch failed\n", c.name);
        return false;
    }
    shelf::gpu::discard_cleanup_result(hipGetLastError());

    ::shelf::kernel::GdnRecurrenceArgs ka{};
    ka.q = (const float*)buf.q;
    ka.k = (const float*)buf.k;
    ka.v = (const bf16t*)buf.v;
    ka.a = (const bf16t*)buf.a;
    ka.b = (const bf16t*)buf.b;
    ka.dt_bias = (const bf16t*)buf.dt_bias;
    ka.a_log = (const bf16t*)buf.a_log;
    ka.recurrent_state = (fp16t*)buf.state_opt;
    ka.output = (float*)buf.out_opt;
    ka.requests = (const ::shelf::runtime::DeviceRequestDescriptor*)buf.requests;
    ka.num_requests = nreq;
    ka.actual_rows = rows;
    ka.q_row_stride = qk;
    ka.k_row_stride = qk;
    ka.v_row_stride = vdim;
    ka.a_row_stride = nvh;
    ka.b_row_stride = nvh;
    ka.output_row_stride = vdim;
    ka.state_index = c.state_index;
    ka.max_sequences = c.max_sequences;
    ka.num_gdn_states = c.num_gdn_states;
    ka.key_heads = c.key_heads;
    ka.num_v_heads = nvh;
    ka.head_k = c.head_k;
    ka.head_v = c.head_v;
    ka.recurrent_slot_stride = slot_stride;
    ka.recurrent_layer_stride = layer_stride;
    ka.skip_state_store = c.skip_store ? 1u : 0u;
    hipError_t he = hipSuccess;
    if (c.verify) {
        he = ::shelf::kernel::launch_gdn_recurrence_f32_wmma_verify(ka, nullptr);
    } else if (c.head_k == 128u && (c.head_v % 64u) == 0u) {
        he = ::shelf::kernel::launch_gdn_recurrence_f32_wmma(ka, nullptr);
    } else {
        he = ::shelf::kernel::launch_gdn_recurrence_f32(ka, nullptr);
    }
    if (he != hipSuccess) {
        std::printf("  [%s] optimized launch failed: %s\n", c.name, hipGetErrorString(he));
        return false;
    }
    shelf::gpu::discard_cleanup_result(hipGetLastError());

    uint32_t ew = 0;
    shelf::gpu::discard_cleanup_result(hipMemcpy(&ew, buf.error_word, sizeof(ew), hipMemcpyDeviceToHost));
    std::vector<float> oc((size_t)rows * vdim), oo((size_t)rows * vdim);
    std::vector<fp16t> sc16(state_total), so16(state_total);
    if (hipMemcpy(oc.data(), buf.out_correct, oc.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(oo.data(), buf.out_opt, oo.size() * 4, hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(sc16.data(), buf.state_correct, sc16.size() * sizeof(fp16t), hipMemcpyDeviceToHost) != hipSuccess) return false;
    if (hipMemcpy(so16.data(), buf.state_opt, so16.size() * sizeof(fp16t), hipMemcpyDeviceToHost) != hipSuccess) return false;
    bool experimental_pass = true;
    bool single_rows = c.rows == nreq && !c.verify && !c.poison_v && c.head_k == 128u && c.head_v % 64u == 0u;
    for (const auto& req : c.reqs) single_rows = single_rows && req.row_count == 1u;
    if (single_rows && ::shelf::kernel::gdn_recurrence_lossy_enabled()) {
        if (hipMemcpy(buf.state_opt, h_state16.data(), h_state16.size() * sizeof(fp16t), hipMemcpyHostToDevice) != hipSuccess ||
            ::shelf::kernel::launch_gdn_recurrence_f32_wmma_single_row_experimental(ka, nullptr) != hipSuccess) return false;
        std::vector<float> candidate_out(oo.size());
        std::vector<fp16t> candidate_state(so16.size());
        if (hipMemcpy(candidate_out.data(), buf.out_opt, candidate_out.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess ||
            hipMemcpy(candidate_state.data(), buf.state_opt, candidate_state.size() * sizeof(fp16t), hipMemcpyDeviceToHost) != hipSuccess) return false;
        size_t output_changed = 0, state_changed = 0;
        double max_output_diff = 0.0, max_state_diff = 0.0;
        for (size_t i = 0; i < oo.size(); ++i) {
            if (std::memcmp(&oo[i], &candidate_out[i], sizeof(float))) ++output_changed;
            max_output_diff = std::fmax(max_output_diff, std::fabs(double(oo[i]) - candidate_out[i]));
        }
        for (size_t i = 0; i < so16.size(); ++i) {
            if (std::memcmp(&so16[i], &candidate_state[i], sizeof(fp16t))) ++state_changed;
            max_state_diff = std::fmax(max_state_diff, std::fabs(double(shelf::fp16_to_f32(so16[i])) - shelf::fp16_to_f32(candidate_state[i])));
        }
        experimental_pass = output_changed == 0u && state_changed == 0u;
        std::printf("  [%s] experimental vs WMMA: changed=%zu/%zu max_abs=%.3e/%.3e %s\n", c.name,
                    output_changed, state_changed, max_output_diff, max_state_diff,
                    experimental_pass ? "PASS" : "FAIL");
    }
    std::vector<float> sc(state_total), so(state_total);
    for (size_t i = 0; i < state_total; ++i) {
        sc[i] = shelf::fp16_to_f32(sc16[i]);
        so[i] = shelf::fp16_to_f32(so16[i]);
        if (c.skip_store) sc[i] = shelf::fp16_to_f32(h_state16[i]);
    }

    const auto out_hit = [&](size_t i) {
        if (!c.poison_v) return false;
        const size_t r = i / vdim;
        const size_t rem = i % vdim;
        return rem / c.head_v == 5u && rem % c.head_v == 7u && r >= 16u;
    };
    const auto state_hit = [&](size_t i) {
        if (!c.poison_v) return false;
        const size_t col = i % c.head_v;
        const size_t head = (i / ((size_t)c.head_k * c.head_v)) % nvh;
        return head == 5u && col == 7u;
    };
    size_t out_nan = 0, state_nan = 0;
    size_t out_nan_hit = 0, state_nan_hit = 0;
    double out_max_abs = 0.0, out_sum_abs = 0.0, out_max_rel = 0.0;
    double state_max_abs = 0.0, state_sum_abs = 0.0, state_max_rel = 0.0;
    size_t out_bigdiff = 0;
    size_t state_bigdiff = 0;
    if (c.poison_v) {
        size_t out_bad_mask = 0, state_bad_mask = 0;
        for (size_t i = 0; i < oc.size(); ++i)
            if (std::isfinite((double)oc[i]) != std::isfinite((double)oo[i])) ++out_bad_mask;
        for (size_t i = 0; i < sc.size(); ++i)
            if (std::isfinite((double)sc[i]) != std::isfinite((double)so[i])) ++state_bad_mask;
        for (size_t i = 0; i < oc.size(); ++i)
            if (!std::isfinite((double)oc[i]) && !std::isfinite((double)oo[i])) ++out_nan_hit;
        for (size_t i = 0; i < sc.size(); ++i)
            if (!std::isfinite((double)sc[i]) && !std::isfinite((double)so[i])) ++state_nan_hit;
        if (out_bad_mask != 0u || state_bad_mask != 0u) ++out_bigdiff;
        if (out_nan_hit == 0u || state_nan_hit == 0u) ++out_bigdiff;
    }
    const bool lossy = []() {
        const char* x = std::getenv("SHELF_GDN_RECURRENCE_EXACT");
        if (x != nullptr && x[0] != '\0' && x[0] != '0') return false;
        const char* l = std::getenv("SHELF_GDN_RECURRENCE_LOSSY");
        if (l != nullptr && l[0] == '0') return false;
        return true;
    }();
    const double out_rel_tol = lossy ? 2e-2 : 2e-4;
    const double state_rel_tol = lossy ? 1e-2 : 1e-3;
    for (size_t i = 0; i < oc.size(); ++i) {
        const double ref = (double)oc[i];
        const double got = (double)oo[i];
        if (!std::isfinite(ref) || !std::isfinite(got)) {
            if (!std::isfinite(got)) ++out_nan;
            continue;
        }
        const double diff = std::fabs(got - ref);
        out_sum_abs += diff;
        if (diff > out_max_abs) out_max_abs = diff;
        const double den = std::fmax(1.0, std::fabs(got));
        const double rel = diff / den;
        if (rel > out_max_rel) out_max_rel = rel;
        if (rel > out_rel_tol) ++out_bigdiff;
    }
    for (size_t i = 0; i < sc.size(); ++i) {
        const double ref = (double)sc[i];
        const double got = (double)so[i];
        if (!std::isfinite(ref) || !std::isfinite(got)) {
            if (!std::isfinite(got)) ++state_nan;
            continue;
        }
        const double diff = std::fabs(got - ref);
        state_sum_abs += diff;
        if (diff > state_max_abs) state_max_abs = diff;
        const double den = std::fmax(1.0, std::fabs(got));
        const double rel = diff / den;
        if (rel > state_max_rel) state_max_rel = rel;
        if (rel > state_rel_tol) ++state_bigdiff;
    }
    const double out_mean_abs = out_sum_abs / (double)oc.size();
    const double state_mean_abs = state_sum_abs / (double)sc.size();

    auto d2 = [&](uint16_t x) { return (double)bf16_to_f32(x); };
    const double max_rel = 0.0;
    double cpu_max_rel = 0.0;
    {
        std::vector<double> S((size_t)c.head_k);
        for (uint32_t r = 0; r < nreq; ++r) {
            const uint32_t slot = c.reqs[r].slot;
            for (uint32_t h = 0; h < nvh; ++h) {
                const uint32_t qkh = h / repeat;
                for (uint32_t vc = 0; vc < c.head_v; ++vc) {
                    const uint64_t base = (uint64_t)slot * slot_stride + (uint64_t)c.state_index * layer_stride +
                        (uint64_t)h * c.head_k * c.head_v + vc;
                    for (uint32_t kk = 0; kk < c.head_k; ++kk)
                        S[kk] = (double)h_state[base + (uint64_t)kk * c.head_v];
                    for (uint32_t t = 0; t < c.reqs[r].row_count; ++t) {
                        const uint32_t row = c.reqs[r].row_begin + t;
                        if (row >= rows) continue;
                        const double beta = 1.0 / (1.0 + std::exp(-d2(h_b[(size_t)row * nvh + h])));
                        const double a_val = d2(h_a[(size_t)row * nvh + h]);
                        const double dt = a_val + d2(h_dt[h]);
                        const double g = -std::exp(d2(h_al[h])) * std::log(1.0 + std::exp(dt));
                        const double d_t = std::exp(g);
                        const double v_val = d2(h_v[(size_t)row * vdim + (size_t)h * c.head_v + vc]);
                        double kv = 0.0;
                        for (uint32_t kk = 0; kk < c.head_k; ++kk)
                            kv += S[kk] * d_t * (double)h_k[(size_t)row * qk + (size_t)qkh * c.head_k + kk];
                        const double delta = (v_val - kv) * beta;
                        for (uint32_t kk = 0; kk < c.head_k; ++kk)
                            S[kk] = S[kk] * d_t + (double)h_k[(size_t)row * qk + (size_t)qkh * c.head_k + kk] * delta;
                        double acc = 0.0;
                        for (uint32_t kk = 0; kk < c.head_k; ++kk)
                            acc += S[kk] * (double)h_q[(size_t)row * qk + (size_t)qkh * c.head_k + kk];
                        if (t == 0u) {
                            const double ref = acc;
                            const double got = (double)oc[(size_t)row * vdim + (size_t)h * c.head_v + vc];
                            const double den = std::fmax(1.0, std::fabs(ref));
                            const double rel = std::fabs(got - ref) / den;
                            if (rel > cpu_max_rel) cpu_max_rel = rel;
                        }
                    }
                }
            }
        }
        (void)max_rel;
    }

    const bool pass = experimental_pass && (c.poison_v || (out_nan == 0u && state_nan == 0u)) &&
                      (out_bigdiff == 0) && (state_bigdiff == 0) && (cpu_max_rel < 2e-2);
    std::printf("  [%s] kh=%u nvh=%u hk=%u hv=%u rows=%u nreq=%u si=%u | ew=%u nan=%zu/%zu "
                "out(maxabs=%.3e meanabs=%.3e maxrel=%.3e bad=%zu) state(maxabs=%.3e meanabs=%.3e maxrel=%.3e bad=%zu) cpu_rel=%.3e mode=%s %s\n",
                c.name, c.key_heads, nvh, c.head_k, c.head_v, rows, nreq, c.state_index,
                ew, out_nan, state_nan,
                out_max_abs, out_mean_abs, out_max_rel, out_bigdiff,
                state_max_abs, state_mean_abs, state_max_rel, state_bigdiff,
                cpu_max_rel, lossy ? "lossy" : "exact", pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    int failures = 0;
    auto run = [&](const Case& c) {
        if (!run_case(c)) ++failures;
    };
    run({"4b-decode-multi", 16, 32, 128, 128, 0, 1, 4,
         {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}}, 4, 101});
    run({"4b-prefill", 16, 32, 128, 128, 0, 1, 1,
         {{0, 0, 64}}, 64, 202});
    run({"headv-tail", 2, 4, 64, 100, 0, 1, 1,
         {{0, 0, 3}}, 3, 303});
    run({"repeat1", 4, 4, 64, 64, 0, 1, 1,
         {{0, 0, 2}}, 2, 404});
    run({"repeat4", 2, 8, 64, 32, 0, 1, 1,
         {{0, 0, 2}}, 2, 505});
    run({"slot-ne-reqidx", 4, 8, 32, 32, 0, 1, 4,
         {{2, 0, 1}}, 1, 606});
    run({"state-index", 4, 8, 32, 32, 1, 3, 2,
         {{1, 0, 2}}, 2, 707});
    run({"row-begin", 4, 8, 32, 32, 0, 1, 2,
         {{0, 0, 2}, {1, 4, 2}}, 6, 808});
    run({"mixed", 8, 16, 64, 64, 0, 1, 3,
         {{0, 0, 1}, {1, 1, 5}, {2, 6, 1}}, 7, 909});
    run({"4b-prefill-128", 16, 32, 128, 128, 0, 1, 1,
         {{0, 0, 128}}, 128, 1010});
    run({"4b-decode-1", 16, 32, 128, 128, 0, 1, 1,
         {{0, 0, 1}}, 1, 1011});
    run({"4b-prefill-192", 16, 32, 128, 128, 0, 1, 2,
         {{0, 0, 160}, {1, 160, 32}}, 192, 1012});
    run({"rows31", 16, 32, 128, 128, 0, 1, 1, {{0, 0, 31}}, 31, 1113});
    run({"rows32", 16, 32, 128, 128, 0, 1, 1, {{0, 0, 32}}, 32, 1114});
    run({"rows33", 16, 32, 128, 128, 0, 1, 1, {{0, 0, 33}}, 33, 1115});
    run({"rows36", 16, 32, 128, 128, 0, 1, 1, {{0, 0, 36}}, 36, 1116});
    run({"rows96", 16, 32, 128, 128, 0, 1, 1, {{0, 0, 96}}, 96, 1117});
    run({"rows96-nr4", 16, 32, 128, 128, 0, 2, 4,
         {{0, 0, 24}, {1, 24, 24}, {2, 48, 24}, {3, 72, 24}}, 96, 1118});
    run({"27b-tp2-decode", 8, 24, 128, 128, 0, 1, 1, {{0, 0, 1}}, 1, 2027});
    run({"27b-tp2-prefill", 8, 24, 128, 128, 0, 1, 1, {{0, 0, 64}}, 64, 2028});
    run({"27b-tp2-decode-multi", 8, 24, 128, 128, 0, 1, 4,
         {{0, 0, 1}, {1, 1, 1}, {2, 2, 1}, {3, 3, 1}}, 4, 2029});
    run({"wmma-verify-m32", 16, 32, 128, 128, 0, 1, 4,
         {{0, 0, 8}, {1, 8, 8}, {2, 16, 8}, {3, 24, 8}}, 32, 3030, false, true});
    run({"wmma-verify-m32-27b", 8, 24, 128, 128, 0, 1, 4,
         {{0, 0, 8}, {1, 8, 8}, {2, 16, 8}, {3, 24, 8}}, 32, 3031, false, true});
    {
        std::vector<Req> eight, sixteen;
        for (uint32_t i = 0; i < 8u; ++i) eight.push_back({i, i * 8u, 8u});
        for (uint32_t i = 0; i < 16u; ++i) sixteen.push_back({i, i * 8u, 8u});
        run({"wmma-verify-nr8-m64", 16, 32, 128, 128, 0, 1, 8, eight, 64, 3032, false, true});
        run({"wmma-verify-nr16-m128-27b", 8, 24, 128, 128, 0, 1, 16, sixteen, 128, 3033, false, true});
        run({"wmma-verify-skip-store", 8, 24, 128, 128, 0, 1, 8, eight, 64, 3034, false, true, true});
    }
    run({"nan-input", 16, 32, 128, 128, 0, 1, 1, {{0, 0, 33}}, 33, 1119, true});
    std::printf("%s: %d failures\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
