#include <phaseshift/core/gpu/cleanup.h>
#include <phaseshift/core/memory/types.h>
#include <phaseshift/kernels/optimized/elementwise.h>
#include <phaseshift/kernels/optimized/gdn/conv1d.h>
#include <phaseshift/kernels/optimized/gdn/fused_gdn_qk_prepare.h>
#include <phaseshift/kernels/optimized/gdn/recurrence.h>
#include <phaseshift/runtime/batch/device_batch_context.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

uint32_t xr(uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

float rndf(uint32_t& s, float scale) {
    return ((float)(int32_t)(xr(s) % 2001u) - 1000.0f) / 1000.0f * scale;
}

float widen_bf16(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct Metrics {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rmse = 0.0;
    std::size_t worst = 0;
};

Metrics diff_metrics(const float* a, const float* b, std::size_t n) {
    Metrics m;
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > m.max_abs) {
            m.max_abs = d;
            m.worst = i;
        }
        const double den = std::fabs((double)b[i]);
        if (den > 1e-30) {
            const double r = d / den;
            if (r > m.max_rel) m.max_rel = r;
        }
        acc += d * d;
    }
    if (n) m.rmse = std::sqrt(acc / (double)n);
    return m;
}

void report(const char* group, const char* name, const Metrics& m) {
    std::printf("METRIC %-10s %-28s max_abs=%.6e max_rel=%.6e rmse=%.6e worst=%zu\n",
                group, name, m.max_abs, m.max_rel, m.rmse, m.worst);
}

void check_close(const Metrics& m, double atol, double rtol, const std::string& msg) {
    const bool ok = m.max_abs <= atol || m.max_rel <= rtol;
    check(ok, msg);
    if (!ok) {
        std::printf("  detail: max_abs=%.6e atol=%.1e max_rel=%.6e rtol=%.1e\n",
                    m.max_abs, atol, m.max_rel, rtol);
    }
}

constexpr uint32_t kKeyHeads = 16;
constexpr uint32_t kValHeads = 48;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQk = kKeyHeads * kHeadDim;
constexpr uint32_t kVal = kValHeads * kHeadDim;
constexpr uint32_t kConvDim = 2 * kQk + kVal;
constexpr float kEps = 1e-6f;
const float kQSpecScale = 1.0f / std::sqrt(128.0f);

bool run_conv_fit() {
    constexpr uint32_t kRows = 72;
    constexpr uint32_t kPhase1 = 64;
    constexpr uint32_t kHist = 3;
    constexpr uint32_t kHistStride = (kConvDim + 7u) & ~7u;
    constexpr uint32_t kLayerStride = kHist * kHistStride;

    void* d_in = nullptr;
    void* d_w = nullptr;
    void* d_state = nullptr;
    void* d_out = nullptr;
    void* d_req = nullptr;
    const size_t in_bytes = (size_t)kRows * kConvDim * 2u;
    const size_t w_bytes = (size_t)kConvDim * 4u * 2u;
    const size_t st_bytes = (size_t)kHist * kHistStride * 2u;
    const size_t out_bytes = (size_t)kRows * kConvDim * 4u;
    bool ok = hipMalloc(&d_in, in_bytes) == hipSuccess &&
              hipMalloc(&d_w, w_bytes) == hipSuccess &&
              hipMalloc(&d_state, st_bytes) == hipSuccess &&
              hipMalloc(&d_out, out_bytes) == hipSuccess &&
              hipMalloc(&d_req, sizeof(::ps::runtime::DeviceRequestDescriptor)) == hipSuccess;
    if (!ok) {
        std::printf("FAIL: conv alloc\n");
        return false;
    }

    std::vector<std::uint16_t> h_in((size_t)kRows * kConvDim);
    std::vector<std::uint16_t> h_w((size_t)kConvDim * 4u);
    std::vector<std::uint16_t> h_state((size_t)kHist * kHistStride);
    uint32_t s = 7717u;
    for (auto& v : h_in) v = ps::f32_to_bf16_rne(rndf(s, 1.0f));
    for (auto& v : h_w) v = ps::f32_to_bf16_rne(rndf(s, 0.01f));
    for (uint32_t i = 0; i < kHist * kHistStride; ++i) {
        h_state[i] = (i < (size_t)kHist * kConvDim)
                         ? ps::f32_to_bf16_rne(rndf(s, 0.5f))
                         : (uint16_t)0u;
    }
    ps::gpu::discard_cleanup_result(hipMemcpy(d_in, h_in.data(), in_bytes, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_w, h_w.data(), w_bytes, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(
        hipMemcpy(d_state, h_state.data(), st_bytes, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemset(d_out, 0, out_bytes));

    std::vector<float> w_f((size_t)kConvDim * 4u);
    for (size_t i = 0; i < w_f.size(); ++i) w_f[i] = widen_bf16(h_w[i]);
    std::vector<float> in_f((size_t)kRows * kConvDim);
    for (size_t i = 0; i < in_f.size(); ++i) in_f[i] = widen_bf16(h_in[i]);
    std::vector<float> st_f((size_t)kHist * kHistStride);
    for (size_t i = 0; i < st_f.size(); ++i) st_f[i] = widen_bf16(h_state[i]);
    std::vector<float> y_g((size_t)kRows * kConvDim, 0.0f);

    auto conv_step = [&](uint32_t row0, uint32_t count, std::vector<float>& state_f) {
        float s0[10240];
        float s1[10240];
        float s2[10240];
        for (uint32_t ch = 0; ch < kConvDim; ++ch) {
            s0[ch] = state_f[ch];
            s1[ch] = state_f[kHistStride + ch];
            s2[ch] = state_f[2u * kHistStride + ch];
        }
        for (uint32_t t = 0; t < count; ++t) {
            const uint32_t row = row0 + t;
            for (uint32_t ch = 0; ch < kConvDim; ++ch) {
                const float x = in_f[(size_t)row * kConvDim + ch];
                const float y = w_f[(size_t)ch * 4u + 0u] * s0[ch] +
                                w_f[(size_t)ch * 4u + 1u] * s1[ch] +
                                w_f[(size_t)ch * 4u + 2u] * s2[ch] +
                                w_f[(size_t)ch * 4u + 3u] * x;
                y_g[(size_t)row * kConvDim + ch] = y;
                s0[ch] = s1[ch];
                s1[ch] = s2[ch];
                s2[ch] = x;
            }
        }
        for (uint32_t ch = 0; ch < kConvDim; ++ch) {
            state_f[ch] = s0[ch];
            state_f[kHistStride + ch] = s1[ch];
            state_f[2u * kHistStride + ch] = s2[ch];
        }
    };

    std::vector<float> oracle_state((size_t)kHist * kHistStride, 0.0f);
    for (size_t i = 0; i < st_f.size(); ++i) oracle_state[i] = st_f[i];

    ::ps::runtime::DeviceRequestDescriptor req{};
    std::memset(&req, 0, sizeof(req));
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = kPhase1;
    ps::gpu::discard_cleanup_result(hipMemcpy(d_req, &req, sizeof(req), hipMemcpyHostToDevice));

    ::ps::kernel::GdnConv1dArgs a{};
    a.input = reinterpret_cast<const ::ps::bf16_t*>(d_in);
    a.output = reinterpret_cast<float*>(d_out);
    a.conv_weight = reinterpret_cast<const ::ps::bf16_t*>(d_w);
    a.conv_state = reinterpret_cast<::ps::bf16_t*>(d_state);
    a.requests = reinterpret_cast<const ::ps::runtime::DeviceRequestDescriptor*>(d_req);
    a.num_requests = 1u;
    a.actual_rows = kRows;
    a.max_request_rows = kPhase1;
    a.input_row_stride = kConvDim;
    a.output_row_stride = kConvDim;
    a.conv_dim = kConvDim;
    a.conv_history = kHist;
    a.state_index = 0u;
    a.max_sequences = 1u;
    a.conv_slot_stride = kLayerStride;
    a.conv_layer_stride = kLayerStride;
    a.conv_history_stride = kHistStride;

    conv_step(0u, kPhase1, oracle_state);
    std::vector<float> oracle_state_after_p1 = oracle_state;
    for (uint32_t i = 0; i < kHist * kHistStride; ++i) {
        oracle_state[i] = widen_bf16(ps::f32_to_bf16_rne(oracle_state[i]));
    }

    hipError_t he = ::ps::kernel::launch_gdn_conv1d_bf16_f32(a, nullptr);
    if (he != hipSuccess) {
        std::printf("FAIL: conv launch1: %s\n", hipGetErrorString(he));
        return false;
    }
    ps::gpu::discard_cleanup_result(hipGetLastError());

    req.row_begin = kPhase1;
    req.row_count = kRows - kPhase1;
    ps::gpu::discard_cleanup_result(hipMemcpy(d_req, &req, sizeof(req), hipMemcpyHostToDevice));
    a.max_request_rows = kRows - kPhase1;
    conv_step(kPhase1, kRows - kPhase1, oracle_state);
    for (uint32_t i = 0; i < kHist * kHistStride; ++i) {
        oracle_state[i] = widen_bf16(ps::f32_to_bf16_rne(oracle_state[i]));
    }
    he = ::ps::kernel::launch_gdn_conv1d_bf16_f32(a, nullptr);
    if (he != hipSuccess) {
        std::printf("FAIL: conv launch2: %s\n", hipGetErrorString(he));
        return false;
    }
    ps::gpu::discard_cleanup_result(hipGetLastError());
    he = hipDeviceSynchronize();
    if (he != hipSuccess) {
        std::printf("FAIL: conv sync: %s\n", hipGetErrorString(he));
        return false;
    }

    std::vector<float> y_c((size_t)kRows * kConvDim);
    std::vector<std::uint16_t> st_c((size_t)kHist * kHistStride);
    ps::gpu::discard_cleanup_result(
        hipMemcpy(y_c.data(), d_out, out_bytes, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(
        hipMemcpy(st_c.data(), d_state, st_bytes, hipMemcpyDeviceToHost));

    const Metrics m = diff_metrics(y_c.data(), y_g.data(), y_c.size());
    report("conv", "qwen4exp 10240x4 prefill64+cont8", m);
    check_close(m, 1.0e-6, 1.0e-6, "conv output vs f32 oracle");

    std::vector<float> st_c_w((size_t)kHist * kHistStride);
    std::size_t st_bits = 0;
    for (size_t i = 0; i < st_c.size(); ++i) {
        st_c_w[i] = widen_bf16(st_c[i]);
        const std::uint16_t g = ps::f32_to_bf16_rne(oracle_state[i]);
        if (st_c[i] == g) ++st_bits;
    }
    const Metrics ms = diff_metrics(st_c_w.data(), oracle_state.data(), st_c_w.size());
    report("conv", "conv state after 72 steps", ms);
    std::printf("INFO   conv state bf16 bitwise %zu/%zu\n", st_bits, st_c.size());
    double st_max = 0.0;
    for (float v : oracle_state) st_max = std::max(st_max, (double)std::fabs(v));
    check(ms.max_abs <= st_max * (4.0 / 128.0) + 1e-7,
          "conv state widened within bf16 step bound");

    ps::gpu::discard_cleanup_result(hipFree(d_in));
    ps::gpu::discard_cleanup_result(hipFree(d_w));
    ps::gpu::discard_cleanup_result(hipFree(d_state));
    ps::gpu::discard_cleanup_result(hipFree(d_out));
    ps::gpu::discard_cleanup_result(hipFree(d_req));
    return true;
}

bool run_qk_fit() {
    constexpr uint32_t kRows = 16;
    const float kScale = 1.0f / std::sqrt((float)kHeadDim);

    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_nq = nullptr;
    void* d_nk = nullptr;
    void* d_sq = nullptr;
    const size_t row_b = (size_t)kRows * kQk;
    bool ok = hipMalloc(&d_q, row_b * 4u) == hipSuccess &&
              hipMalloc(&d_k, row_b * 4u) == hipSuccess &&
              hipMalloc(&d_nq, row_b * 4u) == hipSuccess &&
              hipMalloc(&d_nk, row_b * 4u) == hipSuccess &&
              hipMalloc(&d_sq, row_b * 4u) == hipSuccess;
    if (!ok) {
        std::printf("FAIL: qk alloc\n");
        return false;
    }

    std::vector<std::uint16_t> h_q(row_b), h_k(row_b);
    uint32_t s = 5150u;
    for (auto& v : h_q) v = ps::f32_to_bf16_rne(rndf(s, 1.0f));
    for (auto& v : h_k) v = ps::f32_to_bf16_rne(rndf(s, 1.0f));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_q, h_q.data(), row_b * 2u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_k, h_k.data(), row_b * 2u, hipMemcpyHostToDevice));

    ::ps::kernel::FusedGdnQkPrepareArgs a{};
    a.q = reinterpret_cast<const ::ps::bf16_t*>(d_q);
    a.q_row_stride = kQk;
    a.k = reinterpret_cast<const ::ps::bf16_t*>(d_k);
    a.k_row_stride = kQk;
    a.nq_out = reinterpret_cast<float*>(d_nq);
    a.nq_row_stride = kQk;
    a.nk_out = reinterpret_cast<float*>(d_nk);
    a.nk_row_stride = kQk;
    a.sq_out = reinterpret_cast<float*>(d_sq);
    a.sq_row_stride = kQk;
    a.eps = kEps;
    a.scale = kScale;
    a.rows = kRows;
    a.features = kQk;
    a.group_size = kHeadDim;

    hipError_t he = ::ps::kernel::launch_fused_gdn_qk_prepare(a, nullptr);
    if (he != hipSuccess) {
        std::printf("FAIL: qk launch: %s\n", hipGetErrorString(he));
        return false;
    }
    ps::gpu::discard_cleanup_result(hipGetLastError());
    he = hipDeviceSynchronize();
    if (he != hipSuccess) {
        std::printf("FAIL: qk sync: %s\n", hipGetErrorString(he));
        return false;
    }

    std::vector<float> nq_c(row_b), nk_c(row_b), sq_c(row_b);
    ps::gpu::discard_cleanup_result(hipMemcpy(nq_c.data(), d_nq, row_b * 4u, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(hipMemcpy(nk_c.data(), d_nk, row_b * 4u, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(hipMemcpy(sq_c.data(), d_sq, row_b * 4u, hipMemcpyDeviceToHost));

    std::vector<float> nq_g(row_b), nk_g(row_b), sq_g(row_b);
    for (uint32_t r = 0; r < kRows; ++r) {
        for (uint32_t g = 0; g < kQk / kHeadDim; ++g) {
            const size_t off = (size_t)r * kQk + (size_t)g * kHeadDim;
            double sqn = 0.0;
            double sqk = 0.0;
            for (uint32_t i = 0; i < kHeadDim; ++i) {
                const float qv = widen_bf16(h_q[off + i]);
                const float kv = widen_bf16(h_k[off + i]);
                sqn += (double)qv * (double)qv;
                sqk += (double)kv * (double)kv;
            }
            const float in = (float)(1.0 / std::sqrt(sqn + (double)kEps));
            const float ik = (float)(1.0 / std::sqrt(sqk + (double)kEps));
            for (uint32_t i = 0; i < kHeadDim; ++i) {
                const float qv = widen_bf16(h_q[off + i]);
                const float kv = widen_bf16(h_k[off + i]);
                nq_g[off + i] = qv * in;
                nk_g[off + i] = kv * ik;
                sq_g[off + i] = nq_g[off + i] * kScale;
            }
        }
    }

    const Metrics m_nq = diff_metrics(nq_c.data(), nq_g.data(), row_b);
    const Metrics m_nk = diff_metrics(nk_c.data(), nk_g.data(), row_b);
    const Metrics m_sq = diff_metrics(sq_c.data(), sq_g.data(), row_b);
    report("qk", "qwen4exp nq 2048/128", m_nq);
    report("qk", "qwen4exp nk 2048/128", m_nk);
    report("qk", "qwen4exp sq q*scale", m_sq);
    check_close(m_nq, 1.0e-5, 1.0e-4, "qk nq vs oracle");
    check_close(m_nk, 1.0e-5, 1.0e-4, "qk nk vs oracle");
    check_close(m_sq, 1.0e-5, 1.0e-4, "qk sq vs oracle");

    ps::gpu::discard_cleanup_result(hipFree(d_q));
    ps::gpu::discard_cleanup_result(hipFree(d_k));
    ps::gpu::discard_cleanup_result(hipFree(d_nq));
    ps::gpu::discard_cleanup_result(hipFree(d_nk));
    ps::gpu::discard_cleanup_result(hipFree(d_sq));
    return true;
}

struct RecBuf {
    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    void* dt = nullptr;
    void* al = nullptr;
    void* state = nullptr;
    void* state2 = nullptr;
    void* out = nullptr;
    void* out2 = nullptr;
    void* req = nullptr;

    void free_all() {
        void* bufs[] = {q, k, v, a, b, dt, al, state, state2, out, out2, req};
        for (void* p : bufs)
            if (p) ps::gpu::discard_cleanup_result(hipFree(p));
    }
};

bool alloc_rec(RecBuf& d, uint32_t steps) {
    const size_t qk_b = (size_t)steps * kQk * 4u;
    const size_t v_b = (size_t)steps * kVal * 2u;
    const size_t ab_b = (size_t)steps * kValHeads * 2u;
    const size_t st_b = (size_t)kValHeads * kHeadDim * kHeadDim * sizeof(ps::fp16_t);
    const size_t out_b = (size_t)steps * kVal * 4u;
    return hipMalloc(&d.q, qk_b) == hipSuccess &&
           hipMalloc(&d.k, qk_b) == hipSuccess &&
           hipMalloc(&d.v, v_b) == hipSuccess &&
           hipMalloc(&d.a, ab_b) == hipSuccess &&
           hipMalloc(&d.b, ab_b) == hipSuccess &&
           hipMalloc(&d.dt, kValHeads * 2u) == hipSuccess &&
           hipMalloc(&d.al, kValHeads * 2u) == hipSuccess &&
           hipMalloc(&d.state, st_b) == hipSuccess &&
           hipMalloc(&d.state2, st_b) == hipSuccess &&
           hipMalloc(&d.out, out_b) == hipSuccess &&
           hipMalloc(&d.out2, out_b) == hipSuccess &&
           hipMalloc(&d.req, sizeof(::ps::runtime::DeviceRequestDescriptor)) == hipSuccess;
}

::ps::kernel::GdnRecurrenceArgs make_rec_args(RecBuf& d, uint32_t rows) {
    ::ps::kernel::GdnRecurrenceArgs r{};
    r.q = reinterpret_cast<const float*>(d.q);
    r.k = reinterpret_cast<const float*>(d.k);
    r.v = reinterpret_cast<const ::ps::bf16_t*>(d.v);
    r.a = reinterpret_cast<const ::ps::bf16_t*>(d.a);
    r.b = reinterpret_cast<const ::ps::bf16_t*>(d.b);
    r.dt_bias = reinterpret_cast<const ::ps::bf16_t*>(d.dt);
    r.a_log = reinterpret_cast<const ::ps::bf16_t*>(d.al);
    r.recurrent_state = reinterpret_cast<::ps::fp16_t*>(d.state);
    r.output = reinterpret_cast<float*>(d.out);
    r.requests = reinterpret_cast<const ::ps::runtime::DeviceRequestDescriptor*>(d.req);
    r.num_requests = 1u;
    r.actual_rows = rows;
    r.q_row_stride = kQk;
    r.k_row_stride = kQk;
    r.v_row_stride = kVal;
    r.a_row_stride = kValHeads;
    r.b_row_stride = kValHeads;
    r.output_row_stride = kVal;
    r.state_index = 0u;
    r.max_sequences = 1u;
    r.num_gdn_states = 1u;
    r.key_heads = kKeyHeads;
    r.num_v_heads = kValHeads;
    r.head_k = kHeadDim;
    r.head_v = kHeadDim;
    r.recurrent_slot_stride = (uint64_t)kValHeads * kHeadDim * kHeadDim;
    r.recurrent_layer_stride = (uint64_t)kValHeads * kHeadDim * kHeadDim;
    return r;
}

void rec_norm_inputs(std::vector<float>& q, std::vector<float>& k) {
    for (uint32_t h = 0; h < kKeyHeads; ++h) {
        const size_t off = (size_t)h * kHeadDim;
        double sqn = 0.0;
        double sqk = 0.0;
        for (uint32_t i = 0; i < kHeadDim; ++i) {
            sqn += (double)q[off + i] * (double)q[off + i];
            sqk += (double)k[off + i] * (double)k[off + i];
        }
        const float in = (float)(1.0 / std::sqrt(sqn));
        const float ik = (float)(1.0 / std::sqrt(sqk));
        for (uint32_t i = 0; i < kHeadDim; ++i) {
            q[off + i] = q[off + i] * in * kQSpecScale;
            k[off + i] = k[off + i] * ik;
        }
    }
}

struct RecOracle {
    std::vector<ps::fp16_t> state;
    std::vector<float> work;

    explicit RecOracle(uint32_t seed) {
        state.resize((size_t)kValHeads * kHeadDim * kHeadDim);
        uint32_t s = seed;
        for (auto& v : state) v = ps::f32_to_fp16(rndf(s, 0.05f));
        work.resize(state.size());
    }

    void step(const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& a,
              const std::vector<float>& b, const std::vector<float>& dt,
              const std::vector<float>& al, std::vector<float>& out) {
        for (size_t i = 0; i < state.size(); ++i) work[i] = ps::fp16_to_f32(state[i]);
        for (uint32_t h = 0; h < kValHeads; ++h) {
            const uint32_t qh = h / (kValHeads / kKeyHeads);
            const float av = a[h];
            const float bv = b[h];
            const float g =
                -std::exp(al[h]) * std::log1p(std::exp(av + dt[h]));
            const float beta = 1.0f / (1.0f + std::exp(-bv));
            const float decay = std::exp(g);
            float* S = work.data() + (size_t)h * kHeadDim * kHeadDim;
            for (uint32_t i = 0; i < kHeadDim * kHeadDim; ++i) S[i] *= decay;
            float kv[128];
            float delta[128];
            for (uint32_t j = 0; j < kHeadDim; ++j) {
                float acc = 0.0f;
                for (uint32_t i = 0; i < kHeadDim; ++i)
                    acc += S[(size_t)i * kHeadDim + j] * k[(size_t)qh * kHeadDim + i];
                kv[j] = acc;
            }
            for (uint32_t j = 0; j < kHeadDim; ++j)
                delta[j] = (v[(size_t)h * kHeadDim + j] - kv[j]) * beta;
            for (uint32_t i = 0; i < kHeadDim; ++i) {
                const float ki = k[(size_t)qh * kHeadDim + i];
                for (uint32_t j = 0; j < kHeadDim; ++j)
                    S[(size_t)i * kHeadDim + j] += ki * delta[j];
            }
            for (uint32_t j = 0; j < kHeadDim; ++j) {
                float acc = 0.0f;
                for (uint32_t i = 0; i < kHeadDim; ++i)
                    acc += S[(size_t)i * kHeadDim + j] * q[(size_t)qh * kHeadDim + i];
                out[(size_t)h * kHeadDim + j] = acc;
            }
            for (uint32_t i = 0; i < kHeadDim * kHeadDim; ++i)
                state[(size_t)h * kHeadDim * kHeadDim + i] = ps::f32_to_fp16(S[i]);
        }
    }
};

bool upload_rec_static(RecBuf& d) {
    std::vector<std::uint16_t> h_a(kValHeads), h_b(kValHeads), h_dt(kValHeads),
        h_al(kValHeads);
    uint32_t s = 991u;
    for (auto& v : h_a) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    for (auto& v : h_b) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    for (auto& v : h_dt) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    for (auto& v : h_al) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    ::ps::runtime::DeviceRequestDescriptor req{};
    std::memset(&req, 0, sizeof(req));
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = 1u;
    return hipMemcpy(d.a, h_a.data(), kValHeads * 2u, hipMemcpyHostToDevice) == hipSuccess &&
           hipMemcpy(d.b, h_b.data(), kValHeads * 2u, hipMemcpyHostToDevice) == hipSuccess &&
           hipMemcpy(d.dt, h_dt.data(), kValHeads * 2u, hipMemcpyHostToDevice) == hipSuccess &&
           hipMemcpy(d.al, h_al.data(), kValHeads * 2u, hipMemcpyHostToDevice) == hipSuccess &&
           hipMemcpy(d.req, &req, sizeof(req), hipMemcpyHostToDevice) == hipSuccess;
}

bool capture_static(std::vector<float>& a, std::vector<float>& b,
                    std::vector<float>& dt, std::vector<float>& al) {
    std::vector<std::uint16_t> h_a(kValHeads), h_b(kValHeads), h_dt(kValHeads),
        h_al(kValHeads);
    uint32_t s = 991u;
    for (auto& v : h_a) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    for (auto& v : h_b) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    for (auto& v : h_dt) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    for (auto& v : h_al) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    a.resize(kValHeads);
    b.resize(kValHeads);
    dt.resize(kValHeads);
    al.resize(kValHeads);
    for (uint32_t i = 0; i < kValHeads; ++i) {
        a[i] = widen_bf16(h_a[i]);
        b[i] = widen_bf16(h_b[i]);
        dt[i] = widen_bf16(h_dt[i]);
        al[i] = widen_bf16(h_al[i]);
    }
    return true;
}

bool run_rec_decode_fit() {
    constexpr uint32_t kSteps = 64;
    RecBuf d;
    if (!alloc_rec(d, kSteps)) {
        std::printf("FAIL: rec alloc\n");
        return false;
    }
    if (!upload_rec_static(d)) {
        d.free_all();
        return false;
    }
    std::vector<float> h_a, h_b, h_dt, h_al;
    capture_static(h_a, h_b, h_dt, h_al);

    std::vector<std::uint16_t> init_state((size_t)kValHeads * kHeadDim * kHeadDim);
    {
        uint32_t s = 4242u;
        for (auto& v : init_state) v = ps::f32_to_bf16_rne(rndf(s, 0.05f));
        std::vector<ps::fp16_t> init16(init_state.size());
        for (size_t i = 0; i < init_state.size(); ++i)
            init16[i] = ps::f32_to_fp16(widen_bf16(init_state[i]));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state, init16.data(), init16.size() * sizeof(ps::fp16_t),
                      hipMemcpyHostToDevice));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state2, init16.data(), init16.size() * sizeof(ps::fp16_t),
                      hipMemcpyHostToDevice));
    }

    RecOracle oracle(4242u);

    std::vector<float> step_q(kQk), step_k(kQk);
    std::vector<float> step_v(kVal);
    std::vector<std::uint16_t> hv(kVal), ha(kValHeads), hb(kValHeads);
    std::vector<float> out_c(kVal), out_g(kVal);

    Metrics m_first, m_mid, m_last;
    uint32_t s_in = 31337u;
    for (uint32_t step = 0; step < kSteps; ++step) {
        for (auto& v : step_q) v = rndf(s_in, 0.5f);
        for (auto& v : step_k) v = rndf(s_in, 0.5f);
        for (auto& v : hv) v = ps::f32_to_bf16_rne(rndf(s_in, 0.5f));
        for (auto& v : ha) v = ps::f32_to_bf16_rne(rndf(s_in, 0.5f));
        for (auto& v : hb) v = ps::f32_to_bf16_rne(rndf(s_in, 0.5f));
        rec_norm_inputs(step_q, step_k);

        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.q, step_q.data(), kQk * 4u, hipMemcpyHostToDevice));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.k, step_k.data(), kQk * 4u, hipMemcpyHostToDevice));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.v, hv.data(), kVal * 2u, hipMemcpyHostToDevice));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.a, ha.data(), kValHeads * 2u, hipMemcpyHostToDevice));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.b, hb.data(), kValHeads * 2u, hipMemcpyHostToDevice));

        auto args = make_rec_args(d, 1u);
        const hipError_t he = ::ps::kernel::launch_gdn_recurrence_f32_wmma(args, nullptr);
        if (he != hipSuccess) {
            std::printf("FAIL: rec launch step=%u: %s\n", step, hipGetErrorString(he));
            d.free_all();
            return false;
        }
        ps::gpu::discard_cleanup_result(hipGetLastError());

        std::vector<float> v_f(kVal);
        for (uint32_t i = 0; i < kVal; ++i) v_f[i] = widen_bf16(hv[i]);
        out_g.assign(kVal, 0.0f);
        std::vector<float> a_f(kValHeads), b_f(kValHeads);
        for (uint32_t i = 0; i < kValHeads; ++i) {
            a_f[i] = widen_bf16(ha[i]);
            b_f[i] = widen_bf16(hb[i]);
        }
        oracle.step(step_q, step_k, v_f, a_f, b_f, h_dt, h_al, out_g);

        const bool need = (step == 0) || (step == kSteps / 2 - 1) || (step + 1 == kSteps);
        if (need) {
            ps::gpu::discard_cleanup_result(
                hipMemcpy(out_c.data(), d.out, kVal * 4u, hipMemcpyDeviceToHost));
            const Metrics m = diff_metrics(out_c.data(), out_g.data(), kVal);
            if (step == 0) m_first = m;
            if (step == kSteps / 2 - 1) m_mid = m;
            if (step + 1 == kSteps) m_last = m;
        }
        if (step == 0) {
            const bool sup = ::ps::kernel::gdn_recurrence_decode1_supported(
                make_rec_args(d, 1u));
            check(sup, "decode1 supported for qwen4exp geometry 16/48/128");
            if (sup) {
                ps::gpu::discard_cleanup_result(hipGetLastError());
                ::ps::kernel::GdnRecurrenceArgs ad = make_rec_args(d, 1u);
                ad.recurrent_state = reinterpret_cast<::ps::fp16_t*>(d.state2);
                ad.output = reinterpret_cast<float*>(d.out2);
                const hipError_t hd =
                    ::ps::kernel::launch_gdn_recurrence_f32_wmma_decode1(ad, nullptr);
                check(hd == hipSuccess,
                      std::string("decode1 launch: ") +
                          (hd == hipSuccess ? "ok" : hipGetErrorString(hd)));
                if (hd == hipSuccess) {
                    ps::gpu::discard_cleanup_result(hipDeviceSynchronize());
                    std::vector<float> o2(kVal);
                    ps::gpu::discard_cleanup_result(
                        hipMemcpy(o2.data(), d.out2, kVal * 4u, hipMemcpyDeviceToHost));
                    const Metrics md = diff_metrics(o2.data(), out_c.data(), kVal);
                    report("rec", "decode1 vs wmma step1 same state", md);
                    check_close(md, 1.0e-6, 1.0e-6,
                                "decode1 matches wmma path on same state");
                }
            }
        }
    }
    ps::gpu::discard_cleanup_result(hipDeviceSynchronize());

    ps::gpu::discard_cleanup_result(
        hipMemcpy(out_c.data(), d.out, kVal * 4u, hipMemcpyDeviceToHost));
    m_last = diff_metrics(out_c.data(), out_g.data(), kVal);

    report("rec", "16/48/128 decode step1", m_first);
    report("rec", "16/48/128 decode step32", m_mid);
    report("rec", "16/48/128 decode step64", m_last);
    std::printf("INFO   rec error growth: step1 max_abs=%.6e -> step64 max_abs=%.6e\n",
                m_first.max_abs, m_last.max_abs);
    check_close(m_first, 1.0e-4, 1.0e-3, "recurrence decode step1 vs oracle");
    check_close(m_last, 1.0e-3, 1.0e-3, "recurrence decode step64 vs oracle (fp16 state lossy)");

    std::vector<ps::fp16_t> st_c(kValHeads * kHeadDim * kHeadDim);
    ps::gpu::discard_cleanup_result(
        hipMemcpy(st_c.data(), d.state, st_c.size() * sizeof(ps::fp16_t),
                  hipMemcpyDeviceToHost));
    std::size_t st_bits = 0;
    double st_max = 0.0, st_abs = 0.0;
    for (size_t i = 0; i < st_c.size(); ++i) {
        if (st_c[i].value == oracle.state[i].value) ++st_bits;
        const double dv = std::fabs((double)ps::fp16_to_f32(st_c[i]) -
                                    (double)ps::fp16_to_f32(oracle.state[i]));
        if (dv > st_abs) st_abs = dv;
        st_max = std::max(st_max, (double)std::fabs((double)ps::fp16_to_f32(oracle.state[i])));
    }
    std::printf("INFO   rec state fp16 bitwise %zu/%zu state_diff=%.6e bound=%.6e\n",
                st_bits, st_c.size(), st_abs, st_max * (4.0 / 1024.0));
    check(st_abs <= st_max * (4.0 / 1024.0) + 1e-6,
          "recurrent state within fp16 step bound (lossy contract preserved)");

    d.free_all();
    return true;
}

bool run_rec_prefill_fit() {
    constexpr uint32_t kRows = 64;
    RecBuf d;
    if (!alloc_rec(d, kRows)) {
        std::printf("FAIL: prefill alloc\n");
        return false;
    }
    if (!upload_rec_static(d)) {
        d.free_all();
        return false;
    }
    std::vector<float> h_a, h_b, h_dt, h_al;
    capture_static(h_a, h_b, h_dt, h_al);

    std::vector<ps::fp16_t> init16((size_t)kValHeads * kHeadDim * kHeadDim);
    {
        uint32_t s = 777u;
        for (auto& v : init16) v = ps::f32_to_fp16(rndf(s, 0.05f));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state, init16.data(), init16.size() * sizeof(ps::fp16_t),
                      hipMemcpyHostToDevice));
        ps::gpu::discard_cleanup_result(
            hipMemcpy(d.state2, init16.data(), init16.size() * sizeof(ps::fp16_t),
                      hipMemcpyHostToDevice));
    }

    std::vector<float> hq((size_t)kRows * kQk), hk((size_t)kRows * kQk);
    std::vector<std::uint16_t> hv((size_t)kRows * kVal);
    uint32_t s = 8080u;
    for (auto& v : hq) v = rndf(s, 0.5f);
    for (auto& v : hk) v = rndf(s, 0.5f);
    for (auto& v : hv) v = ps::f32_to_bf16_rne(rndf(s, 0.5f));
    for (uint32_t r = 0; r < kRows; ++r) {
        std::vector<float> qr(hq.begin() + (size_t)r * kQk,
                              hq.begin() + (size_t)(r + 1) * kQk);
        std::vector<float> kr(hk.begin() + (size_t)r * kQk,
                              hk.begin() + (size_t)(r + 1) * kQk);
        rec_norm_inputs(qr, kr);
        std::copy(qr.begin(), qr.end(), hq.begin() + (size_t)r * kQk);
        std::copy(kr.begin(), kr.end(), hk.begin() + (size_t)r * kQk);
    }

    ps::gpu::discard_cleanup_result(hipMemcpy(d.q, hq.data(), hq.size() * 4u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d.k, hk.data(), hk.size() * 4u, hipMemcpyHostToDevice));
    ps::gpu::discard_cleanup_result(hipMemcpy(d.v, hv.data(), hv.size() * 2u, hipMemcpyHostToDevice));

    ::ps::runtime::DeviceRequestDescriptor req{};
    std::memset(&req, 0, sizeof(req));
    req.request_handle.slot = 0u;
    req.row_begin = 0u;
    req.row_count = kRows;
    ps::gpu::discard_cleanup_result(hipMemcpy(d.req, &req, sizeof(req), hipMemcpyHostToDevice));

    auto args = make_rec_args(d, kRows);
    const hipError_t he = ::ps::kernel::launch_gdn_recurrence_f32_wmma(args, nullptr);
    if (he != hipSuccess) {
        std::printf("FAIL: prefill launch: %s\n", hipGetErrorString(he));
        d.free_all();
        return false;
    }
    ps::gpu::discard_cleanup_result(hipGetLastError());
    ps::gpu::discard_cleanup_result(hipDeviceSynchronize());

    RecOracle oracle(777u);
    std::vector<float> out_g(kVal), out_c(kVal);
    Metrics m_first, m_last;
    for (uint32_t r = 0; r < kRows; ++r) {
        std::vector<float> qr(hq.begin() + (size_t)r * kQk,
                              hq.begin() + (size_t)(r + 1) * kQk);
        std::vector<float> kr(hk.begin() + (size_t)r * kQk,
                              hk.begin() + (size_t)(r + 1) * kQk);
        std::vector<float> vf(kVal);
        for (uint32_t i = 0; i < kVal; ++i)
            vf[i] = widen_bf16(hv[(size_t)r * kVal + i]);
        out_g.assign(kVal, 0.0f);
        oracle.step(qr, kr, vf, h_a, h_b, h_dt, h_al, out_g);
        if (r == 0 || r + 1 == kRows) {
            ps::gpu::discard_cleanup_result(
                hipMemcpy(out_c.data(),
                          reinterpret_cast<const char*>(d.out) + (size_t)r * kVal * 4u,
                          kVal * 4u, hipMemcpyDeviceToHost));
            const Metrics m = diff_metrics(out_c.data(), out_g.data(), kVal);
            if (r == 0) m_first = m;
            if (r + 1 == kRows) m_last = m;
        }
    }
    report("rec", "prefill chunk step1 vs oracle", m_first);
    report("rec", "prefill chunk step64 vs oracle", m_last);
    check_close(m_first, 1.0e-4, 1.0e-3, "prefill chunk step1 vs oracle");
    check_close(m_last, 5.0e-3, 1.0e-3, "prefill chunk step64 vs oracle");

    d.free_all();
    return true;
}

bool run_activation_fit() {
    constexpr uint32_t kRows = 8;
    constexpr uint32_t kFeat = kVal;
    void* d_x = nullptr;
    void* d_y1 = nullptr;
    void* d_y2 = nullptr;
    const size_t nb = (size_t)kRows * kFeat;
    if (hipMalloc(&d_x, nb * 2u) != hipSuccess ||
        hipMalloc(&d_y1, nb * 4u) != hipSuccess ||
        hipMalloc(&d_y2, nb * 4u) != hipSuccess) {
        std::printf("FAIL: activation alloc\n");
        return false;
    }
    std::vector<std::uint16_t> hx(nb);
    uint32_t s = 2718u;
    for (auto& v : hx) v = ps::f32_to_bf16_rne(rndf(s, 4.0f));
    ps::gpu::discard_cleanup_result(hipMemcpy(d_x, hx.data(), nb * 2u, hipMemcpyHostToDevice));

    hipError_t he = ::ps::kernel::launch_sigmoid_bf16_f32(
        reinterpret_cast<const ::ps::bf16_t*>(d_x), kFeat,
        reinterpret_cast<float*>(d_y1), kFeat, kRows, kFeat, nullptr);
    if (he != hipSuccess) {
        std::printf("FAIL: sigmoid launch: %s\n", hipGetErrorString(he));
        return false;
    }
    he = ::ps::kernel::launch_silu_bf16_f32(
        reinterpret_cast<const ::ps::bf16_t*>(d_x), kFeat,
        reinterpret_cast<float*>(d_y2), kFeat, kRows, kFeat, nullptr);
    if (he != hipSuccess) {
        std::printf("FAIL: silu launch: %s\n", hipGetErrorString(he));
        return false;
    }
    ps::gpu::discard_cleanup_result(hipGetLastError());
    ps::gpu::discard_cleanup_result(hipDeviceSynchronize());

    std::vector<float> y1(nb), y2(nb), g1(nb), g2(nb);
    ps::gpu::discard_cleanup_result(hipMemcpy(y1.data(), d_y1, nb * 4u, hipMemcpyDeviceToHost));
    ps::gpu::discard_cleanup_result(hipMemcpy(y2.data(), d_y2, nb * 4u, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < nb; ++i) {
        const float x = widen_bf16(hx[i]);
        const float sig = 1.0f / (1.0f + std::exp(-(double)x));
        g1[i] = 1.0f / (1.0f + std::exp(-(double)x));
        g2[i] = x * sig;
    }
    const Metrics m_s = diff_metrics(y1.data(), g1.data(), nb);
    const Metrics m_l = diff_metrics(y2.data(), g2.data(), nb);
    report("gate", "sigmoid bf16->f32 features=6144", m_s);
    report("gate", "silu bf16->f32 features=6144", m_l);
    check_close(m_s, 1.0e-6, 1.0e-6, "sigmoid kernel vs oracle (output_gate sigmoid path)");
    check_close(m_l, 1.0e-6, 1.0e-6, "silu kernel vs oracle (qwen35 reference path)");

    ps::gpu::discard_cleanup_result(hipFree(d_x));
    ps::gpu::discard_cleanup_result(hipFree(d_y1));
    ps::gpu::discard_cleanup_result(hipFree(d_y2));
    return true;
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("SKIP: no GPU available\n");
        return 77;
    }
    bool ok = true;
    ok &= run_conv_fit();
    ok &= run_qk_fit();
    ok &= run_rec_decode_fit();
    ok &= run_rec_prefill_fit();
    ok &= run_activation_fit();
    if (!ok) ++g_fail;
    std::printf("\nQWEN4EXP_GDN_FIT: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
