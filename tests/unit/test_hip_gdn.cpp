// shelf unit test — vendored PhaseShift GDN decode kernels on gfx1201.
//
// Parity vs a fresh independent f64 reference implementing the shelf model
// contract for one decode step:
//   causal conv step (kernel width 4, state [C,3] chronological, silu on conv
//   output) -> split q/k/v -> fused qk_prepare (l2norm eps=1e-6 inside rsqrt,
//   q scaled by dk^-0.5 AFTER norm) -> recurrent gated delta-rule update with
//   fp32 state [nvh,dk,dv] (S <- S*exp(g) + k (x) beta (v - exp(g) S^T k),
//   o = S^T q; beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias)).
// Two sequential 1-row launches per case; state persists in the device buffer
// across launches. Outputs AND final state are compared.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <shelf/backend/hip/kernels/gdn_conv1d.h>
#include <shelf/backend/hip/kernels/gdn_conv_silu.h>
#include <shelf/backend/hip/kernels/gdn_qk_prepare.h>
#include <shelf/backend/hip/kernels/gdn_recurrence.h>
#include <shelf/foundation/numeric.hpp>

namespace {

using shelf::hipk::FusedGdnConvSiluArgs;
using shelf::hipk::FusedGdnQkPrepareArgs;
using shelf::hipk::GdnConv1dArgs;
using shelf::hipk::GdnRecurrenceArgs;
using shelf::hipk::GdnRequestDescriptor;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

#define HIP_OK(call)                                                            \
    do {                                                                        \
        const hipError_t e_ = (call);                                           \
        if (e_ != hipSuccess) {                                                 \
            std::printf("FAIL hip: %s @%d: %s\n", #call, __LINE__,              \
                        hipGetErrorString(e_));                                 \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

// rel-ish comparison: |got-ref| <= tol * max(1, |ref|)
void compare(const std::vector<double>& got, const std::vector<double>& ref,
             double tol, const char* what) {
    double worst = 0.0;
    size_t worst_i = 0;
    bool bad = false;
    for (size_t i = 0; i < got.size(); ++i) {
        const double err = std::fabs(got[i] - ref[i]);
        const double denom = std::max(1.0, std::fabs(ref[i]));
        if (err / denom > worst) { worst = err / denom; worst_i = i; }
        if (err > tol * denom) bad = true;
    }
    if (bad) {
        ++failures;
        std::printf("FAIL: %s: max rel-ish diff %.3e at idx %zu (got %.6f, ref %.6f) tol %.1e n=%zu\n",
                    what, worst, worst_i, got[worst_i], ref[worst_i], tol, got.size());
    } else {
        std::printf("ok: %s (max rel-ish diff %.3e, n=%zu)\n", what, worst, got.size());
    }
}

double softplus_ref(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }
double sigmoid_ref(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double silu_ref(double x) { return x / (1.0 + std::exp(-x)); }

struct CaseDims {
    uint32_t nkh, nvh, dk, dv;
};

void run_case(CaseDims d) {
    const uint32_t nkh = d.nkh, nvh = d.nvh, dk = d.dk, dv = d.dv;
    const uint32_t rep = nvh / nkh;
    const uint32_t C = 2u * nkh * dk + nvh * dv;
    std::printf("== case nkh=%u nvh=%u dk=%u dv=%u (C=%u) ==\n", nkh, nvh, dk, dv, C);

    std::mt19937 rng(0xC0FFEEu + nkh * 131071u + nvh * 8191u + dk * 131u + dv);
    std::uniform_real_distribution<float> u01(-1.0f, 1.0f);
    auto bf = [&](float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return shelf::f32_to_bf16(dist(rng));
    };

    // ------------------------------------------------------------------
    // inputs (bf16 projections as shelf decode would deliver them)
    // ------------------------------------------------------------------
    // q/k raw projections for the two decode steps: per-token row layout
    // [2 tokens][nkh heads][dk] (recurrence indexes q/k by global token row,
    // exactly like the real pipeline where qk_prepare feeds one row/step).
    const size_t qk_row_elems = static_cast<size_t>(nkh) * dk;   // one token row
    std::vector<shelf::bf16_t> qraw(2u * qk_row_elems);
    std::vector<shelf::bf16_t> kraw(2u * qk_row_elems);
    for (auto& x : qraw) x = bf(-1.0f, 1.0f);
    for (auto& x : kraw) x = bf(-1.0f, 1.0f);

    // two decode steps of v/a/b, dt_bias/a_log per value head
    std::vector<shelf::bf16_t> a(2u * nvh), b(2u * nvh);
    std::vector<shelf::bf16_t> v(2u * nvh * dv);
    std::vector<shelf::bf16_t> a_log(nvh), dt_bias(nvh);
    for (auto& x : a) x = bf(-1.5f, 1.5f);
    for (auto& x : b) x = bf(-2.0f, 2.0f);
    for (auto& x : v) x = bf(-1.0f, 1.0f);
    for (auto& x : a_log) x = bf(-1.0f, 0.5f);
    for (auto& x : dt_bias) x = bf(-0.5f, 0.5f);

    // ------------------------------------------------------------------
    // 1) fused qk prepare (l2norm eps inside rsqrt + q scale after norm)
    // ------------------------------------------------------------------
    const float eps = 1e-6f;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dk));
    std::vector<double> nq_ref(qraw.size()), sq_ref(qraw.size()), nk_ref(kraw.size());
    // L2 norm is per dk-group; rows here enumerate 2 tokens x nkh heads.
    for (uint32_t r = 0; r < 2u * nkh; ++r) {
        double qsum = 0.0, ksum = 0.0;
        for (uint32_t i = 0; i < dk; ++i) {
            const double q = shelf::bf16_to_f32(qraw[static_cast<size_t>(r) * dk + i]);
            const double k = shelf::bf16_to_f32(kraw[static_cast<size_t>(r) * dk + i]);
            qsum += q * q;
            ksum += k * k;
        }
        const double qinv = 1.0 / std::sqrt(qsum + eps);
        const double kinv = 1.0 / std::sqrt(ksum + eps);
        for (uint32_t i = 0; i < dk; ++i) {
            const double q = shelf::bf16_to_f32(qraw[static_cast<size_t>(r) * dk + i]);
            const double k = shelf::bf16_to_f32(kraw[static_cast<size_t>(r) * dk + i]);
            nq_ref[static_cast<size_t>(r) * dk + i] = q * qinv;
            sq_ref[static_cast<size_t>(r) * dk + i] = q * qinv * scale;
            nk_ref[static_cast<size_t>(r) * dk + i] = k * kinv;
        }
    }

    void *dq, *dk_, *dnq, *dsq, *dnk;
    const size_t qk_elems = 2u * qk_row_elems;   // two token rows
    HIP_OK(hipMalloc(&dq, qk_elems * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&dk_, qk_elems * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&dnq, qk_elems * sizeof(float)));
    HIP_OK(hipMalloc(&dsq, qk_elems * sizeof(float)));
    HIP_OK(hipMalloc(&dnk, qk_elems * sizeof(float)));
    HIP_OK(hipMemcpy(dq, qraw.data(), qk_elems * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dk_, kraw.data(), qk_elems * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));

    // NOTE qk_prepare semantics: `rows` enumerates contiguous dk-sized head
    // rows (row stride == dk), grid.y = rows, grid.x = 2 (q block + k block).
    FusedGdnQkPrepareArgs qa{};
    qa.q = static_cast<const shelf::bf16_t*>(dq);
    qa.q_row_stride = dk;
    qa.k = static_cast<const shelf::bf16_t*>(dk_);
    qa.k_row_stride = dk;
    qa.nq_out = static_cast<float*>(dnq);
    qa.nq_row_stride = dk;
    qa.nk_out = static_cast<float*>(dnk);
    qa.nk_row_stride = dk;
    qa.sq_out = static_cast<float*>(dsq);
    qa.sq_row_stride = dk;
    qa.eps = eps;
    qa.scale = scale;
    qa.rows = 2u * nkh;    // both token rows (heads x tokens), dk per group
    qa.features = dk;      // group == head dim
    qa.group_size = dk;
    check(shelf::hipk::fused_gdn_qk_prepare_blocks_x(dk, dk) == 2u,
          "qk_prepare blocks_x helper");
    HIP_OK(shelf::hipk::launch_fused_gdn_qk_prepare(qa, 0));

    std::vector<float> nq(qk_elems), sq(qk_elems), nk(qk_elems);
    HIP_OK(hipMemcpy(nq.data(), dnq, qk_elems * sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(sq.data(), dsq, qk_elems * sizeof(float), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(nk.data(), dnk, qk_elems * sizeof(float), hipMemcpyDeviceToHost));
    compare(std::vector<double>(nq.begin(), nq.end()), nq_ref, 5e-3, "qk_prepare nq");
    compare(std::vector<double>(sq.begin(), sq.end()), sq_ref, 5e-3, "qk_prepare sq (scaled q)");
    compare(std::vector<double>(nk.begin(), nk.end()), nk_ref, 5e-3, "qk_prepare nk");

    // ------------------------------------------------------------------
    // 2) recurrence: two sequential decode launches, f32 state persists
    // ------------------------------------------------------------------
    const size_t s_elems = static_cast<size_t>(nvh) * dk * dv;
    std::vector<float> s0_host(s_elems);
    for (auto& x : s0_host) x = u01(rng) * 0.5f;
    float* dS;
    HIP_OK(hipMalloc(&dS, s_elems * sizeof(float)));
    HIP_OK(hipMemcpy(dS, s0_host.data(), s_elems * sizeof(float), hipMemcpyHostToDevice));

    void *da, *db, *d_v, *da_log, *d_dt;
    HIP_OK(hipMalloc(&da, a.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&db, b.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_v, v.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&da_log, a_log.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&d_dt, dt_bias.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMemcpy(da, a.data(), a.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(db, b.data(), b.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_v, v.data(), v.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(da_log, a_log.data(), a_log.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_dt, dt_bias.data(), dt_bias.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));

    GdnRequestDescriptor hreq;
    hreq.slot = 0;
    GdnRequestDescriptor* dreq;
    HIP_OK(hipMalloc(&dreq, sizeof(GdnRequestDescriptor)));

    const size_t out_elems = 2u * nvh * dv;
    float* dout;
    HIP_OK(hipMalloc(&dout, out_elems * sizeof(float)));
    HIP_OK(hipMemset(dout, 0, out_elems * sizeof(float)));

    GdnRecurrenceArgs ra{};
    ra.q = static_cast<const float*>(dsq);   // scaled q from qk_prepare
    ra.q_row_stride = static_cast<uint32_t>(qk_row_elems);
    ra.k = static_cast<const float*>(dnk);   // normalized k from qk_prepare
    ra.k_row_stride = static_cast<uint32_t>(qk_row_elems);
    ra.v = static_cast<const shelf::bf16_t*>(d_v);
    ra.v_row_stride = nvh * dv;
    ra.a = static_cast<const shelf::bf16_t*>(da);
    ra.a_row_stride = nvh;
    ra.b = static_cast<const shelf::bf16_t*>(db);
    ra.b_row_stride = nvh;
    ra.dt_bias = static_cast<const shelf::bf16_t*>(d_dt);
    ra.a_log = static_cast<const shelf::bf16_t*>(da_log);
    ra.recurrent_state = dS;
    ra.recurrent_slot_stride = s_elems;
    ra.recurrent_layer_stride = s_elems;
    ra.output = dout;
    ra.output_row_stride = nvh * dv;
    ra.requests = dreq;
    ra.num_requests = 1;
    ra.actual_rows = 2;   // two-step buffer; each launch consumes 1 row
    ra.max_sequences = 1;
    ra.state_index = 0;
    ra.key_heads = nkh;
    ra.num_v_heads = nvh;
    ra.head_k = dk;
    ra.head_v = dv;

    // f64 reference state chain (shelf formulation: decay, then delta rule)
    std::vector<double> S(s_elems);
    for (size_t i = 0; i < s_elems; ++i) S[i] = s0_host[i];
    std::vector<double> out_ref(out_elems, 0.0);
    for (uint32_t step = 0; step < 2u; ++step) {
        for (uint32_t vh = 0; vh < nvh; ++vh) {
            const uint32_t kh = vh / rep;
            const double bv = shelf::bf16_to_f32(b[static_cast<size_t>(step) * nvh + vh]);
            const double av = shelf::bf16_to_f32(a[static_cast<size_t>(step) * nvh + vh]);
            const double al = shelf::bf16_to_f32(a_log[vh]);
            const double dtb = shelf::bf16_to_f32(dt_bias[vh]);
            const double beta = sigmoid_ref(bv);
            const double g = -std::exp(al) * softplus_ref(av + dtb);
            const double decay = std::exp(g);
            double* Sh = S.data() + static_cast<size_t>(vh) * dk * dv;
            for (size_t i = 0; i < dk * dv; ++i) Sh[i] *= decay;
            std::vector<double> delta(dv);
            for (uint32_t j = 0; j < dv; ++j) {
                double kv_mem = 0.0;
                for (uint32_t i = 0; i < dk; ++i)
                    kv_mem += Sh[static_cast<size_t>(i) * dv + j] *
                              nk_ref[(static_cast<size_t>(step) * nkh + kh) * dk + i];
                delta[j] = (shelf::bf16_to_f32(
                                v[static_cast<size_t>(step) * nvh * dv + static_cast<size_t>(vh) * dv + j]) -
                            kv_mem) * beta;
            }
            for (uint32_t i = 0; i < dk; ++i) {
                const double kk = nk_ref[(static_cast<size_t>(step) * nkh + kh) * dk + i];
                for (uint32_t j = 0; j < dv; ++j)
                    Sh[static_cast<size_t>(i) * dv + j] += kk * delta[j];
            }
            for (uint32_t j = 0; j < dv; ++j) {
                double acc = 0.0;
                for (uint32_t i = 0; i < dk; ++i)
                    acc += Sh[static_cast<size_t>(i) * dv + j] *
                           sq_ref[(static_cast<size_t>(step) * nkh + kh) * dk + i];
                out_ref[static_cast<size_t>(step) * nvh * dv + static_cast<size_t>(vh) * dv + j] = acc;
            }
        }
    }

    std::vector<float> step_out(out_elems, 0.0f);
    for (uint32_t step = 0; step < 2u; ++step) {
        hreq.row_begin = step;
        hreq.row_count = 1;   // decode: one token row per launch
        HIP_OK(hipMemcpy(dreq, &hreq, sizeof(GdnRequestDescriptor), hipMemcpyHostToDevice));
        HIP_OK(shelf::hipk::launch_gdn_recurrence_f32(ra, 0));
        HIP_OK(hipMemcpy(step_out.data(), dout, out_elems * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<double> got_out(out_elems);
        for (size_t i = 0; i < out_elems; ++i) got_out[i] = step_out[i];
        std::vector<double> want(out_elems, 0.0);
        for (size_t i = 0; i < out_elems; ++i) {
            if (i >= static_cast<size_t>(step) * nvh * dv &&
                i < static_cast<size_t>(step + 1u) * nvh * dv)
                want[i] = out_ref[i];
            else
                got_out[i] = 0.0;   // rows not written by this launch
        }
        char label[64];
        std::snprintf(label, sizeof(label), "recurrence out step%u", step);
        compare(got_out, want, 5e-3, label);

        // mid-launch state check: f64 reference state chain after this step
        std::vector<float> sdev(s_elems, 0.0f);
        HIP_OK(hipMemcpy(sdev.data(), dS, s_elems * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<double> got_s(s_elems), want_s(s_elems);
        for (size_t i = 0; i < s_elems; ++i) got_s[i] = sdev[i];
        // recompute reference chain up to this step
        std::vector<double> Sr(s_elems);
        for (size_t i = 0; i < s_elems; ++i) Sr[i] = s0_host[i];
        for (uint32_t s2 = 0; s2 <= step; ++s2) {
            for (uint32_t vh = 0; vh < nvh; ++vh) {
                const uint32_t kh = vh / rep;
                const double bv = shelf::bf16_to_f32(b[static_cast<size_t>(s2) * nvh + vh]);
                const double av = shelf::bf16_to_f32(a[static_cast<size_t>(s2) * nvh + vh]);
                const double beta = sigmoid_ref(bv);
                const double g = -std::exp(shelf::bf16_to_f32(a_log[vh])) *
                                 softplus_ref(av + shelf::bf16_to_f32(dt_bias[vh]));
                const double decay = std::exp(g);
                double* Sh = Sr.data() + static_cast<size_t>(vh) * dk * dv;
                for (size_t i = 0; i < dk * dv; ++i) Sh[i] *= decay;
                std::vector<double> d2(dv);
                for (uint32_t j = 0; j < dv; ++j) {
                    double kv = 0.0;
                    for (uint32_t i = 0; i < dk; ++i)
                        kv += Sh[static_cast<size_t>(i) * dv + j] *
                              nk_ref[(static_cast<size_t>(s2) * nkh + kh) * dk + i];
                    d2[j] = (shelf::bf16_to_f32(
                                 v[static_cast<size_t>(s2) * nvh * dv + static_cast<size_t>(vh) * dv + j]) -
                             kv) * beta;
                }
                for (uint32_t i = 0; i < dk; ++i) {
                    const double kk = nk_ref[(static_cast<size_t>(s2) * nkh + kh) * dk + i];
                    for (uint32_t j = 0; j < dv; ++j)
                        Sh[static_cast<size_t>(i) * dv + j] += kk * d2[j];
                }
            }
        }
        for (size_t i = 0; i < s_elems; ++i) want_s[i] = Sr[i];
        char slabel[64];
        std::snprintf(slabel, sizeof(slabel), "recurrence state after step%u", step);
        compare(got_s, want_s, 5e-3, slabel);
    }

    // ------------------------------------------------------------------
    // 3) fused conv+silu step: two launches, fp32 state [C,3] persists
    // ------------------------------------------------------------------
    std::vector<shelf::bf16_t> convW(static_cast<size_t>(C) * 4u);
    for (auto& x : convW) x = bf(-0.5f, 0.5f);
    std::vector<shelf::bf16_t> xin(2u * C);
    for (auto& x : xin) x = bf(-1.0f, 1.0f);
    std::vector<float> cst0(static_cast<size_t>(C) * 3u);
    for (auto& x : cst0) x = u01(rng) * 0.5f;

    void *dxin, *dconvw, *dcst, *dconvout, *dact;
    HIP_OK(hipMalloc(&dxin, xin.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&dconvw, convW.size() * sizeof(shelf::bf16_t)));
    HIP_OK(hipMalloc(&dcst, cst0.size() * sizeof(float)));
    HIP_OK(hipMalloc(&dconvout, 2u * C * sizeof(float)));
    HIP_OK(hipMalloc(&dact, 2u * C * sizeof(shelf::bf16_t)));
    HIP_OK(hipMemcpy(dxin, xin.data(), xin.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dconvw, convW.data(), convW.size() * sizeof(shelf::bf16_t), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dcst, cst0.data(), cst0.size() * sizeof(float), hipMemcpyHostToDevice));

    FusedGdnConvSiluArgs ca{};
    GdnConv1dArgs& cc = ca.conv;
    cc.input = static_cast<const shelf::bf16_t*>(dxin);
    cc.input_row_stride = C;
    cc.output = static_cast<float*>(dconvout);
    cc.output_row_stride = C;
    cc.conv_weight = static_cast<const shelf::bf16_t*>(dconvw);
    cc.conv_state = static_cast<float*>(dcst);
    cc.conv_slot_stride = static_cast<uint64_t>(C) * 3u;
    cc.conv_layer_stride = static_cast<uint64_t>(C) * 3u;
    cc.requests = dreq;
    cc.num_requests = 1;
    cc.actual_rows = 2;
    cc.max_request_rows = 1;
    cc.conv_dim = C;
    cc.conv_history = 3;
    cc.state_index = 0;
    cc.max_sequences = 1;
    ca.act_out = static_cast<shelf::bf16_t*>(dact);
    ca.act_row_stride = C;

    // f64 reference (shelf causal_conv_step semantics)
    std::vector<double> st(cst0.size());
    for (size_t i = 0; i < st.size(); ++i) st[i] = cst0[i];
    std::vector<double> conv_ref(2u * C), act_ref(2u * C);
    std::vector<double> st_final(st);
    for (uint32_t step = 0; step < 2u; ++step) {
        for (uint32_t c = 0; c < C; ++c) {
            const double u0 = st_final[static_cast<size_t>(c) * 3 + 0];
            const double u1 = st_final[static_cast<size_t>(c) * 3 + 1];
            const double u2 = st_final[static_cast<size_t>(c) * 3 + 2];
            const double x = shelf::bf16_to_f32(
                xin[static_cast<size_t>(step) * C + c]);
            double acc = 0.0;
            acc += shelf::bf16_to_f32(convW[static_cast<size_t>(c) * 4 + 0]) * u0;
            acc += shelf::bf16_to_f32(convW[static_cast<size_t>(c) * 4 + 1]) * u1;
            acc += shelf::bf16_to_f32(convW[static_cast<size_t>(c) * 4 + 2]) * u2;
            acc += shelf::bf16_to_f32(convW[static_cast<size_t>(c) * 4 + 3]) * x;
            conv_ref[static_cast<size_t>(step) * C + c] = acc;
            act_ref[static_cast<size_t>(step) * C + c] = silu_ref(acc);
            st_final[static_cast<size_t>(c) * 3 + 0] = u1;
            st_final[static_cast<size_t>(c) * 3 + 1] = u2;
            st_final[static_cast<size_t>(c) * 3 + 2] = x;
        }
    }

    std::vector<float> conv_out(2u * C, 0.0f);
    std::vector<shelf::bf16_t> act(2u * C, shelf::bf16_t{0});
    for (uint32_t step = 0; step < 2u; ++step) {
        hreq.row_begin = step;
        hreq.row_count = 1;
        HIP_OK(hipMemcpy(dreq, &hreq, sizeof(GdnRequestDescriptor), hipMemcpyHostToDevice));
        HIP_OK(shelf::hipk::launch_fused_gdn_conv_silu(ca, 0));
        HIP_OK(hipMemcpy(conv_out.data(), dconvout, conv_out.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_OK(hipMemcpy(act.data(), dact, act.size() * sizeof(shelf::bf16_t), hipMemcpyDeviceToHost));
        std::vector<double> got_conv(2u * C, 0.0), got_act(2u * C, 0.0);
        for (size_t i = step * C; i < (step + 1) * C; ++i) {
            got_conv[i] = conv_out[i];
            got_act[i] = shelf::bf16_to_f32(act[i]);
        }
        // zero-out other rows so compare() ignores them (ref row is also zeroed)
        std::vector<double> want_conv = conv_ref, want_act = act_ref;
        for (size_t i = 0; i < 2u * C; ++i)
            if (i < step * C || i >= (step + 1) * C) {
                got_conv[i] = want_conv[i] = 0.0;
                got_act[i] = want_act[i] = 0.0;
            }
        char l1[64], l2[64];
        std::snprintf(l1, sizeof(l1), "conv out f32 step%u", step);
        std::snprintf(l2, sizeof(l2), "conv silu act bf16 step%u", step);
        compare(got_conv, want_conv, 1e-3, l1);
        compare(got_act, want_act, 3e-2, l2);

        std::vector<float> cdev(cst0.size(), 0.0f);
        HIP_OK(hipMemcpy(cdev.data(), dcst, cdev.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<double> got_st(cst0.size()), want_st(cst0.size());
        std::vector<double> st_ref(cst0.size());
        for (size_t i = 0; i < cst0.size(); ++i) st_ref[i] = cst0[i];
        for (uint32_t s2 = 0; s2 <= step; ++s2)
            for (uint32_t c = 0; c < C; ++c) {
                const double u1 = st_ref[static_cast<size_t>(c) * 3 + 1];
                const double u2 = st_ref[static_cast<size_t>(c) * 3 + 2];
                const double x = shelf::bf16_to_f32(
                    xin[static_cast<size_t>(s2) * C + c]);
                st_ref[static_cast<size_t>(c) * 3 + 0] = u1;
                st_ref[static_cast<size_t>(c) * 3 + 1] = u2;
                st_ref[static_cast<size_t>(c) * 3 + 2] = x;
            }
        for (size_t i = 0; i < cst0.size(); ++i) {
            got_st[i] = cdev[i];
            want_st[i] = st_ref[i];
        }
        char l3[64];
        std::snprintf(l3, sizeof(l3), "conv state [C,3] after step%u", step);
        compare(got_st, want_st, 1e-3, l3);
    }

    hipFree(dq); hipFree(dk_); hipFree(dnq); hipFree(dsq); hipFree(dnk);
    hipFree(dS); hipFree(da); hipFree(db); hipFree(d_v); hipFree(da_log);
    hipFree(d_dt); hipFree(dreq); hipFree(dout);
    hipFree(dxin); hipFree(dconvw); hipFree(dcst); hipFree(dconvout); hipFree(dact);
}

}  // namespace

int main() {
    HIP_OK(hipSetDevice(0));
    run_case(CaseDims{2u, 4u, 32u, 32u});
    run_case(CaseDims{16u, 32u, 128u, 128u});
    std::printf("test_hip_gdn: failures=%d\n", failures);
    return failures ? 1 : 0;
}
