#include <phaseshift/models/qwen4exp/hyper/gated_residual.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace hyper = ps::qwen4exp::hyper;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++passed;
        printf("PASS: %s\n", msg);
    } else {
        ++failed;
        printf("FAIL: %s\n", msg);
    }
}

struct Rng {
    std::uint64_t state;

    explicit Rng(std::uint64_t seed) : state(seed) {}

    std::uint64_t next() {
        state += 0x9E3779B97F4A7C15ull;
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    double uniform(double lo, double hi) {
        const double u = static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
        return lo + (hi - lo) * u;
    }
};

static double d_sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

static double d_silu(double x) {
    return x * d_sigmoid(x);
}

struct OracleRead {
    std::vector<double> normed;
    std::vector<double> down;
    std::vector<double> up;
    std::vector<double> gate;
    std::vector<double> mixed;
    std::vector<double> inject;
};

static OracleRead oracle_read(const hyper::GatedResidualDims& dims,
                              const hyper::GatedResidualWeights& w,
                              const std::vector<float>& x4, std::uint32_t tokens) {
    const std::uint32_t S = dims.streams;
    const std::uint32_t H = dims.hidden;
    const std::uint32_t L = dims.lowrank;
    const std::uint32_t SH = S * H;
    OracleRead out;
    out.normed.resize(static_cast<std::size_t>(tokens) * SH);
    out.down.resize(static_cast<std::size_t>(tokens) * L);
    out.up.resize(static_cast<std::size_t>(tokens) * SH);
    out.gate.resize(static_cast<std::size_t>(tokens) * SH);
    out.mixed.resize(static_cast<std::size_t>(tokens) * H);
    out.inject.resize(static_cast<std::size_t>(tokens) * S);

    for (std::uint32_t t = 0; t < tokens; ++t) {
        const float* x = x4.data() + static_cast<std::size_t>(t) * SH;
        double* n = out.normed.data() + static_cast<std::size_t>(t) * SH;
        double* d = out.down.data() + static_cast<std::size_t>(t) * L;
        double* u = out.up.data() + static_cast<std::size_t>(t) * SH;
        double* g = out.gate.data() + static_cast<std::size_t>(t) * SH;
        double* m = out.mixed.data() + static_cast<std::size_t>(t) * H;
        double* inj = out.inject.data() + static_cast<std::size_t>(t) * S;

        for (std::uint32_t s = 0; s < S; ++s) {
            double ms = 0.0;
            for (std::uint32_t h = 0; h < H; ++h) {
                const double v = static_cast<double>(x[s * H + h]);
                ms += v * v;
            }
            ms /= static_cast<double>(H);
            const double r = 1.0 / std::sqrt(ms + static_cast<double>(dims.eps));
            for (std::uint32_t h = 0; h < H; ++h) {
                const std::uint32_t idx = s * H + h;
                n[idx] = (1.0 + static_cast<double>(w.hc_norm[idx])) *
                         (static_cast<double>(x[idx]) * r);
            }
        }
        for (std::uint32_t j = 0; j < L; ++j) {
            double acc = 0.0;
            for (std::uint32_t i = 0; i < SH; ++i) {
                acc += n[i] * static_cast<double>(w.mix_down[static_cast<std::size_t>(j) * SH + i]);
            }
            d[j] = acc;
        }
        const double stream_scale = 1.0 / static_cast<double>(S);
        for (std::uint32_t i = 0; i < SH; ++i) {
            double acc = 0.0;
            for (std::uint32_t j = 0; j < L; ++j) {
                acc += d_silu(d[j] * stream_scale) *
                       static_cast<double>(w.mix_up[static_cast<std::size_t>(i) * L + j]);
            }
            u[i] = acc;
            g[i] = d_sigmoid(acc);
        }
        for (std::uint32_t h = 0; h < H; ++h) {
            double acc = 0.0;
            for (std::uint32_t s = 0; s < S; ++s) {
                const std::uint32_t idx = s * H + h;
                acc += g[idx] * n[idx];
            }
            m[h] = acc * stream_scale;
        }
        for (std::uint32_t s = 0; s < S; ++s) {
            double acc = 0.0;
            for (std::uint32_t i = 0; i < SH; ++i) {
                acc += static_cast<double>(w.block_inject[static_cast<std::size_t>(s) * SH + i]) *
                       n[i];
            }
            inj[s] = 2.0 * d_sigmoid(acc * stream_scale);
        }
    }
    return out;
}

static std::vector<double> oracle_write(const hyper::GatedResidualDims& dims,
                                        const std::vector<float>& x4,
                                        const std::vector<float>& sub,
                                        const std::vector<float>& inj,
                                        std::uint32_t tokens) {
    const std::uint32_t S = dims.streams;
    const std::uint32_t H = dims.hidden;
    std::vector<double> out(static_cast<std::size_t>(tokens) * S * H);
    for (std::uint32_t t = 0; t < tokens; ++t) {
        for (std::uint32_t s = 0; s < S; ++s) {
            for (std::uint32_t h = 0; h < H; ++h) {
                const std::size_t idx = (static_cast<std::size_t>(t) * S + s) * H + h;
                out[idx] = static_cast<double>(x4[idx]) +
                           static_cast<double>(sub[static_cast<std::size_t>(t) * H + h]) *
                               static_cast<double>(inj[static_cast<std::size_t>(t) * S + s]);
            }
        }
    }
    return out;
}

static std::vector<double> oracle_collapse(const hyper::GatedResidualDims& dims,
                                           const hyper::GatedResidualCollapseWeights& w,
                                           const std::vector<float>& x4,
                                           std::uint32_t tokens) {
    const std::uint32_t S = dims.streams;
    const std::uint32_t H = dims.hidden;
    const std::uint32_t SH = S * H;
    const std::uint32_t L = dims.lowrank;
    std::vector<double> out(static_cast<std::size_t>(tokens) * H);
    const double stream_scale = 1.0 / static_cast<double>(S);
    for (std::uint32_t t = 0; t < tokens; ++t) {
        const float* x = x4.data() + static_cast<std::size_t>(t) * SH;
        std::vector<double> n(SH);
        for (std::uint32_t s = 0; s < S; ++s) {
            double ms = 0.0;
            for (std::uint32_t h = 0; h < H; ++h) {
                const double v = static_cast<double>(x[s * H + h]);
                ms += v * v;
            }
            ms /= static_cast<double>(H);
            const double r = 1.0 / std::sqrt(ms + static_cast<double>(dims.eps));
            for (std::uint32_t h = 0; h < H; ++h) {
                const std::uint32_t idx = s * H + h;
                n[idx] = (1.0 + static_cast<double>(w.hc_norm[idx])) *
                         (static_cast<double>(x[idx]) * r);
            }
        }
        std::vector<double> d(L);
        for (std::uint32_t j = 0; j < L; ++j) {
            double acc = 0.0;
            for (std::uint32_t i = 0; i < SH; ++i) {
                acc += n[i] * static_cast<double>(w.mix_down[static_cast<std::size_t>(j) * SH + i]);
            }
            d[j] = acc;
        }
        std::vector<double> g(SH);
        for (std::uint32_t i = 0; i < SH; ++i) {
            double acc = 0.0;
            for (std::uint32_t j = 0; j < L; ++j) {
                acc += d_silu(d[j] * stream_scale) *
                       static_cast<double>(w.mix_up[static_cast<std::size_t>(i) * L + j]);
            }
            g[i] = d_sigmoid(acc);
        }
        for (std::uint32_t h = 0; h < H; ++h) {
            double acc = 0.0;
            for (std::uint32_t s = 0; s < S; ++s) {
                const std::uint32_t idx = s * H + h;
                acc += g[idx] * n[idx];
            }
            out[static_cast<std::size_t>(t) * H + h] = acc * stream_scale;
        }
    }
    return out;
}

struct Metrics {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rmse = 0.0;
};

static Metrics compare(const float* got, const std::vector<double>& want) {
    Metrics m;
    double sq = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double diff = std::abs(static_cast<double>(got[i]) - want[i]);
        if (diff > m.max_abs) m.max_abs = diff;
        const double denom = std::max(std::abs(want[i]), 1e-6);
        const double rel = diff / denom;
        if (rel > m.max_rel) m.max_rel = rel;
        sq += diff * diff;
    }
    m.rmse = std::sqrt(sq / static_cast<double>(want.size()));
    return m;
}

static void report(const char* name, const Metrics& m) {
    printf("  %-12s max_abs=%.3e max_rel=%.3e RMSE=%.3e\n", name, m.max_abs, m.max_rel, m.rmse);
}

struct Weights {
    std::vector<float> hc_norm;
    std::vector<float> mix_down;
    std::vector<float> mix_up;
    std::vector<float> block_inject;
};

static Weights make_weights(const hyper::GatedResidualDims& dims, Rng& rng) {
    const std::size_t sh = static_cast<std::size_t>(dims.streams) * dims.hidden;
    Weights w;
    w.hc_norm.resize(sh);
    w.mix_down.resize(static_cast<std::size_t>(dims.lowrank) * sh);
    w.mix_up.resize(sh * dims.lowrank);
    w.block_inject.resize(sh * dims.streams);
    for (float& v : w.hc_norm) v = static_cast<float>(rng.uniform(-0.02, 0.02));
    for (float& v : w.mix_down) v = static_cast<float>(rng.uniform(-0.02, 0.02));
    for (float& v : w.mix_up) v = static_cast<float>(rng.uniform(-0.02, 0.02));
    for (float& v : w.block_inject) v = static_cast<float>(rng.uniform(-0.02, 0.02));
    return w;
}

static void run_case(const hyper::GatedResidualDims& dims, std::uint32_t tokens,
                     std::uint64_t seed) {
    Rng rng(seed);
    const std::size_t sh = static_cast<std::size_t>(tokens) * dims.streams * dims.hidden;
    const std::size_t h = static_cast<std::size_t>(tokens) * dims.hidden;
    const std::size_t s_out = static_cast<std::size_t>(tokens) * dims.streams;

    Weights w = make_weights(dims, rng);
    std::vector<float> x4(sh);
    for (float& v : x4) v = static_cast<float>(rng.uniform(-1.0, 1.0));

    hyper::GatedResidualWeights gw;
    gw.hc_norm = w.hc_norm.data();
    gw.mix_down = w.mix_down.data();
    gw.mix_up = w.mix_up.data();
    gw.block_inject = w.block_inject.data();

    std::vector<float> mixed(h, 0.0f);
    std::vector<float> inject(s_out, 0.0f);
    hyper::GatedResidualReadTrace trace;
    const ps::Status st = hyper::gated_residual_read(dims, gw, x4.data(), tokens,
                                                     mixed.data(), inject.data(), &trace);
    if (!st.ok()) {
        check(false, "read runs");
        printf("  reason: %s\n", st.message().c_str());
        return;
    }

    const OracleRead oracle = oracle_read(dims, gw, x4, tokens);
    const Metrics m_normed = compare(trace.normed.data(), oracle.normed);
    const Metrics m_down = compare(trace.down.data(), oracle.down);
    const Metrics m_up = compare(trace.up.data(), oracle.up);
    const Metrics m_gate = compare(trace.gate.data(), oracle.gate);
    const Metrics m_mixed = compare(mixed.data(), oracle.mixed);
    const Metrics m_inject = compare(inject.data(), oracle.inject);
    report("normed", m_normed);
    report("down", m_down);
    report("up", m_up);
    report("gate", m_gate);
    report("mixed", m_mixed);
    report("inject", m_inject);

    check(m_normed.max_abs <= 1e-5 && m_normed.rmse <= 1e-6,
          "read: normed within threshold");
    check(m_down.max_abs <= 1e-4 && m_down.rmse <= 1e-5, "read: down within threshold");
    check(m_up.max_abs <= 1e-5 && m_up.rmse <= 1e-6, "read: up within threshold");
    check(m_gate.max_abs <= 1e-6 && m_gate.rmse <= 1e-7, "read: gate within threshold");
    check(m_mixed.max_abs <= 1e-5 && m_mixed.rmse <= 1e-6,
          "read: mixed within threshold");
    check(m_inject.max_abs <= 1e-5 && m_inject.rmse <= 1e-6,
          "read: inject within threshold");

    bool gate_range = true;
    for (float v : trace.gate) {
        if (!(v > 0.0f && v < 1.0f)) gate_range = false;
    }
    check(gate_range, "read: gate strictly inside (0,1)");
    bool inj_range = true;
    for (float v : inject) {
        if (!(v > 0.0f && v < 4.0f)) inj_range = false;
    }
    check(inj_range, "read: inject strictly inside (0,4)");

    const bool trace_match =
        trace.mixed.size() == mixed.size() && trace.inject.size() == inject.size() &&
        std::equal(trace.mixed.begin(), trace.mixed.end(), mixed.begin()) &&
        std::equal(trace.inject.begin(), trace.inject.end(), inject.begin());
    check(trace_match, "read: trace mirrors outputs");

    std::vector<float> sub(h);
    for (float& v : sub) v = static_cast<float>(rng.uniform(-0.5, 0.5));
    std::vector<float> out_x4(sh, 0.0f);
    check(hyper::gated_residual_write(dims, x4.data(), sub.data(), inject.data(), tokens,
                                      out_x4.data())
              .ok(),
          "write runs");
    const std::vector<double> oracle_out = oracle_write(dims, x4, sub, inject, tokens);
    const Metrics m_write = compare(out_x4.data(), oracle_out);
    report("write", m_write);
    check(m_write.max_abs <= 1e-6 && m_write.rmse <= 1e-7, "write: within threshold");

    hyper::GatedResidualCollapseWeights cw;
    cw.hc_norm = w.hc_norm.data();
    cw.mix_down = w.mix_down.data();
    cw.mix_up = w.mix_up.data();
    std::vector<float> collapsed(h, 0.0f);
    check(hyper::gated_residual_collapse(dims, cw, x4.data(), tokens, collapsed.data()).ok(),
          "collapse runs");
    const std::vector<double> oracle_col = oracle_collapse(dims, cw, x4, tokens);
    const Metrics m_col = compare(collapsed.data(), oracle_col);
    report("collapse", m_col);
    check(m_col.max_abs <= 1e-5 && m_col.rmse <= 1e-6, "collapse: within threshold");
}

static void test_write_hand_case() {
    hyper::GatedResidualDims dims;
    dims.streams = 4;
    dims.hidden = 2;
    const float x4[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float sub[2] = {0.5f, -0.25f};
    const float inj[4] = {2.0f, 0.5f, 3.0f, 1.0f};
    float out[8] = {};
    check(hyper::gated_residual_write(dims, x4, sub, inj, 1, out).ok(), "write hand: runs");
    const float want[8] = {2.0f, 1.5f, 3.25f, 3.875f, 6.5f, 5.25f, 7.5f, 7.75f};
    bool ok = true;
    for (int i = 0; i < 8; ++i) {
        if (out[i] != want[i]) {
            ok = false;
            printf("  write hand mismatch at %d: got=%f want=%f\n", i, out[i], want[i]);
        }
    }
    check(ok, "write hand: exact scalar injection per stream");
}

static void test_split_equivalence() {
    hyper::GatedResidualDims dims;
    Rng rng(0xC0FFEE);
    const std::uint32_t tokens = 7;
    const std::size_t sh = static_cast<std::size_t>(tokens) * dims.streams * dims.hidden;
    Weights w = make_weights(dims, rng);
    std::vector<float> x4(sh);
    for (float& v : x4) v = static_cast<float>(rng.uniform(-1.0, 1.0));

    hyper::GatedResidualWeights gw;
    gw.hc_norm = w.hc_norm.data();
    gw.mix_down = w.mix_down.data();
    gw.mix_up = w.mix_up.data();
    gw.block_inject = w.block_inject.data();

    std::vector<float> mixed_full(sh / dims.streams * dims.streams, 0.0f);
    mixed_full.resize(static_cast<std::size_t>(tokens) * dims.hidden);
    std::vector<float> inject_full(static_cast<std::size_t>(tokens) * dims.streams, 0.0f);
    check(hyper::gated_residual_read(dims, gw, x4.data(), tokens, mixed_full.data(),
                                     inject_full.data(), nullptr)
              .ok(),
          "split: full read runs");

    std::vector<float> mixed_split(static_cast<std::size_t>(tokens) * dims.hidden, 0.0f);
    std::vector<float> inject_split(static_cast<std::size_t>(tokens) * dims.streams, 0.0f);
    const std::uint32_t hc = dims.streams * dims.hidden;
    check(hyper::gated_residual_read(dims, gw, x4.data(), 3, mixed_split.data(),
                                     inject_split.data(), nullptr)
              .ok(),
          "split: first read runs");
    check(hyper::gated_residual_read(dims, gw, x4.data() + 3ull * hc, tokens - 3,
                                     mixed_split.data() + 3ull * dims.hidden,
                                     inject_split.data() + 3ull * dims.streams, nullptr)
              .ok(),
          "split: second read runs");
    const bool same =
        std::equal(mixed_full.begin(), mixed_full.end(), mixed_split.begin()) &&
        std::equal(inject_full.begin(), inject_full.end(), inject_split.begin());
    check(same, "split: chunked read is bitwise identical to full read");
}

static void test_tiny_dims() {
    hyper::GatedResidualDims dims;
    dims.streams = 2;
    dims.hidden = 8;
    dims.lowrank = 3;
    run_case(dims, 3, 0xABCDEFu);
}

static void test_error_paths() {
    hyper::GatedResidualDims dims;
    Rng rng(7);
    Weights w = make_weights(dims, rng);
    std::vector<float> x4(10240, 0.1f);
    std::vector<float> mixed(2560, 0.0f);
    std::vector<float> inject(4, 0.0f);

    hyper::GatedResidualWeights gw;
    gw.hc_norm = w.hc_norm.data();
    gw.mix_down = w.mix_down.data();
    gw.mix_up = w.mix_up.data();
    gw.block_inject = w.block_inject.data();

    check(!hyper::gated_residual_read(dims, gw, x4.data(), 1, nullptr, inject.data(), nullptr)
              .ok(),
          "err: null mixed rejected");
    hyper::GatedResidualWeights bad = gw;
    bad.block_inject = nullptr;
    check(!hyper::gated_residual_read(dims, bad, x4.data(), 1, mixed.data(), inject.data(),
                                      nullptr)
              .ok(),
          "err: null block_inject rejected");
    hyper::GatedResidualDims zero = dims;
    zero.streams = 0;
    check(!hyper::gated_residual_read(zero, gw, x4.data(), 1, mixed.data(), inject.data(),
                                      nullptr)
              .ok(),
          "err: zero streams rejected");
    check(!hyper::gated_residual_write(dims, x4.data(), nullptr, inject.data(), 1, x4.data())
              .ok(),
          "err: null sub_out rejected");
    hyper::GatedResidualCollapseWeights cw;
    cw.hc_norm = w.hc_norm.data();
    cw.mix_down = w.mix_down.data();
    check(!hyper::gated_residual_collapse(dims, cw, x4.data(), 1, mixed.data()).ok(),
          "err: null collapse mixer rejected");
}

int main() {
    printf("=== Gated Residual / HyperConnection (float32 vs float64 oracle) ===\n");
    hyper::GatedResidualDims full;
    printf("case: streams=%u hidden=%u lowrank=%u tokens=1 seed=1\n", full.streams,
           full.hidden, full.lowrank);
    run_case(full, 1, 1u);
    printf("case: streams=%u hidden=%u lowrank=%u tokens=5 seed=2\n", full.streams,
           full.hidden, full.lowrank);
    run_case(full, 5, 2u);
    printf("case: streams=%u hidden=%u lowrank=%u tokens=17 seed=3\n", full.streams,
           full.hidden, full.lowrank);
    run_case(full, 17, 3u);
    test_write_hand_case();
    test_split_equivalence();
    test_tiny_dims();
    test_error_paths();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
