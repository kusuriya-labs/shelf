#include <phaseshift/core/memory/types.h>
#include <phaseshift/models/common/qwen_hybrid/moe_math.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace qh = ::ps::qwen_hybrid;

namespace {

constexpr std::uint32_t kH = 2560;
constexpr std::uint32_t kI = 640;
constexpr std::uint32_t kE = 512;
constexpr std::uint32_t kK = 10;
constexpr std::uint64_t kSeed = 20261024ull;
constexpr std::uint64_t kGamma = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t kMix1 = 0xBF58476D1CE4E5B9ull;
constexpr std::uint64_t kMix2 = 0x94D049BB133111EBull;

constexpr std::uint64_t kPosRouter = 0ull;
constexpr std::uint64_t kPosGateUp = 1310720ull;
constexpr std::uint64_t kPosDown = 1679032320ull;
constexpr std::uint64_t kPosSg = 2517893120ull;
constexpr std::uint64_t kPosSu = 2519531520ull;
constexpr std::uint64_t kPosSd = 2521169920ull;
constexpr std::uint64_t kPosSw = 2522808320ull;

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

float widen_bf16(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

float uniform_at(std::uint64_t pos) {
    std::uint64_t z = kSeed + (pos + 1ull) * kGamma;
    z = (z ^ (z >> 30)) * kMix1;
    z = (z ^ (z >> 27)) * kMix2;
    z = z ^ (z >> 31);
    const float m = static_cast<float>(static_cast<std::uint32_t>(z >> 40));
    return m * (1.0f / 8388608.0f) - 1.0f;
}

float scaled_bf16_at(std::uint64_t pos, float scale) {
    const float v = uniform_at(pos) * scale;
    return widen_bf16(ps::f32_to_bf16_rne(v));
}

void gen_scaled(float* out, std::uint64_t start, std::size_t count, float scale) {
    for (std::size_t p = 0; p < count; ++p) {
        out[p] = scaled_bf16_at(start + p, scale);
    }
}

const float kScaleH = 1.0f / std::sqrt(static_cast<float>(kH));
const float kScaleI = 1.0f / std::sqrt(static_cast<float>(kI));

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
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > m.max_abs) {
            m.max_abs = d;
            m.worst = i;
        }
        const double den = std::fabs(static_cast<double>(b[i]));
        if (den > 1e-30) {
            const double r = d / den;
            if (r > m.max_rel) m.max_rel = r;
        }
        acc += d * d;
    }
    if (n) m.rmse = std::sqrt(acc / static_cast<double>(n));
    return m;
}

void report(const char* stage, const char* name, const Metrics& m) {
    std::printf("METRIC %-8s %-20s max_abs=%.6e max_rel=%.6e rmse=%.6e worst=%zu\n",
                stage, name, m.max_abs, m.max_rel, m.rmse, m.worst);
}

void check_close(const Metrics& m, double atol, double rtol, const std::string& msg) {
    const bool ok = m.max_abs <= atol || m.max_rel <= rtol;
    check(ok, msg);
    if (!ok) {
        std::printf("  detail: max_abs=%.6e (atol %.1e) max_rel=%.6e (rtol %.1e)\n",
                    m.max_abs, atol, m.max_rel, rtol);
    }
}

bool read_bin(const std::string& dir, const std::string& name, void* dst,
              std::size_t bytes) {
    const std::string path = dir + "/" + name;
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::printf("FAIL: cannot open %s\n", path.c_str());
        return false;
    }
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (static_cast<std::size_t>(f.gcount()) != bytes) {
        std::printf("FAIL: short read %s\n", path.c_str());
        return false;
    }
    return true;
}

std::string find_dir() {
#ifdef PHASESHIFT_MOE_ORACLE_DIR
    {
        std::ifstream f(std::string(PHASESHIFT_MOE_ORACLE_DIR) + "/MANIFEST.txt");
        if (f.good()) return PHASESHIFT_MOE_ORACLE_DIR;
    }
#endif
    const char* cands[] = {
        "tests/data/qwen4exp_moe",
        "../tests/data/qwen4exp_moe",
        "../../tests/data/qwen4exp_moe",
        "../../../tests/data/qwen4exp_moe",
    };
    for (const char* c : cands) {
        std::ifstream f(std::string(c) + "/MANIFEST.txt");
        if (f.good()) return c;
    }
    return "";
}

struct Case {
    const char* name;
    std::uint32_t tokens;
    bool renorm;
};

void gen_router(float* wr) {
    gen_scaled(wr, kPosRouter, static_cast<std::size_t>(kE) * kH, kScaleH);
    for (std::uint32_t e = 0; e < 7; ++e) {
        wr[static_cast<std::size_t>(e) * kH] =
            widen_bf16(ps::f32_to_bf16_rne(static_cast<float>((20 - e) * 0.001)));
    }
    wr[7ull * kH] =
        widen_bf16(ps::f32_to_bf16_rne(static_cast<float>(13 * 0.001)));
    wr[9ull * kH] =
        widen_bf16(ps::f32_to_bf16_rne(static_cast<float>(12 * 0.001)));
    wr[11ull * kH] =
        widen_bf16(ps::f32_to_bf16_rne(static_cast<float>(11 * 0.001)));
    for (std::uint32_t e = 12; e < kE; ++e) {
        wr[static_cast<std::size_t>(e) * kH] =
            widen_bf16(ps::f32_to_bf16_rne(-0.100f));
    }
    std::memcpy(wr + 8ull * kH, wr + 7ull * kH, kH * sizeof(float));
    std::memcpy(wr + 10ull * kH, wr + 9ull * kH, kH * sizeof(float));
}

}  // namespace

int main() {
    const std::string dir = find_dir();
    if (dir.empty()) {
        std::printf("FAIL: oracle fixture dir not found (set PHASESHIFT_MOE_ORACLE_DIR)\n");
        return 1;
    }

    std::vector<float> wr(static_cast<std::size_t>(kE) * kH);
    gen_router(wr.data());

    const Case cases[] = {
        {"random_t3", 3, true},
        {"zero_t2", 2, true},
        {"tie_t2", 2, true},
        {"extreme_t2", 2, true},
        {"same_t4", 4, true},
        {"random_t3_norenorm", 3, false},
    };

    for (const Case& c : cases) {
        const std::string n = c.name;
        const std::size_t th = static_cast<std::size_t>(c.tokens) * kH;
        const std::size_t te = static_cast<std::size_t>(c.tokens) * kE;
        const std::size_t tk = static_cast<std::size_t>(c.tokens) * kK;

        std::vector<std::uint16_t> x_bits(th);
        std::vector<float> probs_g(te), top_w_g(tk), w_used_g(tk);
        std::vector<std::uint16_t> w_used_bf16_g(tk);
        std::vector<std::uint32_t> top_i_g(tk);
        std::vector<float> routed_g(th), shared_g(th), gate_g(c.tokens), moe_g(th);
        if (!read_bin(dir, n + "_x.bin", x_bits.data(), x_bits.size() * 2) ||
            !read_bin(dir, n + "_probs.bin", probs_g.data(), te * 4) ||
            !read_bin(dir, n + "_top_w.bin", top_w_g.data(), tk * 4) ||
            !read_bin(dir, n + "_top_w_used.bin", w_used_g.data(), tk * 4) ||
            !read_bin(dir, n + "_top_w_used_bf16.bin", w_used_bf16_g.data(), tk * 2) ||
            !read_bin(dir, n + "_top_i.bin", top_i_g.data(), tk * 4) ||
            !read_bin(dir, n + "_routed.bin", routed_g.data(), th * 4) ||
            !read_bin(dir, n + "_shared.bin", shared_g.data(), th * 4) ||
            !read_bin(dir, n + "_shared_gate.bin", gate_g.data(), c.tokens * 4) ||
            !read_bin(dir, n + "_moe.bin", moe_g.data(), th * 4)) {
            check(false, n + ": fixture load");
            continue;
        }

        std::vector<float> x(th);
        for (std::size_t i = 0; i < th; ++i) x[i] = widen_bf16(x_bits[i]);

        std::vector<float> probs(te, 0.0f);
        qh::moe_router_logits_f32(x.data(), wr.data(), c.tokens, kH, kE, probs.data());
        qh::moe_softmax_f32(probs.data(), c.tokens, kE);

        std::vector<std::uint32_t> top_i(tk, 0u);
        std::vector<float> top_w(tk, 0.0f);
        qh::moe_topk_stable_f32(probs.data(), c.tokens, kE, kK, top_i.data(),
                                top_w.data());
        std::vector<float> w_used = top_w;
        if (c.renorm) qh::moe_normalize_topk_f32(w_used.data(), c.tokens, kK);
        std::vector<std::uint16_t> w_used_bf16(tk);
        qh::moe_f32_to_bf16(w_used.data(), w_used_bf16.data(), tk);

        const Metrics m_probs = diff_metrics(probs.data(), probs_g.data(), te);
        const Metrics m_topw = diff_metrics(top_w.data(), top_w_g.data(), tk);
        const Metrics m_used = diff_metrics(w_used.data(), w_used_g.data(), tk);
        report("router", (n + "/probs").c_str(), m_probs);
        report("router", (n + "/top_w").c_str(), m_topw);
        report("router", (n + "/top_w_used").c_str(), m_used);

        check(std::memcmp(top_i.data(), top_i_g.data(),
                          tk * sizeof(std::uint32_t)) == 0,
              n + ": router top-k index bitwise");
        check(m_probs.max_abs <= 1.0e-5, n + ": probs max_abs <= 1e-5");
        check(m_topw.max_abs <= 1.0e-5, n + ": top_w max_abs <= 1e-5");
        check(m_used.max_abs <= 1.0e-5, n + ": top_w_used max_abs <= 1e-5");

        std::vector<std::uint16_t> cast_from_golden(tk);
        qh::moe_f32_to_bf16(w_used_g.data(), cast_from_golden.data(), tk);
        check(std::memcmp(cast_from_golden.data(), w_used_bf16_g.data(),
                          tk * sizeof(std::uint16_t)) == 0,
              n + ": bf16 cast function bitwise on golden input");
        std::vector<float> w_pipe(tk), w_gold_bf(tk);
        std::size_t bf_bitwise = 0;
        for (std::size_t i = 0; i < tk; ++i) {
            w_pipe[i] = widen_bf16(w_used_bf16[i]);
            w_gold_bf[i] = widen_bf16(w_used_bf16_g[i]);
            if (w_used_bf16[i] == w_used_bf16_g[i]) ++bf_bitwise;
        }
        const Metrics m_bf = diff_metrics(w_pipe.data(), w_gold_bf.data(), tk);
        report("router", (n + "/w_used_bf16").c_str(), m_bf);
        std::printf("INFO   %-20s bf16 bitwise elements: %zu/%zu\n", n.c_str(),
                    bf_bitwise, tk);
        check(m_bf.max_abs <= 1.0e-2, n + ": widened bf16 weight within 1 step");

        if (c.renorm) {
            for (std::uint32_t t = 0; t < c.tokens; ++t) {
                float s = 0.0f;
                for (std::uint32_t j = 0; j < kK; ++j)
                    s += w_used[static_cast<std::size_t>(t) * kK + j];
                check(std::fabs(s - 1.0f) <= 1.0e-5f,
                      n + ": normalized top-k sums to 1");
            }
        } else {
            float s = 0.0f;
            for (std::uint32_t j = 0; j < kK; ++j) s += w_used[j];
            check(s < 1.0f, n + ": norenorm top-k sum below 1");
        }

        std::vector<float> routed(th, 0.0f);
        std::vector<float> xrow(kH);
        std::vector<float> gu(static_cast<std::size_t>(2) * kI * kH);
        std::vector<float> dn(static_cast<std::size_t>(kH) * kI);
        std::vector<float> out_e(kH);
        for (std::uint32_t t = 0; t < c.tokens; ++t) {
            std::memcpy(xrow.data(), x.data() + static_cast<std::size_t>(t) * kH,
                        kH * sizeof(float));
            float* acc = routed.data() + static_cast<std::size_t>(t) * kH;
            for (std::uint32_t j = 0; j < kK; ++j) {
                const std::uint32_t e =
                    top_i[static_cast<std::size_t>(t) * kK + j];
                const float wgt =
                    widen_bf16(w_used_bf16[static_cast<std::size_t>(t) * kK + j]);
                gen_scaled(gu.data(),
                           kPosGateUp +
                               static_cast<std::uint64_t>(e) * 2ull * kI * kH,
                           gu.size(), kScaleH);
                gen_scaled(dn.data(),
                           kPosDown + static_cast<std::uint64_t>(e) * kH * kI,
                           dn.size(), kScaleI);
                qh::moe_expert_forward_f32(xrow.data(), gu.data(), dn.data(), kI,
                                           kH, out_e.data());
                for (std::uint32_t h = 0; h < kH; ++h) acc[h] += wgt * out_e[h];
            }
        }

        std::vector<float> sg(static_cast<std::size_t>(kI) * kH);
        std::vector<float> su(static_cast<std::size_t>(kI) * kH);
        std::vector<float> sd(static_cast<std::size_t>(kH) * kI);
        std::vector<float> sw(kH);
        gen_scaled(sg.data(), kPosSg, sg.size(), kScaleH);
        gen_scaled(su.data(), kPosSu, su.size(), kScaleH);
        gen_scaled(sd.data(), kPosSd, sd.size(), kScaleI);
        gen_scaled(sw.data(), kPosSw, sw.size(), kScaleH);

        std::vector<float> shared(th, 0.0f);
        std::vector<float> gate(c.tokens, 0.0f);
        qh::moe_shared_expert_f32(x.data(), c.tokens, kH, kI, sg.data(), su.data(),
                                  sd.data(), sw.data(), shared.data(), gate.data());
        std::vector<float> moe(th, 0.0f);
        qh::moe_compose_f32(routed.data(), shared.data(), th, moe.data());

        const Metrics m_routed = diff_metrics(routed.data(), routed_g.data(), th);
        const Metrics m_shared = diff_metrics(shared.data(), shared_g.data(), th);
        const Metrics m_gate = diff_metrics(gate.data(), gate_g.data(), c.tokens);
        const Metrics m_moe = diff_metrics(moe.data(), moe_g.data(), th);
        report("expert", (n + "/routed").c_str(), m_routed);
        report("shared", (n + "/shared").c_str(), m_shared);
        report("shared", (n + "/gate").c_str(), m_gate);
        report("compose", (n + "/moe").c_str(), m_moe);

        check_close(m_routed, 5.0e-2, 5.0e-4, n + ": routed close");
        check_close(m_shared, 5.0e-3, 5.0e-4, n + ": shared close");
        check(m_gate.max_abs <= 1.0e-5, n + ": shared gate max_abs <= 1e-5");
        check_close(m_moe, 5.0e-2, 5.0e-4, n + ": moe compose close");
    }

    {
        const std::string zdir = dir;
        std::vector<std::uint32_t> ti(10);
        if (read_bin(zdir, "zero_t2_top_i.bin", ti.data(), ti.size() * 4)) {
            bool ordered = true;
            for (std::uint32_t i = 0; i < 10; ++i) ordered &= (ti[i] == i);
            check(ordered, "zero_t2: full tie selects experts 0..9 in index order");
        }
        std::vector<float> pr(512);
        if (read_bin(zdir, "tie_t2_probs.bin", pr.data(), pr.size() * 4)) {
            check(std::memcmp(&pr[7], &pr[8], sizeof(float)) == 0,
                  "tie_t2: internal tie pair probs bitwise equal");
            check(std::memcmp(&pr[9], &pr[10], sizeof(float)) == 0,
                  "tie_t2: boundary tie pair probs bitwise equal");
        }
    }

    std::printf("\nQWEN4EXP_MOE: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
