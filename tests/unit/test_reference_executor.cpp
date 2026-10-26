// shelf unit test — CPU reference executor on a synthetic model (PR-005).
// Plan Phase 4 completion: synthetic Model Graph runs on CPU and State
// updates match a golden fixture.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <shelf/backend/cpu/reference_executor.hpp>
#include <shelf/ir/graph.hpp>

namespace {

using namespace shelf;
using namespace shelf::backend::cpu;
using namespace shelf::ir;

int failures = 0;

void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

void check_close(const std::vector<float>* got, const std::vector<double>& want,
                 double tol, const char* what) {
    bool ok = got != nullptr && got->size() == want.size();
    for (std::size_t i = 0; ok && i < want.size(); ++i) {
        ok = std::fabs(static_cast<double>((*got)[i]) - want[i]) <= tol;
    }
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s (got n=%zu want n=%zu)\n", what,
                    got ? got->size() : 0, want.size());
        if (got) {
            for (std::size_t i = 0; i < std::min<std::size_t>(got->size(), 8); ++i) {
                std::printf("  [%zu] got=%.6f want=%.6f\n", i, (*got)[i], want[i]);
            }
        }
    }
}

// deterministic weight values shared by graph data and expected computation
double wval(std::uint64_t i, std::uint64_t j) {
    return static_cast<double>((i * 31 + j * 17) % 13) * 0.125 - 0.5;
}
std::vector<float> wtable(std::size_t rows, std::size_t cols) {
    std::vector<float> w(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) w[i * cols + j] = static_cast<float>(wval(i, j));
    }
    return w;
}

ValueDesc vd(std::uint32_t id, std::initializer_list<std::size_t> dims,
             DataType dt = DataType::f32) {
    return ValueDesc{ValueId::from_uint32(id), TensorShape::create(dims).value(), dt};
}

Status add_p(Graph& g, Kind kind, std::vector<ValueId> ins, std::vector<TensorId> ws,
             ValueDesc out, Attr attr = {}) {
    Primitive p;
    p.kind = kind;
    p.attr = attr;
    p.inputs = std::move(ins);
    p.weights = std::move(ws);
    p.output = out;
    return g.add_primitive(std::move(p));
}

auto V = [](std::uint32_t i) { return ValueId::from_uint32(i); };
auto W = [](std::uint32_t i) { return TensorId::parse(i).value(); };

constexpr std::size_t H = 4, D = 4, VOCAB = 8, F = 6, E = 3, CAP = 4, VOC_OUT = 5;
constexpr float EPS = 1e-5f;

// ---------------------------------------------------------------------------
// full single-decode synthetic model
// ---------------------------------------------------------------------------
Graph build_model() {
    Graph g;
    g.declare_input(vd(0, {1}, DataType::i32));   // tokens
    g.declare_input(vd(1, {1, H}));               // rope cos
    g.declare_input(vd(2, {1, H}));               // rope sin
    g.declare_input(vd(3, {D}));                  // new K
    g.declare_input(vd(4, {D}));                  // new V
    g.declare_input(vd(23, {E, H}));              // stacked expert outputs
    g.declare_state(StateDesc{StateId::from_uint32(0), StateKind::kv_cache,
                               TensorShape::create({CAP, D}).value(), DataType::f32});
    g.declare_weight(W(1), TensorShape::create({VOCAB, H}).value());   // table
    g.declare_weight(W(2), TensorShape::create({H}).value());          // gamma
    g.declare_weight(W(5), TensorShape::create({F, H}).value());       // gate
    g.declare_weight(W(6), TensorShape::create({F, H}).value());       // up
    g.declare_weight(W(7), TensorShape::create({H, F}).value());       // down
    g.declare_weight(W(8), TensorShape::create({E, H}).value());       // router
    g.declare_weight(W(9), TensorShape::create({VOC_OUT, H}).value()); // proj

    Attr eps;
    eps.epsilon = EPS;
    Status st = Status::make_ok();
    st = st.ok() ? add_p(g, Kind::embedding, {V(0)}, {W(1)}, vd(10, {1, H})) : st;
    st = st.ok() ? add_p(g, Kind::rmsnorm, {V(10)}, {W(2)}, vd(11, {1, H}), eps) : st;
    st = st.ok() ? add_p(g, Kind::rope, {V(11), V(1), V(2)}, {}, vd(12, {1, H})) : st;
    {
        Primitive att;
        att.kind = Kind::attention_decode;
        att.inputs = {V(12), V(3), V(4)};
        att.state_read = {StateId::from_uint32(0)};
        att.state_write = {StateId::from_uint32(0)};
        att.output = vd(13, {1, D});
        st = st.ok() ? g.add_primitive(att) : st;
    }
    st = st.ok() ? add_p(g, Kind::residual_add, {V(11), V(13)}, {}, vd(14, {1, H})) : st;
    st = st.ok() ? add_p(g, Kind::linear, {V(14)}, {W(5)}, vd(15, {1, F})) : st;
    st = st.ok() ? add_p(g, Kind::silu, {V(15)}, {}, vd(16, {1, F})) : st;
    st = st.ok() ? add_p(g, Kind::linear, {V(14)}, {W(6)}, vd(17, {1, F})) : st;
    st = st.ok() ? add_p(g, Kind::elem_mul, {V(16), V(17)}, {}, vd(18, {1, F})) : st;
    st = st.ok() ? add_p(g, Kind::linear, {V(18)}, {W(7)}, vd(19, {1, H})) : st;
    st = st.ok() ? add_p(g, Kind::residual_add, {V(14), V(19)}, {}, vd(20, {1, H})) : st;
    st = st.ok() ? add_p(g, Kind::linear, {V(20)}, {W(8)}, vd(21, {1, E})) : st;
    st = st.ok() ? add_p(g, Kind::moe_route, {V(21)}, {}, vd(22, {1, E})) : st;
    st = st.ok() ? add_p(g, Kind::moe_merge, {V(22), V(23)}, {}, vd(24, {H})) : st;
    st = st.ok() ? add_p(g, Kind::output_projection, {V(20)}, {W(9)}, vd(25, {1, VOC_OUT})) : st;
    if (!st.ok()) {
        std::printf("FATAL graph: %s\n", st.message().c_str());
        std::exit(2);
    }
    return g;
}

std::vector<double> linear_ref(const std::vector<double>& x, std::size_t N, std::size_t K) {
    // weights via wval(n,k)
    std::vector<double> out(N, 0.0);
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t k = 0; k < K; ++k) out[n] += x[k] * wval(n, k);
    }
    return out;
}

void test_single_layer_golden() {
    const Graph g = build_model();
    HostStorage s;

    check(s.set_value_int(V(0), {5}).ok(), "set tokens");
    std::vector<float> cosv(H), sinv(H), knew(D), vnew(D), eo(E * H);
    std::vector<double> cosd(H), sind(H), knewd(D), vnewd(D), eod(E * H);
    const double c[2] = {0.9, 0.85};
    const double sn[2] = {0.1, 0.17};
    for (std::size_t i = 0; i < H; ++i) {
        // genuine RoPE table: pair indices share the angle (llama style)
        cosd[i] = c[i % (H / 2)];
        sind[i] = sn[i % (H / 2)];
        cosv[i] = static_cast<float>(cosd[i]); sinv[i] = static_cast<float>(sind[i]);
    }
    for (std::size_t d = 0; d < D; ++d) {
        knewd[d] = 0.2 * d + 1.0; vnewd[d] = 3.0 - 0.3 * d;
        knew[d] = static_cast<float>(knewd[d]); vnew[d] = static_cast<float>(vnewd[d]);
    }
    for (std::size_t i = 0; i < E * H; ++i) {
        eod[i] = wval(i, 7);
        eo[i] = static_cast<float>(eod[i]);
    }
    check(s.set_value(V(1), cosv).ok() && s.set_value(V(2), sinv).ok(), "set rope");
    check(s.set_value(V(3), knew).ok() && s.set_value(V(4), vnew).ok(), "set k/v new");
    check(s.set_value(V(23), eo).ok(), "set expert out");
    check(s.init_state(StateId::from_uint32(0), std::vector<float>(2 * CAP * D, 0.0f)).ok(),
          "init kv state");
    for (auto [id, tbl] : std::vector<std::pair<TensorId, std::vector<float>>>{
             {W(1), wtable(VOCAB, H)}, {W(2), wtable(1, H)}, {W(5), wtable(F, H)},
             {W(6), wtable(F, H)}, {W(7), wtable(H, F)}, {W(8), wtable(E, H)},
             {W(9), wtable(VOC_OUT, H)}}) {
        check(s.set_weight(id, std::move(tbl)).ok(), "set weight");
    }

    check(ReferenceExecutor::run(g, s).ok(), "reference run ok");

    // ---- independent expected computation (double) ----
    std::vector<double> h(H);
    for (std::size_t i = 0; i < H; ++i) h[i] = wval(5, i);  // token 5

    double sum_sq = 0.0;
    for (double v : h) sum_sq += v * v;
    const double inv_rms = 1.0 / std::sqrt(sum_sq / H + EPS);
    std::vector<double> nrm(H), rope(H);
    for (std::size_t i = 0; i < H; ++i) nrm[i] = h[i] * inv_rms * wval(0, i);  // gamma row0

    const std::size_t half = H / 2;
    rope = nrm;
    for (std::size_t i = 0; i < half; ++i) {
        const std::size_t lo = i, hi = i + half;
        rope[lo] = nrm[lo] * cosd[lo] - nrm[hi] * sind[lo];
        rope[hi] = nrm[hi] * cosd[hi] + nrm[lo] * sind[hi];
    }

    // single-row attention -> softmax over one score = 1 -> out = vnew
    std::vector<double> attn = vnewd;

    std::vector<double> h2(H);
    for (std::size_t i = 0; i < H; ++i) h2[i] = nrm[i] + attn[i];

    std::vector<double> gg = linear_ref(h2, F, H), sil(F), u = linear_ref(h2, F, H), m(F);
    for (std::size_t i = 0; i < F; ++i) sil[i] = gg[i] / (1.0 + std::exp(-gg[i]));
    for (std::size_t i = 0; i < F; ++i) m[i] = sil[i] * u[i];

    std::vector<double> f = linear_ref(m, H, F), h3(H);
    for (std::size_t i = 0; i < H; ++i) h3[i] = h2[i] + f[i];

    std::vector<double> logits = linear_ref(h3, E, H);
    double mx = logits[0], ssum = 0.0;
    for (double v : logits) mx = std::max(mx, v);
    std::vector<double> probs(E);
    for (std::size_t e = 0; e < E; ++e) { probs[e] = std::exp(logits[e] - mx); ssum += probs[e]; }
    for (double& p : probs) p /= ssum;

    std::vector<double> moe(H, 0.0), finalv(VOC_OUT);
    for (std::size_t e = 0; e < E; ++e) {
        for (std::size_t hh = 0; hh < H; ++hh) moe[hh] += probs[e] * eod[e * H + hh];
    }
    finalv = linear_ref(h3, VOC_OUT, H);

    // ---- compare executor values ----
    check_close(s.value(V(10)), h, 1e-6, "embedding out");
    check_close(s.value(V(11)), nrm, 1e-5, "rmsnorm out");
    check_close(s.value(V(12)), rope, 1e-5, "rope out");
    check_close(s.value(V(13)), attn, 1e-6, "attention out (single row = v)");
    check_close(s.value(V(14)), h2, 1e-5, "residual h2");
    check_close(s.value(V(18)), m, 1e-5, "swiglu m");
    check_close(s.value(V(20)), h3, 1e-5, "residual h3");
    check_close(s.value(V(22)), probs, 1e-5, "moe probs");
    check_close(s.value(V(24)), moe, 1e-5, "moe merge");
    check_close(s.value(V(25)), finalv, 1e-5, "final logits");

    // ---- State golden: row 0 K/V written, rest zero ----
    const std::vector<float>* kv = s.state_const(StateId::from_uint32(0));
    bool state_ok = kv && kv->size() == 2 * CAP * D;
    for (std::size_t d = 0; state_ok && d < D; ++d) {
        state_ok = (*kv)[d] == knew[d] && (*kv)[CAP * D + d] == vnew[d];
    }
    for (std::size_t r = 1; state_ok && r < CAP; ++r) {
        for (std::size_t d = 0; d < D; ++d) {
            if ((*kv)[r * D + d] != 0.0f || (*kv)[CAP * D + r * D + d] != 0.0f) { state_ok = false; break; }
        }
    }
    check(state_ok, "KV state golden row0 K/V");

    // overwrite guards (SSA / state single init)
    check(!s.set_value(V(10), std::vector<float>(H, 0.0f)).ok(), "value overwrite rejected");
    check(!s.init_state(StateId::from_uint32(0), std::vector<float>(2 * CAP * D, 0.0f)).ok(),
          "state reinit rejected");
}

// ---------------------------------------------------------------------------
// two-step attention: state continuity + capacity guard
// ---------------------------------------------------------------------------
Graph build_two_step(std::size_t cap) {
    Graph g;
    for (std::uint32_t i = 0; i < 6; ++i) g.declare_input(vd(i, {D}));  // q1,k1,v1,q2,k2,v2
    g.declare_state(StateDesc{StateId::from_uint32(0), StateKind::kv_cache,
                               TensorShape::create({cap, D}).value(), DataType::f32});
    {
        Primitive a;
        a.kind = Kind::attention_decode;
        a.inputs = {V(0), V(1), V(2)};
        a.state_read = {StateId::from_uint32(0)};
        a.state_write = {StateId::from_uint32(0)};
        a.output = vd(10, {D});
        g.add_primitive(a);
    }
    {
        Primitive a;
        a.kind = Kind::attention_decode;
        a.inputs = {V(3), V(4), V(5)};
        a.state_read = {StateId::from_uint32(0)};
        a.state_write = {StateId::from_uint32(0)};
        a.output = vd(11, {D});
        g.add_primitive(a);
    }
    return g;
}

void test_two_step_state() {
    const Graph g = build_two_step(CAP);
    HostStorage s;
    std::vector<double> q1(D), k1(D), v1(D), q2(D), k2(D), v2(D);
    for (std::size_t d = 0; d < D; ++d) {
        q1[d] = 0.5 * d - 0.5; k1[d] = 1.0 + 0.25 * d; v1[d] = 2.0 - 0.1 * d;
        q2[d] = 0.1 * d + 0.7; k2[d] = 0.5 - 0.3 * d; v2[d] = 1.0 + 0.5 * d;
    }
    auto sf = [](std::vector<double> x) {
        std::vector<float> y(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
        return y;
    };
    check(s.set_value(V(0), sf(q1)).ok() && s.set_value(V(1), sf(k1)).ok() &&
              s.set_value(V(2), sf(v1)).ok() && s.set_value(V(3), sf(q2)).ok() &&
              s.set_value(V(4), sf(k2)).ok() && s.set_value(V(5), sf(v2)).ok(),
          "set two-step inputs");
    check(s.init_state(StateId::from_uint32(0), std::vector<float>(2 * CAP * D, 0.0f)).ok(),
          "state init");

    check(ReferenceExecutor::run(g, s).ok(), "two-step run ok");

    // step 1: single row -> out = v1
    check_close(s.value(V(10)), v1, 1e-6, "attn step1");

    // step 2: softmax over two rows
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    std::vector<double> sc(2);
    sc[0] = 0.0; sc[1] = 0.0;
    for (std::size_t d = 0; d < D; ++d) {
        sc[0] += q2[d] * k1[d];
        sc[1] += q2[d] * k2[d];
    }
    sc[0] *= scale; sc[1] *= scale;
    const double mx = std::max(sc[0], sc[1]);
    const double e0 = std::exp(sc[0] - mx), e1 = std::exp(sc[1] - mx);
    std::vector<double> want(D);
    for (std::size_t d = 0; d < D; ++d) want[d] = (e0 * v1[d] + e1 * v2[d]) / (e0 + e1);
    check_close(s.value(V(11)), want, 1e-5, "attn step2 softmax over 2 rows");

    // capacity guard: cap=1 -> second decode must fail
    const Graph gcap = build_two_step(1);
    HostStorage s2;
    s2.set_value(V(0), sf(q1)); s2.set_value(V(1), sf(k1)); s2.set_value(V(2), sf(v1));
    s2.set_value(V(3), sf(q2)); s2.set_value(V(4), sf(k2)); s2.set_value(V(5), sf(v2));
    s2.init_state(StateId::from_uint32(0), std::vector<float>(2 * 1 * D, 0.0f));
    auto run2 = ReferenceExecutor::run(gcap, s2);
    check(run2.code() == Status::Code::invalid_state, "KV capacity exceeded detected");
}

void test_missing_input() {
    const Graph g = build_two_step(CAP);
    HostStorage s;
    s.init_state(StateId::from_uint32(0), std::vector<float>(2 * CAP * D, 0.0f));
    auto st = ReferenceExecutor::run(g, s);
    check(!st.ok() && st.code() == Status::Code::invalid_state, "missing inputs -> error");
}

} // namespace

int main() {
    test_single_layer_golden();
    test_two_step_state();
    test_missing_input();
    std::printf("test_reference_executor: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
