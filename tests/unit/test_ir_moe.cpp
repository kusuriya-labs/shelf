// shelf unit test — MoE ブロックの契約（T17 / フェーズ 2）。
//
// 対象:
//   * config.json の MoE 欄（num_experts / top-k / 共有エキスパート / mlp_only_layers）
//   * ir::Kind::moe_mlp の形状検証（重みの並びと top_k）
//   * CPU リファレンスの top-k ルーティング + 共有エキスパートが、
//     独立に書いた double の参照計算と一致すること
//   * step_graph が MoE 層だけ moe_mlp を 1 ノードで組み、dense 層は従来どおり
//     3 本の linear で組むこと
//
// GPU もモデルも使わない（合成の重みだけで閉じる）。
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <shelf/backend/cpu/reference_executor.hpp>
#include <shelf/ir/graph.hpp>
#include <shelf/model/qwen35/step_graph.hpp>
#include <shelf/model/qwen35/text_config.hpp>

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

ValueDesc vd(std::uint32_t id, std::initializer_list<std::size_t> dims) {
    return ValueDesc{ValueId::from_uint32(id), TensorShape::create(dims).value(), DataType::f32};
}

auto V = [](std::uint32_t i) { return ValueId::from_uint32(i); };
auto W = [](std::uint32_t i) { return TensorId::parse(i).value(); };
auto S = [](std::initializer_list<std::size_t> d) { return TensorShape::create(d).value(); };

// ---- 合成の数値（決定的・同値を避ける）----
double wv(std::uint64_t seed, std::uint64_t i, std::uint64_t j) {
    return static_cast<double>((seed * 5 + i * 29 + j * 13) % 17) * 0.0625 - 0.5;
}
std::vector<float> wtable(std::uint64_t seed, std::size_t rows, std::size_t cols) {
    std::vector<float> w(rows * cols);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            w[i * cols + j] = static_cast<float>(wv(seed, i, j));
    return w;
}

constexpr std::size_t H = 4, E = 3, I = 2, IS = 2, R = 2, K = 2;

// 独立参照（double）。moe_mlp の意味論をそのまま書き下す。
std::vector<double> moe_expect(const std::vector<double>& x, bool shared) {
    const auto router = [](std::size_t e, std::size_t h) { return wv(1, e, h); };
    const auto xg = [&x](std::size_t r, std::size_t h) { return x[r * H + h]; };
    std::vector<double> lg(E);
    std::vector<double> out(R * H, 0.0);
    std::vector<double> p(E), pr(E);
    for (std::size_t r = 0; r < R; ++r) {
        double mx = -1e30;
        for (std::size_t e = 0; e < E; ++e) {
            double a = 0.0;
            for (std::size_t h = 0; h < H; ++h) a += xg(r, h) * router(e, h);
            lg[e] = a;
            if (a > mx) mx = a;
        }
        double sum = 0.0;
        for (std::size_t e = 0; e < E; ++e) {
            p[e] = std::exp(lg[e] - mx);
            sum += p[e];
        }
        std::vector<std::size_t> idx;
        std::vector<double> w;
        std::vector<char> used(E, 0);
        double keep = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            std::size_t best = E;
            for (std::size_t e = 0; e < E; ++e) {
                if (used[e]) continue;
                if (best == E || p[e] > p[best]) best = e;
            }
            used[best] = 1;
            idx.push_back(best);
            w.push_back(p[best] / sum);
            keep += p[best] / sum;
        }
        for (auto& v : w) v /= keep;
        for (std::size_t j = 0; j < K; ++j) {
            const std::size_t e = idx[j];
            std::vector<double> mid(I);
            for (std::size_t i = 0; i < I; ++i) {
                double a = 0.0, b = 0.0;
                for (std::size_t h = 0; h < H; ++h) {
                    a += xg(r, h) * wv(10 + e * 3 + 0, i, h);      // gate
                    b += xg(r, h) * wv(10 + e * 3 + 1, i, h);      // up
                }
                mid[i] = (a / (1.0 + std::exp(-a))) * b;
            }
            for (std::size_t h = 0; h < H; ++h) {
                double acc = 0.0;
                for (std::size_t i = 0; i < I; ++i) acc += mid[i] * wv(10 + e * 3 + 2, h, i);
                out[r * H + h] += w[j] * acc;
            }
        }
        // 共有エキスパート
        if (!shared) continue;
        double gv = 0.0;
        for (std::size_t h = 0; h < H; ++h) gv += xg(r, h) * wv(30, 0, h);
        const double sg = 1.0 / (1.0 + std::exp(-gv));
        std::vector<double> smid(IS);
        for (std::size_t i = 0; i < IS; ++i) {
            double a = 0.0, b = 0.0;
            for (std::size_t h = 0; h < H; ++h) {
                a += xg(r, h) * wv(31, i, h);
                b += xg(r, h) * wv(32, i, h);
            }
            smid[i] = (a / (1.0 + std::exp(-a))) * b;
        }
        for (std::size_t h = 0; h < H; ++h) {
            double acc = 0.0;
            for (std::size_t i = 0; i < IS; ++i) acc += smid[i] * wv(33, h, i);
            out[r * H + h] += sg * acc;
        }
    }
    return out;
}

Graph build_moe_graph(bool shared, std::int64_t top_k) {
    Graph g;
    g.declare_input(vd(1, {R, H}));
    g.declare_weight(W(1), S({E, H}));   // router
    std::uint32_t wid = 10;
    for (std::size_t e = 0; e < E; ++e) {
        g.declare_weight(W(wid++), S({I, H}));   // gate
        g.declare_weight(W(wid++), S({I, H}));   // up
        g.declare_weight(W(wid++), S({H, I}));   // down
    }
    if (shared) {
        g.declare_weight(W(31), S({IS, H}));
        g.declare_weight(W(32), S({IS, H}));
        g.declare_weight(W(33), S({H, IS}));
        g.declare_weight(W(30), S({1, H}));
    }
    Primitive p;
    p.kind = Kind::moe_mlp;
    p.attr.top_k = top_k;
    p.inputs = {V(1)};
    p.weights = {W(1)};
    for (std::size_t e = 0; e < E; ++e)
        for (std::uint32_t k = 0; k < 3; ++k) p.weights.push_back(W(10 + e * 3 + k));
    if (shared) {
        p.weights.push_back(W(31));
        p.weights.push_back(W(32));
        p.weights.push_back(W(33));
        p.weights.push_back(W(30));
    }
    p.output = vd(2, {R, H});
    const Status st = g.add_primitive(std::move(p));
    if (!st.ok()) std::printf("graph: %s\n", st.message().c_str());
    return g;
}

void bind_weights(HostStorage& st, bool shared) {
    st.set_weight(W(1), wtable(1, E, H));
    std::uint32_t wid = 10;
    for (std::size_t e = 0; e < E; ++e) {
        st.set_weight(W(wid++), wtable(10 + e * 3 + 0, I, H));
        st.set_weight(W(wid++), wtable(10 + e * 3 + 1, I, H));
        st.set_weight(W(wid++), wtable(10 + e * 3 + 2, H, I));
    }
    if (shared) {
        st.set_weight(W(31), wtable(31, IS, H));
        st.set_weight(W(32), wtable(32, IS, H));
        st.set_weight(W(33), wtable(33, H, IS));
        st.set_weight(W(30), wtable(30, 1, H));
    }
}

void test_reference_moe(bool shared) {
    Graph g = build_moe_graph(shared, static_cast<std::int64_t>(K));
    HostStorage st;
    bind_weights(st, shared);
    std::vector<double> xd(R * H);
    for (std::size_t r = 0; r < R; ++r)
        for (std::size_t h = 0; h < H; ++h) xd[r * H + h] = wv(99, r, h);
    std::vector<float> xf(xd.begin(), xd.end());
    st.set_value(V(1), xf);

    ReferenceExecutor ex;
    const Status rs = ex.run(g, st);
    check(rs.ok(), "moe_mlp graph runs on the CPU reference");
    if (!rs.ok()) {
        std::printf("  run: %s\n", rs.message().c_str());
        return;
    }
    const auto* got = st.value(V(2));
    const std::vector<double> want = moe_expect(xd, shared);
    bool ok = got != nullptr && got->size() == want.size();
    for (std::size_t i = 0; ok && i < want.size(); ++i)
        ok = std::fabs(static_cast<double>((*got)[i]) - want[i]) <= 2e-5;
    if (!ok) {
        ++failures;
        std::printf("FAIL: moe_mlp reference value (shared=%d)\n", shared ? 1 : 0);
        if (got) {
            for (std::size_t i = 0; i < got->size(); ++i)
                std::printf("  [%zu] got=%.6f want=%.6f\n", i, (*got)[i], want[i]);
        }
    }

    // top-k を 1 に落とすと「最上位エキスパートだけ」の結果になる（再正規化で重み 1）。
    Graph g1 = build_moe_graph(shared, 1);
    HostStorage st1;
    bind_weights(st1, shared);
    st1.set_value(V(1), xf);
    ReferenceExecutor ex1;
    check(ex1.run(g1, st1).ok(), "moe_mlp top_k=1 runs");
}

void test_validation_rejects() {
    // top_k = 0 / E 超過
    for (const std::int64_t bad : {std::int64_t{0}, std::int64_t{E + 1}}) {
        Graph g = build_moe_graph(true, bad);
        check(g.size() == 0, "top_k out of range is rejected");
    }
    // エキスパートの down 形状が [H,I] でない
    {
        Graph g;
        g.declare_input(vd(1, {R, H}));
        g.declare_weight(W(1), S({E, H}));
        std::uint32_t wid = 10;
        for (std::size_t e = 0; e < E; ++e) {
            g.declare_weight(W(wid++), S({I, H}));
            g.declare_weight(W(wid++), S({I, H}));
            g.declare_weight(W(wid++), S({I, H}));   // わざと誤り
        }
        Primitive p;
        p.kind = Kind::moe_mlp;
        p.attr.top_k = 1;
        p.inputs = {V(1)};
        p.weights = {W(1)};
        for (std::size_t e = 0; e < E; ++e)
            for (std::uint32_t k = 0; k < 3; ++k) p.weights.push_back(W(10 + e * 3 + k));
        p.output = vd(2, {R, H});
        check(!g.add_primitive(std::move(p)).ok(), "expert down shape is validated");
    }
}

// ---- config / step_graph ----
std::string moe_cfg_json() {
    return R"({
  "text_config": {
    "hidden_size": 8,
    "num_hidden_layers": 2,
    "vocab_size": 16,
    "num_attention_heads": 1,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "intermediate_size": 8,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 4,
    "shared_expert_intermediate_size": 4,
    "mlp_only_layers": [1],
    "rms_norm_eps": 1e-6,
    "eos_token_id": 1,
    "rope_parameters": {"rope_theta": 10000.0},
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 1,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "layer_types": ["full_attention", "full_attention"]
  }
})";
}

void test_config_and_lowering() {
    auto cfg_r = model::qwen35::load_text_config_json(moe_cfg_json());
    check(cfg_r.ok(), "MoE config parses");
    if (!cfg_r.ok()) {
        std::printf("  config: %s\n", cfg_r.status().message().c_str());
        return;
    }
    const auto& cfg = cfg_r.value();
    check(cfg.has_moe(), "has_moe");
    check(cfg.num_experts == 4 && cfg.num_experts_per_tok == 2, "expert counts");
    check(cfg.is_moe_layer(0) && !cfg.is_moe_layer(1), "mlp_only_layers selects dense layers");
    check(cfg.has_shared_expert(), "shared expert present");

    auto plan_r = model::qwen35::build_step_graph(cfg, /*kv_capacity*/ 8);
    check(plan_r.ok(), "MoE step graph builds");
    if (!plan_r.ok()) {
        std::printf("  plan: %s\n", plan_r.status().message().c_str());
        return;
    }
    const auto& plan = plan_r.value();
    std::size_t moe = 0, dense_down = 0;
    std::string router_name;
    for (const auto& p : plan.graph.primitives()) {
        if (p.kind == Kind::moe_mlp) {
            ++moe;
            check(p.weights.size() == 3u * 4u + 1u + 4u, "moe_mlp weight count");
            check(p.attr.top_k == 2, "moe_mlp carries top_k");
            check(p.inputs.size() == 1 && p.output.shape.rank() == 2, "moe_mlp arity");
            router_name = plan.weights.front().checkpoint_name;
        }
        if (p.kind == Kind::linear && p.weights.size() == 1) {
            for (const auto& w : plan.weights)
                if (w.id == p.weights[0] &&
                    w.checkpoint_name.find("layers.1.mlp.down_proj.weight") != std::string::npos)
                    ++dense_down;
        }
    }
    check(moe == 1, "exactly one moe_mlp (layer 0)");
    check(dense_down == 1, "layer 1 keeps the dense MLP");

    bool router_ok = false, expert_ok = false, shared_ok = false;
    for (const auto& w : plan.weights) {
        if (w.checkpoint_name == "model.language_model.layers.0.mlp.gate.weight") {
            router_ok = w.shape.size() == 2 && w.shape[0] == 4 && w.shape[1] == 8;
        }
        if (w.checkpoint_name ==
            "model.language_model.layers.0.mlp.experts.3.down_proj.weight") {
            expert_ok = w.shape.size() == 2 && w.shape[0] == 8 && w.shape[1] == 4;
        }
        if (w.checkpoint_name ==
            "model.language_model.layers.0.mlp.shared_expert_gate.weight") {
            shared_ok = w.shape.size() == 2 && w.shape[0] == 1 && w.shape[1] == 8;
        }
    }
    check(router_ok, "router weight declared [E,H]");
    check(expert_ok, "expert down weight declared [H,I]");
    check(shared_ok, "shared expert gate declared [1,H]");
    (void)router_name;
}

// MoE 専用 config（dense の intermediate_size が無い）が受理されること。
void test_moe_only_config_without_dense_width() {
    std::string j = moe_cfg_json();
    const std::string only = "\"mlp_only_layers\": [1],";
    check(j.find(only) != std::string::npos, "fixture has mlp_only_layers");
    j.replace(j.find(only), only.size(), std::string("\"mlp_only_layers\": [],"));
    const std::string dense_width = "\"intermediate_size\": 8,\n    ";
    check(j.find(dense_width) != std::string::npos, "fixture has intermediate_size");
    j.replace(j.find(dense_width), dense_width.size(), std::string());
    auto cfg = model::qwen35::load_text_config_json(j);
    check(cfg.ok(), "MoE-only config without intermediate_size parses");
    check(cfg.ok() && cfg.value().has_moe() && cfg.value().is_moe_layer(0) &&
              cfg.value().is_moe_layer(1),
          "all layers are MoE");
    if (!cfg.ok()) std::printf("  config: %s\n", cfg.status().message().c_str());
}

} // namespace

int main() {
    test_reference_moe(/*shared=*/true);
    test_reference_moe(/*shared=*/false);
    test_validation_rejects();
    test_config_and_lowering();
    test_moe_only_config_without_dense_width();
    std::printf("test_ir_moe: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
