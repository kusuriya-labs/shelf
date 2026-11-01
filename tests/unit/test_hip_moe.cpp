// shelf unit test — HIP MoE（T17 / フェーズ 2）の数値一致。
//
// 合成の小さな MoE ブロック（router + 4 エキスパート + 共有エキスパート）を
// HIP 実行器と CPU リファレンスで走らせ、bf16 活性化の丸めの範囲で一致する
// ことを確認する。ルーティング（top-k）が両者で同じ選択になることも、
// 重みを「片方のエキスパートだけ大きい」形にして間接的に確かめる。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <shelf/backend/cpu/reference_executor.hpp>
#include <shelf/backend/hip/step_executor.hpp>
#include <shelf/ir/graph.hpp>

namespace {

using namespace shelf;
using namespace shelf::backend;
using namespace shelf::ir;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

ValueDesc vd(std::uint32_t id, std::initializer_list<std::size_t> dims,
             DataType dt = DataType::f32) {
    return ValueDesc{ValueId::from_uint32(id), TensorShape::create(dims).value(), dt};
}
auto V = [](std::uint32_t i) { return ValueId::from_uint32(i); };
auto W = [](std::uint32_t i) { return TensorId::parse(i).value(); };
auto S = [](std::initializer_list<std::size_t> d) { return TensorShape::create(d).value(); };

constexpr std::size_t H = 8, I = 4, IS = 4, E = 4, R = 2, K = 2;

// f32 → bf16 → f32（HIP 側の upload と同じ丸めを CPU 参照にも与える）。
float rbf(float v) {
    const auto* p = reinterpret_cast<const std::uint32_t*>(&v);
    std::uint32_t u = *p;
    u = (u + 0x8000u + ((u >> 16) & 1u)) >> 16;
    const std::uint32_t bits = u << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<float> table(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> w(n);
    for (auto& v : w) v = rbf(nd(rng));
    return w;
}

Graph build_rows(std::vector<std::vector<float>>& weights, std::vector<TensorId>& ids,
                 std::size_t rows) {
    Graph g;
    g.declare_input(vd(1, {rows}, DataType::i32));
    auto add_w = [&](std::uint32_t id, std::vector<std::size_t> shape,
                     std::vector<float> data) {
        g.declare_weight(W(id), TensorShape::create(shape).value());
        ids.push_back(W(id));
        weights.push_back(std::move(data));
    };
    // 埋め込み表（入力 x の行をそのまま作る）
    add_w(1, {rows, H}, {});
    // router
    add_w(2, {E, H}, {});
    // experts: e ごとに gate/up/down
    std::uint32_t wid = 10;
    for (std::size_t e = 0; e < E; ++e) {
        add_w(wid++, {I, H}, {});
        add_w(wid++, {I, H}, {});
        add_w(wid++, {H, I}, {});
    }
    // shared expert + gate vector
    add_w(31, {IS, H}, {});
    add_w(32, {IS, H}, {});
    add_w(33, {H, IS}, {});
    add_w(30, {1, H}, {});

    Primitive emb;
    emb.kind = Kind::embedding;
    emb.inputs = {V(1)};
    emb.weights = {W(1)};
    emb.output = vd(2, {rows, H});
    check(g.add_primitive(std::move(emb)).ok(), "embedding added");

    Primitive p;
    p.kind = Kind::moe_mlp;
    p.attr.top_k = static_cast<std::int64_t>(K);
    p.inputs = {V(2)};
    p.weights = {W(2)};
    for (std::size_t e = 0; e < E; ++e)
        for (std::uint32_t k = 0; k < 3; ++k) p.weights.push_back(W(10 + e * 3 + k));
    p.weights.push_back(W(31));
    p.weights.push_back(W(32));
    p.weights.push_back(W(33));
    p.weights.push_back(W(30));
    p.output = vd(3, {rows, H});
    check(g.add_primitive(std::move(p)).ok(), "moe_mlp added");
    return g;
}

Graph build(std::vector<std::vector<float>>& weights, std::vector<TensorId>& ids) {
    return build_rows(weights, ids, R);
}

} // namespace

int main() {
    std::mt19937 rng(1234);
    std::vector<std::vector<float>> weights;
    std::vector<TensorId> ids;
    Graph g = build(weights, ids);
    if (failures != 0) return 1;

    // 入力 x（2 行）と、router の重みを「行ごとに得意なエキスパート」が
    // 変わるように作る（top-k の選択が固定値にならない）。
    std::vector<float> x = table(rng, R * H, 0.5f);
    std::vector<float> router = table(rng, E * H, 0.5f);
    for (std::size_t r = 0; r < R; ++r) {
        const std::size_t favorite = r % E;
        for (std::size_t h = 0; h < H; ++h) router[favorite * H + h] += x[r * H + h];
    }
    // weights[0] は埋め込み表、weights[1] は router
    weights[0] = x;
    weights[1] = router;
    for (std::size_t i = 2; i < weights.size(); ++i) {
        const auto* sh = g.find_weight(ids[i]);
        const std::size_t n = sh->num_elements().value();
        weights[i] = table(rng, n, 0.3f);
    }

    // ---- CPU 参照 ----
    backend::cpu::HostStorage st;
    for (std::size_t i = 0; i < ids.size(); ++i) st.set_weight(ids[i], weights[i]);
    st.set_value_int(V(1), {0, 1});
    backend::cpu::ReferenceExecutor ref;
    const Status rs = ref.run(g, st);
    check(rs.ok(), "cpu reference runs");
    if (!rs.ok()) {
        std::printf("  %s\n", rs.message().c_str());
        return 1;
    }
    const auto* want = st.value(V(3));

    // ---- HIP ----
    backend::hip::HipStepExecutorOptions ho;
    ho.arena_bytes = 4u << 20;
    ho.rope_theta = 1.0e7;
    ho.rope_rotary_dim = 8;
    backend::hip::HipStepExecutor ex(ho);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const Status us = ex.upload_weight(ids[i], weights[i]);
        if (!us.ok()) {
            std::printf("upload: %s\n", us.message().c_str());
            check(false, "upload weights");
            return 1;
        }
    }
    const auto ds = ex.declare_states(g);
    check(ds.ok(), "declare states");
    if (!ds.ok()) {
        std::printf("  %s\n", ds.message().c_str());
        return 1;
    }
    const std::int32_t toks[R] = {0, 1};
    const auto run = ex.run_step(g, toks, R, 0);
    check(run.ok(), "hip moe step");
    if (!run.ok()) {
        std::printf("  %s\n", run.message().c_str());
        return 1;
    }
    auto got = ex.fetch_value_debug(g, V(3));
    check(got.ok(), "read moe output");
    if (!got.ok() || want == nullptr) return 1;

    check(got.value().size() == want->size(), "output size");
    double worst = 0.0;
    std::size_t worst_i = 0;
    for (std::size_t i = 0; i < got.value().size() && i < want->size(); ++i) {
        const double d = std::fabs(static_cast<double>(got.value()[i]) -
                                   static_cast<double>((*want)[i]));
        if (d > worst) {
            worst = d;
            worst_i = i;
        }
    }
    std::printf("hip moe: worst|diff|=%.5g at %zu (cpu=%.6f hip=%.6f)\n", worst, worst_i,
                (*want)[worst_i], got.value()[worst_i]);
    // bf16 活性化（入力・エキスパート出力・共有エキスパート）と f32 累積の差。
    // 値のスケールは ~0.1 なので 1e-3 は十分厳しい（ルーティング取り違えなら桁で外れる）。
    check(worst < 1e-3, "hip moe matches cpu reference");

    // ---- pinned host のエキスパートを CPU で計算する経路（T17 フェーズ 3）----
    {
        backend::hip::HipStepExecutorOptions ho2 = ho;
        backend::hip::HipStepExecutor ex2(ho2);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const std::uint32_t v = ids[i].value();
            const bool expert = v >= 10u && v < 10u + 3u * E;
            ex2.set_weight_placement(expert);
            const Status us = ex2.upload_weight(ids[i], weights[i]);
            if (!us.ok()) {
                std::printf("upload2: %s\n", us.message().c_str());
                check(false, "upload pinned weights");
                return 1;
            }
        }
        ex2.set_weight_placement(false);
        check(ex2.host_weight_bytes() > 0, "experts are pinned on host");
        const auto ds2 = ex2.declare_states(g);
        check(ds2.ok(), "declare states (cpu experts)");
        const auto run_cpu = ex2.run_step(g, toks, R, 0);
        check(run_cpu.ok(), "hip moe step (cpu experts)");
        if (run_cpu.ok()) {
            auto got_cpu = ex2.fetch_value_debug(g, V(3));
            check(got_cpu.ok(), "read moe output (cpu experts)");
            if (got_cpu.ok() && want != nullptr) {
                double worst_cpu = 0.0;
                for (std::size_t i = 0; i < got_cpu.value().size(); ++i)
                    worst_cpu = std::max(
                        worst_cpu,
                        std::fabs(static_cast<double>(got_cpu.value()[i]) -
                                  static_cast<double>((*want)[i])));
                std::printf("hip moe (cpu experts): worst|diff|=%.5g cpu=%zu gpu=%zu\n",
                            worst_cpu, ex2.moe_cpu_experts(), ex2.moe_gpu_experts());
                check(worst_cpu < 1e-3, "cpu experts match cpu reference");
                check(ex2.moe_cpu_experts() > 0, "some experts computed on cpu");
                check(ex2.moe_gpu_experts() == 0, "no expert computed on gpu");
            }
        }
    }

    // ---- 縮退: VRAM 確保の失敗を pinned host で吸収して継続する（T17 フェーズ 6）----
    {
        setenv("SHELF_TEST_FAIL_DEV_ALLOC", "1000000", 1);
        backend::hip::HipStepExecutorOptions ho3 = ho;
        ho3.allow_host_fallback = true;
        backend::hip::HipStepExecutor ex3(ho3);
        unsetenv("SHELF_TEST_FAIL_DEV_ALLOC");
        bool upload_ok = true;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const Status us = ex3.upload_weight(ids[i], weights[i]);
            if (!us.ok()) {
                std::printf("upload3: %s\n", us.message().c_str());
                upload_ok = false;
            }
        }
        check(upload_ok, "device allocation failure falls back to pinned host");
        check(ex3.host_weight_bytes() > 0, "fallback placed weights on host");
        const auto ds3 = ex3.declare_states(g);
        check(ds3.ok(), "declare states after fallback");
        const auto run3 = ex3.run_step(g, toks, R, 0);
        check(run3.ok(), "hip moe step after fallback");
        if (run3.ok()) {
            auto got3 = ex3.fetch_value_debug(g, V(3));
            check(got3.ok(), "read moe output after fallback");
            if (got3.ok() && want != nullptr) {
                double worst3 = 0.0;
                for (std::size_t i = 0; i < got3.value().size(); ++i)
                    worst3 = std::max(
                        worst3, std::fabs(static_cast<double>(got3.value()[i]) -
                                          static_cast<double>((*want)[i])));
                std::printf("hip moe (fallback): worst|diff|=%.5g cpu=%zu gpu=%zu\n",
                            worst3, ex3.moe_cpu_experts(), ex3.moe_gpu_experts());
                check(worst3 < 1e-3, "fallback path matches cpu reference");
            }
        }
    }

    // 決定性: 同じ入力を 2 回目に流しても同じ値（アリーナの残骸を読まない）。
    check(ex.reset_states().ok(), "reset");
    const auto run2 = ex.run_step(g, toks, R, 0);
    check(run2.ok(), "hip moe step 2");
    auto got2 = ex.fetch_value_debug(g, V(3));
    check(got2.ok(), "read moe output 2");
    if (got2.ok() && want != nullptr) {
        double d2 = 0.0;
        for (std::size_t i = 0; i < got2.value().size(); ++i)
            d2 = std::max(d2, std::fabs(static_cast<double>(got2.value()[i]) -
                                        static_cast<double>(got.value()[i])));
        check(d2 < 1e-6, "deterministic across runs");
    }

    std::printf("test_hip_moe: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
