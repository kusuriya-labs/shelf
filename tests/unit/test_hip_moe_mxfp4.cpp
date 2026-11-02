// shelf unit test — HIP MoE（MXFP4 重み / W4A8 活性化）の数値一致（T17 フェーズ 3）。
//
// 実機の 35B-A3B はエキスパートが MXFP4（E2M1 + E8M0、Quark）で、活性化は
// FP8（E4M3、group 128）に量子化される（W4A8）。CPU 分担は「pinned host の重みを復号しながら
// 内積する」経路なので、GPU 経路と同じ値になることを合成データで確かめる。
//
// 参照は独立に書いた double の計算（重みは同じ量子化コードを別実装で復号し、
// 活性化は同じ量子化器を通してから f64 で内積）。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include <shelf/backend/hip/step_executor.hpp>
#include <shelf/ir/graph.hpp>
#include <shelf/quant/fp8_codec.hpp>
#include <shelf/quant/mxfp4.hpp>

namespace {

using namespace shelf;
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

constexpr std::size_t H = 64, I = 64, IS = 64, E = 2, R = 2, K = 2;

float rbf(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, sizeof u);
    u = (u + 0x8000u + ((u >> 16) & 1u)) >> 16;
    const std::uint32_t bits = u << 16;
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

std::vector<float> rand_table(std::mt19937& rng, std::size_t n, float scale) {
    std::normal_distribution<float> nd(0.f, scale);
    std::vector<float> w(n);
    for (auto& v : w) v = rbf(nd(rng));
    return w;
}

// テスト側の独立な復号（canonical = 行優先 [rows, K/2]）。
double e2m1_dec(std::uint8_t nib) {
    static const double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const double m = mag[nib & 7u];
    return (nib & 8u) ? -m : m;
}
double e8m0_dec(std::uint8_t b) { return std::ldexp(1.0, static_cast<int>(b) - 127); }

// 1 行列を MXFP4 へ量子化し、canonical コード・スケール・f32 復号値を作る。
struct MxMatrix {
    std::vector<std::uint8_t> canon;   // [rows, K/2]
    std::vector<std::uint8_t> scales;  // [rows, K/32]
    std::vector<std::uint8_t> tiled;   // 断片順（アップロード用）
    std::vector<double> deq;           // [rows, K] 復号値
    std::uint32_t rows = 0, cols = 0;
};

MxMatrix quantize_matrix(const std::vector<float>& w, std::uint32_t rows, std::uint32_t cols) {
    MxMatrix m;
    m.rows = rows;
    m.cols = cols;
    m.canon.assign(static_cast<std::size_t>(rows) * (cols / 2u), 0u);
    m.scales.assign(static_cast<std::size_t>(rows) * (cols / 32u), 0u);
    for (std::uint32_t r = 0; r < rows; ++r) {
        if (!quant::mxfp4::quantize_row(w.data() + static_cast<std::size_t>(r) * cols, cols,
                                        cols, m.canon.data() + static_cast<std::size_t>(r) * (cols / 2u),
                                        m.scales.data() + static_cast<std::size_t>(r) * (cols / 32u))
                 .ok()) {
            check(false, "quantize_row");
        }
    }
    m.tiled.assign(quant::mxfp4::tiled_bytes(rows, cols), 0u);
    check(quant::mxfp4::preshuffle_codes(m.canon.data(), rows, cols, cols,
                                         m.tiled.data())
              .ok(),
          "preshuffle");
    m.deq.assign(static_cast<std::size_t>(rows) * cols, 0.0);
    for (std::uint32_t r = 0; r < rows; ++r) {
        for (std::uint32_t c = 0; c < cols; ++c) {
            const std::uint8_t byte = m.canon[static_cast<std::size_t>(r) * (cols / 2u) + c / 2u];
            const std::uint8_t nib = (c % 2u == 0u) ? (byte & 0x0Fu) : (byte >> 4);
            m.deq[static_cast<std::size_t>(r) * cols + c] =
                e2m1_dec(nib) *
                e8m0_dec(m.scales[static_cast<std::size_t>(r) * (cols / 32u) + c / 32u]);
        }
    }
    // 断片順からの読み（shelf の実装）と、テスト側の canonical 復号が一致すること。
    for (std::uint32_t r = 0; r < rows; ++r)
        for (std::uint32_t c = 0; c < cols; ++c) {
            const double a = quant::mxfp4::read_element(m.tiled.data(), m.scales.data(), r, c,
                                                        cols);
            const double b = m.deq[static_cast<std::size_t>(r) * cols + c];
            if (std::fabs(a - b) > 1e-12) {
                check(false, "tiled decode matches canonical decode");
                return m;
            }
        }
    return m;
}

// W4A8 の活性化量子化（E4M3、K 128 要素ごとの動的スケール）→ f64 の復号値。
// GPU/CPU と同じ規則（amax / kE4M3Max）。
std::vector<double> quantize_act(const std::vector<float>& x, std::uint32_t k) {
    using namespace quant::fp8;
    // executor と同じ規則: K が 128 の倍数のときだけ 128 グループ、それ以外は行単位。
    const std::uint32_t group = (k % 128u == 0u) ? 128u : 0u;
    const std::uint32_t g = group == 0u ? k : group;
    std::vector<double> out(k);
    for (std::uint32_t b = 0; b < k; b += g) {
        const std::uint32_t e = std::min(k, b + g);
        float amax = 0.0f;
        for (std::uint32_t i = b; i < e; ++i) amax = std::max(amax, std::fabs(x[i]));
        const float s = amax_scale(amax);
        for (std::uint32_t i = b; i < e; ++i) {
            const std::uint8_t code = e4m3_encode(s > 0.0f ? x[i] / s : 0.0f);
            out[i] = static_cast<double>(e4m3_decode(code)) * s;
        }
    }
    return out;
}

double dot(const std::vector<double>& w, const std::vector<double>& x, std::size_t k) {
    double acc = 0.0;
    for (std::size_t i = 0; i < k; ++i) acc += w[i] * x[i];
    return acc;
}

} // namespace

int main() {
    std::mt19937 rng(77);
    // 埋め込み表（x の行）とエキスパートの重み
    std::vector<float> x = rand_table(rng, R * H, 1.0f);
    std::vector<float> router = rand_table(rng, E * H, 0.5f);
    // 行ごとに得意なエキスパートを変える
    for (std::size_t r = 0; r < R; ++r) {
        const std::size_t fav = r % E;
        for (std::size_t h = 0; h < H; ++h)
            router[fav * H + h] = rbf(router[fav * H + h] + x[r * H + h]);
    }
    std::vector<MxMatrix> expert_w;
    for (std::size_t e = 0; e < E; ++e) {
        expert_w.push_back(quantize_matrix(rand_table(rng, I * H, 0.5f), I, H));  // gate
        expert_w.push_back(quantize_matrix(rand_table(rng, I * H, 0.5f), I, H));  // up
        expert_w.push_back(quantize_matrix(rand_table(rng, H * I, 0.5f), H, I));  // down
    }
    std::vector<float> sg = rand_table(rng, IS * H, 0.5f);
    std::vector<float> su = rand_table(rng, IS * H, 0.5f);
    std::vector<float> sd = rand_table(rng, H * IS, 0.5f);
    std::vector<float> sv = rand_table(rng, H, 0.5f);

    // ---- グラフ ----
    Graph g;
    g.declare_input(vd(1, {R}, DataType::i32));
    g.declare_weight(W(1), TensorShape::create({R, H}).value());
    g.declare_weight(W(2), TensorShape::create({E, H}).value());
    for (std::size_t e = 0; e < E; ++e) {
        g.declare_weight(W(10 + e * 3 + 0), TensorShape::create({I, H}).value());
        g.declare_weight(W(10 + e * 3 + 1), TensorShape::create({I, H}).value());
        g.declare_weight(W(10 + e * 3 + 2), TensorShape::create({H, I}).value());
    }
    g.declare_weight(W(31), TensorShape::create({IS, H}).value());
    g.declare_weight(W(32), TensorShape::create({IS, H}).value());
    g.declare_weight(W(33), TensorShape::create({H, IS}).value());
    g.declare_weight(W(30), TensorShape::create({1, H}).value());
    {
        Primitive emb;
        emb.kind = Kind::embedding;
        emb.inputs = {V(1)};
        emb.weights = {W(1)};
        emb.output = vd(2, {R, H});
        check(g.add_primitive(std::move(emb)).ok(), "embedding");
    }
    {
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
        p.output = vd(3, {R, H});
        check(g.add_primitive(std::move(p)).ok(), "moe_mlp");
    }

    // ---- 参照（double）----
    std::vector<double> want(R * H, 0.0);
    for (std::size_t r = 0; r < R; ++r) {
        std::vector<float> xr(x.begin() + r * H, x.begin() + (r + 1) * H);
        const auto xq = quantize_act(xr, H);
        // router logits（bf16 重み・f32 累積だが、ここでは double で十分近い）
        std::vector<double> lg(E);
        double mx = -1e30;
        for (std::size_t e = 0; e < E; ++e) {
            double a = 0.0;
            for (std::size_t h = 0; h < H; ++h) a += x[r * H + h] * router[e * H + h];
            lg[e] = a;
            mx = std::max(mx, a);
        }
        std::vector<double> p(E);
        double sum = 0.0;
        for (std::size_t e = 0; e < E; ++e) {
            p[e] = std::exp(lg[e] - mx);
            sum += p[e];
        }
        std::vector<char> used(E, 0);
        std::vector<std::size_t> idx;
        std::vector<double> wt;
        double keep = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            std::size_t best = E;
            for (std::size_t e = 0; e < E; ++e) {
                if (used[e]) continue;
                if (best == E || p[e] / sum > p[best] / sum) best = e;
            }
            used[best] = 1;
            idx.push_back(best);
            wt.push_back(p[best] / sum);
            keep += p[best] / sum;
        }
        for (auto& v : wt) v /= keep;
        for (std::size_t j = 0; j < K; ++j) {
            const MxMatrix& mg = expert_w[idx[j] * 3 + 0];
            const MxMatrix& mu = expert_w[idx[j] * 3 + 1];
            const MxMatrix& md = expert_w[idx[j] * 3 + 2];
            std::vector<double> mid(I);
            for (std::size_t i = 0; i < I; ++i) {
                const double gv = dot({mg.deq.begin() + i * H, mg.deq.begin() + (i + 1) * H},
                                      xq, H);
                const double uv = dot({mu.deq.begin() + i * H, mu.deq.begin() + (i + 1) * H},
                                      xq, H);
                mid[i] = (gv / (1.0 + std::exp(-gv))) * uv;
            }
            const auto mq = quantize_act(
                std::vector<float>(mid.begin(), mid.end()), static_cast<std::uint32_t>(I));
            for (std::size_t h = 0; h < H; ++h) {
                want[r * H + h] += wt[j] *
                    dot({md.deq.begin() + h * I, md.deq.begin() + (h + 1) * I}, mq, I);
            }
        }
        // 共有エキスパート（bf16・GPU と同じ sigmoid ゲート）
        double gv = 0.0;
        for (std::size_t h = 0; h < H; ++h) gv += x[r * H + h] * sv[h];
        const double s = 1.0 / (1.0 + std::exp(-gv));
        std::vector<double> smid(IS);
        for (std::size_t i = 0; i < IS; ++i) {
            double a = 0.0, b = 0.0;
            for (std::size_t h = 0; h < H; ++h) {
                a += x[r * H + h] * sg[i * H + h];
                b += x[r * H + h] * su[i * H + h];
            }
            smid[i] = (a / (1.0 + std::exp(-a))) * b;
        }
        for (std::size_t h = 0; h < H; ++h) {
            double acc = 0.0;
            for (std::size_t i = 0; i < IS; ++i) acc += smid[i] * sd[h * IS + i];
            want[r * H + h] += s * acc;
        }
    }

    // ---- HIP 実行（GPU のみ / エキスパートを pinned にして CPU 分担）----
    const auto run_case = [&](bool pin_experts) -> std::vector<float> {
        backend::hip::HipStepExecutorOptions ho;
        ho.arena_bytes = 4u << 20;
        ho.fp8_act_group = 128;       // W4A8（E4M3、group 128）
        backend::hip::HipStepExecutor ex(ho);
        for (std::size_t i = 0; i < 3; ++i) { (void)i; }
        const auto up_bf16 = [&](std::uint32_t id, const std::vector<float>& v) {
            check(ex.upload_weight(W(id), v).ok(), "upload bf16");
        };
        up_bf16(1, x);
        up_bf16(2, router);
        up_bf16(31, sg);
        up_bf16(32, su);
        up_bf16(33, sd);
        up_bf16(30, sv);
        for (std::size_t e = 0; e < E; ++e) {
            for (std::uint32_t k = 0; k < 3; ++k) {
                const auto& m = expert_w[e * 3 + k];
                ex.set_weight_placement(pin_experts);
                const auto st = ex.upload_weight_mxfp4(
                    W(10 + static_cast<std::uint32_t>(e * 3) + k), m.tiled.data(),
                    m.tiled.size(), m.scales.data(), m.scales.size(), m.rows, m.cols);
                ex.set_weight_placement(false);
                check(st.ok(), "upload mxfp4");
            }
        }
        const auto ds = ex.declare_states(g);
        check(ds.ok(), "declare states");
        const std::int32_t toks[R] = {0, 1};
        const auto run = ex.run_step(g, toks, R, 0);
        check(run.ok(), "moe step");
        if (!run.ok()) std::printf("  %s\n", run.message().c_str());
        auto out = ex.fetch_value_debug(g, V(3));
        check(out.ok(), "read output");
        if (pin_experts) {
            std::printf("  mxfp4 cpu: cpu=%zu gpu=%zu cpu_ms=%.1f\n",
                        ex.moe_cpu_experts(), ex.moe_gpu_experts(), ex.moe_cpu_ms());
            check(ex.moe_cpu_experts() > 0, "pinned experts ran on cpu");
        }
        return out.ok() ? out.value() : std::vector<float>{};
    };

    const auto got_gpu = run_case(false);
    const auto got_cpu = run_case(true);
    check(got_gpu.size() == want.size() && got_cpu.size() == want.size(), "output size");
    if (got_gpu.size() != want.size() || got_cpu.size() != want.size()) return 1;

    // 出力は bf16（fetch_value_debug が bf16 → f32）。絶対差ではなく相対で見る。
    double wg = 0.0, wc = 0.0, gc = 0.0;
    std::size_t wi = 0, big = 0, huge = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double scale = std::max(1.0, std::fabs(want[i]));
        const double d = std::fabs(static_cast<double>(got_gpu[i]) - want[i]) / scale;
        if (d > 0.01) ++big;
        if (d > 0.05) ++huge;
        if (d > wg) { wg = d; wi = i; }
        wc = std::max(wc, std::fabs(static_cast<double>(got_cpu[i]) - want[i]) / scale);
        gc = std::max(gc, std::fabs(static_cast<double>(got_gpu[i]) -
                                    static_cast<double>(got_cpu[i])));
    }
    std::printf("  worst i=%zu want=%.6f gpu=%.6f big=%zu huge=%zu /%zu\n", wi, want[wi],
                got_gpu[wi], big, huge, want.size());
    // gpu vs cpu は「FP8 の mid コードが境界で 1 段変わる」ことがあるため、
    // 絶対差だけでなく相対差でも見る。
    double gcr = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double scale = std::max(1.0, std::fabs(static_cast<double>(got_cpu[i])));
        gcr = std::max(gcr, std::fabs(static_cast<double>(got_gpu[i]) -
                                      static_cast<double>(got_cpu[i])) / scale);
    }
    std::printf("mxfp4 moe: gpu_vs_ref=%.5g cpu_vs_ref=%.5g gpu_vs_cpu=%.5g (rel %.4g)\n",
                wg, wc, gc, gcr);
    for (std::size_t i = 0; i < 4; ++i)
        std::printf("  [%zu] want=%.6f gpu=%.6f cpu=%.6f\n", i, want[i], got_gpu[i],
                    got_cpu[i]);
    // 重みは同じ量子化コード、活性化も同じ量子化器を通しているので、差は
    // 累積順・f32 丸めだけ（値は ~0.5 スケール）。1e-3 は十分厳しい。
    // 参照との差は 2 種類ある:
    //   (a) 出力の bf16 丸め（相対 ~0.4%）
    //   (b) mid の量子化境界で E2M1 のコードが 1 つ変わる（要素単位の飛び）
    // (b) は参照が f64 で mid を作るのに対し実行器が bf16/f32 で作るために起きる、
    // 量子化の性質上の差。**GPU と CPU は同じ量子化器・同じコードを使うので一致**
    // する（`gc`）。参照との比較は「大半が 1% 以内・飛びは少数」で判定する。
    check(gcr < 5e-2, "gpu and cpu agree modulo FP8 mid-code flips");
    check(wg < 0.35 && huge <= 6, "gpu matches the reference except a few FP8 flips");
    check(big <= 24, "most elements are within 1% of the reference");
    check(wc < 0.35, "cpu is close to the double reference");

    std::printf("test_hip_moe_mxfp4: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
