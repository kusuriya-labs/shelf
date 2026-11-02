// HIP executor: batched prefill step must equal sequential decode steps
// (same kernels, same state updates) on a tiny synthetic Qwen35-family graph.
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <random>
#include <vector>

#include <shelf/backend/hip/step_executor.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

using namespace shelf;

static int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL: %s\n", msg);                                  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

// 失敗理由（Status の中身）も出す版。declare 相の offsets と実行器の方針が
// 食い違ったとき、どの primitive で止まったかが分かる。
#define CHECK_ST(cond, msg, st)                                              \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL: %s: %s\n", msg, (st).message().c_str());       \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

template <typename C> static std::size_t numelof(const C& c) {
    std::size_t n = 1;
    for (auto d : c) n *= d;
    return n;
}

int main(int argc, char** argv) {
    model::qwen35::TextConfig cfg;
    cfg.hidden_size = 512;
    cfg.num_layers = argc > 2 ? std::stoul(argv[2]) : 2;
    cfg.vocab_size = 1024;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 256;   // rope pair kernel constraint
    cfg.rotary_dim = 64;
    cfg.rope_theta = 1.0e7;
    cfg.lin_key_heads = 2;
    cfg.lin_value_heads = 4;
    cfg.lin_key_head_dim = 128;
    cfg.lin_value_head_dim = 128;
    cfg.conv_kernel = 4;
    cfg.intermediate_size = 1024;
    cfg.rms_eps = 1e-6;
    const bool lin_first = argc > 1 && std::string(argv[1]) == "lin";
    cfg.layer_kinds.clear();
    for (std::size_t l = 0; l < cfg.num_layers; ++l)
        cfg.layer_kinds.push_back(lin_first ? model::qwen35::LayerKind::linear_attention
                                            : model::qwen35::LayerKind::full_attention);
    cfg.tie_word_embeddings = false;
    cfg.eos_token_id = 0;

    auto p1r = model::qwen35::build_step_graph(cfg, 16, 1);
    CHECK(p1r.ok(), "decode graph");
    auto pbr = model::qwen35::build_step_graph(cfg, 16, 4);
    CHECK(pbr.ok(), "batch graph");
    if (!p1r.ok() || !pbr.ok()) return 1;
    auto& pd = p1r.value();
    auto& pb = pbr.value();

    backend::hip::HipStepExecutorOptions ho;
    ho.rope_theta = cfg.rope_theta;
    ho.rope_rotary_dim = cfg.rotary_dim;
    ho.arena_bytes = 16u << 20;
    ho.max_sequences = 2;   // T15: 2 スロットで状態が分離されることを確認する
    backend::hip::HipStepExecutor ex(ho);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.05f);
    for (const auto& w : pd.weights) {
        std::vector<float> f(numelof(w.shape));
        for (auto& x : f) x = nd(rng);
        CHECK(ex.upload_weight(w.id, f).ok(), "upload weight");
    }
    const auto st_decl = ex.declare_states(pd.graph);
    CHECK_ST(st_decl.ok(), "declare states", st_decl);

    std::vector<std::int32_t> toks = {7, 111, 512, 900};

    // ---- 同じ入力の prefill を 2 回（状態を戻して）回し、logits がビット一致すること ----
    {
        std::vector<float> first;
        for (int rep = 0; rep < 2; ++rep) {
            CHECK(ex.reset_states().ok(), "determinism reset");
            const auto st = ex.run_step(pb.graph, toks.data(), toks.size(), 0);
            CHECK_ST(st.ok(), "determinism prefill", st);
            auto lg = ex.fetch_logits(pb.graph);
            CHECK(lg.ok(), "determinism logits");
            if (!lg.ok()) break;
            if (rep == 0) {
                first = lg.value();
            } else {
                const bool same = first.size() == lg.value().size() &&
                                  std::memcmp(first.data(), lg.value().data(),
                                              first.size() * sizeof(float)) == 0;
                if (!same && first.size() == lg.value().size()) {
                    double w = 0.0;
                    std::size_t wi = 0;
                    for (std::size_t i = 0; i < first.size(); ++i) {
                        const double d = std::fabs(double(first[i]) - double(lg.value()[i]));
                        if (d > w) { w = d; wi = i; }
                    }
                    std::printf("prefill determinism: worst|diff|=%.3e at %zu (%.6f vs %.6f)\n",
                                w, wi, first[wi], lg.value()[wi]);
                }
                CHECK(same, "prefill は 2 回実行で logits がビット一致");
            }
        }
    }
    CHECK(ex.reset_states().ok(), "reset after determinism");

    // sequential
    for (std::size_t i = 0; i < toks.size(); ++i) {
        const auto st = ex.run_step(pd.graph, toks[i], static_cast<std::uint32_t>(i));
        CHECK_ST(st.ok(), "seq step", st);
    }
    auto lseq = ex.fetch_logits(pd.graph);
    CHECK(lseq.ok() && lseq.value().size() == cfg.vocab_size, "seq logits");

    CHECK(ex.reset_states().ok(), "reset");

    // batched
    {
        const auto st = ex.run_step(pb.graph, toks.data(), toks.size(), 0);
        CHECK_ST(st.ok(), "batch step", st);
    }
    auto lbat = ex.fetch_logits(pb.graph);
    CHECK(lbat.ok() && lbat.value().size() == toks.size() * cfg.vocab_size,
          "batch logits [P,V]");
    if (!lseq.ok() || !lbat.ok()) return failures ? 1 : 0;

    // last row of batch vs sequential final logits
    const auto& a = lseq.value();
    const auto& bf = lbat.value();
    const std::size_t V = cfg.vocab_size;
    const float* b = bf.data() + (toks.size() - 1) * V;
    double worst = 0.0;
    for (std::size_t i = 0; i < V; ++i) worst = std::max(worst, static_cast<double>(std::abs(a[i] - b[i])));
    std::size_t ta = 0, tb = 0;
    for (std::size_t i = 1; i < V; ++i) {
        if (a[i] > a[ta]) ta = i;
        if (b[i] > b[tb]) tb = i;
    }
    std::printf("batch==seq: worst|diff|=%.4g top1 %zu vs %zu\n", worst, ta, tb);
    // 閾値は prefill attention の実装差で決まる。prefill は転置コア（S^T = K·Q^T）で
    // P を f16 1 成分（11 bit）で持つため、attention 出力の誤差は 4e-4 程度あり、
    // decode 経路（bf16 完全一致）と比べると logits で数 e-2 まで開く。top-1 の一致は
    // 厳しいまま残すので、配線・レイアウトの取り違えはここで捕まる。
    CHECK(worst < 5e-2, "logit deviation batched vs sequential");
    CHECK(ta == tb, "top-1 identity");

    // decode after prefill continues correctly: pos 4 single step both paths
    CHECK(ex.run_step(pd.graph, 42, 4).ok(), "decode after batch");
    auto l2 = ex.fetch_logits(pd.graph);
    CHECK(l2.ok() && l2.value().size() == V, "post-decode logits");

    if (argc > 3 && std::string(argv[3]) == "probe") {
        CHECK(ex.reset_states().ok(), "reset2");
        for (std::size_t i = 0; i < toks.size(); ++i)
            CHECK(ex.run_step(pd.graph, toks[i], static_cast<std::uint32_t>(i)).ok(), "probe seq");
        std::vector<std::vector<float>> seqv;
        for (const auto& pr : pd.graph.primitives()) {
            auto r = ex.fetch_value_debug(pd.graph, pr.output.id);
            seqv.push_back(r.ok() ? r.value() : std::vector<float>{});
        }
        CHECK(ex.reset_states().ok(), "reset3");
        CHECK(ex.run_step(pb.graph, toks.data(), toks.size(), 0).ok(), "probe batch");
        const std::size_t P = toks.size();
        for (std::size_t k = 0; k < pb.graph.primitives().size(); ++k) {
            auto rb = ex.fetch_value_debug(pb.graph, pb.graph.primitives()[k].output.id);
            if (!rb.ok() || seqv[k].empty()) continue;
            const auto& b = rb.value();
            if (b.size() != seqv[k].size() * P) continue;  // non-batched value
            double w = 0.0, wr = 0.0;
            for (std::size_t i = 0; i < seqv[k].size(); ++i) {
                const double d = std::abs(seqv[k][i] - b[(P - 1) * seqv[k].size() + i]);
                const double mag = std::max(static_cast<double>(std::abs(seqv[k][i])), 1e-3);
                w = std::max(w, d); wr = std::max(wr, d / mag);
            }
            if (w > 1e-4 || wr > 1e-2) {
                double mx = 0.0;
                for (float x : b) mx = std::max(mx, static_cast<double>(std::abs(x)));
                std::printf("divergence: prim %zu kind=%s abs=%.4g rel=%.4g batchmax=%.4g\n", k,
                            ir::kind_name(pb.graph.primitives()[k].kind), w, wr, mx);
            }
        }
    }

    // ---- T14: 複数スロットの online KV（4 スロット、host とミラーが同一値）----
    {
        model::qwen35::TextConfig c4 = cfg;
        c4.num_layers = 2;
        c4.layer_kinds.assign(2, model::qwen35::LayerKind::full_attention);
        auto g4 = model::qwen35::build_step_graph(c4, 64, 4);
        CHECK(g4.ok(), "online: graph");
        if (g4.ok()) {
            const auto& plan = g4.value();
            auto make = [&](std::size_t online_pages) {
                backend::hip::HipStepExecutorOptions o = ho;
                o.max_sequences = 4;
                o.kv_pinned_host = true;
                o.online_kv_pages = online_pages;
                o.arena_bytes = 16u << 20;
                auto e = std::make_unique<backend::hip::HipStepExecutor>(o);
                std::mt19937 rng(123);
                std::normal_distribution<float> nd(0.f, 0.05f);
                for (const auto& w : plan.weights) {
                    std::vector<float> f(numelof(w.shape));
                    for (auto& x : f) x = nd(rng);
                    CHECK(e->upload_weight(w.id, f).ok(), "online: upload");
                }
                const auto ds = e->declare_states(plan.graph);
                CHECK_ST(ds.ok(), "online: declare", ds);
                return e;
            };
            auto host = make(0);
            auto mir4 = make(4);
            auto mir1 = make(1);
            std::vector<std::int32_t> toks4(4);
            std::vector<std::uint32_t> slots4 = {0u, 1u, 2u, 3u};
            std::vector<std::uint32_t> pos4(4);
            double worst4 = 0.0, worst1 = 0.0;
            for (std::uint32_t step = 0; step < 40; ++step) {
                for (std::size_t r = 0; r < 4; ++r) {
                    toks4[r] = static_cast<std::int32_t>(1000 + step * 4 + r);
                    pos4[r] = step;
                }
                backend::hip::HipStepExecutor::BatchRows br{slots4.data(), pos4.data()};
                const auto rh = host->run_batch(plan.graph, toks4.data(), nullptr, 4, br);
                const auto r4 = mir4->run_batch(plan.graph, toks4.data(), nullptr, 4, br);
                const auto r1 = mir1->run_batch(plan.graph, toks4.data(), nullptr, 4, br);
                CHECK_ST(rh.ok(), "online: host step", rh);
                CHECK_ST(r4.ok(), "online: mirror4 step", r4);
                CHECK_ST(r1.ok(), "online: mirror1 step", r1);
                if (!rh.ok() || !r4.ok() || !r1.ok()) break;
                auto lh = host->fetch_logits(plan.graph);
                auto l4 = mir4->fetch_logits(plan.graph);
                auto l1 = mir1->fetch_logits(plan.graph);
                CHECK(lh.ok() && l4.ok() && l1.ok(), "online: logits");
                if (!lh.ok() || !l4.ok() || !l1.ok()) break;
                for (std::size_t i = 0; i < lh.value().size(); ++i) {
                    worst4 = std::max(worst4, std::fabs(static_cast<double>(lh.value()[i]) -
                                                        static_cast<double>(l4.value()[i])));
                    worst1 = std::max(worst1, std::fabs(static_cast<double>(lh.value()[i]) -
                                                        static_cast<double>(l1.value()[i])));
                }
            }
            std::printf("online KV (4 slots, 40 steps): worst R=4 %.3e / R=1 %.3e\n", worst4,
                        worst1);
            CHECK(worst4 == 0.0, "online KV 4 slots matches host exactly (R=4)");
            CHECK(worst1 == 0.0, "online KV 4 slots matches host exactly (R=1)");
        }
    }

    // ---- T16: グローバル KV ブロック + ブロック表の書き換え（prefix 共有）----
    {
        model::qwen35::TextConfig c2 = cfg;
        c2.layer_kinds.assign(cfg.num_layers, model::qwen35::LayerKind::full_attention);
        auto pref = model::qwen35::build_step_graph(c2, 64, 16);
        auto dec = model::qwen35::build_step_graph(c2, 64, 1);
        CHECK(pref.ok() && dec.ok(), "t16 graphs");
        if (pref.ok() && dec.ok()) {
            std::vector<std::int32_t> P(16);
            for (std::size_t i = 0; i < P.size(); ++i) P[i] = static_cast<std::int32_t>(30 + i);
            auto upload = [&](backend::hip::HipStepExecutor& e) {
                std::mt19937 rng(7);
                std::normal_distribution<float> nd(0.f, 0.05f);
                for (const auto& w : pref.value().weights) {
                    std::vector<float> f(numelof(w.shape));
                    for (auto& x : f) x = nd(rng);
                    CHECK(e.upload_weight(w.id, f).ok(), "t16 upload");
                }
                return e.declare_states(pref.value().graph);
            };
            backend::hip::HipStepExecutor shared(ho);
            shared.set_weight_placement(false);
            auto st1 = upload(shared);
            CHECK_ST(st1.ok(), "t16 declare shared", st1);
            CHECK(shared.kv_blocks() == 8, "kv_blocks を明示できる");

            // 共有プールの slot0 で 16 トークンを prefill（block0 を使う）。
            std::vector<std::uint32_t> slots0(16, 0u), pos16(16);
            for (std::size_t i = 0; i < 16; ++i) pos16[i] = static_cast<std::uint32_t>(i);
            backend::hip::HipStepExecutor::BatchRows br0{slots0.data(), pos16.data()};
            const auto r0 = shared.run_batch(pref.value().graph, P.data(), nullptr, 16, br0);
            CHECK_ST(r0.ok(), "t16 prefill", r0);
            auto lg0 = shared.fetch_logits(pref.value().graph);
            CHECK(lg0.ok() && lg0.value().size() == 16 * cfg.vocab_size, "t16 logits [16,V]");
            std::int32_t next = 0;
            if (lg0.ok()) {
                const float* last = lg0.value().data() + 15 * cfg.vocab_size;
                for (std::size_t i = 1; i < cfg.vocab_size; ++i)
                    if (last[i] > last[static_cast<std::size_t>(next)])
                        next = static_cast<std::int32_t>(i);
            }

            // slot1 は page0 を block0（slot0 と共有）、page1 を block3 に向ける。
            const std::uint32_t table[4] = {0u, 3u, 0u, 0u};
            const auto as = shared.assign_kv_blocks(1, table, 4);
            CHECK_ST(as.ok(), "t16 assign blocks", as);
            const std::uint32_t bad_table[2] = {0u, 99u};
            CHECK(!shared.assign_kv_blocks(1, bad_table, 2).ok(), "範囲外 block は拒否");

            // slot1 で 17 個目を流す（位置16・共有 prefix を読む）。
            std::uint32_t one_slot = 1u, one_pos = 16u;
            backend::hip::HipStepExecutor::BatchRows br1{&one_slot, &one_pos};
            const auto r1 = shared.run_batch(dec.value().graph, &next, nullptr, 1, br1);
            CHECK_ST(r1.ok(), "t16 continuation", r1);
            auto lg1 = shared.fetch_logits(dec.value().graph);

            // 参照: 別の実行器（従来の固定プール）で同じ 17 トークンを流す。
            backend::hip::HipStepExecutor ref(ho);
            auto st2 = upload(ref);
            CHECK_ST(st2.ok(), "t16 declare ref", st2);
            std::vector<std::uint32_t> rs(16, 0u);
            backend::hip::HipStepExecutor::BatchRows rbr{rs.data(), pos16.data()};
            const auto rrp = ref.run_batch(pref.value().graph, P.data(), nullptr, 16, rbr);
            CHECK_ST(rrp.ok(), "t16 ref prefill", rrp);
            std::uint32_t zslot = 0u, zpos = 16u;
            backend::hip::HipStepExecutor::BatchRows zbr{&zslot, &zpos};
            const auto rrs = ref.run_batch(dec.value().graph, &next, nullptr, 1, zbr);
            CHECK_ST(rrs.ok(), "t16 ref step", rrs);
            auto lgr = ref.fetch_logits(dec.value().graph);
            CHECK(lg1.ok() && lgr.ok(), "t16 ref logits");
            if (lg1.ok() && lgr.ok()) {
                double worst = 0.0;
                for (std::size_t i = 0; i < lg1.value().size(); ++i)
                    worst = std::max(worst, std::abs(static_cast<double>(lg1.value()[i]) -
                                                     static_cast<double>(lgr.value()[i])));
                std::printf("t16 shared-block logits diff = %.3e\n", worst);
                CHECK(worst < 1e-5, "共有ブロック経由の logits が再計算と一致");
            }
        }
    }

    // ---- T16: GDN 状態の export/import（prefix 共有の前提）----
    {
        model::qwen35::TextConfig c3 = cfg;
        c3.num_layers = 2;
        c3.layer_kinds = {model::qwen35::LayerKind::full_attention,
                          model::qwen35::LayerKind::linear_attention};
        auto pref3 = model::qwen35::build_step_graph(c3, 64, 16);
        auto dec3 = model::qwen35::build_step_graph(c3, 64, 1);
        CHECK(pref3.ok() && dec3.ok(), "t16 gdn graphs");
        if (pref3.ok() && dec3.ok()) {
            std::vector<std::int32_t> P(16);
            for (std::size_t i = 0; i < P.size(); ++i) P[i] = static_cast<std::int32_t>(60 + i);
            auto setup = [&](backend::hip::HipStepExecutor& e) {
                std::mt19937 rng(11);
                std::normal_distribution<float> nd(0.f, 0.05f);
                for (const auto& w : pref3.value().weights) {
                    std::vector<float> f(numelof(w.shape));
                    for (auto& x : f) x = nd(rng);
                    CHECK(e.upload_weight(w.id, f).ok(), "t16 gdn upload");
                }
                return e.declare_states(pref3.value().graph);
            };
            backend::hip::HipStepExecutor a(ho);   // ho.kv_blocks は 0（固定プール）
            backend::hip::HipStepExecutor b(ho);
            CHECK_ST(setup(a).ok(), "t16 gdn declare a", setup(a));
            CHECK_ST(setup(b).ok(), "t16 gdn declare b", setup(b));
            std::vector<std::uint32_t> s0(16, 0u), p16(16);
            for (std::size_t i = 0; i < 16; ++i) p16[i] = static_cast<std::uint32_t>(i);
            backend::hip::HipStepExecutor::BatchRows br{s0.data(), p16.data()};
            CHECK_ST(a.run_batch(pref3.value().graph, P.data(), nullptr, 16, br).ok(),
                     "t16 gdn prefill a", a.run_batch(pref3.value().graph, P.data(), nullptr, 16, br));
            CHECK_ST(b.run_batch(pref3.value().graph, P.data(), nullptr, 16, br).ok(),
                     "t16 gdn prefill b", b.run_batch(pref3.value().graph, P.data(), nullptr, 16, br));
            auto blob = a.export_gdn_state(0);
            CHECK(blob.ok() && !blob.value().empty(), "GDN 状態を export できる");
            if (blob.ok()) {
                // b を同じ状態へ戻す（KV はそのまま。ここでは GDN だけを差し替える）。
                CHECK_ST(b.import_gdn_state(0, blob.value()).ok(), "GDN 状態を import",
                         b.import_gdn_state(0, blob.value()));
                CHECK(b.export_gdn_state(0).ok() &&
                          b.export_gdn_state(0).value() == blob.value(),
                      "import 後に export すると同じ値");
            }
            // 17 個目: 両者は同じ状態なので logits が一致する。
            std::uint32_t zs = 0u, zp = 16u;
            backend::hip::HipStepExecutor::BatchRows zbr{&zs, &zp};
            std::int32_t next = 77;
            CHECK_ST(a.run_batch(dec3.value().graph, &next, nullptr, 1, zbr).ok(),
                     "t16 gdn step a", a.run_batch(dec3.value().graph, &next, nullptr, 1, zbr));
            CHECK_ST(b.run_batch(dec3.value().graph, &next, nullptr, 1, zbr).ok(),
                     "t16 gdn step b", b.run_batch(dec3.value().graph, &next, nullptr, 1, zbr));
            auto la = a.fetch_logits(dec3.value().graph);
            auto lb = b.fetch_logits(dec3.value().graph);
            if (la.ok() && lb.ok()) {
                double worst = 0.0;
                for (std::size_t i = 0; i < la.value().size(); ++i)
                    worst = std::max(worst, std::abs(static_cast<double>(la.value()[i]) -
                                                     static_cast<double>(lb.value()[i])));
                std::printf("t16 gdn export/import logits diff = %.3e\n", worst);
                CHECK(worst == 0.0, "GDN 状態の export/import で logits が一致");
            }
        }
    }

    // ---- T15: 2 スロットのバッチ（それぞれ独立した KV / GDN 状態）----
    {
        const std::int32_t a0 = 7, a1 = 111, b0 = 512, b1 = 900;
        CHECK(ex.reset_states().ok(), "ms reset a");
        CHECK(ex.run_step(pd.graph, a0, 0).ok(), "ms a0");
        CHECK(ex.run_step(pd.graph, a1, 1).ok(), "ms a1");
        auto la = ex.fetch_logits(pd.graph);
        CHECK(ex.reset_states().ok(), "ms reset b");
        CHECK(ex.run_step(pd.graph, b0, 0).ok(), "ms b0");
        CHECK(ex.run_step(pd.graph, b1, 1).ok(), "ms b1");
        auto lb = ex.fetch_logits(pd.graph);
        CHECK(ex.reset_states().ok(), "ms reset batch");
        const std::int32_t toks4[4] = {a0, a1, b0, b1};
        const std::uint32_t slots[4] = {0u, 0u, 1u, 1u};
        const std::uint32_t pos[4] = {0u, 1u, 0u, 1u};
        backend::hip::HipStepExecutor::BatchRows rows{slots, pos};
        const auto stb = ex.run_batch(pb.graph, toks4, nullptr, 4, rows);
        CHECK_ST(stb.ok(), "ms batch", stb);
        auto lbat2 = ex.fetch_logits(pb.graph);
        CHECK(lbat2.ok() && lbat2.value().size() == 4 * V, "ms batch logits [4,V]");
        if (la.ok() && lb.ok() && lbat2.ok()) {
            const auto diff = [&](const std::vector<float>& x, const float* y) {
                double w = 0.0;
                for (std::size_t i = 0; i < V; ++i)
                    w = std::max(w, std::abs(static_cast<double>(x[i]) -
                                             static_cast<double>(y[i])));
                return w;
            };
            const double wa = diff(la.value(), lbat2.value().data() + V);
            const double wb = diff(lb.value(), lbat2.value().data() + 3 * V);
            std::printf("multislot: |a|=%.3e |b|=%.3e\n", wa, wb);
            CHECK(wa < 5e-2, "multislot slot0 matches single-slot logits");
            CHECK(wb < 5e-2, "multislot slot1 matches single-slot logits");
        }
    }

    // ---- prefill logits を最終行だけにする（lm_head の重複計算を消す） ----
    // 同じ batch グラフで、head だけ 1 行に絞った版を作る。
    // 期待: logits の要素数は V、値は全行版の最終行と一致（同じ hidden 行を
    // 同じ重みで読むので差は rows=1 側 GEMM の累加順に由来する微小差）、argmax は一致。
    {
        model::qwen35::GraphOptions go;
        go.logits_last_row_only = true;
        auto plr = model::qwen35::build_step_graph(cfg, 16, 4, {}, go);
        CHECK(plr.ok(), "last-row graph");
        if (plr.ok()) {
            auto& pl = plr.value();
            CHECK(pl.logits_rows == 1u, "last-row: logits_rows == 1");
            CHECK(ex.reset_states().ok(), "reset before last-row");
            CHECK(ex.run_step(pl.graph, toks.data(), toks.size(), 0).ok(), "last-row step");
            auto ll = ex.fetch_logits(pl.graph);
            CHECK(ll.ok() && ll.value().size() == V, "last-row logits size");
            if (ll.ok() && lbat.ok() && lbat.value().size() == toks.size() * V) {
                const auto& a = ll.value();
                const auto& b = lbat.value();
                const std::size_t off = (toks.size() - 1) * V;
                double mx = 0.0;
                for (std::size_t i = 0; i < V; ++i)
                    mx = std::max(mx, std::fabs(static_cast<double>(a[i]) -
                                                  static_cast<double>(b[off + i])));
                std::printf("last-row logits: worst|diff| vs batch last row = %.3e\n", mx);
                CHECK(mx <= 1e-5, "last-row logits == batch last row (<=1e-5)");
                std::size_t ba = 0, bb = 0;
                for (std::size_t i = 1; i < V; ++i) { if (a[i] > a[ba]) ba = i; }
                for (std::size_t i = 1; i < V; ++i) { if (b[off + i] > b[off + bb]) bb = i; }
                CHECK(ba == bb, "last-row argmax == batch argmax");
            }
        }
    }

    // ---- rows=64 の prefill でも 2 回実行でビット一致するか ----
    // エンジン（実モデル）で観測された「同じプロンプトの prefill が実行毎に
    // わずかに変わる」現象の再現を狙う。小さい形で再現すれば切り分けが速い。
    {
        auto p64 = model::qwen35::build_step_graph(cfg, 128, 64);
        CHECK(p64.ok(), "64-row graph");
        if (p64.ok()) {
            backend::hip::HipStepExecutorOptions ho64 = ho;
            ho64.arena_bytes = 64u << 20;
            backend::hip::HipStepExecutor ex64(ho64);
            std::mt19937 rng64(7);
            std::normal_distribution<float> nd64(0.f, 0.05f);
            for (const auto& w : p64.value().weights) {
                std::vector<float> f(numelof(w.shape));
                for (auto& x : f) x = nd64(rng64);
                CHECK(ex64.upload_weight(w.id, f).ok(), "64-row upload weight");
            }
            CHECK_ST(ex64.declare_states(p64.value().graph).ok(), "64-row declare",
                     ex64.declare_states(p64.value().graph));
            std::vector<std::int32_t> t64(64);
            for (std::size_t i = 0; i < t64.size(); ++i)
                t64[i] = static_cast<std::int32_t>((i * 37u) % cfg.vocab_size);
            std::vector<float> first;
            for (int rep = 0; rep < 2; ++rep) {
                CHECK(ex64.reset_states().ok(), "64-row reset");
                const auto st = ex64.run_step(p64.value().graph, t64.data(), t64.size(), 0);
                CHECK_ST(st.ok(), "64-row prefill", st);
                auto lg = ex64.fetch_logits(p64.value().graph);
                CHECK(lg.ok(), "64-row logits");
                if (!lg.ok()) break;
                if (rep == 0) {
                    first = lg.value();
                } else {
                    const bool same = first.size() == lg.value().size() &&
                                      std::memcmp(first.data(), lg.value().data(),
                                                  first.size() * sizeof(float)) == 0;
                    if (!same && first.size() == lg.value().size()) {
                        double w = 0.0;
                        for (std::size_t i = 0; i < first.size(); ++i)
                            w = std::max(w, std::fabs(double(first[i]) - double(lg.value()[i])));
                        std::printf("64-row prefill determinism: worst|diff|=%.3e\n", w);
                    }
                    CHECK(same, "64-row prefill は 2 回実行でビット一致");
                }
            }
        }
    }

    // ---- reset_slot(1) が slot0 の状態・KV を壊さない（所有権の分離） ----
    {
        auto p16 = model::qwen35::build_step_graph(cfg, 16, 16);
        CHECK(p16.ok(), "ownership graph");
        if (p16.ok()) {
            backend::hip::HipStepExecutorOptions ho2;
            ho2.rope_theta = cfg.rope_theta;
            ho2.rope_rotary_dim = cfg.rotary_dim;
            ho2.arena_bytes = 0;
            ho2.max_sequences = 2;
            ho2.kv_pinned_host = true;
            ho2.online_kv_pages = 2;
            backend::hip::HipStepExecutor ex2(ho2);
            std::mt19937 rng2(21);
            std::normal_distribution<float> nd2(0.f, 0.05f);
            for (const auto& w : p16.value().weights) {
                std::vector<float> f(numelof(w.shape));
                for (auto& x : f) x = nd2(rng2);
                CHECK(ex2.upload_weight(w.id, f).ok(), "ownership upload");
            }
            CHECK_ST(ex2.declare_states(p16.value().graph).ok(), "ownership declare",
                     ex2.declare_states(p16.value().graph));
            std::vector<std::int32_t> t16(16);
            for (std::size_t i = 0; i < t16.size(); ++i)
                t16[i] = static_cast<std::int32_t>((i * 29u) % cfg.vocab_size);
            CHECK_ST(ex2.run_step(p16.value().graph, t16.data(), t16.size(), 0).ok(),
                     "ownership prefill", ex2.run_step(p16.value().graph, t16.data(),
                                                      t16.size(), 0));
            auto st_a = ex2.export_gdn_state(0);
            auto kv_a = ex2.debug_kv_hash(0, ex2.kv_pages_per_sequence());
            CHECK_ST(ex2.reset_slot(1).ok(), "reset_slot(1)",
                     ex2.reset_slot(1));
            auto st_b = ex2.export_gdn_state(0);
            auto kv_b = ex2.debug_kv_hash(0, ex2.kv_pages_per_sequence());
            CHECK(st_a.ok() && st_b.ok() && st_a.value() == st_b.value(),
                  "reset_slot(1) は slot0 の GDN 状態を変えない");
            CHECK(kv_a.ok() && kv_b.ok() && kv_a.value() == kv_b.value(),
                  "reset_slot(1) は slot0 の KV を変えない");
            auto st_s1 = ex2.export_gdn_state(1);
            bool zero = st_s1.ok();
            if (st_s1.ok())
                for (const float v : st_s1.value())
                    if (v != 0.0f) zero = false;
            CHECK(zero, "reset_slot(1) は slot1 の状態を 0 にする");
        }
    }

    CHECK(ex.device_bytes() > 0u, "budget: 通常実行器も確保を計上している");

    std::printf("test_hip_step_executor: failures=%d\n", failures);
    return failures ? 1 : 0;
}
