// shelf unit test — decode 1 ステップの実行間決定性（2026-11-01 の調査）。
//
// 実モデル（9B / 27B）で「同じプロンプトを prefill し直すと最初の decode の
// logits がわずかに変わる」現象が観測された。prefill の logits はビット一致する
// ため、疑わしいのは decode 経路である。ここでは実モデルに近い形（hidden 4096・
// heads 16 / kv 4・head_dim 256・linear attention あり）で
//   ① 同じ状態から decode を 2 回（状態を戻して）回し、logits をビット比較
// する。小さい形（hidden 512）では再現しなかったため、形を実物に寄せている。
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <shelf/backend/hip/step_executor.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

namespace {

using namespace shelf;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}
void check_st(const Status& st, const char* what, const Status& err) {
    if (!st.ok()) {
        ++failures;
        std::printf("FAIL: %s: %s\n", what, err.message().c_str());
    }
}

template <typename C>
std::size_t numelof(const C& c) {
    std::size_t n = 1;
    for (auto d : c) n *= d;
    return n;
}

// prefill を 2 回（状態を戻して）回し、logits がビット一致するか。
int run_prefill(const model::qwen35::TextConfig& cfg, std::size_t kv_capacity,
                std::size_t prefill_rows, std::size_t layers,
                model::qwen35::LayerKind kind) {
    model::qwen35::TextConfig c = cfg;
    c.num_layers = layers;
    c.layer_kinds.clear();
    for (std::size_t l = 0; l < layers; ++l) c.layer_kinds.push_back(kind);
    auto pre = model::qwen35::build_step_graph(c, kv_capacity, prefill_rows);
    check(pre.ok(), "prefill graph");
    if (!pre.ok()) return 1;
    backend::hip::HipStepExecutorOptions ho;
    ho.rope_theta = c.rope_theta;
    ho.rope_rotary_dim = c.rotary_dim;
    ho.arena_bytes = 0;
    backend::hip::HipStepExecutor ex(ho);
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.f, 0.05f);
    for (const auto& w : pre.value().weights) {
        std::vector<float> f(numelof(w.shape));
        for (auto& x : f) x = nd(rng);
        const auto st = ex.upload_weight(w.id, f);
        check_st(st, "prefill upload", st);
    }
    { const auto st = ex.declare_states(pre.value().graph); check_st(st, "prefill declare", st); }
    std::vector<std::int32_t> prompt(prefill_rows);
    for (std::size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = static_cast<std::int32_t>((i * 41u) % c.vocab_size);
    std::vector<float> first;
    for (int rep = 0; rep < 2; ++rep) {
        { const auto st = ex.reset_states(); check_st(st, "prefill reset", st); }
        { const auto st = ex.run_step(pre.value().graph, prompt.data(), prompt.size(), 0);
          check_st(st, "prefill step", st); }
        auto lg = ex.fetch_logits(pre.value().graph);
        check(lg.ok(), "prefill logits");
        if (!lg.ok()) return 1;
        if (rep == 0) {
            first = lg.value();
        } else {
            const bool same = first.size() == lg.value().size() &&
                              std::memcmp(first.data(), lg.value().data(),
                                          first.size() * sizeof(float)) == 0;
            if (!same && first.size() == lg.value().size()) {
                std::size_t nd = 0;
                double w = 0.0;
                for (std::size_t i = 0; i < first.size(); ++i) {
                    if (first[i] != lg.value()[i]) ++nd;
                    w = std::max(w, std::fabs(double(first[i]) - double(lg.value()[i])));
                }
                std::printf("  prefill kind=%d layers=%zu: ndiff=%zu worst=%.3e\n",
                            static_cast<int>(kind), layers, nd, w);
            }
            check(same, "prefill logits は 2 回実行でビット一致");
        }
    }
    return 0;
}

int run(const model::qwen35::TextConfig& cfg, std::size_t kv_capacity, std::size_t prefill_rows,
        std::size_t layers) {
    model::qwen35::TextConfig c = cfg;
    c.num_layers = layers;
    c.layer_kinds.clear();
    for (std::size_t l = 0; l < layers; ++l)
        c.layer_kinds.push_back(l % 2u == 0u ? model::qwen35::LayerKind::linear_attention
                                             : model::qwen35::LayerKind::full_attention);
    auto dec = model::qwen35::build_step_graph(c, kv_capacity, 1);
    auto pre = model::qwen35::build_step_graph(c, kv_capacity, prefill_rows);
    check(dec.ok() && pre.ok(), "graph");
    if (!dec.ok() || !pre.ok()) return 1;

    backend::hip::HipStepExecutorOptions ho;
    ho.rope_theta = c.rope_theta;
    ho.rope_rotary_dim = c.rotary_dim;
    ho.arena_bytes = 0;   // auto
    ho.max_sequences = 1;
    ho.arena_reuse = false;   // 値を個別に読めるように（デバッグ優先）
    backend::hip::HipStepExecutor ex(ho);

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 0.05f);
    for (const auto& w : dec.value().weights) {
        std::vector<float> f(numelof(w.shape));
        for (auto& x : f) x = nd(rng);
        check_st(ex.upload_weight(w.id, f), "upload", ex.upload_weight(w.id, f));
    }
    { const auto st = ex.declare_states(dec.value().graph); check_st(st, "declare", st); }

    std::vector<std::int32_t> prompt(prefill_rows);
    for (std::size_t i = 0; i < prompt.size(); ++i)
        prompt[i] = static_cast<std::int32_t>((i * 37u) % c.vocab_size);

    std::vector<float> first;
    std::vector<std::vector<float>> runs;
    for (int rep = 0; rep < 2; ++rep) {
        std::fprintf(stderr, "[rep] %d\n", rep);
        { const auto st = ex.reset_states(); check_st(st, "reset", st); }
        { const auto st = ex.run_step(pre.value().graph, prompt.data(), prompt.size(), 0);
          check_st(st, "prefill", st); }
        // prefill 直後の状態（GDN と KV）を要約して、状態が同じかを見る。
        {
            std::uint64_t hs = 1469598103934665603ull;
            if (const auto st = ex.export_gdn_state(0); st.ok()) {
                for (const float v : st.value()) {
                    std::uint32_t b = 0;
                    std::memcpy(&b, &v, sizeof(b));
                    hs = (hs ^ b) * 1099511628211ull;
                }
            }
            std::uint64_t hk = 0;
            if (const auto kv = ex.debug_kv_hash(0, ex.kv_pages_per_sequence()); kv.ok())
                hk = kv.value();
            std::printf("  state hash=%016llx kv=%016llx\n",
                        static_cast<unsigned long long>(hs),
                        static_cast<unsigned long long>(hk));
        }
        { const auto st = ex.run_step(dec.value().graph, prompt[0], prefill_rows);
        check_st(st, "decode", st); }
        auto lg = ex.fetch_logits(dec.value().graph);
        check(lg.ok(), "logits");
        if (!lg.ok()) return 1;
        runs.push_back(lg.value());
        if (rep == 0) {
            first = lg.value();
        } else {
            const bool same = first.size() == lg.value().size() &&
                              std::memcmp(first.data(), lg.value().data(),
                                          first.size() * sizeof(float)) == 0;
            if (!same && first.size() == lg.value().size()) {
                double w = 0.0;
                std::size_t wi = 0, ndiff = 0;
                std::size_t first_bad = first.size();
                for (std::size_t i = 0; i < first.size(); ++i) {
                    if (first[i] != lg.value()[i]) {
                        ++ndiff;
                        first_bad = std::min(first_bad, i);
                    }
                    const double d = std::fabs(double(first[i]) - double(lg.value()[i]));
                    if (d > w) { w = d; wi = i; }
                }
                std::printf("  layers=%zu: ndiff=%zu first=%zu worst|diff|=%.3e at %zu "
                            "(%.6f vs %.6f)\n", layers, ndiff, first_bad, w, wi, first[wi],
                            lg.value()[wi]);
            }
            check(same, "decode logits は 2 回実行でビット一致");
        }
    }
    return 0;
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }
    model::qwen35::TextConfig cfg;
    cfg.hidden_size = 4096;
    cfg.vocab_size = 1024;
    cfg.num_heads = 16;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 256;
    cfg.rotary_dim = 64;
    cfg.rope_theta = 1.0e7;
    cfg.lin_key_heads = 16;
    cfg.lin_value_heads = 32;
    cfg.lin_key_head_dim = 128;
    cfg.lin_value_head_dim = 128;
    cfg.conv_kernel = 4;
    cfg.intermediate_size = 512;
    cfg.rms_eps = 1e-6;
    cfg.tie_word_embeddings = false;
    cfg.eos_token_id = 0;

    std::printf("test_hip_decode_determinism（hidden=4096・heads=16/kv=4）\n");
    run_prefill(cfg, 256u, 64u, 2u, model::qwen35::LayerKind::linear_attention);
    run_prefill(cfg, 256u, 64u, 2u, model::qwen35::LayerKind::full_attention);
    {
        // 27B 相当（heads 24 / kv 4 = GQA 6）。prefill attention の形依存を疑う。
        model::qwen35::TextConfig c27 = cfg;
        c27.num_heads = 24;
        c27.num_kv_heads = 4;
        run_prefill(c27, 256u, 64u, 2u, model::qwen35::LayerKind::full_attention);
        run_prefill(c27, 256u, 64u, 2u, model::qwen35::LayerKind::linear_attention);
    }
    for (const std::size_t layers : {1u, 2u}) {
        if (run(cfg, 256u, 64u, layers) != 0) break;
    }
    std::printf("test_hip_decode_determinism: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
