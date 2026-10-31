// shelf unit test — 状態の初期化（同一プロセスで別シーケンスを流せるか）。
//
// `shelf-serve` で「2 件目の generate が 1 件目と違う」事象が出たため、
// 合成モデル（重みは乱数・GPU）で最小の再現を作る。
//
// 見ていること:
//   1. 同じトークン列を **reset を挟んで 2 回流す** と logits がビット一致する
//      （= 新しいシーケンスがプロセス起動直後と同じ条件に戻る）
//   2. n=1 ずつ積む経路（serve と同じ）と、バッチで積む経路の両方で成り立つ
//   3. 3 回目も同じ（1 回だけ直るのではなく、毎回戻る）
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <shelf/backend/hip/step_executor.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

using namespace shelf;

namespace {

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

std::size_t numelof(const std::vector<std::size_t>& s) {
    std::size_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

// 名前から決まる重み（構成が同じなら常に同じ値）。
std::vector<float> weight_for(const std::string& name, std::size_t n) {
    std::mt19937 rng(static_cast<std::uint32_t>(std::hash<std::string>{}(name) & 0x7fffffffu));
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> f(n);
    for (auto& x : f) x = nd(rng);
    return f;
}

model::qwen35::TextConfig make_cfg() {
    model::qwen35::TextConfig cfg;
    cfg.hidden_size = 512;
    cfg.num_layers = 4;
    cfg.vocab_size = 1024;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 256;
    cfg.rotary_dim = 64;
    cfg.rope_theta = 1.0e7;
    cfg.lin_key_heads = 2;
    cfg.lin_value_heads = 4;
    cfg.lin_key_head_dim = 128;
    cfg.lin_value_head_dim = 128;
    cfg.conv_kernel = 4;
    cfg.intermediate_size = 1024;
    cfg.rms_eps = 1e-6;
    cfg.tie_word_embeddings = false;
    cfg.eos_token_id = 0;
    for (std::size_t l = 0; l < cfg.num_layers; ++l)
        cfg.layer_kinds.push_back((l % 2 == 0) ? model::qwen35::LayerKind::linear_attention
                                               : model::qwen35::LayerKind::full_attention);
    return cfg;
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double w = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
        w = std::max(w, std::abs(double(a[i]) - double(b[i])));
    return w;
}

}  // namespace

int main() {
    if (hipSetDevice(0) != hipSuccess) {
        std::printf("FAIL: hipSetDevice\n");
        return 1;
    }
    const auto cfg = make_cfg();
    auto dr = model::qwen35::build_step_graph(cfg, 32, 1);
    auto br = model::qwen35::build_step_graph(cfg, 32, 4);
    if (!dr.ok() || !br.ok()) {
        std::printf("FAIL: グラフを作れない\n");
        return 1;
    }
    auto dec = dr.release();
    auto bat = br.release();

    backend::hip::HipStepExecutorOptions ho;
    ho.rope_theta = cfg.rope_theta;
    ho.rope_rotary_dim = cfg.rotary_dim;
    backend::hip::HipStepExecutor ex(ho);
    for (const auto& w : dec.weights) {
        const auto f = weight_for(w.checkpoint_name, numelof(w.shape));
        if (!ex.upload_weight(w.id, f).ok()) {
            std::printf("FAIL: upload\n");
            return 1;
        }
    }
    if (!ex.declare_states(bat.graph).ok()) {   // バッチ側で宣言（大きい方）
        std::printf("FAIL: declare\n");
        return 1;
    }

    // バッチ側のグラフは rows=4 なので、4 トークンで両経路を同じ条件にする。
    const std::vector<std::int32_t> toks = {11, 222, 333, 444};

    // ---- 1) n=1 ずつ積む経路（serve と同じ形）----
    const auto run_seq = [&](bool use_batch) {
        std::vector<float> logits;
        if (!ex.reset_states().ok()) return logits;
        if (use_batch) {
            if (!ex.run_step(bat.graph, toks.data(), toks.size(), 0).ok()) return logits;
            auto lg = ex.fetch_logits(bat.graph);
            if (!lg.ok()) return logits;
            // 最終行だけ取り出す
            const std::size_t rows = lg.value().size() / cfg.vocab_size;
            logits.assign(lg.value().end() - static_cast<std::ptrdiff_t>(rows),
                          lg.value().end());
            return logits;
        }
        for (std::size_t i = 0; i < toks.size(); ++i) {
            if (!ex.run_step(dec.graph, toks[i], static_cast<std::uint32_t>(i)).ok())
                return logits;
        }
        auto lg = ex.fetch_logits(dec.graph);
        if (lg.ok()) logits = lg.value();
        return logits;
    };

    // reset を挟んで 3 回。2 回目以降が 1 回目と一致することを見る。
    for (const bool use_batch : {false, true}) {
        const char* what = use_batch ? "バッチ prefill" : "n=1 ずつ";
        std::vector<float> first;
        for (int round = 0; round < 3; ++round) {
            const auto lg = run_seq(use_batch);
            if (lg.empty()) {
                std::printf("FAIL: %s の %d 回目で logits が無い\n", what, round + 1);
                ++failures;
                break;
            }
            if (round == 0) {
                first = lg;
                continue;
            }
            if (!bit_equal(first, lg)) {
                std::printf("FAIL: %s で %d 回目が 1 回目と違う（worst|diff|=%.3g）\n", what,
                            round + 1, max_abs_diff(first, lg));
                ++failures;
            }
        }
        // 経路が違えば数値も違いうる（丸め）。ここでは「同じ経路なら一致」を見る。
        (void)first;
    }

    std::printf("test_hip_reset: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
