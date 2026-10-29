// shelf architecture test — 性能の回帰ゲート（コストモデル、GPU 不要）。
//
// 「時間」ではなく**グラフから決まる量**を固定する。時間は装置・負荷でぶれ、
// CI では回帰を検出できないため、ここでは次の 4 つを回帰対象にする:
//
//   1. decode 1 ステップで**触る重みのバイト数**（= 重みを 1 回だけ読むこと）
//   2. prefill 64 行でも同じ（行数ぶん重みを読み直さないこと）
//   3. 行あたりの活性化バイト数（アリーナ必要量の下限。行数に比例すること）
//   4. primitive 種別の個数（グラフ形状。不要なノード追加・融合漏れの検出）
//
// 値を変える正当な理由（融合の追加など）があるときは、実測し直して下の
// `kBaseline` を更新する（コミットメッセージに理由を書く）。
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <shelf/foundation/numeric.hpp>
#include <shelf/ir/graph.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

namespace {

using namespace shelf;

int failures = 0;

std::size_t numel(const std::vector<std::size_t>& s) {
    std::size_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

std::size_t numel(const TensorShape& s) {
    std::size_t n = 1;
    for (std::size_t i = 0; i < s.rank(); ++i) n *= s.dim(i);
    return n;
}

// 基準値（2026-10-29 実測。`test_cost_model --dump` で取り直して更新する）。
// 変更理由（融合・ノード追加など）はコミットメッセージに書く。
constexpr std::size_t kBaselineWeights = 27307552;
constexpr std::size_t kBaselineActRows1 = 159808;
constexpr std::size_t kBaselineActRows64 = 10227712;
const std::map<std::string, std::size_t> kBaselinePrims = {
    {"attention_gqa_step", 2},
    {"causal_conv_step", 2},
    {"elem_mul", 4},
    {"embedding", 1},
    {"gdn_step", 2},
    {"linear", 30},
    {"output_projection", 1},
    {"reshape", 30},
    {"residual_add", 8},
    {"rmsnorm", 13},
    {"rmsnorm_gated", 2},
    {"rope", 4},
    {"sigmoid_gate", 2},
    {"silu", 4},
    {"slice", 10},
};

struct Baseline {
    // decode（rows=1）で触る重みのバイト数（bf16 = 2 B/要素）
    std::size_t weights_bytes = 0;
    // rows=1 / rows=64 の活性化バイト数（bf16 = 2 B/要素、出力の総和）
    std::size_t act_bytes_rows1 = 0;
    std::size_t act_bytes_rows64 = 0;
    // primitive 種別ごとの個数
    std::map<std::string, std::size_t> prims;
};

Baseline measure(const model::qwen35::StepGraphPlan& plan, std::size_t rows) {
    Baseline b;
    std::size_t wb = 0, ab = 0;
    for (const auto& p : plan.graph.primitives()) {
        for (const auto id : p.weights) {
            const auto* sh = plan.graph.find_weight(id);
            if (sh) wb += numel(*sh) * sizeof(bf16_t);
        }
        ab += numel(p.output.shape) * sizeof(bf16_t);
        ++b.prims[ir::kind_name(p.kind)];
    }
    b.weights_bytes = wb;
    if (rows == 1) b.act_bytes_rows1 = ab;
    else b.act_bytes_rows64 = ab;
    return b;
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

void expect_eq(const char* what, std::size_t got, std::size_t want) {
    if (got != want) {
        ++failures;
        std::printf("FAIL: %s: 実測 %zu / 基準 %zu\n", what, got, want);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const auto cfg = make_cfg();
    auto d = model::qwen35::build_step_graph(cfg, 16, 1);
    auto p64 = model::qwen35::build_step_graph(cfg, 16, 64);
    if (!d.ok() || !p64.ok()) {
        std::printf("FAIL: グラフを作れない\n");
        return 1;
    }
    const Baseline b1 = measure(d.value(), 1);
    const Baseline b64 = measure(p64.value(), 64);

    // 基準値の取り直し用（値をそのまま貼れる形で出す）。
    if (argc > 1 && std::string(argv[1]) == "--dump") {
        std::printf("constexpr std::size_t kBaselineWeights = %zu;\n", b1.weights_bytes);
        std::printf("constexpr std::size_t kBaselineActRows1 = %zu;\n", b1.act_bytes_rows1);
        std::printf("constexpr std::size_t kBaselineActRows64 = %zu;\n",
                    b64.act_bytes_rows64);
        std::printf("const std::map<std::string, std::size_t> kBaselinePrims = {\n");
        for (const auto& [kind, n] : b1.prims)
            std::printf("    {\"%s\", %zu},\n", kind.c_str(), n);
        std::printf("};\n");
        return 0;
    }

    // ---- 1. 重みは 1 ステップで 1 回だけ読む ----
    expect_eq("decode の重みバイト数", b1.weights_bytes, kBaselineWeights);
    // ---- 2. prefill でも重みの読み直しが無い（行数に比例しない）----
    expect_eq("prefill(64 行) の重みバイト数", b64.weights_bytes, kBaselineWeights);
    // ---- 3. 活性化は行数に比例する ----
    expect_eq("rows=1 の活性化バイト数", b1.act_bytes_rows1, kBaselineActRows1);
    expect_eq("rows=64 の活性化バイト数", b64.act_bytes_rows64, kBaselineActRows64);
    if (b64.act_bytes_rows64 < 60 * b1.act_bytes_rows1) {
        ++failures;
        std::printf("FAIL: 活性化が行数に比例していない（%zu < 60 × %zu）\n",
                    b64.act_bytes_rows64, b1.act_bytes_rows1);
    }
    // ---- 4. primitive 個数（グラフ形状）----
    for (const auto& [kind, want] : kBaselinePrims) {
        const auto it = b1.prims.find(kind);
        const std::size_t got = it == b1.prims.end() ? 0 : it->second;
        if (got != want) {
            ++failures;
            std::printf("FAIL: primitive %s: 実測 %zu / 基準 %zu\n", kind.c_str(), got, want);
        }
    }
    for (const auto& [kind, got] : b1.prims) {
        if (!kBaselinePrims.count(kind)) {
            ++failures;
            std::printf("FAIL: 基準に無い primitive: %s = %zu\n", kind.c_str(), got);
        }
    }

    std::printf("test_cost_model: weights=%zu B, act(1)=%zu B, act(64)=%zu B, "
                "prims=%zu kinds / failures=%d\n",
                b1.weights_bytes, b1.act_bytes_rows1, b64.act_bytes_rows64, b1.prims.size(),
                failures);
    return failures == 0 ? 0 : 1;
}
