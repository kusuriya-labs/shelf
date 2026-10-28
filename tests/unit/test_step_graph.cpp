// shelf unit test — Qwen3.5 step graph lowering: 出力投影（lm_head）の束縛。
//
// untied（tie_word_embeddings = false）では出力投影が lm_head.weight を使い、
// tied のときだけ埋め込み表を共有する。2026-10-28 のバグの回帰防止:
// untied でも embed_tokens を束ねており、hidden は HF と一致するのに
// logits が埋め込みとの内積になって出力が劣化していた（Qwen3.5-9B で観測）。
#include <cstdio>
#include <string>

#include <shelf/model/qwen35/step_graph.hpp>

namespace {

using namespace shelf;
using namespace shelf::model::qwen35;

int failures = 0;
void check(bool cond, const char* what) {
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

// 解析器が要求する最小の config（1 層の full_attention・テキストのみ）
std::string cfg_json(bool tie) {
    static const char* kHead = R"({
  "text_config": {
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "vocab_size": 128,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 32,
    "rms_norm_eps": 1e-6,
    "eos_token_id": 1,
    "rope_parameters": {"rope_theta": 10000.0},
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 1,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 4,
    "layer_types": ["full_attention"],
    "tie_word_embeddings": )";
    static const char* kTail = R"(
  }
})";
    return std::string(kHead) + (tie ? "true" : "false") + kTail;
}

void test_case(bool tie) {
    auto cfg_r = load_text_config_json(cfg_json(tie));
    if (!cfg_r.ok()) {
        std::printf("config: %s\n", cfg_r.status().message().c_str());
        check(false, "config parses");
        return;
    }
    check(cfg_r.value().tie_word_embeddings == tie, "tie parsed from text_config");

    auto plan_r = build_step_graph(cfg_r.value(), /*kv_capacity*/ 8);
    if (!plan_r.ok()) {
        std::printf("plan: %s\n", plan_r.status().message().c_str());
        check(false, "graph builds");
        return;
    }
    const auto& plan = plan_r.value();
    check(plan.tied_embeddings == tie, "plan.tied_embeddings mirrors config");

    int n_out = 0, n_emb = 0;
    TensorId out_w, emb_w;
    for (const auto& p : plan.graph.primitives()) {
        if (p.kind == ir::Kind::output_projection && !p.weights.empty()) {
            ++n_out;
            out_w = p.weights[0];
        }
        if (p.kind == ir::Kind::embedding && !p.weights.empty()) {
            ++n_emb;
            emb_w = p.weights[0];
        }
    }
    check(n_out == 1, "exactly one output_projection");
    check(n_emb == 1, "exactly one embedding");
    if (n_out != 1 || n_emb != 1) return;

    std::string out_name;
    bool has_lm_head = false;
    for (const auto& w : plan.weights) {
        if (w.id == out_w) out_name = w.checkpoint_name;
        if (w.checkpoint_name == "lm_head.weight") has_lm_head = true;
    }

    if (tie) {
        check(out_w == emb_w, "tied: output shares the embedding weight id");
        check(!has_lm_head, "tied: plan carries no separate lm_head");
        check(out_name == "model.language_model.embed_tokens.weight",
              "tied: output binds embed_tokens");
    } else {
        check(out_w != emb_w, "untied: output must not reuse the embedding id");
        check(has_lm_head, "untied: plan carries lm_head.weight");
        check(out_name == "lm_head.weight", "untied: output binds lm_head.weight");
        for (const auto& w : plan.weights) {
            if (w.checkpoint_name == "lm_head.weight") {
                check(w.shape.size() == 2 && w.shape[0] == 128 && w.shape[1] == 16,
                      "untied: lm_head shape is [vocab, hidden]");
            }
        }
    }
}

} // namespace

int main() {
    test_case(false);   // untied（Qwen3.5-9B 相当）
    test_case(true);    // tied
    std::printf("test_step_graph: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
