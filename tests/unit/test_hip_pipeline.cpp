// shelf unit test — pipeline-parallel（レイヤ分割）が数値を変えないこと。
//
// 検証: 同じ重み・同じ入力で
//   ① 1 ステージ（全層を 1 実行器で）= 基準
//   ② 2 ステージ（レイヤで分割し、bf16 生ビットを host 経由で受け渡し）
//   ③ 不均等な 2 分割（層数 5 で [0,2) / [2,5)）
// の logits が **ビット一致** することを見る（Strata の `--split-device 0` と同型）。
// 全ステージを同じ GPU に置くので 1 台で走る。
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <shelf/backend/hip/step_executor.hpp>
#include <shelf/model/qwen35/step_graph.hpp>

#include "common/pipeline.hpp"

using namespace shelf;

static int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL: %s\n", msg);                                  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

// Status 付き（失敗理由を出す）
#define CHECK_ST(cond, msg, st)                                              \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL: %s: %s\n", msg, (st).message().c_str());       \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

namespace {

std::size_t numelof(const std::vector<std::size_t>& s) {
    std::size_t n = 1;
    for (auto d : s) n *= d;
    return n;
}

// 名前から決まる重み（同じ名前なら構成が違っても同じ値になる）。
std::vector<float> weight_for(const std::string& name, std::size_t n) {
    std::mt19937 rng(static_cast<std::uint32_t>(std::hash<std::string>{}(name) & 0x7fffffffu));
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> f(n);
    for (auto& x : f) x = nd(rng);
    return f;
}

model::qwen35::TextConfig make_cfg(std::size_t layers) {
    model::qwen35::TextConfig cfg;
    cfg.hidden_size = 512;
    cfg.num_layers = layers;
    cfg.vocab_size = 1024;
    cfg.num_heads = 4;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 256;   // rope pair kernel の制約
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
    // linear_attention と full_attention を交互に混ぜる（Hybrid の経路を通す）。
    for (std::size_t l = 0; l < layers; ++l)
        cfg.layer_kinds.push_back((l % 2 == 0) ? model::qwen35::LayerKind::linear_attention
                                               : model::qwen35::LayerKind::full_attention);
    return cfg;
}

backend::hip::HipStepExecutorOptions exec_opts(const model::qwen35::TextConfig& cfg) {
    backend::hip::HipStepExecutorOptions ho;
    ho.device = 0;
    ho.rope_theta = cfg.rope_theta;
    ho.rope_rotary_dim = cfg.rotary_dim;
    return ho;
}

void upload_synthetic(backend::hip::HipStepExecutor& ex,
                      const model::qwen35::StepGraphPlan& plan) {
    for (const auto& w : plan.weights) {
        const auto f = weight_for(w.checkpoint_name, numelof(w.shape));
        auto st = ex.upload_weight(w.id, f);
        CHECK_ST(st.ok(), "upload weight", st);
    }
    auto st = ex.declare_states(plan.graph);
    CHECK_ST(st.ok(), "declare states", st);
}

// グラフ構築は失敗したら理由を出して即終了（release() は ok 前提のため）。
model::qwen35::StepGraphPlan must(Result<model::qwen35::StepGraphPlan> r, const char* what) {
    if (!r.ok()) {
        std::printf("FAIL: %s: %s\n", what, r.status().message().c_str());
        std::exit(1);
    }
    return r.release();
}

// 1 ステージ構成（全層）。
struct Single {
    model::qwen35::StepGraphPlan decode;
    std::unique_ptr<model::qwen35::StepGraphPlan> prefill;
    std::unique_ptr<backend::hip::HipStepExecutor> exec;
};

Single make_single(const model::qwen35::TextConfig& cfg, std::size_t chunk) {
    Single s;
    s.decode = must(model::qwen35::build_step_graph(cfg, 16, 1), "single decode graph");
    if (chunk > 1) {
        s.prefill = std::make_unique<model::qwen35::StepGraphPlan>(
            must(model::qwen35::build_step_graph(cfg, 16, chunk), "single prefill graph"));
    }
    s.exec = std::make_unique<backend::hip::HipStepExecutor>(exec_opts(cfg));
    upload_synthetic(*s.exec, s.prefill ? *s.prefill : s.decode);
    return s;
}

// K ステージ構成（全ステージを同じ GPU に置く）。
app::Pipeline make_staged(const model::qwen35::TextConfig& cfg,
                          const std::vector<model::qwen35::StageRange>& ranges,
                          std::size_t chunk) {
    app::Pipeline p;
    p.hidden_size = cfg.hidden_size;
    p.handoff.assign(cfg.hidden_size * (chunk > 1 ? chunk : 1), bf16_t{0u});
    for (const auto& r : ranges) {
        app::PipelineStage st;
        st.layer_lo = r.layer_lo;
        st.layer_hi = r.layer_hi;
        st.decode = must(model::qwen35::build_step_graph(cfg, 16, 1, r), "stage decode graph");
        if (chunk > 1) {
            st.prefill = std::make_unique<model::qwen35::StepGraphPlan>(
                must(model::qwen35::build_step_graph(cfg, 16, chunk, r), "stage prefill graph"));
        }
        st.exec = std::make_unique<backend::hip::HipStepExecutor>(exec_opts(cfg));
        upload_synthetic(*st.exec, st.prefill ? *st.prefill : st.decode);
        p.stages.push_back(std::move(st));
    }
    return p;
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}


}  // namespace

int main() {
    CHECK(hipSetDevice(0) == hipSuccess, "hipSetDevice");

    // ---- 均等分割の性質 ----
    {
        const auto r4 = app::even_stage_ranges(4, 2);
        CHECK(r4.size() == 2 && r4[0].layer_lo == 0 && r4[0].layer_hi == 2 &&
                  r4[1].layer_lo == 2 && r4[1].layer_hi == 4,
              "4 層を 2 分割");
        const auto r5 = app::even_stage_ranges(5, 2);   // 余りは最後へ
        CHECK(r5.size() == 2 && r5[0].layer_hi == 2 && r5[1].layer_lo == 2 &&
                  r5[1].layer_hi == 5,
              "5 層を 2 分割（余りは最終ステージ）");
        CHECK(app::even_stage_ranges(4, 8).empty(), "層数より多い分割は不可");
        CHECK(app::even_stage_ranges(4, 0).empty(), "0 分割は不可");
        CHECK(app::even_stage_ranges(4, 1).size() == 1, "1 分割はそのまま");
    }

    // ---- ① 1 ステージ vs ② 2 ステージ（4 層を 2 分割）----
    {
        const auto cfg = make_cfg(4);
        Single s = make_single(cfg, 1);
        app::Pipeline p = make_staged(cfg, app::even_stage_ranges(cfg.num_layers, 2), 1);

        const std::vector<std::int32_t> toks = {7, 111, 512, 900};
        for (std::size_t i = 0; i < toks.size(); ++i) {
            const auto pos = static_cast<std::uint32_t>(i);
            auto st1 = s.exec->run_step(s.decode.graph, toks[i], pos);
            CHECK_ST(st1.ok(), "single step", st1);
            auto ref = s.exec->fetch_logits(s.decode.graph);
            CHECK(ref.ok(), "single logits");

            std::vector<float> got;
            auto st4 = app::pipeline_step(p, &toks[i], 1, pos, &got);
            CHECK_ST(st4.ok(), "pipeline step", st4);
            const std::vector<float> refv = ref.ok() ? ref.value() : std::vector<float>{};
            const bool same = bit_equal(refv, got);
            CHECK(same, "2 ステージの logits が 1 ステージとビット一致");
            if (!same) {
                double worst = 0.0;
                for (std::size_t j = 0; j < got.size() && j < refv.size(); ++j)
                    worst = std::max(worst, std::abs(double(got[j]) - double(refv[j])));
                std::printf("  step %zu: worst|diff|=%.3g (sizes %zu vs %zu, ref_ok=%d)\n", i,
                            worst, got.size(), refv.size(), int(ref.ok()));
            }
        }

        // 分割位置を変えても同じ（[0,1)/[1,4) と [0,3)/[3,4)）
        for (const auto& ranges : {std::vector<model::qwen35::StageRange>{
                                      {0, 1}, {1, 2}, {2, 3}, {3, 4}},
                                  std::vector<model::qwen35::StageRange>{{0, 3}, {3, 4}}}) {
            Single ref = make_single(cfg, 1);
            app::Pipeline pp = make_staged(cfg, ranges, 1);
            std::vector<float> a, b;
            CHECK(ref.exec->run_step(ref.decode.graph, 7, 0).ok(), "ref step");
            auto rl = ref.exec->fetch_logits(ref.decode.graph);
            CHECK(rl.ok(), "ref logits");
            CHECK(app::pipeline_step(pp, &toks[0], 1, 0, &b).ok(), "split step");
            const std::vector<float> rlv = rl.ok() ? rl.value() : std::vector<float>{};
            CHECK(bit_equal(rlv, b), "分割数を変えてもビット一致（4 分割・不均等 2 分割）");
        }
    }

    // ---- ③ prefill チャンク（4 行）でも一致 ----
    {
        const auto cfg = make_cfg(4);
        const std::size_t chunk = 4;
        Single s = make_single(cfg, chunk);
        app::Pipeline p = make_staged(cfg, app::even_stage_ranges(cfg.num_layers, 2), chunk);
        const std::vector<std::int32_t> toks = {7, 111, 512, 900};

        auto stc = s.exec->run_step(s.prefill->graph, toks.data(), chunk, 0);
        CHECK_ST(stc.ok(), "single chunk", stc);
        auto ref = s.exec->fetch_logits(s.prefill->graph);
        CHECK(ref.ok(), "single chunk logits");

        std::vector<float> got;
        auto stp = app::pipeline_step(p, toks.data(), chunk, 0, &got);
        CHECK_ST(stp.ok(), "pipeline chunk", stp);
        const std::vector<float> refv = ref.ok() ? ref.value() : std::vector<float>{};
        CHECK(bit_equal(refv, got), "prefill 4 行チャンクでもビット一致");
        if (!bit_equal(refv, got)) {
            double worst = 0.0;
            for (std::size_t j = 0; j < got.size() && j < refv.size(); ++j)
                worst = std::max(worst, std::abs(double(got[j]) - double(refv[j])));
            std::printf("  chunk: worst|diff|=%.3g (sizes %zu vs %zu, ref_ok=%d)\n", worst,
                        got.size(), refv.size(), int(ref.ok()));
        }
    }

    // ---- ④ 5 層（奇数）を 2 分割して decode 継続 ----
    {
        const auto cfg = make_cfg(5);
        Single s = make_single(cfg, 1);
        app::Pipeline p = make_staged(cfg, app::even_stage_ranges(cfg.num_layers, 2), 1);
        const std::vector<std::int32_t> toks = {3, 42, 777};
        bool all_eq = true;
        for (std::size_t i = 0; i < toks.size(); ++i) {
            auto st2 = s.exec->run_step(s.decode.graph, toks[i], std::uint32_t(i));
            CHECK_ST(st2.ok(), "odd single step", st2);
            auto ref = s.exec->fetch_logits(s.decode.graph);
            std::vector<float> got;
            auto st3 = app::pipeline_step(p, &toks[i], 1, std::uint32_t(i), &got);
            CHECK_ST(st3.ok(), "odd pipe", st3);
            if (!ref.ok() || !bit_equal(ref.value(), got)) all_eq = false;
        }
        CHECK(all_eq, "5 層の不均等 2 分割でもビット一致");
        CHECK(app::even_stage_ranges(cfg.num_layers, 2)[0].layer_hi == 2,
              "奇数層の境界（[0,2)/[2,5)）");
    }

    std::printf("test_hip_pipeline: failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
