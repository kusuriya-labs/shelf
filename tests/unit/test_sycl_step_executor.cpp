// shelf unit test — SYCL executor: batched prefill step must equal
// sequential decode steps (port of test_hip_step_executor.cpp).
//
// On devices without 32-lane sub-groups (e.g. the AdaptiveCpp OpenMP
// backend) run_step reports unsupported and the test SKIPs with success —
// the wave32 kernel family needs GPU wave semantics.
#include <cmath>
#include <cstdio>
#include <string>
#include <random>
#include <vector>

#include <shelf/backend/sycl/step_executor.hpp>
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

    backend::sycl::SyclStepExecutorOptions ho;
    ho.rope_theta = cfg.rope_theta;
    ho.rope_rotary_dim = cfg.rotary_dim;
    ho.arena_bytes = 16u << 20;
    backend::sycl::SyclStepExecutor ex(ho);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 0.05f);
    for (const auto& w : pd.weights) {
        std::vector<float> f(numelof(w.shape));
        for (auto& x : f) x = nd(rng);
        CHECK(ex.upload_weight(w.id, f).ok(), "upload weight");
    }
    CHECK(ex.declare_states(pd.graph).ok(), "declare states");

    std::vector<std::int32_t> toks = {7, 111, 512, 900};

    const Status first = ex.run_step(pd.graph, toks[0], 0u);
    if (first.code() == Status::Code::unsupported) {
        std::printf("test_sycl_step_executor: SKIP (%s)\n", first.message().c_str());
        return 0;
    }
    CHECK(first.ok(), "seq step 0");

    // sequential
    for (std::size_t i = 1; i < toks.size(); ++i)
        CHECK(ex.run_step(pd.graph, toks[i], static_cast<std::uint32_t>(i)).ok(),
              "seq step");
    auto lseq = ex.fetch_logits(pd.graph);
    CHECK(lseq.ok() && lseq.value().size() == cfg.vocab_size, "seq logits");

    CHECK(ex.reset_states().ok(), "reset");

    // batched
    CHECK(ex.run_step(pb.graph, toks.data(), toks.size(), 0).ok(), "batch step");
    auto lbat = ex.fetch_logits(pb.graph);
    CHECK(lbat.ok() && lbat.value().size() == toks.size() * cfg.vocab_size,
          "batch logits [P,V]");
    if (!lseq.ok() || !lbat.ok()) return failures ? 1 : 0;

    // last row of batch vs sequential final logits (bf16 GPU numerics vs
    // themselves across launch shapes: same kernels, same order)
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
    CHECK(worst < 1e-2, "logit deviation batched vs sequential");
    CHECK(ta == tb, "top-1 identity");

    // decode after prefill continues correctly: pos 4 single step both paths
    CHECK(ex.run_step(pd.graph, 42, 4).ok(), "decode after batch");
    auto l2 = ex.fetch_logits(pd.graph);
    CHECK(l2.ok() && l2.value().size() == V, "post-decode logits");

    std::printf("test_sycl_step_executor: failures=%d\n", failures);
    return failures ? 1 : 0;
}
