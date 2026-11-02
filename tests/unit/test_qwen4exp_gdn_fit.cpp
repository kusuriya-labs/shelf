#include <phaseshift/models/qwen35/model/lower_to_primitives.h>
#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/inference/execution/selectors/elementwise_selector.h>
#include <phaseshift/inference/execution/selectors/gdn_conv_selector.h>
#include <phaseshift/inference/execution/selectors/gdn_recurrence_selector.h>
#include <phaseshift/inference/execution/selectors/l2_normalize_selector.h>
#include <phaseshift/models/qwen4exp/model/qwen4exp_config.h>
#include <phaseshift/runtime/graph/primitive_graph.h>
#include <phaseshift/runtime/graph/primitive_node.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace rt = ::ps::runtime;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL: %s\n", msg.c_str());
    }
}

static const char* kLayerTypes =
    "\"layer_types\":["
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\","
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"]";

static std::string make_config(const std::string& gate_type) {
    return std::string("{\"model_type\":\"qwen4_exp\",\"text_config\":{")
           + "\"vocab_size\":248320,\"hidden_size\":2560,\"num_hidden_layers\":48,"
           + "\"num_attention_heads\":24,\"num_key_value_heads\":2,\"head_dim\":256,"
           + "\"full_attention_interval\":4,"
           + "\"linear_num_key_heads\":16,\"linear_num_value_heads\":48,"
           + "\"linear_key_head_dim\":128,\"linear_value_head_dim\":128,"
           + "\"linear_conv_kernel_dim\":4,"
           + "\"intermediate_size\":9216,"
           + "\"num_experts\":512,\"num_experts_per_tok\":10,"
           + "\"moe_intermediate_size\":640,\"shared_expert_intermediate_size\":640,"
           + "\"hc_count\":4,\"hc_lowrank\":320,"
           + "\"ngram_size\":3,\"heads_per_ngram\":8,\"ngram_vocab_size_base\":20000000,"
           + "\"make_ngram_vocab_size_divisible_by\":128,\"split_ngram_parts\":128,"
           + "\"ple_layer_ids\":[2],\"ple_embed_dim\":2560,\"ple_conv_kernel_size\":4,"
           + "\"indexer_n_heads\":4,\"indexer_kv_heads\":1,\"indexer_head_dim\":128,"
           + "\"indexer_budget\":2048,\"indexer_compress_ratio\":4,"
           + "\"rms_norm_eps\":1e-6,"
           + "\"output_gate_type\":\"" + gate_type + "\","
           + "\"tie_word_embeddings\":false,"
           + "\"rope_parameters\":{\"rope_theta\":10000000,"
           + "\"partial_rotary_factor\":0.25,\"mrope_section\":[11,11,10],"
           + "\"mrope_interleaved\":true,\"rope_type\":\"default\"},"
           + "\"eos_token_id\":248044,"
           + kLayerTypes + "}}";
}

static rt::PrimitiveKind kind_of(const rt::PrimitiveGraphNode& n) noexcept {
    return rt::primitive_kind_of(n.node);
}

struct GdnFitFixture {
    constexpr static std::uint32_t kHidden = 2560;
    constexpr static std::uint32_t kInter = 640;
    constexpr static std::uint32_t kKeyHeads = 16;
    constexpr static std::uint32_t kValHeads = 48;
    constexpr static std::uint32_t kHeadDim = 128;
    constexpr static std::uint32_t kConv = 4;
    constexpr static std::uint32_t kQk = kKeyHeads * kHeadDim;
    constexpr static std::uint32_t kVal = kValHeads * kHeadDim;
    constexpr static std::uint32_t kConvF = 2 * kQk + kVal;

    ::ps::qwen35::Qwen35TextConfig cfg;
    ::ps::qwen35::Qwen35ModelWeights w;

    static ::ps::weights::MatrixWeight make_matrix(std::uint32_t rows,
                                                   std::uint32_t cols) {
        ::ps::weights::MatrixWeight m;
        m.rows = rows;
        m.cols = cols;
        m.k_padded = cols;
        return m;
    }

    explicit GdnFitFixture(bool output_gate_silu) {
        cfg.vocab_size = 8;
        cfg.hidden_size = kHidden;
        cfg.intermediate_size = kInter;
        cfg.num_hidden_layers = 1;
        cfg.linear_num_key_heads = kKeyHeads;
        cfg.linear_num_value_heads = kValHeads;
        cfg.linear_key_head_dim = kHeadDim;
        cfg.linear_value_head_dim = kHeadDim;
        cfg.linear_conv_kernel_dim = kConv;
        cfg.num_attention_heads = 24;
        cfg.num_key_value_heads = 2;
        cfg.attention_head_dim = kHeadDim;
        cfg.rms_norm_eps = 1e-6f;
        cfg.rope_theta = 1e7f;
        cfg.partial_rotary_factor = 0.25f;
        cfg.tie_word_embeddings = true;
        cfg.output_gate_silu = output_gate_silu;
        cfg.layer_types = {0};

        w.lm_head_tied = true;
        w.embed_tokens = make_matrix(8, kHidden);

        ::ps::qwen35::Qwen35LayerWeights g;
        g.is_gdn = true;
        g.mlp_gate_proj = make_matrix(kInter, kHidden);
        g.mlp_up_proj = make_matrix(kInter, kHidden);
        g.mlp_down_proj = make_matrix(kHidden, kInter);
        g.attn_in_proj_qkv = make_matrix(kConvF, kHidden);
        g.attn_in_proj_z = make_matrix(kVal, kHidden);
        g.attn_in_proj_b = make_matrix(kValHeads, kHidden);
        g.attn_in_proj_a = make_matrix(kValHeads, kHidden);
        g.attn_out_proj = make_matrix(kHidden, kVal);
        w.layers.push_back(std::move(g));
    }
};

static const ::ps::runtime::PrimitiveGraphNode* find_named(
    const ::ps::runtime::PrimitiveGraph& g, const std::string& name) {
    for (const auto& n : g.nodes) {
        if (n.debug_name == name) return &n;
    }
    return nullptr;
}

static void verify_gdn_nodes(const ::ps::runtime::PrimitiveGraph& g,
                             bool expect_sigmoid, const std::string& label) {
    const auto* gate = find_named(g, "L0.gdn_z_gate");
    check(gate != nullptr, label + ": gdn_z_gate node present");
    if (gate != nullptr) {
        const auto k = kind_of(*gate);
        check(k == (expect_sigmoid ? rt::PrimitiveKind::SIGMOID
                                   : rt::PrimitiveKind::SILU),
              label + ": gdn_z_gate kind matches output_gate_type");
    }
    const rt::GdnShapeKey expect{16, 48, 128, 128, 4};
    const auto* rec = find_named(g, "L0.gdn_recurrence");
    check(rec != nullptr, label + ": gdn_recurrence node present");
    if (rec != nullptr) {
        check(std::holds_alternative<rt::GdnRecurrenceNode>(rec->node),
              label + ": gdn_recurrence variant");
        if (std::holds_alternative<rt::GdnRecurrenceNode>(rec->node)) {
            check(std::get<rt::GdnRecurrenceNode>(rec->node).gdn == expect,
                  label + ": recurrence GdnShapeKey 16/48/128/128/4");
        }
    }
    const auto* conv = find_named(g, "L0.gdn_conv");
    check(conv != nullptr, label + ": gdn_conv node present");
    if (conv != nullptr &&
        std::holds_alternative<rt::StatefulCausalConv1DNode>(conv->node)) {
        check(std::get<rt::StatefulCausalConv1DNode>(conv->node).gdn == expect,
              label + ": conv GdnShapeKey 16/48/128/128/4");
    }
    const auto* nq = find_named(g, "L0.gdn_q_normalize");
    check(nq != nullptr, label + ": gdn_q_normalize node present");
    const auto* nk = find_named(g, "L0.gdn_k_normalize");
    check(nk != nullptr, label + ": gdn_k_normalize node present");
    const auto* sq = find_named(g, "L0.gdn_q_scale");
    check(sq != nullptr, label + ": gdn_q_scale node present");
    const auto* norm = find_named(g, "L0.gdn_norm");
    check(norm != nullptr, label + ": gdn_norm node present");
    const auto* gated = find_named(g, "L0.gdn_gated");
    check(gated != nullptr, label + ": gdn_gated node present");
    const auto* out = find_named(g, "L0.gdn_out");
    check(out != nullptr, label + ": gdn_out node present");

    std::uint32_t rec_states = 0;
    std::uint32_t conv_states = 0;
    for (const auto& s : g.states) {
        if (s.kind == rt::PrimitiveStateKind::GDN_RECURRENCE_STATE) ++rec_states;
        if (s.kind == rt::PrimitiveStateKind::GDN_CONV_STATE) ++conv_states;
    }
    check(rec_states == 1, label + ": exactly one recurrent state allocated");
    check(conv_states == 1, label + ": exactly one conv state allocated");
}

int main() {
    const fs::path root = fs::temp_directory_path() / "qwen4exp_gdn_fit";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);

    {
        const fs::path dir = root / "sigmoid";
        fs::create_directories(dir, ec);
        std::ofstream f(dir / "config.json", std::ios::binary | std::ios::trunc);
        f << make_config("sigmoid");
    }
    {
        const fs::path dir = root / "silu";
        fs::create_directories(dir, ec);
        std::ofstream f(dir / "config.json", std::ios::binary | std::ios::trunc);
        f << make_config("silu");
    }

    const std::string sig_dir = (root / "sigmoid").string();
    const std::string silu_dir = (root / "silu").string();

    auto sig_cfg = ps::qwen4exp::read_qwen4exp_text_config(sig_dir);
    check(sig_cfg.ok(), "qwen4exp sigmoid config parse: " +
                            (sig_cfg.ok() ? "" : sig_cfg.status().message()));
    auto silu_cfg = ps::qwen4exp::read_qwen4exp_text_config(silu_dir);
    check(silu_cfg.ok(), "qwen4exp silu config parse: " +
                             (silu_cfg.ok() ? "" : silu_cfg.status().message()));

    if (!sig_cfg.ok() || !silu_cfg.ok()) {
        fs::remove_all(root, ec);
        std::printf("%d passed, %d failed\n", g_pass, g_fail);
        return 1;
    }

    check(sig_cfg.value().output_gate_silu == false,
          "qwen4exp sigmoid config -> output_gate_silu=false");
    check(silu_cfg.value().output_gate_silu == true,
          "qwen4exp silu config -> output_gate_silu=true");
    check(sig_cfg.value().linear_num_key_heads == 16,
          "qwen4exp linear_num_key_heads == 16");
    check(sig_cfg.value().linear_num_value_heads == 48,
          "qwen4exp linear_num_value_heads == 48");
    check(sig_cfg.value().linear_key_head_dim == 128,
          "qwen4exp linear_key_head_dim == 128");
    check(sig_cfg.value().linear_value_head_dim == 128,
          "qwen4exp linear_value_head_dim == 128");
    check(sig_cfg.value().linear_conv_kernel_dim == 4,
          "qwen4exp linear_conv_kernel_dim == 4");
    check(sig_cfg.value().num_experts == 512, "qwen4exp num_experts == 512");
    check(sig_cfg.value().num_experts_per_tok == 10,
          "qwen4exp num_experts_per_tok == 10");
    check(sig_cfg.value().moe_intermediate_size == 640,
          "qwen4exp moe_intermediate_size == 640");
    check(sig_cfg.value().shared_expert_intermediate_size == 640,
          "qwen4exp shared_expert_intermediate_size == 640");
    check(sig_cfg.value().norm_topk_prob == true,
          "qwen4exp norm_topk_prob == true");

    {
        GdnFitFixture sig_fx(false);
        auto sig_lower = ps::qwen35::lower_qwen35_to_primitives(sig_fx.cfg, sig_fx.w);
        check(sig_lower.ok(), "gdn sigmoid lowering: " +
                                  (sig_lower.ok() ? ""
                                                  : sig_lower.status().message()));
        if (sig_lower.ok()) {
            verify_gdn_nodes(sig_lower.value().graph, true, "sigmoid");
        }
    }
    {
        GdnFitFixture silu_fx(true);
        auto silu_lower = ps::qwen35::lower_qwen35_to_primitives(silu_fx.cfg, silu_fx.w);
        check(silu_lower.ok(), "gdn silu lowering: " +
                                   (silu_lower.ok() ? ""
                                                    : silu_lower.status().message()));
        if (silu_lower.ok()) {
            verify_gdn_nodes(silu_lower.value().graph, false, "silu");
        }
    }

    {
        ps::execution::ElementwiseSelectorInput ew;
        ew.profile = ps::execution::ElementwiseProfile::SigmoidBf16ToF32;
        ew.rows = 8;
        ew.features = 6144;
        ew.aux = 0;
        check(ps::execution::select_elementwise_implementation(ew) ==
                  ps::execution::ElementwiseImplementation::Optimized,
              "selector: sigmoid bf16->f32 6144 rows picks Optimized");

        ps::execution::GdnConvSelectorInput cv;
        cv.input_dtype = rt::ValueDType::BF16;
        cv.output_dtype = rt::ValueDType::F32;
        cv.conv_dim = 10240;
        cv.history = 3;
        cv.rows = 64;
        cv.num_requests = 1;
        cv.max_request_rows = 64;
        check(ps::execution::select_gdn_conv_implementation(cv) ==
                  ps::execution::GdnConvImplementation::Optimized,
              "selector: conv_dim 10240 history 3 picks Optimized");

        ps::execution::GdnRecurrenceSelectorInput rc;
        rc.rows = 64;
        rc.num_requests = 1;
        rc.max_request_rows = 64;
        rc.key_heads = 16;
        rc.num_v_heads = 48;
        rc.head_k = 128;
        rc.head_v = 128;
        check(ps::execution::select_gdn_recurrence_implementation(rc) ==
                  ps::execution::GdnRecurrenceImplementation::Optimized,
              "selector: geometry 16/48/128/128 picks Optimized");

        ps::execution::L2NormalizeSelectorInput l2;
        l2.input_dtype = rt::ValueDType::BF16;
        l2.output_dtype = rt::ValueDType::F32;
        l2.features = 2048;
        l2.group_size = 128;
        l2.rows = 64;
        check(ps::execution::select_l2_normalize_implementation(l2) ==
                  ps::execution::L2NormalizeImplementation::Optimized,
              "selector: l2 2048/128 picks Optimized");
    }

    fs::remove_all(root, ec);

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
