#include <shelf/models/qwen35/model/qwen35_config.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using shelf::qwen35::Qwen35TextConfig;
using shelf::qwen35::read_qwen35_text_config;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

fs::path g_root;

void write_config(const std::string& name, const std::string& text) {
    const fs::path dir = g_root / name;
    fs::create_directories(dir);
    std::ofstream f(dir / "config.json", std::ios::binary | std::ios::trunc);
    f << text;
}

std::string with_text_config(const std::string& text_body) {
    return R"json({"model_type":"qwen3_5","text_config":{)json" + text_body + R"json(}})json";
}

Qwen35TextConfig expect_ok(const std::string& name, const std::string& text) {
    write_config(name, text);
    auto cfg = read_qwen35_text_config((g_root / name).string());
    check(cfg.ok(), name + ": parse: " + (cfg.ok() ? "" : cfg.status().message()));
    if (!cfg.ok())
        return Qwen35TextConfig{};
    return cfg.value();
}

void expect_error(const std::string& name, const std::string& text) {
    write_config(name, text);
    auto cfg = read_qwen35_text_config((g_root / name).string());
    check(!cfg.ok(), name + ": config must be rejected");
}

// Measured values of models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 text_config.
constexpr const char* kMoeBase =
    R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256)json";

constexpr const char* kDenseBase =
    R"json("vocab_size":248320,"hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,"num_attention_heads":48,"num_key_value_heads":8,"head_dim":128)json";

}  // namespace

int main() {
    g_root = fs::temp_directory_path() / "shelf_test_moe_config";
    fs::remove_all(g_root);
    fs::create_directories(g_root);

    {
        const auto cfg = expect_ok("moe", with_text_config(kMoeBase));
        check(cfg.is_moe(), "moe: num_experts > 0 selects the MoE contract");
        check(cfg.num_experts == 256u, "moe: num_experts");
        check(cfg.num_experts_per_tok == 8u, "moe: num_experts_per_tok");
        check(cfg.moe_intermediate_size == 512u, "moe: moe_intermediate_size");
        check(cfg.shared_expert_intermediate_size == 512u,
              "moe: shared_expert_intermediate_size");
        check(cfg.intermediate_size == 0u, "moe: no dense FFN field");
        check(cfg.output_gate_silu, "moe: a missing output_gate_type defaults to silu");
        check(cfg.hidden_size == 2048u && cfg.num_hidden_layers == 40u, "moe: geometry kept");
    }

    {
        const auto cfg = expect_ok("dense", with_text_config(
            std::string(kDenseBase) + R"json(,"output_gate_type":"swish")json"));
        check(!cfg.is_moe(), "dense: num_experts absent is not MoE");
        check(cfg.intermediate_size == 17408u, "dense: intermediate_size");
        check(cfg.output_gate_silu, "dense: swish gate");
    }

    {
        const auto cfg = expect_ok("dense_sigmoid", with_text_config(
            std::string(kDenseBase) + R"json(,"output_gate_type":"sigmoid")json"));
        check(!cfg.output_gate_silu, "dense: explicit sigmoid is honoured");
    }

    {
        const auto cfg = expect_ok("dense_missing_gate", with_text_config(kDenseBase));
        check(cfg.output_gate_silu, "a missing output_gate_type defaults to silu");
    }

    expect_error("bad_gate", with_text_config(
        std::string(kDenseBase) + R"json(,"output_gate_type":"gelu")json"));
    expect_error("moe_topk_too_large",
                 with_text_config(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":257,"moe_intermediate_size":512,"shared_expert_intermediate_size":512)json"));
    expect_error("moe_topk_zero",
                 with_text_config(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":0,"moe_intermediate_size":512,"shared_expert_intermediate_size":512)json"));
    expect_error("moe_no_moe_intermediate",
                 with_text_config(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":8,"shared_expert_intermediate_size":512)json"));
    expect_error("moe_no_shared_expert_size",
                 with_text_config(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512)json"));
    expect_error("moe_with_dense_ffn", with_text_config(
        std::string(kMoeBase) + R"json(,"intermediate_size":512)json"));

    fs::remove_all(g_root);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
