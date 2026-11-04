#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;
using shelf::qwen35::Qwen35TextConfig;
using shelf::qwen35::read_qwen35_text_config;
using shelf::qwen35moe::Qwen35MoeTextConfig;
using shelf::qwen35moe::read_qwen35_moe_text_config;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

bool message_contains(const shelf::Status& st, const std::string& needle) {
    return std::string(st.message()).find(needle) != std::string::npos;
}

fs::path g_root;

void write_config(const std::string& name, const std::string& text) {
    const fs::path dir = g_root / name;
    fs::create_directories(dir);
    std::ofstream f(dir / "config.json", std::ios::binary | std::ios::trunc);
    f << text;
}

std::string with_moe_identity(const std::string& text_body) {
    return R"json({"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text",)json" +
           text_body + R"json(}})json";
}

// The structure-level rules below are checked with a config that declares itself dense,
// because the dense reader also refuses a routed-expert identity: with such an identity the
// rejection would come from the identity and the structural rule being tested would be
// hidden.
std::string with_dense_identity(const std::string& text_body) {
    return R"json({"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text",)json" +
           text_body + R"json(}})json";
}

Qwen35MoeTextConfig expect_moe_ok(const std::string& name, const std::string& text) {
    write_config(name, text);
    auto cfg = read_qwen35_moe_text_config((g_root / name).string());
    check(cfg.ok(), name + ": moe config parse: " + (cfg.ok() ? "" : cfg.status().message()));
    if (!cfg.ok())
        return Qwen35MoeTextConfig{};
    return cfg.value();
}

void expect_moe_error(const std::string& name, const std::string& text) {
    write_config(name, text);
    auto cfg = read_qwen35_moe_text_config((g_root / name).string());
    check(!cfg.ok(), name + ": moe config must be rejected");
}

// Measured values of models/Qwen3.6-35B-A3B-Quark-AWQ-MXFP4 text_config.
constexpr const char* kMoeBase =
    R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256)json";

constexpr const char* kDenseBase =
    R"json("vocab_size":248320,"hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,"num_attention_heads":48,"num_key_value_heads":8,"head_dim":128)json";

template <typename T>
concept has_dense_intermediate = requires(const T& c) { c.ffn.intermediate_size; };

template <typename T>
concept has_moe_ffn = requires(const T& c) {
    c.ffn.num_experts;
    c.ffn.moe_intermediate_size;
};

// The MoE config must not be able to state a dense FFN width, and the dense config must
// not be able to state an expert count. Both are type-level facts, checked here so a
// future field cannot quietly reintroduce the mixed form.
static_assert(!has_dense_intermediate<Qwen35MoeTextConfig>,
              "the MoE config carries no dense intermediate size");
static_assert(!has_moe_ffn<Qwen35TextConfig>, "the dense config carries no MoE FFN fields");
static_assert(has_dense_intermediate<Qwen35TextConfig>, "the dense config owns intermediate_size");
static_assert(has_moe_ffn<Qwen35MoeTextConfig>, "the MoE config owns the expert geometry");

}  // namespace

int main() {
    g_root = fs::temp_directory_path() / "shelf_test_moe_config";
    fs::remove_all(g_root);
    fs::create_directories(g_root);

    {
        const auto cfg = expect_moe_ok("moe", with_moe_identity(kMoeBase));
        check(cfg.ffn.num_experts == 256u, "moe: num_experts");
        check(cfg.ffn.num_experts_per_tok == 8u, "moe: num_experts_per_tok");
        check(cfg.ffn.moe_intermediate_size == 512u, "moe: moe_intermediate_size");
        check(cfg.ffn.shared_expert_intermediate_size == 512u,
              "moe: shared_expert_intermediate_size");
        check(cfg.hidden_size == 2048u, "moe: hybrid geometry reaches the MoE config");
        check(cfg.hidden_size == 2048u && cfg.num_hidden_layers == 40u, "moe: base geometry kept");
        check(cfg.vocab_size == 248320u, "moe: base vocab kept");
    }

    // The reference fixes both gate activations (GDN gated RMSNorm = silu, attention
    // output gate = sigmoid), so output_gate_type is not part of the config contract. A
    // checkpoint that still declares it has to be accepted whatever the value is.
    {
        const auto cfg = expect_moe_ok("moe_gate_key_ignored", with_moe_identity(
            std::string(kMoeBase) + R"json(,"output_gate_type":"sigmoid")json"));
        check(cfg.ffn.num_experts == 256u,
              "an output_gate_type declaration does not disturb the config");
    }

    // The MoE reader owns the expert geometry; the shared Qwen35 reader must not grow
    // MoE fields, so a dense config stays dense and a MoE config is refused there.
    {
        write_config("dense", with_dense_identity(std::string(kDenseBase) +
                                               R"json(,"output_gate_type":"swish")json"));
        auto dense = read_qwen35_text_config((g_root / "dense").string());
        check(dense.ok(), "dense: base reader accepts");
        if (dense.ok()) {
            check(dense.value().ffn.intermediate_size == 17408u, "dense: intermediate_size");
        }
        check(!read_qwen35_moe_text_config((g_root / "dense").string()).ok(),
              "a dense config is not a MoE config");
    }

    // num_experts is the authority for MoE: a config that declares experts is not a dense
    // model, so the dense reader refuses it instead of reading an FFN the checkpoint does
    // not carry. num_experts: 0 keeps the model dense.
    {
        write_config("dense_with_experts", with_dense_identity(
            std::string(kDenseBase) + R"json(,"num_experts":256,"num_experts_per_tok":8)json"));
        auto dense = read_qwen35_text_config((g_root / "dense_with_experts").string());
        check(!dense.ok() && message_contains(dense.status(), "must not declare num_experts"),
              "the dense reader rejects a config that declares experts");
    }
    {
        write_config("dense_zero_experts", with_dense_identity(
            std::string(kDenseBase) + R"json(,"num_experts":0)json"));
        check(read_qwen35_text_config((g_root / "dense_zero_experts").string()).ok(),
              "num_experts: 0 keeps the model dense");
    }
    {
        write_config("dense_no_intermediate", with_dense_identity(
            R"json("vocab_size":248320,"hidden_size":5120,"num_hidden_layers":64)json"));
        auto dense = read_qwen35_text_config((g_root / "dense_no_intermediate").string());
        check(!dense.ok() && message_contains(dense.status(), "requires intermediate_size"),
              "a dense config without intermediate_size is rejected");
    }

    expect_moe_error("moe_topk_too_large",
                     with_moe_identity(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":257,"moe_intermediate_size":512,"shared_expert_intermediate_size":512)json"));
    expect_moe_error("moe_topk_zero",
                     with_moe_identity(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":0,"moe_intermediate_size":512,"shared_expert_intermediate_size":512)json"));
    expect_moe_error("moe_no_num_experts",
                     with_moe_identity(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512)json"));
    expect_moe_error("moe_no_moe_intermediate",
                     with_moe_identity(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":8,"shared_expert_intermediate_size":512)json"));
    expect_moe_error("moe_no_shared_expert_size",
                     with_moe_identity(R"json("vocab_size":248320,"hidden_size":2048,"num_hidden_layers":40,"num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512)json"));
    expect_moe_error("moe_with_dense_ffn", with_moe_identity(
        std::string(kMoeBase) + R"json(,"intermediate_size":512)json"));

    fs::remove_all(g_root);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
