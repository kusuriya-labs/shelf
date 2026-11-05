#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/model/rope_profile.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using shelf::qwen35::read_qwen35_text_config;
using shelf::qwen35moe::read_declared_num_experts;
using shelf::qwen35moe::read_qwen35_moe_text_config;

namespace {

int g_fail = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

bool holds(const shelf::Status& st, const std::string& needle) {
    return std::string(st.message()).find(needle) != std::string::npos;
}

fs::path g_root;

void write_config(const std::string& name, const std::string& text) {
    const fs::path dir = g_root / name;
    fs::create_directories(dir);
    std::ofstream f(dir / "config.json", std::ios::binary | std::ios::trunc);
    f << text;
}

constexpr const char* kDenseBody =
    R"json("vocab_size":1024,"hidden_size":256,"intermediate_size":512,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":64)json";

constexpr const char* kMoeBody =
    R"json("vocab_size":1024,"hidden_size":256,"num_hidden_layers":2,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":128,"shared_expert_intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":64)json";

}  // namespace

int main() {
    g_root = fs::temp_directory_path() / "shelf_test_model_identity";
    fs::remove_all(g_root);
    fs::create_directories(g_root);

    // Supported identities are accepted by their own family reader.
    {
        write_config("dense",
                     R"json({"model_type":"qwen3_5","text_config":{)json" +
                         std::string(kDenseBody) + R"json(}})json");
        auto dense = read_qwen35_text_config((g_root / "dense").string());
        check(dense.ok(), "qwen3_5 dense identity is accepted by the dense reader");
        if (dense.ok()) {
            check(dense.value().declared_model_type == "qwen3_5", "declared identity is kept");
        }
        auto moe = read_qwen35_moe_text_config((g_root / "dense").string());
        check(!moe.ok() && holds(moe.status(), "num_experts is required"),
              "a dense config is refused by the MoE reader");
    }
    {
        write_config("moe",
                     R"json({"model_type":"qwen3_5_moe","text_config":{)json" +
                         std::string(kMoeBody) + R"json(}})json");
        auto moe = read_qwen35_moe_text_config((g_root / "moe").string());
        check(moe.ok(), "qwen3_5_moe identity is accepted by the MoE reader");
        auto dense = read_qwen35_text_config((g_root / "moe").string());
        check(!dense.ok() && holds(dense.status(), "intermediate_size"),
              "the dense reader refuses a MoE config on its structure");
    }
    {
        // The structure here looks dense, and the expert count is spelled in a way this
        // build does not read. Only the declared identity shows what the checkpoint is.
        write_config("expert_key_unknown",
                     R"json({"model_type":"qwen3_5_moe","text_config":{)json" +
                         std::string(kDenseBody) +
                         R"json(,"n_routed_experts":256}})json");
        auto dense = read_qwen35_text_config((g_root / "expert_key_unknown").string());
        check(!dense.ok() && holds(dense.status(), "routed-expert model"),
              "a routed-expert identity is refused by the dense reader");
    }

    // A model this build does not implement is named as such, at every config entry
    // point, before any tensor is opened. The keys below follow a real routed checkpoint
    // that declares its expert count under a different name (n_routed_experts) and keeps
    // a flat config.
    {
        write_config("foreign",
                     R"json({"model_type":"mimo_v2","architectures":["MiMoV2ForCausalLM"],)json"
                     R"json("hidden_size":4096,"intermediate_size":16384,"moe_intermediate_size":2048,)json"
                     R"json("n_routed_experts":256,"num_experts_per_tok":8,"num_hidden_layers":48,)json"
                     R"json("num_attention_heads":64,"num_key_value_heads":4,"head_dim":192})json");
        const std::string dir = (g_root / "foreign").string();
        auto dense = read_qwen35_text_config(dir);
        check(!dense.ok() && holds(dense.status(), "unsupported model identity"),
              "an unimplemented architecture is refused by the dense reader");
        auto moe = read_qwen35_moe_text_config(dir);
        check(!moe.ok() && holds(moe.status(), "unsupported model identity"),
              "an unimplemented architecture is refused by the MoE reader");
        auto probe = read_declared_num_experts(dir);
        check(!probe.ok() && holds(probe.status(), "mimo_v2"),
              "the family probe names the rejected identity");
        check(!shelf::qwen_hybrid::check_config_identity(dir).ok(),
              "check_config_identity rejects it standalone");
        auto rope = shelf::qwen35::read_qwen35_rope_parameters(dir);
        check(!rope.ok() && holds(rope.status(), "unsupported model identity"),
              "the rope reader, which the apps read first, rejects it too");
    }

    // An identity has to be declared: reading a config that states nothing about what it
    // is would fall back to guessing from tensor names.
    {
        write_config("silent", R"json({"text_config":{)json" + std::string(kDenseBody) +
                                   R"json(}})json");
        auto st = shelf::qwen_hybrid::check_config_identity((g_root / "silent").string());
        check(!st.ok() && holds(st, "no model identity"),
              "a config without model_type is rejected instead of assumed");
    }

    // A declaration that contradicts the loaded structure is caught: the identity says
    // routed experts, the structure says dense.
    {
        write_config("contradicts",
                     R"json({"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_moe",)json" +
                         std::string(kDenseBody) + R"json(}})json");
        auto dense = read_qwen35_text_config((g_root / "contradicts").string());
        check(!dense.ok() && holds(dense.status(), "routed-expert model"),
              "a _moe text identity is not read as dense even without an expert count");
    }

    // The gate's vocabulary is a build-time allowlist, exposed so the rejection message and
    // the tests can name what this build implements.
    {
        const auto& supported = shelf::qwen_hybrid::supported_model_identities();
        check(supported == std::vector<std::string>{"qwen3_5", "qwen3_5_text", "qwen3_5_moe",
                                                    "qwen3_5_moe_text", "qwen4_exp",
                                                    "qwen4_exp_text"},
              "the supported identity set includes the Qwen3.5 families and Qwen4Exp");
    }

    fs::remove_all(g_root);
    std::printf(g_fail == 0 ? "PASS\n" : "FAIL (%d)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
