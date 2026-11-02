#include <shelf/models/qwen35/eagle4/config.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

using ps::qwen35::eagle4::Eagle4Config;
using ps::qwen35::eagle4::read_eagle4_config;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static fs::path g_root;

static const char* kBaseConfig = R"json({
  "architectures": ["LlamaForCausalLMEagle4"],
  "draft_vocab_size": 256,
  "eagle_config": {"eagle_aux_hidden_state_layer_ids": [3, 31, 59], "use_aux_hidden_state": true},
  "efficient_lm_head_config": {"num_candidates": 64, "rank": 32, "type": "low_rank"},
  "fc_norm": true,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "intermediate_size": 16384,
  "layer_types": ["sliding_attention"],
  "norm_output": true,
  "num_attention_heads": 40,
  "num_aux_hidden_states": 3,
  "num_hidden_layers": 1,
  "num_key_value_heads": 40,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
  "sliding_window": 4096,
  "target_hidden_size": 5120,
  "tie_word_embeddings": false,
  "vocab_size": 248320
}
)json";

static bool replace_first(std::string& text, const std::string& from, const std::string& to) {
    const std::size_t pos = text.find(from);
    if (pos == std::string::npos) {
        check(false, "test setup: pattern not found: " + from);
        return false;
    }
    text.replace(pos, from.size(), to);
    return true;
}

static void write_config(const std::string& name, const std::string& text) {
    const fs::path dir = g_root / name;
    fs::create_directories(dir);
    std::ofstream f(dir / "config.json", std::ios::binary | std::ios::trunc);
    f << text;
}

static void expect_error(
    const std::string& name, const std::string& text, const std::string& what) {
    write_config(name, text);
    auto result = read_eagle4_config((g_root / name).string());
    check(!result.ok(), what);
}

static void expect_ok(const std::string& name, const std::string& text) {
    write_config(name, text);
    auto result = read_eagle4_config((g_root / name).string());
    check(result.ok(), "expected ok: " + name);
    if (result.ok()) {
        const Eagle4Config& c = result.value();
        check(c.hidden_size == 5120, "hidden size");
        check(c.attention_q_rows() == 5120, "q rows");
        check(c.tap_feature_size() == c.num_aux_layers * c.target_hidden_size,
              "tap features");
    }
}

int main() {
    g_root = fs::temp_directory_path() /
             ("ps_eagle4_config_" + std::to_string(static_cast<long>(getpid())));
    fs::remove_all(g_root);
    fs::create_directories(g_root);

    expect_ok("base", kBaseConfig);

    {
        std::string text = kBaseConfig;
        replace_first(text, "\"num_hidden_layers\": 1", "\"num_hidden_layers\": 2");
        expect_error("two_layers", text, "two layers rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"num_key_value_heads\": 40", "\"num_key_value_heads\": 8");
        expect_error("gqa", text, "GQA draft rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"sliding_window\": 4096", "\"sliding_window\": 0");
        expect_error("zero_window", text, "zero sliding window rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "[3, 31, 59]", "[31, 3, 59]");
        expect_error("aux_order", text, "unordered aux layers rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"num_candidates\": 64", "\"num_candidates\": 0");
        expect_error("zero_candidates", text, "zero candidates rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"rank\": 32", "\"rank\": 0");
        expect_error("zero_rank", text, "zero candidate rank rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"type\": \"low_rank\"", "\"type\": \"full\"");
        expect_error("head_type", text, "unsupported head type rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"LlamaForCausalLMEagle4\"", "\"Eagle3LlamaForCausalLM\"");
        expect_error("arch", text, "unsupported architecture rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"num_aux_hidden_states\": 3", "\"num_aux_hidden_states\": 4");
        expect_error("aux_count", text, "aux count mismatch rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"eagle_aux_hidden_state_layer_ids\": [3, 31, 59]",
                      "\"eagle_aux_hidden_state_layer_ids\": [3, 31]");
        replace_first(text, "\"num_aux_hidden_states\": 3", "\"num_aux_hidden_states\": 2");
        expect_ok("two_aux", text);
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"hidden_size\": 5120", "\"hidden_size\": 4096");
        expect_error("hidden_mismatch", text, "hidden size / head geometry mismatch rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true");
        expect_error("tied", text, "tied embeddings rejected");
    }
    {
        std::string text = kBaseConfig;
        replace_first(text, "\"fc_norm\": true", "\"fc_norm\": false");
        replace_first(text, "\"norm_output\": true", "\"norm_output\": false");
        expect_ok("no_fc_norm", text);
    }

    auto missing_dir = read_eagle4_config((g_root / "does_not_exist").string());
    check(!missing_dir.ok(), "missing config rejected");

    fs::remove_all(g_root);
    if (g_fail != 0) {
        std::printf("FAILED (%d)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
