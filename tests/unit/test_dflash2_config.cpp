#include <phaseshift/models/qwen35/dflash2/config.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

using ps::qwen35::dflash2::DFlash2Config;
using ps::qwen35::dflash2::read_dflash2_config;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static fs::path g_root;

static const char* kBaseConfig = R"json({
  "architectures": ["DFlash2DraftModel"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "is_causal": false,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61]
  },
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "intermediate_size": 17408,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
  "max_position_embeddings": 262144,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 64,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
  "sliding_window": 2048,
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

static void expect_error(const std::string& name, const std::string& text) {
    write_config(name, text);
    auto r = read_dflash2_config((g_root / name).string());
    check(!r.ok(), "expected error for " + name);
}

static void expect_ok(const std::string& name, const std::string& text) {
    write_config(name, text);
    auto r = read_dflash2_config((g_root / name).string());
    check(r.ok(), "expected ok for " + name + ": " + (r.ok() ? "" : r.status().message()));
}

int main() {
    char tmpl[] = "/tmp/dflash2_config_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    if (dir == nullptr) {
        std::printf("FAIL mkdtemp\n");
        return 1;
    }
    g_root = fs::path(dir);

    write_config("ok", kBaseConfig);
    {
        auto r = read_dflash2_config((g_root / "ok").string());
        check(r.ok(), "real contract config must parse: " +
                          (r.ok() ? "" : r.status().message()));
        if (r.ok()) {
            const DFlash2Config& c = r.value();
            check(c.architecture == "DFlash2DraftModel", "architecture");
            check(c.vocab_size == 248320, "vocab_size");
            check(c.hidden_size == 5120, "hidden_size");
            check(c.intermediate_size == 17408, "intermediate_size");
            check(c.num_hidden_layers == 5, "num_hidden_layers");
            check(c.num_attention_heads == 32, "num_attention_heads");
            check(c.num_key_value_heads == 8, "num_key_value_heads");
            check(c.head_dim == 128, "head_dim");
            check(c.sliding_window == 2048, "sliding_window");
            check(!c.is_causal, "is_causal");
            check(c.block_size == 8, "block_size");
            check(c.conv_group_size == 16, "conv_group_size");
            check(c.conv_kernel_size == 2, "conv_kernel_size");
            check(c.mask_token_id == 248070, "mask_token_id");
            check(c.selector_rank == 256, "selector_rank");
            check(c.selector_top_k == 16, "selector_top_k");
            check(c.num_target_layers == 64, "num_target_layers");
            check(c.num_target_layer_ids == 5, "num_target_layer_ids");
            check(c.target_layer_ids[0] == 5 && c.target_layer_ids[1] == 19 &&
                      c.target_layer_ids[2] == 33 && c.target_layer_ids[3] == 47 &&
                      c.target_layer_ids[4] == 61,
                  "target_layer_ids");
            check(!c.tie_word_embeddings, "tie_word_embeddings");
            check(c.max_draft_tokens() == 7, "max_draft_tokens");
            check(c.conv_groups() == 320, "conv_groups");
            check(c.conv_projection_rows() == 1280, "conv_projection_rows");
            check(c.attention_q_rows() == 4096, "attention_q_rows");
            check(c.attention_kv_rows() == 1024, "attention_kv_rows");
            check(c.tap_feature_size() == 25600, "tap_feature_size");
        }
    }

    {
        std::string t = kBaseConfig;
        replace_first(t, "DFlash2DraftModel", "MtpDraftModel");
        expect_error("bad_arch", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"is_causal\": false", "\"is_causal\": true");
        expect_error("causal_true", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true");
        expect_error("tied_true", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"conv_group_size\": 16", "\"conv_group_size\": 17");
        expect_error("bad_conv_group", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "[5, 19, 33, 47, 61]", "[5, 19, 19, 47, 61]");
        expect_error("bad_target_order", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "[5, 19, 33, 47, 61]", "[5, 19, 33, 47, 64]");
        expect_error("bad_target_range", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "[5, 19, 33, 47, 61]", "[]");
        expect_error("empty_target_ids", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"selector_top_k\": 16", "\"selector_top_k\": 999999999");
        expect_error("bad_topk", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"rope_type\": \"default\"", "\"rope_type\": \"yarn\"");
        expect_error("bad_rope", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"sliding_attention\", \"sliding_attention\"",
                      "\"full_attention\", \"sliding_attention\"");
        expect_error("bad_layer_type", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"hidden_size\": 5120,", "");
        expect_error("missing_hidden_size", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"dflash_config\": {", "\"dflash_config_disabled\": {");
        expect_error("no_dflash_config", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"mask_token_id\": 248070", "\"mask_token_id\": 999999999");
        expect_error("mask_out_of_range", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"block_size\": 8", "\"block_size\": 1");
        expect_error("block_too_small", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"block_size\": 8", "\"block_size\": 17");
        expect_error("block_too_large", t);
    }
    {
        std::string t = kBaseConfig;
        replace_first(t, "\"num_attention_heads\": 32", "\"num_attention_heads\": 33");
        expect_error("heads_not_multiple", t);
    }
    expect_error("malformed", "{\"architectures\": [");
    expect_error("not_object", "[1, 2, 3]");
    expect_error("empty", "");

    fs::create_directories(g_root / "no_config");
    {
        auto r = read_dflash2_config((g_root / "no_config").string());
        check(!r.ok(), "missing config.json must error");
    }

    std::printf("test_dflash2_config: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
