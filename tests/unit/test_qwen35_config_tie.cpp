#include <phaseshift/models/qwen35/model/qwen35_config.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

using ps::qwen35::read_qwen35_text_config;
using ps::qwen35::read_qwen35_tie_word_embeddings;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static fs::path g_root;

static const char* kTextConfig =
    R"json("vocab_size":248320,"hidden_size":2560,"num_hidden_layers":32)json";

static void write_config(const std::string& name, const std::string& text) {
    const fs::path dir = g_root / name;
    fs::create_directories(dir);
    std::ofstream f(dir / "config.json", std::ios::binary | std::ios::trunc);
    f << text;
}

static void expect_tie(const std::string& name, const std::string& text, bool expected) {
    write_config(name, text);
    const std::string dir = (g_root / name).string();

    auto cfg = read_qwen35_text_config(dir);
    check(cfg.ok(), name + ": text config parse: " + (cfg.ok() ? "" : cfg.status().message()));
    if (cfg.ok()) {
        check(cfg.value().tie_word_embeddings == expected,
              name + ": text config tie_word_embeddings");
    }

    auto tie = read_qwen35_tie_word_embeddings(dir);
    check(tie.ok(), name + ": tie reader: " + (tie.ok() ? "" : tie.status().message()));
    if (tie.ok()) {
        check(tie.value() == expected, name + ": tie reader value");
    }
}

static std::string with_text_config(const std::string& root_tie, const std::string& text_tie) {
    std::string text = R"json({"model_type":"qwen3_5")json";
    if (!root_tie.empty()) {
        text += R"json(,"tie_word_embeddings":)json";
        text += root_tie;
    }
    text += R"json(,"text_config":{)json";
    text += kTextConfig;
    if (!text_tie.empty()) {
        text += R"json(,"tie_word_embeddings":)json";
        text += text_tie;
    }
    text += "}}";
    return text;
}

int main() {
    char tmpl[] = "/tmp/qwen35_config_tie_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    if (dir == nullptr) {
        std::printf("FAIL mkdtemp\n");
        return 1;
    }
    g_root = fs::path(dir);

    expect_tie("root_false", with_text_config("false", ""), false);
    expect_tie("root_true", with_text_config("true", ""), true);
    expect_tie("root_false_wins", with_text_config("false", "true"), false);
    expect_tie("text_true", with_text_config("", "true"), true);
    expect_tie("text_false", with_text_config("", "false"), false);
    expect_tie("absent", with_text_config("", ""), false);

    std::error_code ec;
    fs::remove_all(g_root, ec);

    std::printf("test_qwen35_config_tie: failed=%d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
