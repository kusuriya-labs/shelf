#include <shelf/models/qwen35/model/rope_profile.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

using ps::qwen35::EffectiveRopeProfile;
using ps::qwen35::Qwen35RopeParameters;
using ps::qwen35::RopeFactorSelection;
using ps::qwen35::parse_rope_factor_selection;
using ps::qwen35::read_qwen35_rope_parameters;
using ps::qwen35::resolve_effective_rope_profile;
using ps::qwen35::rope_max_seq_len_cap;

static int g_fail = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) {
        ++g_fail;
        std::printf("FAIL %s\n", msg.c_str());
    }
}

static void check_close(double a, double b, double tol, const std::string& msg) {
    check(std::fabs(a - b) <= tol, msg + " (" + std::to_string(a) + " vs " + std::to_string(b) + ")");
}

static fs::path g_root;

static void write_config(const std::string& name, const std::string& body) {
    const fs::path dir = g_root / name;
    fs::create_directories(dir);
    std::ofstream f(dir / "config.json");
    f << body;
}

static const char* kDefaultConfig = R"json({
  "text_config": {
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {
      "mrope_interleaved": true,
      "mrope_section": [11, 11, 10],
      "partial_rotary_factor": 0.25,
      "rope_theta": 10000000,
      "rope_type": "default"
    }
  }
}
)json";

static const char* kYarn2Config = R"json({
  "text_config": {
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "rope_parameters": {
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 2.0,
      "original_max_position_embeddings": 262144
    }
  }
}
)json";

static const char* kYarn4Config = R"json({
  "text_config": {
    "head_dim": 256,
    "max_position_embeddings": 262144,
    "rope_parameters": {
      "rope_type": "yarn",
      "rope_theta": 10000000,
      "partial_rotary_factor": 0.25,
      "factor": 4.0,
      "original_max_position_embeddings": 262144
    }
  }
}
)json";

int main() {
    char tmpl[] = "/tmp/rope_profile_testXXXXXX";
    const char* dir = mkdtemp(tmpl);
    if (dir == nullptr) {
        std::printf("FAIL mkdtemp\n");
        return 1;
    }
    g_root = fs::path(dir);

    write_config("default", kDefaultConfig);
    write_config("yarn2", kYarn2Config);
    write_config("yarn4", kYarn4Config);

    {
        auto r = read_qwen35_rope_parameters((g_root / "missing").string());
        check(!r.ok(), "missing config.json is rejected");
    }
    {
        write_config("badjson", "{\"text_config\": ");
        auto r = read_qwen35_rope_parameters((g_root / "badjson").string());
        check(!r.ok(), "bad json is rejected");
    }
    {
        write_config("dynamic", R"json({
          "text_config": {"head_dim": 256, "max_position_embeddings": 131072,
            "rope_parameters": {"rope_type": "dynamic", "rope_theta": 5000000}}
        })json");
        auto r = read_qwen35_rope_parameters((g_root / "dynamic").string());
        check(!r.ok(), "unsupported rope_type is rejected");
    }
    {
        write_config("nan_theta", R"json({
          "text_config": {"head_dim": 256, "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e999}}
        })json");
        auto r = read_qwen35_rope_parameters((g_root / "nan_theta").string());
        check(!r.ok(), "overflowing rope_theta is rejected");
    }
    {
        write_config("bad_partial", R"json({
          "text_config": {"head_dim": 256, "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                                "partial_rotary_factor": 1.5}}
        })json");
        auto r = read_qwen35_rope_parameters((g_root / "bad_partial").string());
        check(!r.ok(), "partial_rotary_factor > 1 is rejected");
    }
    {
        write_config("mscale", R"json({
          "text_config": {"head_dim": 256, "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "yarn", "rope_theta": 10000000,
                                "factor": 2.0, "mscale": 0.7,
                                "mscale_all_dim": 0.7}}
        })json");
        auto r = read_qwen35_rope_parameters((g_root / "mscale").string());
        check(!r.ok(), "mscale is rejected");
    }

    {
        auto params = read_qwen35_rope_parameters((g_root / "default").string());
        check(params.ok(), "default config reads");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Auto);
            check(profile.ok(), "default auto resolves");
            if (profile.ok()) {
                const auto& p = profile.value();
                check(p.type == EffectiveRopeProfile::Type::Default, "auto + default is native");
                check(p.factor == 1.0, "native factor is 1");
                check(p.attention_factor == 1.0, "native attention factor is 1");
                check(p.rotary_dim == 64, "rotary dim is 64");
                check(p.original_max_position_embeddings == 262144, "native L is 262144");
                check(rope_max_seq_len_cap(p) == 262144, "native cap is 262144");
            }
        }
    }
    {
        auto params = read_qwen35_rope_parameters((g_root / "yarn2").string());
        check(params.ok(), "yarn2 config reads");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Auto);
            check(profile.ok(), "yarn2 auto resolves");
            if (profile.ok()) {
                const auto& p = profile.value();
                check(p.type == EffectiveRopeProfile::Type::Yarn, "auto + yarn is yarn");
                check(p.factor == 2.0, "yarn2 factor");
                check_close(p.attention_factor, 1.0693147180559945, 1e-12,
                            "yarn2 attention factor matches transformers");
                check(!p.from_cli, "auto source is config");
                check(rope_max_seq_len_cap(p) == 524288, "yarn2 cap is 524288");
            }
        }
    }
    {
        auto params = read_qwen35_rope_parameters((g_root / "yarn4").string());
        check(params.ok(), "yarn4 config reads");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Auto);
            check(profile.ok(), "yarn4 auto resolves");
            if (profile.ok()) {
                const auto& p = profile.value();
                check(p.factor == 4.0, "yarn4 factor");
                check_close(p.attention_factor, 1.138629436111989, 1e-12,
                            "yarn4 attention factor matches transformers");
                check(rope_max_seq_len_cap(p) == 1000000,
                      "yarn4 cap is capped at official 1000000");
            }
            auto forced = resolve_effective_rope_profile(params.value(),
                                                         RopeFactorSelection::Two);
            check(forced.ok(), "yarn4 config + cli 2 resolves");
            if (forced.ok()) {
                check(forced.value().factor == 2.0, "cli two overrides config factor");
                check(forced.value().from_cli, "cli override marks from_cli");
            }
            auto native = resolve_effective_rope_profile(params.value(),
                                                         RopeFactorSelection::One);
            check(native.ok(), "yarn4 config + cli 1 resolves");
            if (native.ok()) {
                check(native.value().type == EffectiveRopeProfile::Type::Default,
                      "cli one forces native");
                check(native.value().theta == 10000000.0, "cli one keeps theta");
                check(rope_max_seq_len_cap(native.value()) == 262144,
                      "forced native cap returns to L");
            }
        }
    }
    {
        auto params = read_qwen35_rope_parameters((g_root / "default").string());
        check(params.ok(), "default config reads for cli four");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Four);
            check(profile.ok(), "default config + cli four resolves without explicit factor");
            if (profile.ok()) {
                check(profile.value().factor == 4.0, "cli four factor");
                check(profile.value().original_max_position_embeddings == 262144,
                      "cli four takes L from max_position_embeddings");
                check_close(profile.value().attention_factor, 1.138629436111989, 1e-12,
                            "cli four attention factor");
            }
        }
    }
    {
        write_config("yarn_nolen", R"json({
          "text_config": {"head_dim": 256,
            "rope_parameters": {"rope_type": "yarn", "rope_theta": 10000000,
                                "partial_rotary_factor": 0.25, "factor": 2.0}}
        })json");
        auto params = read_qwen35_rope_parameters((g_root / "yarn_nolen").string());
        check(params.ok(), "yarn without length reads");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Auto);
            check(!profile.ok(), "yarn without original length is rejected");
        }
    }
    {
        write_config("yarn_nofactor", R"json({
          "text_config": {"head_dim": 256, "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "yarn", "rope_theta": 10000000,
                                "partial_rotary_factor": 0.25}}
        })json");
        auto params = read_qwen35_rope_parameters((g_root / "yarn_nofactor").string());
        check(params.ok(), "yarn without factor reads");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Auto);
            check(!profile.ok(), "yarn without factor is rejected on auto");
        }
    }
    {
        write_config("odd_rotary", R"json({
          "text_config": {"head_dim": 256, "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                                "partial_rotary_factor": 0.25390625}}
        })json");
        auto params = read_qwen35_rope_parameters((g_root / "odd_rotary").string());
        check(params.ok(), "odd rotary config reads");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Auto);
            check(!profile.ok(), "odd rotary dim is rejected");
        }
    }
    {
        Qwen35RopeParameters params;
        params.head_dim = 256;
        params.max_position_embeddings = 262144;
        params.has_factor = true;
        params.factor = 0.0;
        params.rope_type = "yarn";
        auto profile = resolve_effective_rope_profile(params, RopeFactorSelection::Auto);
        check(!profile.ok(), "factor zero is rejected");
        params.factor = -2.0;
        profile = resolve_effective_rope_profile(params, RopeFactorSelection::Auto);
        check(!profile.ok(), "negative factor is rejected");
    }
    {
        auto s = parse_rope_factor_selection("auto");
        check(s.ok() && s.value() == RopeFactorSelection::Auto, "parse auto");
        s = parse_rope_factor_selection("1");
        check(s.ok() && s.value() == RopeFactorSelection::One, "parse 1");
        s = parse_rope_factor_selection("2");
        check(s.ok() && s.value() == RopeFactorSelection::Two, "parse 2");
        s = parse_rope_factor_selection("4");
        check(s.ok() && s.value() == RopeFactorSelection::Four, "parse 4");
        check(!parse_rope_factor_selection("3").ok(), "parse 3 rejected");
        check(!parse_rope_factor_selection("yarn").ok(), "parse yarn rejected");
        check(!parse_rope_factor_selection("").ok(), "parse empty rejected");
    }
    {
        write_config("explicit_af", R"json({
          "text_config": {"head_dim": 256, "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "yarn", "rope_theta": 10000000,
                                "partial_rotary_factor": 0.25, "factor": 2.0,
                                "attention_factor": 1.25}}
        })json");
        auto params = read_qwen35_rope_parameters((g_root / "explicit_af").string());
        check(params.ok(), "explicit attention factor reads");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Auto);
            check(profile.ok(), "explicit attention factor resolves");
            if (profile.ok()) {
                check(profile.value().attention_factor == 1.25,
                      "explicit attention factor is kept");
            }
        }
    }
    {
        write_config("beta", R"json({
          "text_config": {"head_dim": 256, "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "yarn", "rope_theta": 10000000,
                                "partial_rotary_factor": 0.25, "factor": 4.0,
                                "beta_fast": 40, "beta_slow": 2,
                                "truncate": false}}
        })json");
        auto params = read_qwen35_rope_parameters((g_root / "beta").string());
        check(params.ok(), "beta options read");
        if (params.ok()) {
            auto profile = resolve_effective_rope_profile(params.value(),
                                                          RopeFactorSelection::Auto);
            check(profile.ok(), "beta options resolve");
            if (profile.ok()) {
                check(profile.value().beta_fast == 40.0, "beta_fast kept");
                check(profile.value().beta_slow == 2.0, "beta_slow kept");
                check(!profile.value().truncate, "truncate kept");
            }
        }
    }

    fs::remove_all(g_root);
    if (g_fail != 0) {
        std::printf("%d FAIL\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
