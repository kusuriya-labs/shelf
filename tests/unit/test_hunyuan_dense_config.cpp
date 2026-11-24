#include <shelf/models/hunyuan_dense/model/config.h>
#include <shelf/models/qwen35/model/gdn_state_layout.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

bool write_text(const fs::path& path, const std::string& content) {
    std::ofstream out(path);
    if (!out) return false;
    out << content;
    return out.good();
}

std::string make_base_config(const std::string& extra = {}) {
    std::string json = R"({
  "architectures": ["HunYuanDenseV1ForCausalLM"],
  "model_type": "hunyuan_v1_dense",
  "num_hidden_layers": 32,
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "intermediate_size": 6144,
  "vocab_size": 120818,
  "max_position_embeddings": 262144,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "tie_word_embeddings": true,
  "hidden_act": "silu",
  "norm_type": "rms",
  "use_cla": false,
  "use_qk_norm": true,
  "use_rotary_pos_emb": true,
  "attention_bias": false,
  "mlp_bias": false,
  "rope_scaling": {"type": "dynamic", "alpha": 1000.0, "factor": 1.0},
  "eos_token_id": 120020)";
    if (!extra.empty()) {
        json += ",\n  " + extra;
    }
    json += "\n}\n";
    return json;
}

}

int main() {
    const fs::path root = fs::temp_directory_path() / "shelf_test_hunyuan_dense_config";
    fs::remove_all(root);
    fs::create_directories(root);

    const fs::path valid_18b = root / "valid_18b";
    fs::create_directories(valid_18b);
    if (!write_text(valid_18b / "config.json", make_base_config())) return 1;
    if (!write_text(valid_18b / "generation_config.json", R"({"eos_token_id": [127960, 127967]})")) {
        return 1;
    }

    auto detected = shelf::hunyuan_dense::is_hunyuan_dense_config(valid_18b.string());
    if (!detected.ok() || !detected.value()) return 1;

    auto cfg = shelf::hunyuan_dense::read_config(valid_18b.string());
    if (!cfg.ok()) {
        std::fprintf(stderr, "read_config failed: %s\n", cfg.status().message().c_str());
        return 1;
    }
    const auto& c = cfg.value();
    if (c.num_hidden_layers != 32 || c.hidden_size != 2048 || c.num_attention_heads != 16 ||
        c.num_key_value_heads != 4 || c.head_dim != 128 || c.intermediate_size != 6144 ||
        c.vocab_size != 120818 || c.max_position_embeddings != 262144 ||
        !c.tie_word_embeddings || std::abs(c.rms_norm_eps - 1e-5) > 1e-12 ||
        std::abs(c.rope_theta - 10000.0) > 1e-6 || c.rope_scaling.type != "dynamic" ||
        std::abs(c.rope_scaling.alpha - 1000.0) > 1e-6) {
        return 1;
    }
    if (c.stop_tokens.size() != 2 || c.stop_tokens[0] != 127960 || c.stop_tokens[1] != 127967) {
        return 1;
    }

    auto rope = shelf::hunyuan_dense::resolve_rope_profile(c);
    if (!rope.ok()) return 1;
    if (rope.value().type != shelf::runtime::EffectiveRopeProfile::Type::NtkAlpha ||
        std::abs(rope.value().alpha - 1000.0) > 1e-6 ||
        rope.value().rotary_dim != 128u || !rope.value().float32_frequencies) {
        return 1;
    }
    if (shelf::hunyuan_dense::resolve_rope_profile(c, "2").ok()) return 1;

    shelf::qwen_hybrid::QwenHybridConfig hybrid;
    hybrid.declared_model_type = "hunyuan_v1_dense";
    hybrid.num_hidden_layers = 32;
    hybrid.layer_types.assign(32, 1);
    if (shelf::qwen35::make_qwen35_gdn_state_layout(hybrid).num_gdn_states != 0) return 1;

    const char* bad_cases[] = {
        R"("use_cla": true)",
        R"("attention_bias": true)",
        R"("mlp_bias": true)",
        R"("add_classification_head": true)",
        R"("use_qk_norm": false)",
        R"("use_rotary_pos_emb": false)",
        R"("hidden_act": "gelu")",
        R"("norm_type": "layer")",
        R"("rope_scaling": {"type": "linear", "factor": 2.0})",
        R"("rope_scaling": {"type": "dynamic", "factor": 2.0})",
        R"("architectures": ["Qwen2ForCausalLM"])",
    };
    for (std::size_t i = 0; i < sizeof(bad_cases) / sizeof(bad_cases[0]); ++i) {
        const fs::path bad_dir = root / ("bad_" + std::to_string(i));
        fs::create_directories(bad_dir);
        std::string text = make_base_config();
        std::string override_cfg = R"({
  "architectures": ["HunYuanDenseV1ForCausalLM"],
  "model_type": "hunyuan_v1_dense",
  "num_hidden_layers": 32,
  "hidden_size": 2048,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "intermediate_size": 6144,
  "vocab_size": 120818,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
)";
        override_cfg += std::string("  ") + bad_cases[i] + "\n}\n";
        if (!write_text(bad_dir / "config.json", override_cfg)) return 1;
        auto bad = shelf::hunyuan_dense::read_config(bad_dir.string());
        if (bad.ok() || bad.status().code() != shelf::Status::Code::unsupported) {
            std::fprintf(stderr, "expected Unsupported for %s\n", bad_cases[i]);
            return 1;
        }
    }

    fs::remove_all(root);
    std::printf("test_hunyuan_dense_config: PASS\n");
    return 0;
}
