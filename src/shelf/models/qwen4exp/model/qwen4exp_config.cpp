#include <shelf/models/qwen4exp/model/qwen4exp_config.h>

#include <fstream>
#include <nlohmann/json.hpp>

namespace ps::qwen4exp {
namespace {

constexpr std::uint64_t kDefaultNgramSeed = 1234;

}  // namespace

bool Qwen4ExpTextConfig::layer_is_qsa(std::size_t layer_index) const noexcept {
    if (!layer_types.empty()) {
        if (layer_index >= layer_types.size()) return false;
        return layer_types[layer_index] == 1;
    }
    if (full_attention_interval == 0) return false;
    return (layer_index + 1) % full_attention_interval == 0;
}

Result<Qwen4ExpTextConfig> read_qwen4exp_text_config(const std::string& model_dir) {
    const std::string path = model_dir + "/config.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: cannot open config.json", __FILE__, __LINE__);
    }

    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("read_qwen4exp_text_config: bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object() || !root.contains("text_config") ||
        !root["text_config"].is_object()) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: config.json missing text_config", __FILE__, __LINE__);
    }

    const nlohmann::json& t = root["text_config"];
    Qwen4ExpTextConfig cfg;

    auto read_size = [&](const char* key, std::size_t& out) -> bool {
        if (!t.contains(key) || !t[key].is_number()) return false;
        out = t[key].get<std::size_t>();
        return true;
    };
    auto read_float = [&](const char* key, float& out) -> bool {
        if (!t.contains(key) || !t[key].is_number()) return false;
        out = t[key].get<float>();
        return true;
    };
    auto read_bool = [&](const char* key, bool& out) -> bool {
        if (!t.contains(key) || !t[key].is_boolean()) return false;
        out = t[key].get<bool>();
        return true;
    };

    read_size("vocab_size", cfg.vocab_size);
    read_size("hidden_size", cfg.hidden_size);
    read_size("num_hidden_layers", cfg.num_hidden_layers);
    read_size("num_attention_heads", cfg.num_attention_heads);
    read_size("num_key_value_heads", cfg.num_key_value_heads);
    read_size("head_dim", cfg.head_dim);
    read_size("full_attention_interval", cfg.full_attention_interval);

    read_size("linear_num_key_heads", cfg.linear_num_key_heads);
    read_size("linear_num_value_heads", cfg.linear_num_value_heads);
    read_size("linear_key_head_dim", cfg.linear_key_head_dim);
    read_size("linear_value_head_dim", cfg.linear_value_head_dim);
    read_size("linear_conv_kernel_dim", cfg.linear_conv_kernel_dim);

    read_size("num_experts", cfg.num_experts);
    read_size("num_experts_per_tok", cfg.num_experts_per_tok);
    read_size("moe_intermediate_size", cfg.moe_intermediate_size);
    read_size("shared_expert_intermediate_size", cfg.shared_expert_intermediate_size);

    read_size("hc_count", cfg.hc_count);
    read_size("hc_lowrank", cfg.hc_lowrank);

    read_size("ngram_size", cfg.ngram_size);
    read_size("heads_per_ngram", cfg.heads_per_ngram);
    read_size("ngram_vocab_size_base", cfg.ngram_vocab_size_base);
    read_size("make_ngram_vocab_size_divisible_by",
              cfg.make_ngram_vocab_size_divisible_by);
    read_size("split_ngram_parts", cfg.split_ngram_parts);
    if (t.contains("seed") && t["seed"].is_number_unsigned()) {
        cfg.ngram_seed = t["seed"].get<std::uint64_t>();
    } else {
        cfg.ngram_seed = kDefaultNgramSeed;
    }
    read_size("ple_embed_dim", cfg.ple_embed_dim);
    read_size("ple_conv_kernel_size", cfg.ple_conv_kernel_size);

    read_size("indexer_n_heads", cfg.indexer_n_heads);
    read_size("indexer_kv_heads", cfg.indexer_kv_heads);
    read_size("indexer_head_dim", cfg.indexer_head_dim);
    read_size("indexer_budget", cfg.indexer_budget);
    read_size("indexer_compress_ratio", cfg.indexer_compress_ratio);

    if (cfg.indexer_compress_ratio != 0 && cfg.indexer_budget != 0) {
        cfg.block_topk = cfg.indexer_budget / cfg.indexer_compress_ratio;
    }

    read_float("rms_norm_eps", cfg.rms_norm_eps);
    read_bool("tie_word_embeddings", cfg.tie_word_embeddings);

    if (t.contains("rope_parameters") && t["rope_parameters"].is_object()) {
        const nlohmann::json& rp = t["rope_parameters"];
        if (rp.contains("rope_theta") && rp["rope_theta"].is_number()) {
            cfg.rope_theta = rp["rope_theta"].get<float>();
        }
        if (rp.contains("partial_rotary_factor") && rp["partial_rotary_factor"].is_number()) {
            cfg.partial_rotary_factor = rp["partial_rotary_factor"].get<float>();
        }
    } else {
        read_float("rope_theta", cfg.rope_theta);
        read_float("partial_rotary_factor", cfg.partial_rotary_factor);
    }

    if (cfg.head_dim != 0) {
        cfg.rotary_dim = static_cast<std::size_t>(
            static_cast<float>(cfg.head_dim) * cfg.partial_rotary_factor);
    }

    if (t.contains("output_gate_type") && t["output_gate_type"].is_string()) {
        const std::string gate = t["output_gate_type"].get<std::string>();
        if (gate == "silu" || gate == "swish") {
            cfg.output_gate_silu = true;
        } else if (gate != "sigmoid") {
            return Status::invalid_argument(
                ("read_qwen4exp_text_config: unsupported output_gate_type '" + gate + "'")
                    .c_str(),
                __FILE__, __LINE__);
        }
    }

    if (t.contains("norm_topk_prob") && t["norm_topk_prob"].is_boolean()) {
        cfg.norm_topk_prob = t["norm_topk_prob"].get<bool>();
    }

    if (t.contains("layer_types") && t["layer_types"].is_array()) {
        cfg.layer_types.reserve(t["layer_types"].size());
        for (const auto& lt : t["layer_types"]) {
            if (!lt.is_string()) {
                cfg.layer_types.push_back(-1);
                continue;
            }
            const std::string s = lt.get<std::string>();
            if (s == "full_attention") {
                cfg.layer_types.push_back(1);
            } else if (s == "linear_attention") {
                cfg.layer_types.push_back(0);
            } else {
                cfg.layer_types.push_back(-1);
            }
        }
    }

    cfg.ple_slot = 0;
    if (t.contains("ple_layer_ids") && t["ple_layer_ids"].is_array() &&
        !t["ple_layer_ids"].empty()) {
        const auto& ids = t["ple_layer_ids"];
        for (std::size_t slot = 0; slot < ids.size(); ++slot) {
            if (!ids[slot].is_number()) {
                return Status::invalid_argument(
                    "read_qwen4exp_text_config: ple_layer_ids must be integers",
                    __FILE__, __LINE__);
            }
            if (ids[slot].get<std::int64_t>() <= 0) {
                return Status::invalid_argument(
                    "read_qwen4exp_text_config: ple_layer_ids is 1-indexed",
                    __FILE__, __LINE__);
            }
        }
        cfg.ple_layer_index = static_cast<std::size_t>(
            ids[0].get<std::int64_t>() - 1);
        cfg.ple_slot = 0;
        for (std::size_t slot = 0; slot < ids.size(); ++slot) {
            if (static_cast<std::size_t>(ids[slot].get<std::int64_t>() - 1) ==
                cfg.ple_layer_index) {
                cfg.ple_slot = slot;
                break;
            }
        }
    }

    {
        std::ifstream gen_file(model_dir + "/generation_config.json");
        if (gen_file.is_open()) {
            nlohmann::json gen;
            try {
                gen_file >> gen;
            } catch (const nlohmann::json::exception&) {
                gen = nlohmann::json();
            }
            if (gen.is_object() && gen.contains("eos_token_id")) {
                const nlohmann::json& eos = gen["eos_token_id"];
                if (eos.is_number_integer()) {
                    cfg.stop_tokens.push_back(eos.get<int32_t>());
                } else if (eos.is_array()) {
                    for (const auto& value : eos) {
                        if (value.is_number_integer()) {
                            cfg.stop_tokens.push_back(value.get<int32_t>());
                        }
                    }
                }
            }
        }
    }
    if (cfg.stop_tokens.empty() && t.contains("eos_token_id")) {
        const nlohmann::json& eos = t["eos_token_id"];
        if (eos.is_number_integer()) {
            cfg.stop_tokens.push_back(eos.get<int32_t>());
        } else if (eos.is_array()) {
            for (const auto& value : eos) {
                if (value.is_number_integer()) {
                    cfg.stop_tokens.push_back(value.get<int32_t>());
                }
            }
        }
    }

    if (cfg.vocab_size == 0 || cfg.hidden_size == 0 || cfg.num_hidden_layers == 0) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: required dimensions missing", __FILE__, __LINE__);
    }
    if (cfg.num_experts == 0 || cfg.num_experts_per_tok == 0 ||
        cfg.moe_intermediate_size == 0) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: MoE dimensions missing", __FILE__, __LINE__);
    }
    if (cfg.hc_count == 0 || cfg.hc_lowrank == 0) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: hyper connection dimensions missing",
            __FILE__, __LINE__);
    }
    if (cfg.indexer_n_heads != 0 && cfg.indexer_kv_heads != 1) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: indexer_kv_heads must be 1", __FILE__, __LINE__);
    }
    if (cfg.indexer_budget != 0 && cfg.indexer_compress_ratio != 0 &&
        cfg.indexer_budget % cfg.indexer_compress_ratio != 0) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: indexer_budget must divide by "
            "indexer_compress_ratio",
            __FILE__, __LINE__);
    }
    if (!cfg.layer_types.empty() &&
        cfg.layer_types.size() != cfg.num_hidden_layers) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: layer_types length mismatch", __FILE__, __LINE__);
    }

    return cfg;
}

}  // namespace ps::qwen4exp
