#include <shelf/models/qwen4exp/model/qwen4exp_config.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace shelf {
namespace qwen4exp {
namespace {

constexpr std::uint64_t kSplitmixGamma = 0x9E3779B97F4A7C15ull;
constexpr std::uint64_t kSplitmixM1 = 0xBF58476D1CE4E5B9ull;
constexpr std::uint64_t kSplitmixM2 = 0x94D049BB133111EBull;
constexpr std::uint64_t kMask64 = ~std::uint64_t{0};
constexpr std::uint64_t kPrime1 = 11400714819323198485ull;

bool read_size(const nlohmann::json& object, const char* key, std::size_t& out) {
    if (!object.contains(key) || !object[key].is_number()) return false;
    out = object[key].get<std::size_t>();
    return true;
}

bool read_int32(const nlohmann::json& object, const char* key, std::int32_t& out) {
    if (!object.contains(key)) return false;
    const nlohmann::json& value = object[key];
    if (value.is_number_integer()) {
        out = value.get<std::int32_t>();
        return true;
    }
    if (value.is_array() && !value.empty() && value[0].is_number_integer()) {
        out = value[0].get<std::int32_t>();
        return true;
    }
    return false;
}

bool read_string(const nlohmann::json& object, const char* key, std::string& out) {
    if (!object.contains(key) || !object[key].is_string()) return false;
    out = object[key].get<std::string>();
    return true;
}

bool is_prime(std::uint64_t value) {
    if (value < 2u) return false;
    if ((value % 2u) == 0u) return value == 2u;
    for (std::uint64_t divisor = 3u; divisor <= value / divisor; divisor += 2u) {
        if ((value % divisor) == 0u) return false;
    }
    return true;
}

std::uint64_t find_nth_prime_after(std::uint64_t start, std::uint64_t count) {
    std::uint64_t prime = start;
    for (std::uint64_t i = 0; i < count; ++i) {
        ++prime;
        while (!is_prime(prime)) ++prime;
    }
    return prime;
}

std::uint64_t splitmix64(std::uint64_t value) {
    value = (value + kSplitmixGamma) & kMask64;
    value = ((value ^ (value >> 30)) * kSplitmixM1) & kMask64;
    value = ((value ^ (value >> 27)) * kSplitmixM2) & kMask64;
    return (value ^ (value >> 31)) & kMask64;
}

Status require_positive(std::size_t value, const char* what) {
    if (value == 0u) {
        return Status::invalid_argument(
            (std::string("read_qwen4exp_text_config: ") + what + " must be positive").c_str(),
            __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

bool Qwen4ExpTextConfig::layer_has_ple(std::size_t layer) const noexcept {
    for (const std::size_t id : ple.layer_ids) {
        if (id == layer + 1u) return true;
    }
    return false;
}

std::vector<std::size_t> Qwen4ExpTextConfig::ple_layers() const {
    std::vector<std::size_t> out;
    out.reserve(ple.layer_ids.size());
    for (const std::size_t id : ple.layer_ids) {
        if (id == 0u) continue;
        out.push_back(id - 1u);
    }
    return out;
}

Result<bool> declares_qwen4exp_structure(const std::string& model_dir) {
    std::ifstream file(model_dir + "/config.json");
    if (!file.is_open()) {
        return Status::invalid_argument(
            "declares_qwen4exp_structure: cannot open config.json", __FILE__, __LINE__);
    }
    nlohmann::json root;
    try {
        file >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            (std::string("declares_qwen4exp_structure: bad config.json: ") + e.what()).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object() || !root.contains("text_config") ||
        !root["text_config"].is_object()) {
        return false;
    }
    const nlohmann::json& text = root["text_config"];
    const bool has_ple = text.contains("ple_layer_ids") && text["ple_layer_ids"].is_array();
    const bool has_hc = text.contains("hc_count") && text["hc_count"].is_number();
    const bool has_ngram = text.contains("ngram_size") && text["ngram_size"].is_number();
    return has_ple && has_hc && has_ngram;
}

Result<Qwen4ExpTextConfig> read_qwen4exp_text_config(const std::string& model_dir) {
    auto hybrid = shelf::qwen_hybrid::read_qwen_hybrid_config(model_dir);
    if (!hybrid.ok()) return hybrid.status();

    std::ifstream file(model_dir + "/config.json");
    if (!file.is_open()) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: cannot open config.json", __FILE__, __LINE__);
    }
    nlohmann::json root;
    try {
        file >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            (std::string("read_qwen4exp_text_config: bad config.json: ") + e.what()).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object() || !root.contains("text_config") ||
        !root["text_config"].is_object()) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: config.json missing text_config", __FILE__, __LINE__);
    }
    const nlohmann::json& text = root["text_config"];

    Qwen4ExpTextConfig cfg;
    static_cast<shelf::qwen_hybrid::QwenHybridConfig&>(cfg) = hybrid.release();

    read_size(text, "num_experts", cfg.ffn.num_experts);
    read_size(text, "num_experts_per_tok", cfg.ffn.num_experts_per_tok);
    read_size(text, "moe_intermediate_size", cfg.ffn.moe_intermediate_size);
    read_size(text, "shared_expert_intermediate_size",
              cfg.ffn.shared_expert_intermediate_size);
    read_size(text, "hc_count", cfg.hc_count);
    read_size(text, "hc_lowrank", cfg.hc_lowrank);

    read_size(text, "indexer_n_heads", cfg.indexer.n_heads);
    read_size(text, "indexer_kv_heads", cfg.indexer.kv_heads);
    read_size(text, "indexer_head_dim", cfg.indexer.head_dim);
    read_size(text, "indexer_budget", cfg.indexer.budget);
    read_size(text, "indexer_compress_ratio", cfg.indexer.compress_ratio);

    read_size(text, "ple_embed_dim", cfg.ple.embed_dim);
    read_size(text, "ngram_size", cfg.ple.ngram_size);
    read_size(text, "heads_per_ngram", cfg.ple.heads_per_ngram);
    read_size(text, "ngram_vocab_size_base", cfg.ple.ngram_vocab_size_base);
    read_size(text, "make_ngram_vocab_size_divisible_by",
              cfg.ple.make_ngram_vocab_size_divisible_by);
    read_size(text, "split_ngram_parts", cfg.ple.split_ngram_parts);
    read_size(text, "ple_conv_kernel_size", cfg.ple.conv_kernel_size);
    read_size(text, "seed", cfg.ple.seed);
    read_int32(text, "eos_token_id", cfg.ple.eos_token_id);
    read_int32(root, "image_token_id", cfg.image_token_id);
    read_int32(root, "video_token_id", cfg.video_token_id);
    read_string(text, "output_gate_type", cfg.output_gate_type);

    if (text.contains("ple_layer_ids") && text["ple_layer_ids"].is_array()) {
        for (const auto& value : text["ple_layer_ids"]) {
            if (value.is_number_unsigned() || value.is_number_integer()) {
                cfg.ple.layer_ids.push_back(value.get<std::size_t>());
            }
        }
    }

    Status st = require_positive(cfg.ffn.num_experts, "num_experts");
    if (!st.ok()) return st;
    st = require_positive(cfg.ffn.num_experts_per_tok, "num_experts_per_tok");
    if (!st.ok()) return st;
    if (cfg.ffn.num_experts_per_tok > cfg.ffn.num_experts) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: num_experts_per_tok exceeds num_experts", __FILE__,
            __LINE__);
    }
    st = require_positive(cfg.ffn.moe_intermediate_size, "moe_intermediate_size");
    if (!st.ok()) return st;
    st = require_positive(cfg.ffn.shared_expert_intermediate_size,
                          "shared_expert_intermediate_size");
    if (!st.ok()) return st;
    st = require_positive(cfg.hc_count, "hc_count");
    if (!st.ok()) return st;
    st = require_positive(cfg.hc_lowrank, "hc_lowrank");
    if (!st.ok()) return st;
    st = require_positive(cfg.indexer.n_heads, "indexer_n_heads");
    if (!st.ok()) return st;
    st = require_positive(cfg.indexer.kv_heads, "indexer_kv_heads");
    if (!st.ok()) return st;
    st = require_positive(cfg.indexer.head_dim, "indexer_head_dim");
    if (!st.ok()) return st;
    st = require_positive(cfg.indexer.compress_ratio, "indexer_compress_ratio");
    if (!st.ok()) return st;

    if (cfg.ple.layer_ids.empty()) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: ple_layer_ids is required", __FILE__, __LINE__);
    }
    for (const std::size_t id : cfg.ple.layer_ids) {
        if (id == 0u || id > cfg.num_hidden_layers) {
            return Status::invalid_argument(
                "read_qwen4exp_text_config: ple_layer_ids is outside the layer range",
                __FILE__, __LINE__);
        }
    }
    st = require_positive(cfg.ple.embed_dim, "ple_embed_dim");
    if (!st.ok()) return st;
    st = require_positive(cfg.ple.ngram_size, "ngram_size");
    if (!st.ok()) return st;
    st = require_positive(cfg.ple.heads_per_ngram, "heads_per_ngram");
    if (!st.ok()) return st;
    st = require_positive(cfg.ple.ngram_vocab_size_base, "ngram_vocab_size_base");
    if (!st.ok()) return st;
    st = require_positive(cfg.ple.make_ngram_vocab_size_divisible_by,
                          "make_ngram_vocab_size_divisible_by");
    if (!st.ok()) return st;
    st = require_positive(cfg.ple.split_ngram_parts, "split_ngram_parts");
    if (!st.ok()) return st;
    st = require_positive(cfg.ple.conv_kernel_size, "ple_conv_kernel_size");
    if (!st.ok()) return st;
    if (cfg.ple.eos_token_id == 0) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: eos_token_id is required for PLE hashing", __FILE__,
            __LINE__);
    }

    if (cfg.ple.ngram_size < 2u) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: ngram_size must be at least 2", __FILE__, __LINE__);
    }
    const std::size_t ngram_heads = (cfg.ple.ngram_size - 1u) * cfg.ple.heads_per_ngram;
    if (ngram_heads == 0u || (cfg.ple.embed_dim % ngram_heads) != 0u) {
        return Status::invalid_argument(
            "read_qwen4exp_text_config: ple_embed_dim must divide by the n-gram head count",
            __FILE__, __LINE__);
    }
    if ((cfg.ple.embed_dim / ngram_heads) != 160u) {
        return Status::unsupported(
            "read_qwen4exp_text_config: only 160-element PLE rows are implemented", __FILE__,
            __LINE__);
    }
    return cfg;
}

Result<std::size_t> qwen4exp_ple_padded_rows(const Qwen4ExpTextConfig& config,
                                             std::size_t ple_layer_index) {
    auto sizes = qwen4exp_ple_head_vocab_sizes(config, ple_layer_index);
    if (!sizes.ok()) return sizes.status();
    std::size_t total = 0;
    for (const std::size_t size : sizes.value()) total += size;
    const std::size_t divisor = config.ple.make_ngram_vocab_size_divisible_by;
    if (divisor == 0u) {
        return Status::invalid_argument(
            "qwen4exp_ple_padded_rows: divisor must be positive", __FILE__, __LINE__);
    }
    const std::size_t padded = ((total + divisor - 1u) / divisor) * divisor;
    return padded;
}

Result<std::vector<std::size_t>> qwen4exp_ple_head_vocab_sizes(
    const Qwen4ExpTextConfig& config, std::size_t ple_layer_index) {
    const std::size_t heads = (config.ple.ngram_size - 1u) * config.ple.heads_per_ngram;
    if (heads == 0u || config.ple.ngram_vocab_size_base == 0u) {
        return Status::invalid_argument(
            "qwen4exp_ple_head_vocab_sizes: invalid PLE hash geometry", __FILE__, __LINE__);
    }
    std::vector<std::size_t> sizes;
    sizes.reserve(heads);
    for (std::size_t head = 0; head < heads; ++head) {
        const std::size_t global_head = ple_layer_index * heads + head;
        sizes.push_back(static_cast<std::size_t>(
            find_nth_prime_after(config.ple.ngram_vocab_size_base - 1u,
                                 static_cast<std::uint64_t>(global_head) + 1u)));
    }
    return sizes;
}

Result<std::vector<std::size_t>> qwen4exp_ple_head_offsets(
    const Qwen4ExpTextConfig& config, std::size_t ple_layer_index) {
    auto sizes = qwen4exp_ple_head_vocab_sizes(config, ple_layer_index);
    if (!sizes.ok()) return sizes.status();
    std::vector<std::size_t> offsets;
    offsets.reserve(sizes.value().size());
    std::size_t running = 0;
    for (const std::size_t size : sizes.value()) {
        offsets.push_back(running);
        running += size;
    }
    return offsets;
}

Result<std::vector<std::int64_t>> qwen4exp_ple_layer_multipliers(
    const Qwen4ExpTextConfig& config, std::size_t ple_layer_index) {
    if (config.ple.ngram_size == 0u || config.vocab_size == 0u) {
        return Status::invalid_argument(
            "qwen4exp_ple_layer_multipliers: invalid multiplier geometry", __FILE__, __LINE__);
    }
    const std::uint64_t max_long = static_cast<std::uint64_t>((1ull << 63) - 1ull);
    const std::uint64_t multiplier_max = max_long / config.vocab_size;
    const std::uint64_t half_bound = std::max<std::uint64_t>(1u, multiplier_max / 2u);
    const std::uint64_t base_seed =
        static_cast<std::uint64_t>(config.ple.seed) + kPrime1 * ple_layer_index;
    std::vector<std::int64_t> multipliers;
    multipliers.reserve(config.ple.ngram_size);
    for (std::size_t index = 0; index < config.ple.ngram_size; ++index) {
        const std::uint64_t value =
            (base_seed + kSplitmixGamma * (static_cast<std::uint64_t>(index) + 1u)) & kMask64;
        const std::uint64_t multiplier = 2u * (splitmix64(value) % half_bound) + 1u;
        multipliers.push_back(static_cast<std::int64_t>(multiplier));
    }
    return multipliers;
}

}  // namespace qwen4exp
}  // namespace shelf
