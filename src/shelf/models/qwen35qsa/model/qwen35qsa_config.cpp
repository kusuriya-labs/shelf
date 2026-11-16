#include <shelf/models/qwen35qsa/model/qwen35qsa_config.h>

#include <fstream>
#include <nlohmann/json.hpp>

namespace shelf {
namespace qwen35qsa {

namespace {

Result<nlohmann::json> read_root(const std::string& model_dir) {
    std::ifstream f(model_dir + "/config.json");
    if (!f.is_open()) {
        return Status::invalid_argument(
            "read_qwen35qsa_text_config: cannot open config.json", __FILE__, __LINE__);
    }
    nlohmann::json root;
    try {
        f >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status::invalid_argument(
            ("read_qwen35qsa_text_config: bad config.json: " + std::string(e.what())).c_str(),
            __FILE__, __LINE__);
    }
    if (!root.is_object()) {
        return Status::invalid_argument(
            "read_qwen35qsa_text_config: config.json is not an object", __FILE__, __LINE__);
    }
    return root;
}

}  // namespace

Status validate_qsa_indexer_spec(const QsaIndexerSpec& spec) {
    if (!spec.declared) {
        return Status::unsupported(
            "qwen35qsa: the checkpoint does not declare a QSA indexer", __FILE__, __LINE__);
    }
    if (spec.budget == 0 || spec.block_size == 0) {
        return Status::invalid_argument(
            "qwen35qsa: indexer budget and block_size must be non-zero", __FILE__,
            __LINE__);
    }
    if ((spec.block_size & (spec.block_size - 1)) != 0) {
        return Status::invalid_argument(
            "qwen35qsa: indexer block_size must be a power of two", __FILE__, __LINE__);
    }
    if (spec.budget % spec.block_size != 0) {
        return Status::invalid_argument(
            "qwen35qsa: indexer budget must be a multiple of block_size", __FILE__,
            __LINE__);
    }
    if (spec.heads == 0 || spec.head_dim == 0) {
        return Status::invalid_argument(
            "qwen35qsa: indexer heads and head_dim must be non-zero", __FILE__, __LINE__);
    }
    if (spec.rope_dim == 0 || (spec.rope_dim & 1u) != 0 || spec.rope_dim > spec.head_dim) {
        return Status::invalid_argument(
            "qwen35qsa: indexer rope_dim must be even and at most head_dim", __FILE__,
            __LINE__);
    }
    return Status::make_ok();
}

Result<Qwen35QsaTextConfig> read_qwen35qsa_text_config(const std::string& model_dir) {
    auto dense = shelf::qwen35::read_qwen35_text_config(model_dir);
    if (!dense.ok()) return dense.status();
    auto root = read_root(model_dir);
    if (!root.ok()) return root.status();

    Qwen35QsaTextConfig config;
    static_cast<shelf::qwen35::Qwen35TextConfig&>(config) = dense.release();

    const nlohmann::json& r = root.value();
    if (!r.contains("qsa") || !r["qsa"].is_object()) {
        return Status::unsupported(
            "qwen35qsa: config.json does not declare a qsa block", __FILE__, __LINE__);
    }
    const nlohmann::json& q = r["qsa"];
    config.indexer.declared = true;
    auto read_size = [&](const char* key, std::size_t& out) {
        if (q.contains(key) && q[key].is_number()) out = q[key].get<std::size_t>();
    };
    read_size("budget", config.indexer.budget);
    read_size("block_size", config.indexer.block_size);
    read_size("indexer_heads", config.indexer.heads);
    read_size("indexer_head_dim", config.indexer.head_dim);
    read_size("rope_dim", config.indexer.rope_dim);

    // The image/video route is the dense one; another policy is not implemented.
    if (q.contains("multimodal_policy") && q["multimodal_policy"].is_string() &&
        q["multimodal_policy"].get<std::string>() != "dense") {
        return Status::unsupported(
            "qwen35qsa: only the dense multimodal policy is implemented", __FILE__,
            __LINE__);
    }

    Status st = validate_qsa_indexer_spec(config.indexer);
    if (!st.ok()) return st;
    return config;
}

}
}
