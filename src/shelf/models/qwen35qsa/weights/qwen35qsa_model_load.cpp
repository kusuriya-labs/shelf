#include <shelf/models/qwen35qsa/weights/model_weights.h>

#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/quantization/bundle/hf_manifest.h>
#include <shelf/weights/weight_loader.h>

#include <string>

namespace shelf {
namespace qwen35qsa {

namespace {

namespace qqb = shelf::quantization::bundle;

std::string indexer_prefix(std::size_t layer) {
    return "model.language_model.layers." + std::to_string(layer) +
           ".self_attn._qsa_indexer.";
}

Status load_indexer(
    const shelf::weights::SafetensorsCollection& collection,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::weights::WeightLoadOptions& weight_options,
    std::size_t layer,
    QsaIndexerLayerWeights& w) {
    const std::string prefix = indexer_prefix(layer);
    auto q = shelf::weights::load_bf16_matrix(collection, prefix + "q_proj.weight", arena,
                                              stream, weight_options);
    if (!q.ok()) return q.status();
    w.q_proj = q.release();
    auto k = shelf::weights::load_bf16_matrix(collection, prefix + "k_proj.weight", arena,
                                              stream, weight_options);
    if (!k.ok()) return k.status();
    w.k_proj = k.release();
    auto qn = shelf::weights::load_bf16_or_f32_tensor(collection, prefix + "q_norm", arena,
                                                      stream);
    if (!qn.ok()) return qn.status();
    w.q_norm_weight = qn.release();
    auto kn = shelf::weights::load_bf16_or_f32_tensor(collection, prefix + "k_norm", arena,
                                                      stream);
    if (!kn.ok()) return kn.status();
    w.k_norm_weight = kn.release();
    auto bn = shelf::weights::load_bf16_or_f32_tensor(collection, prefix + "k_block_norm",
                                                      arena, stream);
    if (!bn.ok()) return bn.status();
    w.k_block_norm_weight = bn.release();
    return Status::make_ok();
}

}  // namespace

Result<Qwen35QsaModelWeights> load_qwen35qsa_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35QsaLoadOptions& options) {
    Qwen35QsaModelWeights weights;
    const bool quantized = qqb::is_hf_quantized_model_dir(model_dir);
    auto backbone = quantized
        ? shelf::qwen35::load_qwen35_weights_from_quantized_safetensors(
              model_dir, arena, stream, options)
        : shelf::qwen35::load_qwen35_weights_from_safetensors(
              model_dir, arena, stream, options);
    if (!backbone.ok()) return backbone.status();
    weights.backbone = backbone.release();

    auto collection_result = shelf::weights::SafetensorsCollection::open(model_dir);
    if (!collection_result.ok()) return collection_result.status();
    shelf::weights::SafetensorsCollection collection = collection_result.release();

    weights.indexer.resize(weights.backbone.layers.size());
    for (std::size_t layer = 0; layer < weights.backbone.layers.size(); ++layer) {
        if (weights.backbone.layers[layer].is_gdn) continue;
        Status st = load_indexer(collection, arena, stream, options.weights, layer,
                                 weights.indexer[layer]);
        if (!st.ok()) return st;
    }
    return weights;
}

Status validate_qwen35qsa_weights(
    const Qwen35QsaModelWeights& weights,
    const Qwen35QsaTextConfig& config) {
    Status st = shelf::qwen35::validate_qwen35_weights(weights.backbone, config);
    if (!st.ok()) return st;
    st = validate_qsa_indexer_spec(config.indexer);
    if (!st.ok()) return st;

    const std::uint64_t hidden = config.hidden_size;
    const std::uint64_t q_rows = config.indexer.heads * config.indexer.head_dim;
    if (weights.indexer.size() != weights.backbone.layers.size()) {
        return Status::invalid_argument(
            "qwen35qsa: indexer count does not match the layer count", __FILE__, __LINE__);
    }
    for (std::size_t layer = 0; layer < weights.backbone.layers.size(); ++layer) {
        const auto& w = weights.indexer[layer];
        if (weights.backbone.layers[layer].is_gdn) {
            if (w.q_proj.rows != 0) {
                return Status::invalid_argument(
                    "qwen35qsa: a GDN layer carries an indexer", __FILE__, __LINE__);
            }
            continue;
        }
        if (w.q_proj.rows != q_rows || w.q_proj.cols != hidden ||
            w.k_proj.rows != config.indexer.head_dim || w.k_proj.cols != hidden) {
            return Status::invalid_argument(
                "qwen35qsa: indexer projection shape mismatch", __FILE__, __LINE__);
        }
        const auto elements = [](const gpu::Tensor& t) {
            std::size_t n = 1;
            for (std::size_t i = 0; i < t.ndim(); ++i) n *= t.dim(i);
            return n;
        };
        if (elements(w.q_norm_weight) != config.indexer.head_dim ||
            elements(w.k_norm_weight) != config.indexer.head_dim ||
            elements(w.k_block_norm_weight) != config.indexer.head_dim) {
            return Status::invalid_argument(
                "qwen35qsa: indexer norm shape mismatch", __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

}
}
