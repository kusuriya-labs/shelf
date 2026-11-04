#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/common/qwen_hybrid/weight_validation.h>

#include <cstddef>
#include <iterator>
#include <utility>

namespace shelf::qwen35 {

namespace {

// gate / up are [intermediate, hidden] and down is [hidden, intermediate], as the model
// states them: a tensor-parallel rank holds one equal slice of the intermediate axis.
Status validate_dense_ffn_weights(const DenseFfnWeights& ffn, std::size_t hidden,
                                  std::size_t intermediate) {
    namespace qhy = shelf::qwen_hybrid;
    Status st = qhy::check_matrix_shape(ffn.gate, intermediate, hidden, "qwen35 ffn.gate_proj");
    if (!st.ok()) return st;
    st = qhy::check_matrix_shape(ffn.up, intermediate, hidden, "qwen35 ffn.up_proj");
    if (!st.ok()) return st;
    st = qhy::check_matrix_shape(ffn.down, hidden, intermediate, "qwen35 ffn.down_proj");
    if (!st.ok()) return st;
    return Status::make_ok();
}

}  // namespace

Status validate_qwen35_weights(
    const Qwen35ModelWeights& weights,
    const Qwen35TextConfig& config)
{
    namespace qhy = shelf::qwen_hybrid;

    if (config.ffn.intermediate_size == 0) {
        return Status::invalid_argument(
            "Qwen35Model: intermediate_size is required",
            __FILE__, __LINE__);
    }

    Status st = qhy::validate_hybrid_model_level(weights, config, weights.layers.size());
    if (!st.ok()) return st;

    const std::size_t hidden = config.hidden_size;
    const std::size_t inter = config.ffn.intermediate_size;
    for (std::size_t layer = 0; layer < weights.layers.size(); ++layer) {
        st = qhy::validate_hybrid_layer_body(weights.layers[layer], config, layer);
        if (!st.ok()) return st;
        st = validate_dense_ffn_weights(weights.layers[layer].ffn, hidden, inter);
        if (!st.ok()) return st;
    }

    if (!weights.mtp.present) return Status::make_ok();

    st = qhy::validate_hybrid_layer_body(weights.mtp.layer, hidden, false, "qwen35 mtp layer");
    if (!st.ok()) return st;
    st = validate_dense_ffn_weights(weights.mtp.layer.ffn, hidden, inter);
    if (!st.ok()) return st;
    if (weights.mtp.fc.rows == 0 || weights.mtp.fc.cols == 0) {
        return Status::invalid_argument("qwen35 mtp: fc is not loaded", __FILE__, __LINE__);
    }
    const gpu::Tensor mtp_norms[] = {
        weights.mtp.pre_fc_norm_embedding_weight,
        weights.mtp.pre_fc_norm_hidden_weight,
        weights.mtp.final_norm_weight,
    };
    for (const gpu::Tensor& t : mtp_norms) {
        if (t.ndim() != 1 || static_cast<std::size_t>(t.dim(0)) != hidden) {
            return Status::invalid_argument("qwen35 mtp: a norm is not [hidden]",
                                            __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Result<Qwen35Model> Qwen35Model::load_from_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) {
        return config_result.status();
    }
    Qwen35TextConfig text_config = config_result.release();

    auto weights_result =
        load_qwen35_weights_from_safetensors(model_dir, arena, stream, options);
    if (!weights_result.ok()) {
        return weights_result.status();
    }
    Qwen35ModelWeights weights = weights_result.release();

    auto validation = validate_qwen35_weights(weights, text_config);
    if (!validation.ok()) {
        return validation;
    }

    return Qwen35Model(std::move(weights), text_config);
}

Result<Qwen35Model> Qwen35Model::load_tensor_parallel_from_safetensors(
    const std::string& model_dir,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options)
{
    auto config_result = read_qwen35_text_config(model_dir);
    if (!config_result.ok()) {
        return config_result.status();
    }
    Qwen35TextConfig text_config = config_result.release();

    auto weights_result = load_qwen35_weights_tensor_parallel_rank(
        model_dir, tp_size, tp_rank, arena, stream, options);
    if (!weights_result.ok()) {
        return weights_result.status();
    }
    Qwen35ModelWeights weights = weights_result.release();

    auto validation = validate_qwen35_weights(weights, text_config);
    if (!validation.ok()) {
        return validation;
    }

    return Qwen35Model(std::move(weights), text_config);
}

Result<Qwen35Model> Qwen35Model::adopt(
    Qwen35ModelWeights&& weights,
    Qwen35TextConfig text_config)
{
    auto validation = validate_qwen35_weights(weights, text_config);
    if (!validation.ok()) {
        return validation;
    }
    return Qwen35Model(std::move(weights), text_config);
}

Qwen35Model::Qwen35Model(
    Qwen35ModelWeights&& weights,
    Qwen35TextConfig text_config)
    : weights_(std::move(weights)),
      text_config_(std::move(text_config))
{
}

} // namespace shelf::qwen35
