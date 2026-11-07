#pragma once
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/layer_weights.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/models/common/qwen_hybrid/model_weights.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/models/qwen35/weights/dense_ffn_weights.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace shelf {
namespace qwen35 {

using ::shelf::qwen_hybrid::PipelineStage;

// A dense qwen35 layer: the shared hybrid backbone (layernorms and whichever attention
// the layer carries) plus the dense FFN. The FFN is a member, so a layer of this type
// cannot be built without it and cannot hold any other FFN form.
struct Qwen35LayerWeights : shelf::qwen_hybrid::HybridLayerWeights {
    DenseFfnWeights ffn;
};

struct Qwen35MtpWeights {
    bool present = false;

    gpu::Tensor pre_fc_norm_embedding_weight;
    gpu::Tensor pre_fc_norm_hidden_weight;
    gpu::Tensor final_norm_weight;

    shelf::weights::MatrixWeight fc;
    Qwen35LayerWeights layer;
};

// A dense qwen35 model: the shared model-level weights, one dense layer per transformer
// layer, and the optional MTP block.
struct Qwen35ModelWeights : shelf::qwen_hybrid::HybridModelLevelWeights {
    std::vector<Qwen35LayerWeights> layers;

    Qwen35MtpWeights mtp;
};

// True when any linear of this weight set is carried in the rotated-domain trellis form,
// including the LM head and the MTP block. The memory plan uses it to decide whether the
// executor workspace must hold that form's staging planes.
inline bool uses_exl3_linear(const Qwen35ModelWeights& weights) noexcept {
    if (weights.lm_head.encoding == shelf::weights::MatrixEncoding::Exl3) return true;
    if (weights.mtp.present &&
        weights.mtp.fc.encoding == shelf::weights::MatrixEncoding::Exl3)
        return true;
    if (weights.mtp.present &&
        (shelf::qwen_hybrid::layer_uses_exl3_linear(weights.mtp.layer) ||
         ffn_uses_exl3_linear(weights.mtp.layer.ffn)))
        return true;
    for (const Qwen35LayerWeights& layer : weights.layers) {
        if (shelf::qwen_hybrid::layer_uses_exl3_linear(layer) ||
            ffn_uses_exl3_linear(layer.ffn))
            return true;
    }
    return false;
}

// Dense-only load options on top of the shared hybrid options: an MTP block and a
// vocabulary-sharded LM head exist for the dense model only.
struct Qwen35LoadOptions : shelf::qwen_hybrid::HybridLoadOptions {
    bool load_mtp_layers = false;
    bool vocab_parallel_lm_head = true;
    PipelineStage stage;
};

Result<Qwen35ModelWeights> load_qwen35_weights_from_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options = {});

Result<Qwen35ModelWeights> load_qwen35_weights_from_quantized_safetensors(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options = {});

Result<Qwen35ModelWeights> load_qwen35_weights_tensor_parallel_rank(
    const std::string& model_dir,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options = {});

}
}
