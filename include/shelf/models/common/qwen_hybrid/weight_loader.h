#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/common/qwen_hybrid/layer_weights.h>
#include <shelf/models/common/qwen_hybrid/model_weights.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace shelf {
namespace qwen_hybrid {

// Loading the part of a Qwen hybrid checkpoint that does not depend on how a layer's FFN
// is built: the model-level tensors, and each layer's layernorms and attention. A dense
// model and an MoE model both call these and then load their own FFN, so no FFN tensor
// name appears here. A model family owns the names of its FFN; this module owns the
// physical load of everything around them.

struct HybridLoadOptions {
    bool verify_quantized_payload_crc = false;
    bool load_embedding = true;
    bool load_lm_head = true;
    bool load_final_norm = true;
    shelf::weights::WeightLoadOptions weights;
};

// Checkpoint name prefix of one transformer layer:
// "model.language_model.layers.<layer>."
std::string layer_tensor_prefix(std::size_t layer);

// Body logical names: embed_tokens, lm_head, the final norm, and per layer the two
// layernorms plus the attention tensors of that layer's topology. No FFN names.
std::vector<std::string> hf_quantized_body_names(const QwenHybridConfig& config);

std::vector<std::string> hf_quantized_layer_body_names(const QwenHybridConfig& config,
                                                       std::size_t layer);

// Body logical names at an explicit checkpoint prefix and topology: a checkpoint block
// that is not a transformer layer (the MTP block) gives its own prefix.
std::vector<std::string> hf_layer_body_names(const std::string& prefix, bool is_gdn);

// Opens a Shelf quantized model directory, or synthesizes an HF real-quantized one from
// the given logical names. A name the checkpoint does not carry is skipped by the
// synthesis, so the caller owns the completeness check.
Result<shelf::quantization::bundle::QuantizedModelReader> open_quantized_model_reader(
    const std::string& model_dir,
    const std::vector<std::string>& logical_names,
    bool verify_payload_crc);

Status load_hybrid_model_level_quantized(
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    const QwenHybridConfig& config,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const HybridLoadOptions& options,
    HybridModelLevelWeights& weights);

// One layer's layernorms and attention. The layer topology comes from the config, so the
// resulting is_gdn flag cannot disagree with it.
Status load_hybrid_layer_body_quantized(
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    const QwenHybridConfig& config,
    std::size_t layer,
    const HybridLoadOptions& options,
    HybridLayerWeights& w);

// The same body at an explicit prefix and topology.
Status load_hybrid_layer_body_quantized_at(
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    const std::string& prefix,
    bool is_gdn,
    const HybridLoadOptions& options,
    HybridLayerWeights& w);

}  // namespace qwen_hybrid
}  // namespace shelf
