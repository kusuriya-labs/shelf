#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/model/qwen35_config.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/quantization/bundle/quantized_model_reader.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace shelf {
namespace qwen35 {

// The part of a qwen35 checkpoint that does not depend on what the layer's FFN is: the
// model-level tensors, and each layer's layernorms and attention. Both are split out of
// load_qwen35_weights_from_quantized_safetensors so that a loader for a qwen35 variant
// whose FFN is named differently can reuse them and supply its own FFN names and loads.
// The dense loader calls them too, and adds the FFN trio itself.

// Layer topology rule: layer_types when it is complete, otherwise derived from
// full_attention_interval.
bool qwen35_layer_is_gdn(const Qwen35TextConfig& config, std::size_t layer);

// Body logical names: embed_tokens, lm_head, final norm, and per layer the layernorms
// plus the attention tensors of its topology. No FFN name appears here.
std::vector<std::string> qwen35_hf_quantized_body_names(const Qwen35TextConfig& config);

// Opens a Shelf quantized model directory, or synthesizes an HF real-quantized one from
// the given logical names. A name the checkpoint does not carry is skipped by the
// synthesis, so the caller owns the completeness check.
Result<shelf::quantization::bundle::QuantizedModelReader> qwen35_open_quantized_reader(
    const std::string& model_dir,
    const std::vector<std::string>& logical_names,
    bool verify_payload_crc);

// embed_tokens, lm_head (honoring tie_word_embeddings) and the final norm.
Status qwen35_load_model_level_quantized(
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    const Qwen35TextConfig& config,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35LoadOptions& options,
    Qwen35ModelWeights& weights);

// One layer's layernorms and attention, and its is_gdn flag.
Status qwen35_load_layer_body_quantized(
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::quantization::bundle::QuantizedModelReader& reader,
    std::size_t layer,
    bool is_gdn,
    const shelf::weights::WeightLoadOptions& weight_options,
    Qwen35LayerWeights& w);

}  // namespace qwen35
}  // namespace shelf
