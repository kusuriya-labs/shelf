#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35qsa/model/qwen35qsa_config.h>

#include <hip/hip_runtime.h>

#include <vector>

namespace shelf {
namespace qwen35qsa {

// The indexer of one full-attention layer. The reference projects one query head set of
// `heads * head_dim` and one shared key head of `head_dim`, normalizes the keys per head,
// and pools every `block_size` keys into one scored block key.
struct QsaIndexerLayerWeights {
    shelf::weights::MatrixWeight q_proj;
    shelf::weights::MatrixWeight k_proj;
    gpu::Tensor q_norm_weight;
    gpu::Tensor k_norm_weight;
    gpu::Tensor k_block_norm_weight;
};

// A QSA checkpoint: the dense backbone read through the dense reader plus the per-layer
// indexer. GDN layers carry no indexer, so their entry stays empty.
struct Qwen35QsaModelWeights {
    shelf::qwen35::Qwen35ModelWeights backbone;
    std::vector<QsaIndexerLayerWeights> indexer;
};

using Qwen35QsaLoadOptions = shelf::qwen35::Qwen35LoadOptions;

Result<Qwen35QsaModelWeights> load_qwen35qsa_weights(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35QsaLoadOptions& options = {});

// Loads one rank's backbone through the dense tensor-parallel plan and keeps the indexer
// replicated: every rank scores the same pooled keys and must produce the same block mask.
Result<Qwen35QsaModelWeights> load_qwen35qsa_weights_tensor_parallel_rank(
    const std::string& model_dir,
    std::uint32_t tp_size,
    std::uint32_t tp_rank,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen35QsaLoadOptions& options = {});

Status validate_qwen35qsa_weights(
    const Qwen35QsaModelWeights& weights,
    const Qwen35QsaTextConfig& config,
    const shelf::qwen_hybrid::PipelineStage& stage = {});

}
}
