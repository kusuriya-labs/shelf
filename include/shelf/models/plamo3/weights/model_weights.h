#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/models/plamo3/model/config.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace shelf::plamo3 {

struct LayerWeights {
    gpu::Tensor pre_mixer_norm, post_mixer_norm, pre_mlp_norm, post_mlp_norm;
    gpu::Tensor q_norm, k_norm;
    shelf::weights::MatrixWeight qkv, q, k, v, o_proj, gate_up, gate, up, down;
    bool split_projections = false;
};

struct ModelWeights {
    shelf::weights::MatrixWeight embed_tokens, lm_head;
    gpu::Tensor final_norm;
    std::vector<LayerWeights> layers;
};

// Check all names, dtypes and shapes before allocating device memory.
Status validate_checkpoint(const shelf::weights::SafetensorsCollection& collection,
                           const Config& config);
Result<ModelWeights> load_weights(const std::string& model_dir, const Config& config,
                                  gpu::GpuArena& arena, hipStream_t stream);

}  // namespace shelf::plamo3
