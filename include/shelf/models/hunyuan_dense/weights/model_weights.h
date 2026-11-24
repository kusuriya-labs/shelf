#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/models/hunyuan_dense/model/config.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include <string>
#include <vector>

namespace shelf::hunyuan_dense {

struct LoadOptions {
    bool verify_quantized_payload_crc = false;
    bool quantize_lm_head_int8_w32 = false;
};

struct LayerWeights {
    gpu::Tensor input_layernorm;
    gpu::Tensor post_attention_layernorm;
    gpu::Tensor query_layernorm;
    gpu::Tensor key_layernorm;
    shelf::weights::MatrixWeight q_proj;
    shelf::weights::MatrixWeight k_proj;
    shelf::weights::MatrixWeight v_proj;
    shelf::weights::MatrixWeight o_proj;
    shelf::weights::MatrixWeight gate_proj;
    shelf::weights::MatrixWeight up_proj;
    shelf::weights::MatrixWeight down_proj;
};

struct ModelWeights {
    shelf::weights::MatrixWeight embed_tokens;
    shelf::weights::MatrixWeight lm_head;
    bool lm_head_tied = false;
    gpu::Tensor final_norm;
    std::vector<LayerWeights> layers;
};

Status validate_checkpoint(const shelf::weights::SafetensorsCollection& collection,
                           const Config& config);
Result<ModelWeights> load_weights(const std::string& model_dir,
                                  const Config& config,
                                  gpu::GpuArena& arena,
                                  hipStream_t stream,
                                  const LoadOptions& options = {});

}
