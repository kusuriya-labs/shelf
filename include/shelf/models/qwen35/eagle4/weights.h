#pragma once
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/eagle4/config.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace ps::qwen35::eagle4 {

struct Eagle4LayerWeights {
    gpu::Tensor input_layernorm_weight;
    gpu::Tensor hidden_norm_weight;
    gpu::Tensor post_attention_layernorm_weight;

    ps::weights::MatrixWeight attn_q_proj;
    ps::weights::MatrixWeight attn_k_proj;
    ps::weights::MatrixWeight attn_v_proj;
    ps::weights::MatrixWeight attn_o_proj;

    ps::weights::MatrixWeight mlp_gate_proj;
    ps::weights::MatrixWeight mlp_up_proj;
    ps::weights::MatrixWeight mlp_down_proj;
};

struct Eagle4CandidateWeights {
    ps::weights::MatrixWeight lm_head;
    ps::weights::MatrixWeight efficient_lm_head;
    ps::weights::MatrixWeight efficient_lm_head_down_proj;
};

struct Eagle4Weights {
    bool present = false;

    ps::weights::MatrixWeight fc;
    std::array<gpu::Tensor, kEagle4AuxLayers> fc_norm_weight;
    gpu::Tensor final_norm_weight;
    ps::weights::MatrixWeight embed_tokens;
    Eagle4CandidateWeights candidate;
    std::vector<Eagle4LayerWeights> layers;
};

struct Eagle4WeightLoadOptions {
    bool preshuffle = true;
    bool load_embed_tokens = true;
    bool load_lm_head = true;
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
};

struct Eagle4ExpectedTensor {
    std::string name;
    std::vector<std::size_t> shape;
};

std::vector<Eagle4ExpectedTensor> eagle4_expected_tensors(const Eagle4Config& config);

Status validate_eagle4_tensor_contract(
    const std::string& model_dir,
    const Eagle4Config& config);

Result<Eagle4Weights> load_eagle4_weights(
    const std::string& model_dir,
    const Eagle4Config& config,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Eagle4WeightLoadOptions& options = {});

}  // namespace ps::qwen35::eagle4
