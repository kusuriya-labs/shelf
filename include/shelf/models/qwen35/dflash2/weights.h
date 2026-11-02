#pragma once
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/weight_loader.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <string>
#include <vector>

namespace ps {
namespace qwen35 {
namespace dflash2 {

struct DFlash2LayerWeights {
    gpu::Tensor input_layernorm_weight;
    gpu::Tensor post_attention_layernorm_weight;

    ps::weights::MatrixWeight attn_q_proj;
    ps::weights::MatrixWeight attn_k_proj;
    ps::weights::MatrixWeight attn_v_proj;
    ps::weights::MatrixWeight attn_o_proj;
    gpu::Tensor attn_q_norm_weight;
    gpu::Tensor attn_k_norm_weight;

    ps::weights::MatrixWeight mlp_gate_proj;
    ps::weights::MatrixWeight mlp_up_proj;
    ps::weights::MatrixWeight mlp_down_proj;

    gpu::Tensor attention_conv_base_kernel;
    ps::weights::MatrixWeight attention_conv_kernel_projection;
    gpu::Tensor mlp_conv_base_kernel;
    ps::weights::MatrixWeight mlp_conv_kernel_projection;
};

struct DFlash2SelectorWeights {
    ps::weights::MatrixWeight hidden_projection;
    gpu::Tensor predecessor_codebook;
    gpu::Tensor successor_codebook;
};

struct DFlash2Weights {
    bool present = false;

    gpu::Tensor hidden_norm_weight;
    gpu::Tensor final_norm_weight;

    ps::weights::MatrixWeight fc;
    DFlash2SelectorWeights selector;
    std::vector<DFlash2LayerWeights> layers;
};

struct DFlash2ExpectedTensor {
    std::string name;
    std::vector<std::size_t> shape;
};

std::vector<DFlash2ExpectedTensor> dflash2_expected_tensors(const DFlash2Config& config);

Status validate_dflash2_tensor_contract(
    const std::string& model_dir,
    const DFlash2Config& config);

Result<DFlash2Weights> load_dflash2_weights(
    const std::string& model_dir,
    const DFlash2Config& config,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const ps::weights::WeightLoadOptions& options = {});

}  // namespace dflash2
}  // namespace qwen35
}  // namespace ps
