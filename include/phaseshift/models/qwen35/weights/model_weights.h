#pragma once
#include <phaseshift/core/memory/tensor.h>
#include <phaseshift/core/status.h>
#include <phaseshift/weights/matrix_weight.h>
#include <phaseshift/weights/weight_loader.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <string>
#include <vector>

namespace ps {
namespace qwen35 {

struct Qwen35LayerWeights {
    bool is_gdn = false;

    gpu::Tensor input_layernorm_weight;
    gpu::Tensor post_attention_layernorm_weight;

    ps::weights::MatrixWeight mlp_gate_proj;
    ps::weights::MatrixWeight mlp_up_proj;
    ps::weights::MatrixWeight mlp_down_proj;

    ps::weights::MatrixWeight attn_q_proj;
    ps::weights::MatrixWeight attn_k_proj;
    ps::weights::MatrixWeight attn_v_proj;
    ps::weights::MatrixWeight attn_o_proj;
    gpu::Tensor attn_q_norm_weight;
    gpu::Tensor attn_k_norm_weight;

    ps::weights::MatrixWeight attn_in_proj_a;
    ps::weights::MatrixWeight attn_in_proj_b;
    ps::weights::MatrixWeight attn_in_proj_qkv;
    ps::weights::MatrixWeight attn_in_proj_z;
    gpu::Tensor attn_conv1d_weight;
    ps::weights::MatrixWeight attn_out_proj;
    gpu::Tensor attn_norm_weight;
    gpu::Tensor attn_dt_bias;
    gpu::Tensor attn_A_log;
};

struct Qwen35MtpWeights {
    bool present = false;

    gpu::Tensor pre_fc_norm_embedding_weight;
    gpu::Tensor pre_fc_norm_hidden_weight;
    gpu::Tensor final_norm_weight;

    ps::weights::MatrixWeight fc;
    Qwen35LayerWeights layer;
};

struct Qwen35ModelWeights {
    using Tensor = gpu::Tensor;

    ps::weights::MatrixWeight embed_tokens;
    Tensor final_norm_weight;

    ps::weights::MatrixWeight lm_head;
    bool lm_head_tied = true;

    std::vector<Qwen35LayerWeights> layers;

    Qwen35MtpWeights mtp;
};

struct Qwen35LoadOptions {
    bool verify_quantized_payload_crc = false;
    bool load_mtp_layers = false;
    bool vocab_parallel_lm_head = true;
    ps::weights::WeightLoadOptions weights;
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
