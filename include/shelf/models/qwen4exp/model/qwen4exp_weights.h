#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/weight_loader.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ps::qwen4exp {

struct Qwen4ExpHyperWeights {
    gpu::Tensor hc_norm;
    weights::MatrixWeight mix_down;
    weights::MatrixWeight mix_up;
    weights::MatrixWeight block_inject;
    bool has_block_inject = false;
};

struct Qwen4ExpGdnWeights {
    weights::MatrixWeight qkv;
    weights::MatrixWeight z;
    weights::MatrixWeight b;
    weights::MatrixWeight a;
    weights::MatrixWeight out;
    gpu::Tensor conv1d;
    gpu::Tensor norm;
    gpu::Tensor dt_bias;
    gpu::Tensor a_log;
};

struct Qwen4ExpIndexerWeights {
    weights::MatrixWeight index_qk;
    gpu::Tensor q_layernorm;
    gpu::Tensor k_layernorm;
    bool present = false;
};

struct Qwen4ExpQsaWeights {
    weights::MatrixWeight q;
    weights::MatrixWeight k;
    weights::MatrixWeight v;
    weights::MatrixWeight o;
    gpu::Tensor q_norm;
    gpu::Tensor k_norm;
    Qwen4ExpIndexerWeights indexer;
};

struct Qwen4ExpSharedExpertWeights {
    weights::MatrixWeight gate;
    weights::MatrixWeight up;
    weights::MatrixWeight down;
    gpu::Tensor gate_scale;
};

struct Qwen4ExpMoeWeights {
    weights::MatrixWeight router;
    Qwen4ExpSharedExpertWeights shared;
    std::string expert_gate_up_tensor;
    std::string expert_down_tensor;
    std::uint32_t expert_count = 0;
    std::vector<weights::MatrixWeight> expert_gate_up;
    std::vector<weights::MatrixWeight> expert_down;
};

struct Qwen4ExpPleWeights {
    weights::MatrixWeight key_proj;
    weights::MatrixWeight value_proj;
    gpu::Tensor conv1d;
    gpu::Tensor norm_key;
    gpu::Tensor norm_query;
    gpu::Tensor norm_conv;
    std::vector<std::int64_t> layer_multipliers;
    std::vector<std::int64_t> vocab_sizes;
    std::vector<std::int64_t> offsets;
    bool present = false;
};

struct Qwen4ExpLayerWeights {
    bool is_gdn = false;
    Qwen4ExpHyperWeights attn_hyper;
    Qwen4ExpHyperWeights mlp_hyper;
    Qwen4ExpGdnWeights gdn;
    Qwen4ExpQsaWeights qsa;
    Qwen4ExpMoeWeights moe;
    Qwen4ExpPleWeights ple;
};

struct Qwen4ExpModelWeights {
    using Tensor = gpu::Tensor;

    weights::MatrixWeight embed_tokens;
    weights::MatrixWeight lm_head;
    bool lm_head_tied = false;
    Qwen4ExpHyperWeights final_hyper;
    std::vector<Qwen4ExpLayerWeights> layers;
};

struct Qwen4ExpLoadOptions {
    bool load_ngram_table = false;
    bool load_expert_weights = false;
    bool verify_quantized_payload_crc = false;
    std::uint32_t layer_begin = 0;
    std::uint32_t layer_end = 0;
    weights::WeightLoadOptions weights;
};

Result<Qwen4ExpTextConfig> load_qwen4exp_config(const std::string& model_dir);

Result<Qwen4ExpModelWeights> load_qwen4exp_weights_bf16(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen4ExpLoadOptions& options = {});

Result<Qwen4ExpModelWeights> load_qwen4exp_weights_quantized(
    const std::string& model_dir,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const Qwen4ExpLoadOptions& options = {});

}  // namespace ps::qwen4exp
