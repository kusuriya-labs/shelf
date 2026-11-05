#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace shelf {
namespace qwen4exp {

struct Qwen4ExpHyperConnectionWeights {
    shelf::weights::MatrixWeight input_mix_down;  // [lowrank, hc*hidden]
    shelf::weights::MatrixWeight input_mix_up;    // [hc*hidden, lowrank]
    shelf::weights::MatrixWeight block_inject;    // [hc, hc*hidden]
    shelf::gpu::Tensor hc_norm;                   // [hc*hidden]
};

struct Qwen4ExpAttentionWeights {
    bool is_gdn = false;

    // Gated DeltaNet (linear_attention).
    shelf::weights::MatrixWeight in_proj_qkv;
    shelf::weights::MatrixWeight in_proj_z;
    shelf::weights::MatrixWeight in_proj_a;
    shelf::weights::MatrixWeight in_proj_b;
    shelf::weights::MatrixWeight out_proj;
    shelf::gpu::Tensor conv1d;
    shelf::gpu::Tensor norm;
    shelf::gpu::Tensor a_log;
    shelf::gpu::Tensor dt_bias;

    // Qwen Sparse Attention (full_attention).
    shelf::weights::MatrixWeight q_proj;
    shelf::weights::MatrixWeight k_proj;
    shelf::weights::MatrixWeight v_proj;
    shelf::weights::MatrixWeight o_proj;
    shelf::gpu::Tensor q_norm;
    shelf::gpu::Tensor k_norm;

    // The indexer is replicated: its KV head count does not divide by TP=2.
    shelf::weights::MatrixWeight index_qk_proj;
    shelf::gpu::Tensor index_q_layernorm;
    shelf::gpu::Tensor index_k_layernorm;
};

struct Qwen4ExpPleWeights {
    bool present = false;
    shelf::weights::MatrixWeight key_proj;    // [hc*hidden, ple_embed_dim]
    shelf::weights::MatrixWeight value_proj;  // [hidden, ple_embed_dim]
    shelf::gpu::Tensor conv1d;                // [hc*hidden, 1, conv_kernel]
    shelf::gpu::Tensor norm_key;
    shelf::gpu::Tensor norm_query;
    shelf::gpu::Tensor norm_conv;
};

// One routed-expert projection of one rank. The payload is the checkpoint's EXL3
// trellis, left packed. experts are contiguous global indices
// [expert_begin, expert_begin + local_experts). The per-expert strides are the same for
// every expert in a layer because a layer's routed experts share one bit rate and shape.
struct Qwen4ExpExpertPlane {
    std::uint32_t global_experts = 0;
    std::uint32_t local_experts = 0;
    std::uint32_t expert_begin = 0;
    std::uint32_t rows = 0;  // logical output rows (n)
    std::uint32_t cols = 0;  // logical input columns (k)
    std::uint32_t k_padded = 0;
    std::uint32_t ka = 0;
    bool half = false;

    std::uint64_t trellis_bytes_per_expert = 0;
    std::uint64_t suh_bytes_per_expert = 0;
    std::uint64_t svh_bytes_per_expert = 0;

    shelf::gpu::Tensor codes;   // [local_experts, trellis_bytes]
    shelf::gpu::Tensor suh;     // [local_experts, k_padded] F16
    shelf::gpu::Tensor svh;     // [local_experts, rows] F16
    shelf::gpu::Tensor marker;  // [local_experts, 4] U8 (mul1)

    [[nodiscard]] std::uint64_t resident_bytes() const noexcept;
};

// A rank-3 MatrixWeight view over one packed expert plane. The runtime weight table
// consumes this shape; it does not own the GPU memory, which stays in the rank arena.
shelf::weights::MatrixWeight make_expert_matrix(const Qwen4ExpExpertPlane& plane);

struct Qwen4ExpMoeWeights {
    shelf::weights::MatrixWeight router;       // [global experts, hidden] replicated
    shelf::weights::MatrixWeight shared_gate;  // [1, hidden] replicated
    shelf::weights::MatrixWeight shared_gate_proj;
    shelf::weights::MatrixWeight shared_up_proj;
    shelf::weights::MatrixWeight shared_down_proj;

    Qwen4ExpExpertPlane gate_experts;
    Qwen4ExpExpertPlane up_experts;
    Qwen4ExpExpertPlane down_experts;
};

struct Qwen4ExpLayerWeights {
    Qwen4ExpHyperConnectionWeights attn_hyper_connection;
    Qwen4ExpHyperConnectionWeights mlp_hyper_connection;
    Qwen4ExpAttentionWeights attention;
    Qwen4ExpMoeWeights mlp;
    Qwen4ExpPleWeights ple;
};

struct Qwen4ExpRankWeights {
    shelf::weights::MatrixWeight embed_tokens;  // replicated BF16
    shelf::weights::MatrixWeight lm_head;       // replicated EXL3
    Qwen4ExpHyperConnectionWeights mixer;       // model-level stream collapse
    std::vector<Qwen4ExpLayerWeights> layers;

    [[nodiscard]] std::uint64_t resident_bytes() const noexcept;
};

struct Qwen4ExpLoadOptions {
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    bool verify_quantized_payload_crc = false;
    // Text-only is the first target. MTP and Vision are not loaded by this path.
    bool load_mtp = false;
    bool load_vision = false;
};

// Logical checkpoint names this rank asks the HF reader for. Exposed for tests and for
// the memory plan; the list excludes the PLE n-gram table itself, which is not in the
// safetensors index.
std::vector<std::string> qwen4exp_rank_logical_names(const Qwen4ExpTextConfig& config,
                                                     std::uint32_t tp_size,
                                                     std::uint32_t tp_rank);

// Loads one tensor-parallel rank of a Qwen4Exp checkpoint. Routed experts are split on
// the expert axis. The canonical partition contract does not split EXL3 matrices yet, so
// the shared expert, GDN/full attention, router, hyper-connections, embedding, LM head and
// PLE small weights are replicated on every rank for this first load path. The PLE table
// is not loaded here.
Result<Qwen4ExpRankWeights> load_qwen4exp_rank(const std::string& model_dir,
                                               gpu::GpuArena& arena,
                                               hipStream_t stream,
                                               const Qwen4ExpLoadOptions& options = {});

// Shape and encoding invariants of one loaded rank.
Status validate_qwen4exp_rank(const Qwen4ExpRankWeights& weights,
                              const Qwen4ExpTextConfig& config,
                              std::uint32_t tp_size,
                              std::uint32_t tp_rank);

}  // namespace qwen4exp
}  // namespace shelf
