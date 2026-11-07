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

// One routed-expert projection of one rank is the shared rank-3 MatrixWeight form:
// experts are contiguous global indices [expert_begin, expert_begin + experts), and the
// encoding-specific plane (EXL3 trellis / MXFP4 native) is described by MatrixWeight.
struct Qwen4ExpExpertRange {
    std::uint32_t global_experts = 0;
    std::uint32_t local_experts = 0;
    std::uint32_t expert_begin = 0;
};

struct Qwen4ExpMoeWeights {
    shelf::weights::MatrixWeight router;       // [global experts, hidden] replicated
    shelf::weights::MatrixWeight shared_gate;  // [1, hidden] replicated
    shelf::weights::MatrixWeight shared_gate_proj;
    shelf::weights::MatrixWeight shared_up_proj;
    shelf::weights::MatrixWeight shared_down_proj;

    shelf::weights::MatrixWeight gate_experts;
    shelf::weights::MatrixWeight up_experts;
    shelf::weights::MatrixWeight down_experts;
    Qwen4ExpExpertRange experts;
};

struct Qwen4ExpLayerWeights {
    Qwen4ExpHyperConnectionWeights attn_hyper_connection;
    Qwen4ExpHyperConnectionWeights mlp_hyper_connection;
    Qwen4ExpAttentionWeights attention;
    Qwen4ExpMoeWeights mlp;
    Qwen4ExpPleWeights ple;
};

struct Qwen4ExpMtpWeights {
    bool present = false;
    shelf::weights::MatrixWeight fc_hidden;
    shelf::weights::MatrixWeight fc_embedding;
    shelf::gpu::Tensor pre_fc_norm_hidden;
    shelf::gpu::Tensor pre_fc_norm_embedding;
    Qwen4ExpHyperConnectionWeights mixer;
    Qwen4ExpLayerWeights layer;
};

struct Qwen4ExpRankWeights {
    shelf::weights::MatrixWeight embed_tokens;  // replicated BF16
    shelf::weights::MatrixWeight lm_head;       // replicated EXL3
    Qwen4ExpHyperConnectionWeights mixer;       // model-level stream collapse
    std::vector<Qwen4ExpLayerWeights> layers;
    // The draft head is a rank-local copy on rank 0. It is not sharded by the TP plan.
    Qwen4ExpMtpWeights mtp;

    [[nodiscard]] std::uint64_t resident_bytes() const noexcept;
};

inline void note_exl3_linear(const Qwen4ExpHyperConnectionWeights& w,
                             shelf::weights::Exl3LinearShape& shape) noexcept {
    shelf::weights::note_exl3_linear(w.input_mix_down, shape);
    shelf::weights::note_exl3_linear(w.input_mix_up, shape);
    shelf::weights::note_exl3_linear(w.block_inject, shape);
}

inline void note_exl3_linear(const Qwen4ExpAttentionWeights& w,
                             shelf::weights::Exl3LinearShape& shape) noexcept {
    const shelf::weights::MatrixWeight* projections[] = {
        &w.in_proj_qkv, &w.in_proj_z,  &w.in_proj_a, &w.in_proj_b,
        &w.out_proj,    &w.q_proj,     &w.k_proj,    &w.v_proj,
        &w.o_proj,      &w.index_qk_proj};
    for (const shelf::weights::MatrixWeight* m : projections) {
        shelf::weights::note_exl3_linear(*m, shape);
    }
}

inline void note_exl3_linear(const Qwen4ExpMoeWeights& w,
                             shelf::weights::Exl3LinearShape& shape) noexcept {
    shelf::weights::note_exl3_linear(w.router, shape);
    shelf::weights::note_exl3_linear(w.shared_gate, shape);
    shelf::weights::note_exl3_linear(w.shared_gate_proj, shape);
    shelf::weights::note_exl3_linear(w.shared_up_proj, shape);
    shelf::weights::note_exl3_linear(w.shared_down_proj, shape);
    shelf::weights::note_exl3_linear(w.gate_experts, shape);
    shelf::weights::note_exl3_linear(w.up_experts, shape);
    shelf::weights::note_exl3_linear(w.down_experts, shape);
}

inline void note_exl3_linear(const Qwen4ExpPleWeights& w,
                             shelf::weights::Exl3LinearShape& shape) noexcept {
    shelf::weights::note_exl3_linear(w.key_proj, shape);
    shelf::weights::note_exl3_linear(w.value_proj, shape);
}

// The staging-plane geometry of this rank's linears. The LM head and the MTP block are
// excluded: the LM head stays on the fused path, and the MTP block runs in its own executor.
inline shelf::weights::Exl3LinearShape exl3_linear_shape(
    const Qwen4ExpRankWeights& weights) noexcept {
    shelf::weights::Exl3LinearShape shape;
    note_exl3_linear(weights.mixer, shape);
    for (const Qwen4ExpLayerWeights& layer : weights.layers) {
        note_exl3_linear(layer.attn_hyper_connection, shape);
        note_exl3_linear(layer.mlp_hyper_connection, shape);
        note_exl3_linear(layer.attention, shape);
        note_exl3_linear(layer.mlp, shape);
        note_exl3_linear(layer.ple, shape);
    }
    return shape;
}

struct Qwen4ExpLoadOptions {
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    bool verify_quantized_payload_crc = false;
    // Text-only is the first target. MTP and Vision are not loaded by this path.
    bool load_mtp = false;
    bool load_vision = false;
    // Load the embedding table, LM head and MTP block only. Used by the MTP draft-head
    // test and by a rank that does not need the trunk.
    bool mtp_only = false;
};

// Logical checkpoint names this rank asks the HF reader for. Exposed for tests and for
// the memory plan; the list excludes the PLE n-gram table itself, which is not in the
// safetensors index.
std::vector<std::string> qwen4exp_rank_logical_names(const Qwen4ExpTextConfig& config,
                                                     std::uint32_t tp_size,
                                                     std::uint32_t tp_rank,
                                                     bool load_mtp = false,
                                                     bool mtp_only = false);

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
