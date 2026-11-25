#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/weights/matrix_weight.h>
#include <shelf/weights/pinned_host_arena.h>

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
    // Cold experts of the same layer. Empty when the rank keeps every expert in VRAM.
    // A host plane's expert index is the local expert index minus `vram_experts`.
    shelf::weights::MatrixWeight gate_experts_host;
    shelf::weights::MatrixWeight up_experts_host;
    shelf::weights::MatrixWeight down_experts_host;
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
    shelf::weights::MatrixWeight embed_tokens;  // BF16, pinned host
    shelf::weights::MatrixWeight lm_head;       // replicated EXL3
    Qwen4ExpHyperConnectionWeights mixer;       // model-level stream collapse
    std::vector<Qwen4ExpLayerWeights> layers;
    // The draft head is a rank-local copy on rank 0. It is not sharded by the TP plan.
    Qwen4ExpMtpWeights mtp;
    // Backs the embedding table. Pinned host memory, device-readable.
    shelf::weights::PinnedHostArena pinned_host;
    // Cold routed experts of every layer. Pinned host memory, device-readable.
    shelf::weights::PinnedHostArena expert_host;
    // Device int32 [layers][local_experts]: >= 0 is a VRAM slot, -1 - slot is a host
    // slot. Empty when no expert is offloaded.
    shelf::gpu::Tensor expert_slot_map;
    // Device uint32 [layers][local_experts]: routes observed in the current batch.
    // The expert cache reads and clears it between steps.
    shelf::gpu::Tensor expert_route_freq;
    std::uint32_t vram_experts = 0;
    std::uint32_t host_experts = 0;

    [[nodiscard]] std::uint64_t resident_bytes() const noexcept;
    [[nodiscard]] std::uint64_t host_pinned_bytes() const noexcept;
    [[nodiscard]] std::uint64_t host_expert_bytes() const noexcept;
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

// The staging-plane geometry of the MTP block and the LM head it ends with. The drafter
// runs in its own executor, so both shapes share that executor's staging plane.
inline shelf::weights::Exl3LinearShape mtp_exl3_linear_shape(
    const Qwen4ExpRankWeights& weights) noexcept {
    shelf::weights::Exl3LinearShape shape;
    if (!weights.mtp.present) return shape;
    const Qwen4ExpMtpWeights& mtp = weights.mtp;
    shelf::weights::note_exl3_linear(mtp.fc_hidden, shape);
    shelf::weights::note_exl3_linear(mtp.fc_embedding, shape);
    note_exl3_linear(mtp.mixer, shape);
    note_exl3_linear(mtp.layer.attn_hyper_connection, shape);
    note_exl3_linear(mtp.layer.mlp_hyper_connection, shape);
    note_exl3_linear(mtp.layer.attention, shape);
    note_exl3_linear(mtp.layer.mlp, shape);
    note_exl3_linear(mtp.layer.ple, shape);
    const shelf::weights::MatrixWeight& head = weights.lm_head;
    if (head.encoding == shelf::weights::MatrixEncoding::Exl3) {
        const std::size_t head_k = head.k_padded > head.cols ? head.k_padded : head.cols;
        if (head_k > shape.max_k) shape.max_k = head_k;
        const std::size_t head_accum = head.rows * sizeof(float) * 32u;
        if (head_accum > shape.max_int8_decode_accum_bytes)
            shape.max_int8_decode_accum_bytes = head_accum;
    }
    return shape;
}

struct Qwen4ExpLoadOptions {
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    // Routed expert IDs follow the expert-parallel rank. The backbone, shared expert and
    // each local expert's matrix follow tp_size / tp_rank. ep_size == 0 means EP=1.
    std::uint32_t ep_size = 0;
    std::uint32_t ep_rank = 0;
    bool ep_enabled = false;
    // The backbone's TP geometry when it differs from the routed-expert TP. EP>1 with
    // TP==1 uses the EP ranks as the backbone's tensor group, so the trunk is not
    // replicated. Zero means "same as tp_size / tp_rank".
    std::uint32_t backbone_tp_size = 0;
    std::uint32_t backbone_tp_rank = 0;
    bool verify_quantized_payload_crc = false;
    // Text-only is the first target. MTP and Vision are not loaded by this path.
    bool load_mtp = false;
    // Load the MTP block on every rank and split its attention / expert planes. When
    // false only rank 0 loads the block and it stays whole.
    bool mtp_split = false;
    bool load_vision = false;
    // Load the embedding table, LM head and MTP block only. Used by the MTP draft-head
    // test and by a rank that does not need the trunk.
    bool mtp_only = false;
    // Routed experts kept in pinned host memory per layer. 0 keeps every expert in VRAM.
    // The remaining local experts stay in VRAM and are addressed by the slot map.
    std::uint32_t moe_host_experts = 0;
    // Quantize the BF16 lm head to INT8 W32 while loading. Disabled by SHELF_LM_HEAD_INT8=0.
    bool quantize_lm_head_int8_w32 = true;
    // A pipeline stage loads only its own layer range. Embedding and final mixer / LM head
    // follow the first / last rule. A stage combined with an expert-axis rank is not wired.
    shelf::qwen_hybrid::PipelineStage stage;
};

// Logical checkpoint names this rank asks the HF reader for. Exposed for tests and for
// the memory plan; the list excludes the PLE n-gram table itself, which is not in the
// safetensors index.
std::vector<std::string> qwen4exp_rank_logical_names(
    const Qwen4ExpTextConfig& config, std::uint32_t tp_size, std::uint32_t tp_rank,
    bool load_mtp = false, bool mtp_only = false,
    const shelf::qwen_hybrid::PipelineStage& stage = {},
    std::uint32_t ep_size = 0u, std::uint32_t ep_rank = 0u,
    bool mtp_split = false);

// Loads one tensor-parallel rank of a Qwen4Exp checkpoint. Routed expert IDs split on the
// expert axis, and each local expert's matrix splits on tp_size / tp_rank (gate/up on N,
// down on K). The shared expert, router, hyper-connections and PLE small weights stay
// replicated; attention / GDN and the embedding / LM head follow the tensor partition plan.
// EXL3 INT8 requires the local N and K to stay 128-aligned. The PLE table is not loaded
// here.
Result<Qwen4ExpRankWeights> load_qwen4exp_rank(const std::string& model_dir,
                                               gpu::GpuArena& arena,
                                               hipStream_t stream,
                                               const Qwen4ExpLoadOptions& options = {});

// Shape and encoding invariants of one loaded rank.
Status validate_qwen4exp_rank(const Qwen4ExpRankWeights& weights,
                              const Qwen4ExpTextConfig& config,
                              std::uint32_t tp_size,
                              std::uint32_t tp_rank,
                              std::uint32_t ep_size = 0u,
                              std::uint32_t ep_rank = 0u,
                              std::uint32_t expert_tp_size = 0u);

Status validate_qwen4exp_stage(const Qwen4ExpRankWeights& weights,
                               const Qwen4ExpTextConfig& config,
                               const shelf::qwen_hybrid::PipelineStage& stage);

}  // namespace qwen4exp
}  // namespace shelf
