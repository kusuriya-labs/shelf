#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/memory/tensor.h>
#include <shelf/core/memory/types.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/dflash2/weights.h>
#include <shelf/weights/matrix_weight.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ps::qwen35::dflash2 {

struct DFlash2ContextState;

struct DFlash2ExecutorConfig {
    uint32_t max_rows = 0;
    uint32_t max_batch_rows = 0;
    uint32_t max_context_rows = 0;
    const ps::weights::MatrixWeight* target_lm_head = nullptr;
    const ps::weights::MatrixWeight* target_embed_tokens = nullptr;
};

struct DFlash2Executor {
    const DFlash2Config* config = nullptr;
    const DFlash2Weights* weights = nullptr;
    uint32_t max_rows = 0;
    uint32_t max_context_rows = 0;

    gpu::Tensor target_concat;
    gpu::Tensor target_fc;
    gpu::Tensor target_feature;

    gpu::Tensor conv_dynamic;
    gpu::Tensor conv_output;

    gpu::Tensor layer_input_norm;
    gpu::Tensor attention_prepared;
    gpu::Tensor q_raw;
    gpu::Tensor q_norm;
    gpu::Tensor q_rope;
    gpu::Tensor k_noise_raw;
    gpu::Tensor k_noise_norm;
    gpu::Tensor k_noise_rope;
    gpu::Tensor v_noise;
    gpu::Tensor attention_context;
    gpu::Tensor attention_o;
    gpu::Tensor attention_finished;
    gpu::Tensor attention_residual;
    gpu::Tensor post_attention_norm;
    gpu::Tensor mlp_prepared;
    gpu::Tensor mlp_gate;
    gpu::Tensor mlp_up;
    gpu::Tensor mlp_swiglu;
    gpu::Tensor mlp_down;
    gpu::Tensor mlp_finished;

    gpu::Tensor k_ctx_raw;
    gpu::Tensor k_ctx_norm;
    gpu::Tensor k_ctx_rope;
    gpu::Tensor v_ctx;

    gpu::Tensor backbone_hidden_a;
    gpu::Tensor backbone_hidden_b;
    gpu::Tensor backbone_output;

    const ps::weights::MatrixWeight* target_lm_head = nullptr;
    const ps::weights::MatrixWeight* target_embed_tokens = nullptr;
    uint32_t draft_rows = 0;

    gpu::Tensor noise_token_ids;
    gpu::Tensor noise_embedding;

    gpu::Tensor selector_hidden_proj;
    gpu::Tensor proposal_logits;
    gpu::Tensor proposal_topk_ids;
    gpu::Tensor proposal_topk_logits;
    gpu::Tensor proposal_tokens;
    gpu::Tensor topk_scratch_ids;
    gpu::Tensor topk_scratch_logits;
    gpu::Tensor activation_codes;
    gpu::Tensor activation_scales;

    gpu::Tensor conv_row_has_prev;
    gpu::Tensor batch_descriptors;

    bool initialized = false;
};

Result<DFlash2Executor> create_dflash2_executor(
    const DFlash2Config& config,
    const DFlash2Weights& weights,
    gpu::GpuArena& arena,
    const DFlash2ExecutorConfig& exec_config);

Status dflash2_executor_shutdown(DFlash2Executor& executor) noexcept;

Status dflash2_linear(
    const ps::weights::MatrixWeight& weight,
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    const char* tag,
    hipStream_t stream,
    bool allow_exact_rows = true);

Status dflash2_project_target_features(
    DFlash2Executor& executor,
    const bf16_t* const* target_taps,
    uint32_t tap_count,
    uint32_t rows,
    hipStream_t stream);

Status dflash2_grouped_conv_prepare(
    DFlash2Executor& executor,
    const bf16_t* hidden,
    const DFlash2LayerWeights& layer,
    bool attention_conv,
    uint32_t rows,
    bf16_t* output,
    hipStream_t stream);

Status dflash2_grouped_conv_finish(
    DFlash2Executor& executor,
    const bf16_t* hidden,
    const DFlash2LayerWeights& layer,
    bool attention_conv,
    uint32_t rows,
    bf16_t* output,
    hipStream_t stream);

Status dflash2_forward_layer_stateless(
    DFlash2Executor& executor,
    uint32_t layer_index,
    const bf16_t* hidden_in,
    const bf16_t* target_feature,
    uint32_t context_rows,
    uint32_t context_position_start,
    uint32_t block_position_start,
    uint32_t block_rows,
    bf16_t* output,
    hipStream_t stream);

Status dflash2_forward_backbone_stateless(
    DFlash2Executor& executor,
    const bf16_t* noise_embedding,
    const bf16_t* target_feature,
    uint32_t context_rows,
    uint32_t context_position_start,
    uint32_t block_position_start,
    uint32_t block_rows,
    bf16_t* output,
    hipStream_t stream);

Status dflash2_compute_target_logits(
    DFlash2Executor& executor,
    const bf16_t* hidden,
    uint32_t rows,
    float* logits,
    uint32_t logits_row_stride,
    hipStream_t stream);

Status dflash2_run_candidate_selector(
    DFlash2Executor& executor,
    const bf16_t* hidden_proj,
    const int32_t* topk_ids,
    const float* topk_logits,
    uint32_t draft_rows,
    int32_t anchor_token,
    int32_t* output_draft_tokens,
    hipStream_t stream);

Status dflash2_select_draft_tokens(
    DFlash2Executor& executor,
    const bf16_t* final_hidden,
    uint32_t block_rows,
    int32_t anchor_token,
    int32_t* output_draft_tokens,
    hipStream_t stream);

Status dflash2_forward_and_select_stateless(
    DFlash2Executor& executor,
    const bf16_t* noise_embedding,
    const bf16_t* target_feature,
    uint32_t context_rows,
    uint32_t context_position_start,
    uint32_t block_position_start,
    uint32_t block_rows,
    int32_t anchor_token,
    int32_t* output_draft_tokens,
    hipStream_t stream);

Status dflash2_forward_layer_cached(
    DFlash2Executor& executor,
    const DFlash2ContextState& context,
    uint32_t layer_index,
    const bf16_t* hidden_in,
    uint32_t block_position_start,
    uint32_t block_rows,
    bf16_t* output,
    hipStream_t stream);

Status dflash2_forward_backbone_cached(
    DFlash2Executor& executor,
    const DFlash2ContextState& context,
    const bf16_t* noise_embedding,
    uint32_t block_position_start,
    uint32_t block_rows,
    bf16_t* output,
    hipStream_t stream);

Status dflash2_forward_and_select_cached(
    DFlash2Executor& executor,
    const DFlash2ContextState& context,
    const bf16_t* noise_embedding,
    uint32_t block_position_start,
    uint32_t block_rows,
    int32_t anchor_token,
    int32_t* output_draft_tokens,
    hipStream_t stream);

Status dflash2_prepare_noise_embedding(
    DFlash2Executor& executor,
    int32_t anchor_token,
    uint32_t block_rows,
    hipStream_t stream);

Status dflash2_propose_cached(
    DFlash2Executor& executor,
    const DFlash2ContextState& context,
    int32_t anchor_token,
    uint32_t num_drafts,
    int32_t* output_drafts,
    hipStream_t stream);

struct DFlash2ProposeRequest {
    const DFlash2ContextState* context = nullptr;
    int32_t anchor_token = 0;
    uint32_t num_drafts = 0u;
};

Status dflash2_propose_batch(
    DFlash2Executor& executor,
    const DFlash2ProposeRequest* requests,
    int32_t* const* output_drafts,
    uint32_t request_count,
    hipStream_t stream);

Status dflash2_append_target_taps(
    DFlash2Executor& executor,
    DFlash2ContextState& context,
    const bf16_t* const* target_taps,
    uint32_t tap_count,
    uint32_t rows,
    uint32_t position_start,
    hipStream_t stream);

}
