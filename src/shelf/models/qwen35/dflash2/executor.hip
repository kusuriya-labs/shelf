#include <shelf/models/qwen35/dflash2/executor.h>
#include <shelf/quantization/quant_contract.h>
#include <shelf/models/qwen35/dflash2/context_state.h>
#include <shelf/models/qwen35/kernels/dflash2/attention.h>
#include <shelf/models/qwen35/kernels/dflash2/candidate_selector.h>
#include <shelf/models/qwen35/kernels/dflash2/feature_concat.h>
#include <shelf/models/qwen35/kernels/dflash2/grouped_dynamic_conv.h>
#include <shelf/models/qwen35/kernels/dflash2/kv_ring.h>
#include <shelf/models/qwen35/kernels/dflash2/noise_input.h>
#include <shelf/models/qwen35/kernels/dflash2/rmsnorm.h>
#include <shelf/models/qwen35/kernels/dflash2/rope.h>
#include <shelf/models/qwen35/kernels/dflash2/swiglu.h>
#include <shelf/models/qwen35/kernels/dflash2/topk.h>
#include <shelf/kernels/optimized/activation_quantize.h>
#include <shelf/kernels/optimized/elementwise.h>
#include <shelf/kernels/optimized/embedding.h>
#include <shelf/kernels/optimized/linear/bf16.h>
#include <shelf/kernels/optimized/linear/fp8.h>
#include <shelf/kernels/optimized/linear/exl3.h>
#include <shelf/kernels/optimized/linear/exl3_int8.h>
#include <shelf/runtime/program/int8_activation_workspace.h>
#include <shelf/models/qwen35/runtime/selector_types.h>
#include <cstdint>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace shelf::qwen35::dflash2 {

namespace {

template <typename T>
Status allocate_contiguous_tensor(
    gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    gpu::Tensor& out) {
    if (shape.empty()) {
        return Status::invalid_argument("tensor shape must be non-empty", __FILE__, __LINE__);
    }
    std::size_t count = 1;
    for (std::size_t d : shape) {
        if (d == 0) {
            return Status::invalid_argument("tensor shape has a zero dim", __FILE__, __LINE__);
        }
        if (std::numeric_limits<std::size_t>::max() / count < d) {
            return Status::invalid_argument("tensor element count overflow", __FILE__, __LINE__);
        }
        count *= d;
    }
    if (std::numeric_limits<std::size_t>::max() / count < sizeof(T)) {
        return Status::invalid_argument("tensor byte count overflow", __FILE__, __LINE__);
    }
    auto view = arena.allocate_aligned(count * sizeof(T), 128);
    if (!view.ok()) return view.status();
    auto backing = view.release();
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    auto result = gpu::Tensor::create<T>(backing, shape, strides);
    if (!result.ok()) return result.status();
    out = result.release();
    return Status::make_ok();
}

std::optional<shelf::kernel::Bf16GemmConfig> select_bf16_gemm(
    const shelf::weights::MatrixWeight& weight,
    uint32_t rows,
    uint32_t input_row_stride,
    const void* weight_ptr,
    const void* input_ptr,
    bool allow_exact_rows = true) {
    runtime::Bf16GemmSelectorInput lin;
    lin.rows = rows;
    lin.out_features = weight.rows;
    lin.k = weight.cols;
    lin.input_row_stride = input_row_stride;
    lin.weight_aligned16 = (reinterpret_cast<uintptr_t>(weight_ptr) & 15u) == 0u;
    lin.input_aligned16 = (reinterpret_cast<uintptr_t>(input_ptr) & 15u) == 0u;
    lin.allow_exact_rows = allow_exact_rows;
    lin.allow_dot2_rows = allow_exact_rows;
    return runtime::select_bf16_gemm_config(lin);
}

const gpu::Tensor& conv_base_kernel(
    const DFlash2LayerWeights& layer,
    bool attention_conv) {
    return attention_conv ? layer.attention_conv_base_kernel : layer.mlp_conv_base_kernel;
}

bool topk_reference_enabled() {
    const char* env = std::getenv("SHELF_DFLASH2_TOPK_REFERENCE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool attention_reference_enabled() {
    const char* env = std::getenv("SHELF_DFLASH2_ATTENTION_REFERENCE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

const shelf::kernel::DFlash2TopKPlan& topk_plan() {
    static const shelf::kernel::DFlash2TopKPlan plan = shelf::kernel::dflash2_topk_default_plan();
    return plan;
}

const shelf::weights::MatrixWeight& conv_kernel_projection(
    const DFlash2LayerWeights& layer,
    bool attention_conv) {
    return attention_conv ? layer.attention_conv_kernel_projection
                          : layer.mlp_conv_kernel_projection;
}

Status check_conv_args(
    const DFlash2Executor& executor,
    const bf16_t* hidden,
    const bf16_t* output,
    uint32_t rows) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (hidden == nullptr || output == nullptr) {
        return Status::invalid_argument("dflash2 conv buffers are null", __FILE__, __LINE__);
    }
    if (rows > executor.max_rows) {
        return Status::invalid_argument("dflash2 conv rows exceed max_rows", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status conv_apply_dynamic_conv(
    DFlash2Executor& executor,
    const DFlash2LayerWeights& layer,
    bool attention_conv,
    const bf16_t* conv_input,
    uint32_t rows,
    bf16_t* output,
    uint32_t phase,
    const uint32_t* row_has_prev,
    const char* tag,
    hipStream_t stream) {
    const uint32_t hidden_size = static_cast<uint32_t>(executor.config->hidden_size);
    const gpu::Tensor& base = conv_base_kernel(layer, attention_conv);
    const hipError_t err = shelf::kernel::launch_dflash2_grouped_dynamic_conv_rows(
        conv_input, executor.conv_dynamic.data<bf16_t>(), base.data<bf16_t>(), output, rows,
        hidden_size, static_cast<uint32_t>(executor.config->conv_group_size), phase,
        row_has_prev, stream);
    if (err != hipSuccess) {
        return Status::hip_error((std::string("dflash2 ") + tag + " launch").c_str(),
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status executor_linear(DFlash2Executor& executor, const shelf::weights::MatrixWeight& weight,
                       const bf16_t* input, uint32_t input_row_stride, bf16_t* output,
                       uint32_t output_row_stride, uint32_t rows, const char* tag,
                       hipStream_t stream, bool allow_exact_rows = true);

Status grouped_conv_prepare_rows(
    DFlash2Executor& executor,
    const bf16_t* hidden,
    const DFlash2LayerWeights& layer,
    bool attention_conv,
    uint32_t rows,
    bf16_t* output,
    const uint32_t* row_has_prev,
    hipStream_t stream) {
    Status st = check_conv_args(executor, hidden, output, rows);
    if (!st.ok()) return st;
    if (rows == 0u) {
        return Status::make_ok();
    }

    const uint32_t hidden_size = static_cast<uint32_t>(executor.config->hidden_size);
    const shelf::weights::MatrixWeight& projection = conv_kernel_projection(layer, attention_conv);

    Status gemm_st = executor_linear(executor, projection, hidden, hidden_size,
                                    executor.conv_dynamic.data<bf16_t>(),
                                    static_cast<uint32_t>(executor.conv_dynamic.dim(1)), rows,
                                    "conv kernel_projection", stream);
    if (!gemm_st.ok()) return gemm_st;

    return conv_apply_dynamic_conv(executor, layer, attention_conv, hidden, rows, output, 0u,
                                   row_has_prev, "conv prepare", stream);
}

Status grouped_conv_finish_rows(
    DFlash2Executor& executor,
    const bf16_t* hidden,
    const DFlash2LayerWeights& layer,
    bool attention_conv,
    uint32_t rows,
    bf16_t* output,
    const uint32_t* row_has_prev,
    hipStream_t stream) {
    Status st = check_conv_args(executor, hidden, output, rows);
    if (!st.ok()) return st;
    if (rows == 0u) {
        return Status::make_ok();
    }

    const uint32_t hidden_size = static_cast<uint32_t>(executor.config->hidden_size);
    const gpu::Tensor& base = conv_base_kernel(layer, attention_conv);
    const hipError_t err = shelf::kernel::launch_dflash2_grouped_dynamic_conv_rows(
        hidden, executor.conv_dynamic.data<bf16_t>(), base.data<bf16_t>(), output, rows,
        hidden_size, static_cast<uint32_t>(executor.config->conv_group_size), 1u, row_has_prev,
        stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 conv finish launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status layer_gemm_bf16(
    const shelf::weights::MatrixWeight& weight,
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    const char* tag,
    hipStream_t stream,
    bool allow_exact_rows = true) {
    const auto config = select_bf16_gemm(weight, rows, input_row_stride,
                                         weight.bf16.data<void>(), input, allow_exact_rows);
    if (!config) {
        return Status::unsupported(
            (std::string("dflash2 ") + tag + " has no optimized bf16 gemm geometry").c_str(),
            __FILE__, __LINE__);
    }
    const hipError_t err = shelf::kernel::launch_gemm_bf16(
        *config, weight.bf16.data<bf16_t>(), input, output,
        shelf::kernel::Bf16GemmOutputDType::BF16, rows, weight.rows, weight.cols,
        input_row_stride, output_row_stride, stream);
    if (err != hipSuccess) {
        return Status::hip_error((std::string("dflash2 ") + tag + " launch").c_str(),
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}


Status executor_linear(DFlash2Executor& executor, const shelf::weights::MatrixWeight& weight,
                       const bf16_t* input, uint32_t input_row_stride, bf16_t* output,
                       uint32_t output_row_stride, uint32_t rows, const char* tag,
                       hipStream_t stream, bool allow_exact_rows) {
    if (weight.encoding == shelf::weights::MatrixEncoding::Bf16)
        return layer_gemm_bf16(weight, input, input_row_stride, output, output_row_stride,
                               rows, tag, stream, allow_exact_rows);
    return shelf::drafter::quantized_linear(weight, input, input_row_stride, output,
                                            output_row_stride, rows, executor.linear_workspace, stream);
}




}  // namespace

Status dflash2_linear(
    const shelf::weights::MatrixWeight& weight,
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    const char* tag,
    hipStream_t stream,
    bool allow_exact_rows) {
    if (weight.encoding != shelf::weights::MatrixEncoding::Bf16) {
        return Status::unsupported(
            (std::string("dflash2 ") + tag + " weight encoding unsupported").c_str(),
            __FILE__, __LINE__);
    }
    return layer_gemm_bf16(weight, input, input_row_stride, output, output_row_stride, rows,
                           tag, stream, allow_exact_rows);
}

Result<DFlash2Executor> create_dflash2_executor(
    const DFlash2Config& config,
    const DFlash2Weights& weights,
    gpu::GpuArena& arena,
    const DFlash2ExecutorConfig& exec_config) {
    if (exec_config.max_rows == 0u) {
        return Status::invalid_argument("dflash2 max_rows must be non-zero", __FILE__, __LINE__);
    }
    if (!weights.present) {
        return Status::invalid_argument("dflash2 weights are not loaded", __FILE__, __LINE__);
    }
    if (weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument("dflash2 layer count mismatch", __FILE__, __LINE__);
    }
    if (config.num_target_layer_ids != shelf::kernel::kDFlash2TargetTaps) {
        return Status::invalid_argument("dflash2 target tap count mismatch", __FILE__, __LINE__);
    }
    if (config.conv_kernel_size != shelf::kernel::kDFlash2ConvKernelSize) {
        return Status::invalid_argument("dflash2 conv kernel size mismatch", __FILE__, __LINE__);
    }

    DFlash2Executor ex;
    ex.config = &config;
    ex.weights = &weights;
    ex.target_lm_head = exec_config.target_lm_head;
    ex.max_rows = std::max(exec_config.max_rows, exec_config.max_batch_rows);
    ex.max_context_rows = exec_config.max_context_rows != 0u ? exec_config.max_context_rows
                                                             : ex.max_rows;

    const std::size_t rows = ex.max_rows;
    const std::size_t ctx_rows = ex.max_context_rows;
    const std::size_t hidden = config.hidden_size;
    const std::size_t q_features = config.attention_q_rows();
    const std::size_t kv_features = config.attention_kv_rows();
    const std::size_t intermediate = config.intermediate_size;

    auto alloc = [&](const std::vector<std::size_t>& shape, gpu::Tensor& dst) -> Status {
        return allocate_contiguous_tensor<bf16_t>(arena, shape, dst);
    };

    Status st = alloc({rows, config.tap_feature_size()}, ex.target_concat);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.target_fc);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.target_feature);
    if (!st.ok()) return st;
    st = alloc({rows, config.conv_projection_rows()}, ex.conv_dynamic);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.conv_output);
    if (!st.ok()) return st;

    st = alloc({rows, hidden}, ex.layer_input_norm);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.attention_prepared);
    if (!st.ok()) return st;
    st = alloc({rows, q_features}, ex.q_raw);
    if (!st.ok()) return st;
    st = alloc({rows, q_features}, ex.q_norm);
    if (!st.ok()) return st;
    st = alloc({rows, q_features}, ex.q_rope);
    if (!st.ok()) return st;
    st = alloc({rows, kv_features}, ex.k_noise_raw);
    if (!st.ok()) return st;
    st = alloc({rows, kv_features}, ex.k_noise_norm);
    if (!st.ok()) return st;
    st = alloc({rows, kv_features}, ex.k_noise_rope);
    if (!st.ok()) return st;
    st = alloc({rows, kv_features}, ex.v_noise);
    if (!st.ok()) return st;
    st = alloc({rows, q_features}, ex.attention_context);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.attention_o);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.attention_finished);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.attention_residual);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.post_attention_norm);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.mlp_prepared);
    if (!st.ok()) return st;
    st = alloc({rows, intermediate}, ex.mlp_gate);
    if (!st.ok()) return st;
    st = alloc({rows, intermediate}, ex.mlp_up);
    if (!st.ok()) return st;
    st = alloc({rows, intermediate}, ex.mlp_swiglu);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.mlp_down);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.mlp_finished);
    if (!st.ok()) return st;

    st = alloc({ctx_rows, kv_features}, ex.k_ctx_raw);
    if (!st.ok()) return st;
    st = alloc({ctx_rows, kv_features}, ex.k_ctx_norm);
    if (!st.ok()) return st;
    st = alloc({ctx_rows, kv_features}, ex.k_ctx_rope);
    if (!st.ok()) return st;
    st = alloc({ctx_rows, kv_features}, ex.v_ctx);
    if (!st.ok()) return st;

    st = alloc({rows, hidden}, ex.backbone_hidden_a);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.backbone_hidden_b);
    if (!st.ok()) return st;
    st = alloc({rows, hidden}, ex.backbone_output);
    if (!st.ok()) return st;

    ex.draft_rows = config.block_size > 0u ? config.block_size - 1u : 0u;
    const std::size_t draft = ex.max_rows;
    if (ex.draft_rows == 0u) {
        return Status::invalid_argument("dflash2 block_size must be > 1", __FILE__, __LINE__);
    }
    if (draft < ex.draft_rows) {
        return Status::invalid_argument("dflash2 max_rows is below draft_rows", __FILE__,
                                        __LINE__);
    }

    st = alloc({draft, config.selector_rank}, ex.selector_hidden_proj);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<float>(arena, {draft, config.vocab_size},
                                           ex.proposal_logits);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<int32_t>(arena, {draft, config.selector_top_k},
                                             ex.proposal_topk_ids);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<float>(arena, {draft, config.selector_top_k},
                                           ex.proposal_topk_logits);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<int32_t>(arena, {draft}, ex.proposal_tokens);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<int32_t>(
        arena, {draft, shelf::kernel::kDflash2TopKScratchPerRow}, ex.topk_scratch_ids);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<float>(
        arena, {draft, shelf::kernel::kDflash2TopKScratchPerRow}, ex.topk_scratch_logits);
    if (!st.ok()) return st;

    uint32_t act_k = static_cast<uint32_t>(hidden);
    if (exec_config.target_lm_head != nullptr) {
        act_k = std::max(act_k, exec_config.target_lm_head->k_padded);
    }
    const auto act_layout =
        shelf::runtime::Int8ActivationWorkspaceLayout::make(act_k, static_cast<uint32_t>(draft));
    st = allocate_contiguous_tensor<int8_t>(
        arena, {act_layout.codes_bytes()}, ex.activation_codes);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<float>(
        arena, {act_layout.scales_bytes() / sizeof(float)}, ex.activation_scales);
    if (!st.ok()) return st;
    if (exec_config.target_lm_head != nullptr &&
        exec_config.target_lm_head->quant_spec == quantization::QuantSpecId::EXL3_MUL24_INT8_V1) {
        st = allocate_contiguous_tensor<float>(arena, {draft, config.vocab_size},
                                                ex.exl3_lm_head_accum);
        if (!st.ok()) return st;
    }

    bool quantized_linear = weights.fc.encoding != shelf::weights::MatrixEncoding::Bf16;
    for (const auto& layer : weights.layers) {
        for (const auto* w : {&layer.attn_q_proj, &layer.attn_k_proj, &layer.attn_v_proj,
                              &layer.attn_o_proj, &layer.mlp_gate_proj, &layer.mlp_up_proj,
                              &layer.mlp_down_proj})
            quantized_linear |= w->encoding != shelf::weights::MatrixEncoding::Bf16;
    }
    if (quantized_linear) {
        const auto workspace_rows = std::max(rows, ctx_rows);
        const auto max_k = std::max({hidden, intermediate, config.tap_feature_size()});
        const auto max_n = std::max({hidden, intermediate, q_features, kv_features});
        const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(
            static_cast<uint32_t>(max_k), static_cast<uint32_t>(workspace_rows));
        st = allocate_contiguous_tensor<int8_t>(arena, {layout.codes_bytes()}, ex.linear_workspace.codes);
        if (!st.ok()) return st;
        st = allocate_contiguous_tensor<float>(arena, {layout.scales_bytes() / sizeof(float)}, ex.linear_workspace.scales);
        if (!st.ok()) return st;
        st = allocate_contiguous_tensor<float>(arena, {std::max(rows * 4u, ctx_rows) * max_n}, ex.linear_workspace.accum);
        if (!st.ok()) return st;
    }

    st = allocate_contiguous_tensor<uint32_t>(arena, {rows}, ex.conv_row_has_prev);
    if (!st.ok()) return st;
    st = allocate_contiguous_tensor<uint8_t>(
        arena, {rows * sizeof(shelf::kernel::DFlash2RequestDescriptor)}, ex.batch_descriptors);
    if (!st.ok()) return st;

    ex.target_embed_tokens = exec_config.target_embed_tokens;
    if (exec_config.target_embed_tokens != nullptr) {
        if (exec_config.target_embed_tokens->rows != config.vocab_size ||
            exec_config.target_embed_tokens->cols != config.hidden_size) {
            return Status::invalid_argument("dflash2 target embed_tokens shape mismatch",
                                            __FILE__, __LINE__);
        }
        st = allocate_contiguous_tensor<int32_t>(arena, {rows}, ex.noise_token_ids);
        if (!st.ok()) return st;
        st = alloc({rows, hidden}, ex.noise_embedding);
        if (!st.ok()) return st;
    }

    ex.initialized = true;
    return ex;
}

Status dflash2_executor_shutdown(DFlash2Executor& executor) noexcept {
    executor.target_concat = gpu::Tensor{};
    executor.target_fc = gpu::Tensor{};
    executor.target_feature = gpu::Tensor{};
    executor.conv_dynamic = gpu::Tensor{};
    executor.conv_output = gpu::Tensor{};
    executor.layer_input_norm = gpu::Tensor{};
    executor.attention_prepared = gpu::Tensor{};
    executor.q_raw = gpu::Tensor{};
    executor.q_norm = gpu::Tensor{};
    executor.q_rope = gpu::Tensor{};
    executor.k_noise_raw = gpu::Tensor{};
    executor.k_noise_norm = gpu::Tensor{};
    executor.k_noise_rope = gpu::Tensor{};
    executor.v_noise = gpu::Tensor{};
    executor.attention_context = gpu::Tensor{};
    executor.attention_o = gpu::Tensor{};
    executor.attention_finished = gpu::Tensor{};
    executor.attention_residual = gpu::Tensor{};
    executor.post_attention_norm = gpu::Tensor{};
    executor.mlp_prepared = gpu::Tensor{};
    executor.mlp_gate = gpu::Tensor{};
    executor.mlp_up = gpu::Tensor{};
    executor.mlp_swiglu = gpu::Tensor{};
    executor.mlp_down = gpu::Tensor{};
    executor.mlp_finished = gpu::Tensor{};
    executor.k_ctx_raw = gpu::Tensor{};
    executor.k_ctx_norm = gpu::Tensor{};
    executor.k_ctx_rope = gpu::Tensor{};
    executor.v_ctx = gpu::Tensor{};
    executor.backbone_hidden_a = gpu::Tensor{};
    executor.backbone_hidden_b = gpu::Tensor{};
    executor.backbone_output = gpu::Tensor{};
    executor.target_lm_head = nullptr;
    executor.target_embed_tokens = nullptr;
    executor.draft_rows = 0;
    executor.noise_token_ids = gpu::Tensor{};
    executor.noise_embedding = gpu::Tensor{};
    executor.selector_hidden_proj = gpu::Tensor{};
    executor.proposal_logits = gpu::Tensor{};
    executor.proposal_topk_ids = gpu::Tensor{};
    executor.proposal_topk_logits = gpu::Tensor{};
    executor.proposal_tokens = gpu::Tensor{};
    executor.topk_scratch_ids = gpu::Tensor{};
    executor.topk_scratch_logits = gpu::Tensor{};
    executor.linear_workspace = shelf::drafter::LinearWorkspace{};
    executor.activation_codes = gpu::Tensor{};
    executor.activation_scales = gpu::Tensor{};
    executor.exl3_lm_head_accum = gpu::Tensor{};
    executor.conv_row_has_prev = gpu::Tensor{};
    executor.batch_descriptors = gpu::Tensor{};
    executor.max_rows = 0;
    executor.max_context_rows = 0;
    executor.weights = nullptr;
    executor.config = nullptr;
    executor.initialized = false;
    return Status::make_ok();
}

Status dflash2_project_target_features(
    DFlash2Executor& executor,
    const bf16_t* const* target_taps,
    uint32_t tap_count,
    uint32_t rows,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (target_taps == nullptr) {
        return Status::invalid_argument("dflash2 target taps are null", __FILE__, __LINE__);
    }
    if (tap_count != shelf::kernel::kDFlash2TargetTaps) {
        return Status::invalid_argument("dflash2 target tap count mismatch", __FILE__, __LINE__);
    }
    if (rows > executor.max_rows) {
        return Status::invalid_argument("dflash2 rows exceed max_rows", __FILE__, __LINE__);
    }
    if (rows == 0u) {
        return Status::make_ok();
    }
    for (uint32_t t = 0; t < tap_count; ++t) {
        if (target_taps[t] == nullptr) {
            return Status::invalid_argument("dflash2 target tap is null", __FILE__, __LINE__);
        }
    }

    const uint32_t hidden = static_cast<uint32_t>(executor.config->hidden_size);
    const uint32_t concat_stride = static_cast<uint32_t>(executor.target_concat.dim(1));

    hipError_t err = shelf::kernel::launch_dflash2_feature_concat(
        target_taps[0], target_taps[1], target_taps[2], target_taps[3], target_taps[4],
        hidden, rows, executor.target_concat.data<bf16_t>(), concat_stride, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 feature concat launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    const shelf::weights::MatrixWeight& fc = executor.weights->fc;
    Status st = executor_linear(executor, fc, executor.target_concat.data<bf16_t>(),
                               concat_stride, executor.target_fc.data<bf16_t>(),
                               static_cast<uint32_t>(executor.target_fc.dim(1)), rows, "fc",
                               stream);
    if (!st.ok()) return st;

    err = shelf::kernel::launch_dflash2_rmsnorm_direct_bf16(
        executor.target_fc.data<bf16_t>(), executor.weights->hidden_norm_weight.data<bf16_t>(),
        executor.target_feature.data<bf16_t>(), rows, hidden, hidden,
        executor.config->rms_norm_eps, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 hidden norm launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status dflash2_grouped_conv_prepare(
    DFlash2Executor& executor,
    const bf16_t* hidden,
    const DFlash2LayerWeights& layer,
    bool attention_conv,
    uint32_t rows,
    bf16_t* output,
    hipStream_t stream) {
    return grouped_conv_prepare_rows(executor, hidden, layer, attention_conv, rows, output,
                                     nullptr, stream);
}

Status dflash2_grouped_conv_finish(
    DFlash2Executor& executor,
    const bf16_t* hidden,
    const DFlash2LayerWeights& layer,
    bool attention_conv,
    uint32_t rows,
    bf16_t* output,
    hipStream_t stream) {
    return grouped_conv_finish_rows(executor, hidden, layer, attention_conv, rows, output,
                                    nullptr, stream);
}

namespace {

Status layer_rmsnorm_direct(
    const bf16_t* input,
    const gpu::Tensor& weight,
    bf16_t* output,
    uint32_t rows,
    uint32_t features,
    uint32_t group_size,
    float eps,
    const char* tag,
    hipStream_t stream) {
    const hipError_t err = shelf::kernel::launch_dflash2_rmsnorm_direct_bf16(
        input, weight.data<bf16_t>(), output, rows, features, group_size, eps, stream);
    if (err != hipSuccess) {
        return Status::hip_error((std::string("dflash2 ") + tag + " launch").c_str(),
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status layer_attention_tail(
    DFlash2Executor& executor,
    const DFlash2LayerWeights& layer,
    const bf16_t* hidden_in,
    uint32_t block_rows,
    const uint32_t* row_has_prev,
    bf16_t* output,
    hipStream_t stream) {
    const DFlash2Config& config = *executor.config;
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t q_features = static_cast<uint32_t>(config.attention_q_rows());
    const uint32_t intermediate = static_cast<uint32_t>(config.intermediate_size);
    const float eps = config.rms_norm_eps;

    Status st = executor_linear(executor, layer.attn_o_proj,
                               executor.attention_context.data<bf16_t>(), q_features,
                               executor.attention_o.data<bf16_t>(), hidden, block_rows,
                               "o_proj", stream);
    if (!st.ok()) return st;

    st = grouped_conv_finish_rows(executor, executor.attention_o.data<bf16_t>(), layer,
                                  true, block_rows,
                                  executor.attention_finished.data<bf16_t>(), row_has_prev,
                                  stream);
    if (!st.ok()) return st;

    hipError_t err = hipSuccess;
    err = shelf::kernel::launch_residual_add_bf16(
        hidden_in, hidden, executor.attention_finished.data<bf16_t>(), hidden,
        executor.attention_residual.data<bf16_t>(), hidden, block_rows, hidden, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 attention residual launch",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }

    st = layer_rmsnorm_direct(
        executor.attention_residual.data<bf16_t>(), layer.post_attention_layernorm_weight,
        executor.post_attention_norm.data<bf16_t>(), block_rows, hidden, hidden, eps,
        "post attention norm", stream);
    if (!st.ok()) return st;

    st = grouped_conv_prepare_rows(executor, executor.post_attention_norm.data<bf16_t>(),
                                   layer, false, block_rows,
                                   executor.mlp_prepared.data<bf16_t>(), row_has_prev, stream);
    if (!st.ok()) return st;

    const bf16_t* mlp_prepared = executor.mlp_prepared.data<bf16_t>();

        st = executor_linear(executor, layer.mlp_gate_proj, mlp_prepared, hidden,
                             executor.mlp_gate.data<bf16_t>(), intermediate, block_rows,
                             "mlp_gate", stream);
        if (!st.ok()) return st;
        st = executor_linear(executor, layer.mlp_up_proj, mlp_prepared, hidden,
                             executor.mlp_up.data<bf16_t>(), intermediate, block_rows, "mlp_up",
                             stream);
        if (!st.ok()) return st;
    

    err = shelf::kernel::launch_dflash2_swiglu_bf16(
        executor.mlp_gate.data<bf16_t>(), intermediate, executor.mlp_up.data<bf16_t>(),
        intermediate, executor.mlp_swiglu.data<bf16_t>(), intermediate, block_rows,
        intermediate, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 mlp swiglu launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    st = executor_linear(executor, layer.mlp_down_proj, executor.mlp_swiglu.data<bf16_t>(),
                        intermediate, executor.mlp_down.data<bf16_t>(), hidden, block_rows,
                        "mlp_down", stream);
    if (!st.ok()) return st;

    st = grouped_conv_finish_rows(executor, executor.mlp_down.data<bf16_t>(), layer, false,
                                  block_rows, executor.mlp_finished.data<bf16_t>(),
                                  row_has_prev, stream);
    if (!st.ok()) return st;

    err = shelf::kernel::launch_residual_add_bf16(
        executor.attention_residual.data<bf16_t>(), hidden,
        executor.mlp_finished.data<bf16_t>(), hidden, output, hidden, block_rows, hidden,
        stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 mlp residual launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Status dflash2_context_append(
    DFlash2Executor& executor,
    DFlash2ContextState& state,
    const bf16_t* target_feature,
    uint32_t rows,
    uint32_t position_start,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (!state.initialized) {
        return Status::invalid_state("dflash2 context state not initialized", __FILE__,
                                     __LINE__);
    }
    if (target_feature == nullptr) {
        return Status::invalid_argument("dflash2 context feature is null", __FILE__, __LINE__);
    }
    if (executor.config == nullptr || executor.weights == nullptr) {
        return Status::invalid_state("dflash2 executor has no weights", __FILE__, __LINE__);
    }
    if (state.num_layers != executor.config->num_hidden_layers ||
        state.kv_features != executor.config->attention_kv_rows() ||
        state.capacity != executor.config->sliding_window) {
        return Status::invalid_argument("dflash2 context state contract mismatch", __FILE__,
                                        __LINE__);
    }
    if (position_start != state.next_position) {
        return Status::invalid_argument("dflash2 context append is not contiguous", __FILE__,
                                        __LINE__);
    }
    if (rows == 0u) {
        return Status::make_ok();
    }

    const DFlash2Config& config = *executor.config;
    const uint32_t capacity = state.capacity;
    const uint32_t skip = rows > capacity ? rows - capacity : 0u;
    const uint32_t effective_rows = rows > capacity ? capacity : rows;
    if (effective_rows > executor.max_context_rows) {
        return Status::invalid_argument("dflash2 context append exceeds max_context_rows",
                                        __FILE__, __LINE__);
    }

    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t kv_features = state.kv_features;
    const uint32_t head_dim = static_cast<uint32_t>(config.head_dim);
    const bf16_t* feature = target_feature + static_cast<std::size_t>(skip) * hidden;
    const uint32_t effective_position = position_start + skip;
    const std::size_t layer_stride =
        static_cast<std::size_t>(capacity) * kv_features;
    bf16_t* k_ring = state.k_ring.data<bf16_t>();
    bf16_t* v_ring = state.v_ring.data<bf16_t>();


    for (uint32_t li = 0; li < state.num_layers; ++li) {
        const DFlash2LayerWeights& layer = executor.weights->layers[li];
        Status st = Status::make_ok();
    
            st = executor_linear(executor, layer.attn_k_proj, feature, hidden,
                                 executor.k_ctx_raw.data<bf16_t>(), kv_features,
                                 effective_rows, "k_ctx_proj", stream, false);
            if (!st.ok()) return st;
            st = executor_linear(executor, layer.attn_v_proj, feature, hidden,
                                 executor.v_ctx.data<bf16_t>(), kv_features, effective_rows,
                                 "v_ctx_proj", stream, false);
            if (!st.ok()) return st;
        
        st = layer_rmsnorm_direct(
            executor.k_ctx_raw.data<bf16_t>(), layer.attn_k_norm_weight,
            executor.k_ctx_norm.data<bf16_t>(), effective_rows, kv_features, head_dim,
            config.rms_norm_eps, "k_ctx_norm", stream);
        if (!st.ok()) return st;
        const hipError_t err = shelf::kernel::launch_dflash2_rope_bf16(
            executor.k_ctx_norm.data<bf16_t>(), kv_features,
            executor.k_ctx_rope.data<bf16_t>(), kv_features, effective_rows, kv_features,
            head_dim, effective_position, config.rope_theta, stream);
        if (err != hipSuccess) {
            return Status::hip_error("dflash2 k_ctx rope launch", hipGetErrorString(err),
                                     __FILE__, __LINE__);
        }
        const hipError_t append_err = shelf::kernel::launch_dflash2_kv_ring_append_bf16(
            executor.k_ctx_rope.data<bf16_t>(), executor.v_ctx.data<bf16_t>(),
            k_ring + static_cast<std::size_t>(li) * layer_stride,
            v_ring + static_cast<std::size_t>(li) * layer_stride, effective_rows, kv_features,
            capacity, effective_position, stream);
        if (append_err != hipSuccess) {
            return Status::hip_error("dflash2 kv ring append launch",
                                     hipGetErrorString(append_err), __FILE__, __LINE__);
        }
    }

    const uint64_t new_length = static_cast<uint64_t>(state.length) + rows;
    state.length = new_length > capacity ? capacity : static_cast<uint32_t>(new_length);
    state.next_position = position_start + rows;
    return Status::make_ok();
}

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
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (hidden_in == nullptr || target_feature == nullptr || output == nullptr) {
        return Status::invalid_argument("dflash2 layer buffers are null", __FILE__, __LINE__);
    }
    if (executor.weights == nullptr || executor.config == nullptr) {
        return Status::invalid_state("dflash2 executor has no weights", __FILE__, __LINE__);
    }
    if (layer_index >= executor.weights->layers.size()) {
        return Status::invalid_argument("dflash2 layer index out of range", __FILE__, __LINE__);
    }
    if (block_rows == 0u || block_rows > executor.max_rows ||
        block_rows > executor.config->block_size) {
        return Status::invalid_argument("dflash2 block rows out of range", __FILE__, __LINE__);
    }
    if (context_rows == 0u || context_rows > executor.max_context_rows) {
        return Status::invalid_argument("dflash2 context rows out of range", __FILE__, __LINE__);
    }
    if (static_cast<uint64_t>(context_position_start) + context_rows !=
        static_cast<uint64_t>(block_position_start)) {
        return Status::invalid_argument(
            "dflash2 requires contiguous context and block positions", __FILE__, __LINE__);
    }

    const DFlash2LayerWeights& layer = executor.weights->layers[layer_index];
    const DFlash2Config& config = *executor.config;
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t q_features = static_cast<uint32_t>(config.attention_q_rows());
    const uint32_t kv_features = static_cast<uint32_t>(config.attention_kv_rows());
    const uint32_t head_dim = static_cast<uint32_t>(config.head_dim);
    const uint32_t q_heads = static_cast<uint32_t>(config.num_attention_heads);
    const uint32_t kv_heads = static_cast<uint32_t>(config.num_key_value_heads);
    const float eps = config.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    Status st = Status::make_ok();
    st = layer_rmsnorm_direct(
        hidden_in, layer.input_layernorm_weight, executor.layer_input_norm.data<bf16_t>(),
        block_rows, hidden, hidden, eps,
        "input norm", stream);
    if (!st.ok()) return st;

    st = dflash2_grouped_conv_prepare(executor, executor.layer_input_norm.data<bf16_t>(),
                                      layer, true, block_rows,
                                      executor.attention_prepared.data<bf16_t>(), stream);
    if (!st.ok()) return st;

    const bf16_t* attention_prepared = executor.attention_prepared.data<bf16_t>();


        st = executor_linear(executor, layer.attn_q_proj, attention_prepared, hidden,
                             executor.q_raw.data<bf16_t>(), q_features, block_rows, "q_proj",
                             stream);
        if (!st.ok()) return st;
        st = executor_linear(executor, layer.attn_k_proj, target_feature, hidden,
                             executor.k_ctx_raw.data<bf16_t>(), kv_features, context_rows,
                             "k_ctx_proj", stream);
        if (!st.ok()) return st;
        st = executor_linear(executor, layer.attn_v_proj, target_feature, hidden,
                             executor.v_ctx.data<bf16_t>(), kv_features, context_rows,
                             "v_ctx_proj", stream);
        if (!st.ok()) return st;
        st = executor_linear(executor, layer.attn_k_proj, attention_prepared, hidden,
                             executor.k_noise_raw.data<bf16_t>(), kv_features, block_rows,
                             "k_noise_proj", stream);
        if (!st.ok()) return st;
        st = executor_linear(executor, layer.attn_v_proj, attention_prepared, hidden,
                             executor.v_noise.data<bf16_t>(), kv_features, block_rows,
                             "v_noise_proj", stream);
        if (!st.ok()) return st;
    

    hipError_t err = hipSuccess;
    st = layer_rmsnorm_direct(executor.q_raw.data<bf16_t>(), layer.attn_q_norm_weight,
                              executor.q_norm.data<bf16_t>(), block_rows, q_features,
                              head_dim, eps, "q_norm", stream);
    if (!st.ok()) return st;
    st = layer_rmsnorm_direct(executor.k_ctx_raw.data<bf16_t>(), layer.attn_k_norm_weight,
                              executor.k_ctx_norm.data<bf16_t>(), context_rows, kv_features,
                              head_dim, eps, "k_ctx_norm", stream);
    if (!st.ok()) return st;
    st = layer_rmsnorm_direct(executor.k_noise_raw.data<bf16_t>(), layer.attn_k_norm_weight,
                              executor.k_noise_norm.data<bf16_t>(), block_rows, kv_features,
                              head_dim, eps, "k_noise_norm", stream);
    if (!st.ok()) return st;

    err = shelf::kernel::launch_dflash2_rope_bf16(
        executor.q_norm.data<bf16_t>(), q_features, executor.q_rope.data<bf16_t>(),
        q_features, block_rows, q_features, head_dim, block_position_start,
        config.rope_theta, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 q rope launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    err = shelf::kernel::launch_dflash2_rope_bf16(
        executor.k_ctx_norm.data<bf16_t>(), kv_features, executor.k_ctx_rope.data<bf16_t>(),
        kv_features, context_rows, kv_features, head_dim, context_position_start,
        config.rope_theta, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 k_ctx rope launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    err = shelf::kernel::launch_dflash2_rope_bf16(
        executor.k_noise_norm.data<bf16_t>(), kv_features,
        executor.k_noise_rope.data<bf16_t>(), kv_features, block_rows, kv_features, head_dim,
        block_position_start, config.rope_theta, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 k_noise rope launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    err = shelf::kernel::launch_dflash2_attention_bf16(
        executor.q_rope.data<bf16_t>(), q_features, executor.k_ctx_rope.data<bf16_t>(),
        executor.v_ctx.data<bf16_t>(), context_rows, kv_features,
        executor.k_noise_rope.data<bf16_t>(), executor.v_noise.data<bf16_t>(), block_rows,
        kv_features, executor.attention_context.data<bf16_t>(), q_features, q_heads, kv_heads,
        head_dim, context_position_start, block_position_start,
        static_cast<uint32_t>(config.sliding_window), scale, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 attention launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }

    return layer_attention_tail(executor, layer, hidden_in, block_rows, nullptr, output,
                                stream);
}

Status dflash2_forward_backbone_stateless(
    DFlash2Executor& executor,
    const bf16_t* noise_embedding,
    const bf16_t* target_feature,
    uint32_t context_rows,
    uint32_t context_position_start,
    uint32_t block_position_start,
    uint32_t block_rows,
    bf16_t* output,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (noise_embedding == nullptr || target_feature == nullptr || output == nullptr) {
        return Status::invalid_argument("dflash2 backbone buffers are null", __FILE__, __LINE__);
    }
    if (executor.weights == nullptr || executor.config == nullptr) {
        return Status::invalid_state("dflash2 executor has no weights", __FILE__, __LINE__);
    }
    const uint32_t layers = static_cast<uint32_t>(executor.weights->layers.size());
    if (layers == 0u || layers != executor.config->num_hidden_layers) {
        return Status::invalid_state("dflash2 backbone layer count mismatch", __FILE__,
                                     __LINE__);
    }
    if (block_rows == 0u || block_rows > executor.max_rows) {
        return Status::invalid_argument("dflash2 backbone block rows out of range", __FILE__,
                                        __LINE__);
    }
    if (block_rows > executor.config->block_size) {
        return Status::invalid_argument("dflash2 backbone block rows exceed block_size",
                                        __FILE__, __LINE__);
    }
    if (context_rows == 0u || context_rows > executor.max_context_rows) {
        return Status::invalid_argument("dflash2 backbone context rows out of range", __FILE__,
                                        __LINE__);
    }
    if (static_cast<uint64_t>(context_position_start) + context_rows !=
        static_cast<uint64_t>(block_position_start)) {
        return Status::invalid_argument(
            "dflash2 requires contiguous context and block positions", __FILE__, __LINE__);
    }

    bf16_t* hidden_a = executor.backbone_hidden_a.data<bf16_t>();
    bf16_t* hidden_b = executor.backbone_hidden_b.data<bf16_t>();
    const bf16_t* src = noise_embedding;
    for (uint32_t li = 0; li < layers; ++li) {
        bf16_t* dst = (li % 2u == 0u) ? hidden_a : hidden_b;
        const Status st = dflash2_forward_layer_stateless(
            executor, li, src, target_feature, context_rows, context_position_start,
            block_position_start, block_rows, dst, stream);
        if (!st.ok()) return st;
        src = dst;
    }

    const uint32_t hidden = static_cast<uint32_t>(executor.config->hidden_size);
    const hipError_t err = shelf::kernel::launch_dflash2_rmsnorm_direct_bf16(
        src, executor.weights->final_norm_weight.data<bf16_t>(), output, block_rows, hidden,
        hidden, executor.config->rms_norm_eps, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 final norm launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

namespace {

Status lm_head_validate(
    const DFlash2Executor& executor,
    const shelf::weights::MatrixWeight& weight,
    uint32_t rows) {
    const DFlash2Config& config = *executor.config;
    const bool bf16 = weight.encoding == shelf::weights::MatrixEncoding::Bf16;
    const bool fp8 = weight.encoding == shelf::weights::MatrixEncoding::Fp8Finegrained;
    const bool exl3 = weight.encoding == shelf::weights::MatrixEncoding::Exl3;
    if (!bf16 && !fp8 && !exl3) {
        return Status::unsupported("dflash2 target lm_head encoding unsupported", __FILE__,
                                   __LINE__);
    }
    if (weight.rows != config.vocab_size || weight.cols != config.hidden_size) {
        return Status::invalid_argument("dflash2 target lm_head shape mismatch", __FILE__,
                                        __LINE__);
    }
    if (bf16) {
        if (weight.bf16.ndim() == 0) {
            return Status::invalid_argument("dflash2 target lm_head payload missing", __FILE__,
                                            __LINE__);
        }
    } else if (fp8) {
        if (!weight.preshuffled ||
            !quantization::weight_scale_group_is_default(weight.weight_scale_group, quantization::QuantKernelKind::LinearFp8Finegrained) ||
            (weight.k_padded % 128u) != 0u || weight.codes.ndim() == 0 ||
            weight.scales.ndim() == 0) {
            return Status::invalid_argument("dflash2 target lm_head layout invalid", __FILE__,
                                            __LINE__);
        }
    } else {
        if (weight.preshuffled || weight.codes.ndim() == 0 ||
            weight.exl3_suh.ndim() == 0 || weight.exl3_svh.ndim() == 0 ||
            (weight.k_padded % 128u) != 0u || (weight.rows % 128u) != 0u ||
            weight.exl3_ka < 1u || weight.exl3_ka > 8u ||
            (weight.exl3_half && weight.exl3_ka > 3u)) {
            return Status::invalid_argument("dflash2 target lm_head exl3 layout invalid",
                                            __FILE__, __LINE__);
        }
        if (weight.quant_spec != quantization::QuantSpecId::EXL3_W16A16_V1 &&
            weight.quant_spec != quantization::QuantSpecId::EXL3_MUL24_INT8_V1) {
            return Status::unsupported("dflash2 target lm_head exl3 format unsupported",
                                       __FILE__, __LINE__);
        }
        if (weight.quant_spec == quantization::QuantSpecId::EXL3_MUL24_INT8_V1 &&
            (weight.exl3_half || weight.k_padded != weight.cols ||
             executor.exl3_lm_head_accum.ndim() == 0)) {
            return Status::invalid_argument("dflash2 target lm_head exl3 int8 layout invalid",
                                            __FILE__, __LINE__);
        }
    }
    if (rows == 0u || rows > executor.max_rows) {
        return Status::invalid_argument("dflash2 logits rows out of range", __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status lm_head_quantize(
    DFlash2Executor& executor,
    const shelf::weights::MatrixWeight& weight,
    const bf16_t* hidden,
    uint32_t rows,
    hipStream_t stream) {
    if (weight.encoding == shelf::weights::MatrixEncoding::Bf16 ||
        weight.encoding == shelf::weights::MatrixEncoding::Exl3) {
        return Status::make_ok();
    }
    const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(
        weight.k_padded, executor.max_rows);
    if (layout.k_padded != weight.k_padded) {
        return Status::invalid_state("dflash2 lm_head activation k mismatch", __FILE__,
                                     __LINE__);
    }
    if (layout.codes_bytes() > static_cast<uint64_t>(executor.activation_codes.dim(0)) ||
        layout.scales_bytes() >
            static_cast<uint64_t>(executor.activation_scales.dim(0)) * sizeof(float)) {
        return Status::invalid_state("dflash2 activation workspace too small", __FILE__,
                                     __LINE__);
    }
    const hipError_t err = shelf::kernel::launch_activation_quantize_fp8_finegrained(
        hidden, reinterpret_cast<uint8_t*>(executor.activation_codes.data<int8_t>()),
        executor.activation_scales.data<float>(), rows, weight.cols, weight.k_padded,
        weight.cols, layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 lm_head activation quantize launch",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status lm_head_gemm(
    DFlash2Executor& executor,
    const shelf::weights::MatrixWeight& weight,
    const bf16_t* hidden,
    uint32_t rows,
    float* logits,
    uint32_t logits_row_stride,
    hipStream_t stream) {
    if (weight.encoding == shelf::weights::MatrixEncoding::Exl3) {
        if (weight.quant_spec == quantization::QuantSpecId::EXL3_MUL24_INT8_V1) {
            const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(
                weight.k_padded, executor.max_rows);
            auto* codes = executor.activation_codes.data<int8_t>();
            auto* scales = executor.activation_scales.data<float>();
            auto* accum = executor.exl3_lm_head_accum.data<float>();
            hipError_t err = shelf::kernel::launch_exl3_int8_rotate_quantize(
                hidden, weight.cols, weight.exl3_suh.data<uint16_t>(), codes, scales, rows,
                weight.cols, layout.code_row_stride_bytes, layout.scale_row_stride_bytes, stream);
            if (err == hipSuccess) {
                err = shelf::kernel::launch_exl3_int8_gemm(
                    codes, scales, layout.code_row_stride_bytes, layout.scale_row_stride_bytes,
                    weight.codes.data<uint16_t>(), weight.exl3_ka, accum, rows, weight.rows,
                    weight.cols, 1u, stream);
            }
            if (err == hipSuccess) {
                err = shelf::kernel::launch_exl3_int8_rotate_out(
                    accum, weight.exl3_svh.data<uint16_t>(), logits,
                    shelf::kernel::Exl3Int8OutputDType::F32, rows, weight.rows,
                    logits_row_stride, 1u, stream);
            }
            if (err != hipSuccess) {
                return Status::hip_error("dflash2 target lm_head exl3 int8 launch",
                                         hipGetErrorString(err), __FILE__, __LINE__);
            }
            return Status::make_ok();
        }
        ::shelf::kernel::Exl3GemmArgs args;
        args.trellis = weight.codes.data<void>();
        args.suh = weight.exl3_suh.data<void>();
        args.svh = weight.exl3_svh.data<void>();
        args.input = hidden;
        args.output = logits;
        args.rows = rows;
        args.out_features = weight.rows;
        args.k = weight.cols;
        args.input_row_stride = weight.cols;
        args.output_row_stride = logits_row_stride;
        args.output_dtype = 1u;
        args.ka = weight.exl3_ka;
        args.half = weight.exl3_half ? 1u : 0u;
        args.splitk_scratch = nullptr;
        args.splitk_floats = 0u;
        args.splitk_splits = 1u;
        args.xrot = nullptr;
        args.groups = 0u;
        const hipError_t err = ::shelf::kernel::launch_gemm_exl3_w16a16(args, stream);
        if (err != hipSuccess) {
            return Status::hip_error("dflash2 target lm_head exl3 gemm launch",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    if (weight.encoding == shelf::weights::MatrixEncoding::Bf16) {
        const auto config =
            select_bf16_gemm(weight, rows, weight.cols, weight.bf16.data<void>(), hidden,
                             true);
        if (!config) {
            return Status::unsupported("dflash2 target lm_head has no bf16 gemm geometry",
                                       __FILE__, __LINE__);
        }
        const hipError_t err = shelf::kernel::launch_gemm_bf16(
            *config, weight.bf16.data<bf16_t>(), hidden, logits,
            shelf::kernel::Bf16GemmOutputDType::F32, rows, weight.rows, weight.cols, weight.cols,
            logits_row_stride, stream);
        if (err != hipSuccess) {
            return Status::hip_error("dflash2 target lm_head bf16 gemm launch",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    const auto layout = shelf::runtime::Int8ActivationWorkspaceLayout::make(
        weight.k_padded, executor.max_rows);
    const int8_t* acodes = executor.activation_codes.data<int8_t>();
    const float* ascales = executor.activation_scales.data<float>();
    if (weight.encoding == shelf::weights::MatrixEncoding::Fp8Finegrained) {
        const shelf::execution::Fp8FinegrainedGemmSelectorInput lin{
            rows, weight.rows, weight.cols, weight.k_padded};
        const auto config = shelf::execution::select_fp8_finegrained_gemm_config(lin);
        if (!config) {
            return Status::unsupported("dflash2 target lm_head has no fp8 gemm geometry",
                                       __FILE__, __LINE__);
        }
        const hipError_t err = shelf::kernel::launch_gemm_fp8_finegrained_w8a8_wmma_auto(
            *config, weight.codes.data<uint8_t>(), weight.scales.data<float>(), acodes,
            ascales, logits, shelf::kernel::Fp8FinegrainedGemmOutputDType::F32, rows, weight.rows,
            weight.k_padded, weight.storage_scale_stride_bytes,
            layout.code_row_stride_bytes, layout.scale_row_stride_bytes, logits_row_stride,
            stream);
        if (err != hipSuccess) {
            return Status::hip_error("dflash2 target lm_head fp8 gemm launch",
                                     hipGetErrorString(err), __FILE__, __LINE__);
        }
        return Status::make_ok();
    }
    (void)acodes;
    (void)ascales;
    (void)layout;
    return Status::unsupported("dflash2 target lm_head encoding unsupported", __FILE__,
                               __LINE__);
}

Status lm_head_logits(
    DFlash2Executor& executor,
    const shelf::weights::MatrixWeight& weight,
    const bf16_t* hidden,
    uint32_t rows,
    float* logits,
    uint32_t logits_row_stride,
    hipStream_t stream) {
    Status st = lm_head_validate(executor, weight, rows);
    if (!st.ok())
        return st;
    st = lm_head_quantize(executor, weight, hidden, rows, stream);
    if (!st.ok())
        return st;
    return lm_head_gemm(executor, weight, hidden, rows, logits, logits_row_stride, stream);
}

}  // namespace


Status dflash2_compute_target_logits(
    DFlash2Executor& executor,
    const bf16_t* hidden,
    uint32_t rows,
    float* logits,
    uint32_t logits_row_stride,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (hidden == nullptr || logits == nullptr) {
        return Status::invalid_argument("dflash2 logits buffers are null", __FILE__, __LINE__);
    }
    if (executor.config == nullptr || executor.target_lm_head == nullptr) {
        return Status::invalid_state("dflash2 executor has no target lm_head", __FILE__,
                                     __LINE__);
    }
    if (rows == 0u || rows > executor.draft_rows) {
        return Status::invalid_argument("dflash2 logits rows out of range", __FILE__,
                                        __LINE__);
    }
    return lm_head_logits(executor, *executor.target_lm_head, hidden, rows, logits,
                          logits_row_stride, stream);
}

Status dflash2_run_candidate_selector(
    DFlash2Executor& executor,
    const bf16_t* hidden_proj,
    const int32_t* topk_ids,
    const float* topk_logits,
    uint32_t draft_rows,
    int32_t anchor_token,
    int32_t* output_draft_tokens,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (hidden_proj == nullptr || topk_ids == nullptr || topk_logits == nullptr ||
        output_draft_tokens == nullptr) {
        return Status::invalid_argument("dflash2 selector buffers are null", __FILE__,
                                        __LINE__);
    }
    if (executor.weights == nullptr || executor.config == nullptr) {
        return Status::invalid_state("dflash2 executor has no weights", __FILE__, __LINE__);
    }
    if (draft_rows == 0u || draft_rows > executor.draft_rows) {
        return Status::invalid_argument("dflash2 selector draft rows out of range", __FILE__,
                                        __LINE__);
    }
    shelf::kernel::DFlash2CandidateSelectorArgs args;
    args.hidden_proj = hidden_proj;
    args.topk_ids = topk_ids;
    args.topk_logits = topk_logits;
    args.predecessor_codebook =
        executor.weights->selector.predecessor_codebook.data<bf16_t>();
    args.successor_codebook =
        executor.weights->selector.successor_codebook.data<bf16_t>();
    args.anchor_token = anchor_token;
    args.output_tokens = output_draft_tokens;
    args.draft_rows = draft_rows;
    args.rank = executor.config->selector_rank;
    args.top_k = executor.config->selector_top_k;
    args.vocab_size = executor.config->vocab_size;
    const hipError_t err =
        shelf::kernel::launch_dflash2_candidate_selector(args, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 candidate selector launch",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status dflash2_select_draft_tokens(
    DFlash2Executor& executor,
    const bf16_t* final_hidden,
    uint32_t block_rows,
    int32_t anchor_token,
    int32_t* output_draft_tokens,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (final_hidden == nullptr || output_draft_tokens == nullptr) {
        return Status::invalid_argument("dflash2 select buffers are null", __FILE__, __LINE__);
    }
    if (executor.config == nullptr || executor.weights == nullptr) {
        return Status::invalid_state("dflash2 executor has no weights", __FILE__, __LINE__);
    }
    if (executor.target_lm_head == nullptr) {
        return Status::invalid_state("dflash2 executor has no target lm_head", __FILE__,
                                     __LINE__);
    }
    if (block_rows < 2u || block_rows > executor.config->block_size ||
        block_rows - 1u > executor.draft_rows) {
        return Status::invalid_argument("dflash2 select block rows out of range", __FILE__,
                                        __LINE__);
    }
    const uint32_t draft = block_rows - 1u;
    const uint32_t hidden = static_cast<uint32_t>(executor.config->hidden_size);
    const uint32_t rank = executor.config->selector_rank;
    const shelf::weights::MatrixWeight& draft_head = *executor.target_lm_head;
    const uint32_t vocab = executor.config->vocab_size;
    const uint32_t top_k = executor.config->selector_top_k;
    const bf16_t* draft_hidden = final_hidden + hidden;

    const shelf::weights::MatrixWeight& projection =
        executor.weights->selector.hidden_projection;
    const auto gemm = select_bf16_gemm(projection, draft, hidden,
                                         projection.bf16.data<void>(), draft_hidden);
    if (!gemm) {
        return Status::unsupported(
            "dflash2 selector hidden projection has no exact bf16 gemm geometry", __FILE__,
            __LINE__);
    }
    hipError_t err = shelf::kernel::launch_gemm_bf16(
        *gemm, projection.bf16.data<bf16_t>(), draft_hidden,
        executor.selector_hidden_proj.data<bf16_t>(),
        shelf::kernel::Bf16GemmOutputDType::BF16, draft, projection.rows, projection.cols,
        hidden, rank, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 selector hidden projection launch",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }

    Status st = lm_head_logits(executor, draft_head, draft_hidden, draft,
                               executor.proposal_logits.data<float>(), vocab, stream);
    if (!st.ok()) return st;

    const bool use_reference =
        topk_reference_enabled() || top_k != shelf::kernel::kDflash2TopKMaxK;
    if (use_reference) {
        err = shelf::kernel::launch_dflash2_topk_f32(
            executor.proposal_logits.data<float>(), draft, vocab, vocab, top_k,
            executor.proposal_topk_ids.data<int32_t>(),
            executor.proposal_topk_logits.data<float>(),
            executor.topk_scratch_ids.data<int32_t>(),
            executor.topk_scratch_logits.data<float>(), stream);
    } else {
        err = shelf::kernel::launch_dflash2_topk_f32_optimized(
            executor.proposal_logits.data<float>(), draft, vocab, vocab, top_k,
            executor.proposal_topk_ids.data<int32_t>(),
            executor.proposal_topk_logits.data<float>(),
            executor.topk_scratch_ids.data<int32_t>(),
            executor.topk_scratch_logits.data<float>(), topk_plan(), stream);
    }
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 topk launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    return dflash2_run_candidate_selector(
        executor, executor.selector_hidden_proj.data<bf16_t>(),
        executor.proposal_topk_ids.data<int32_t>(),
        executor.proposal_topk_logits.data<float>(), draft, anchor_token,
        output_draft_tokens, stream);
}

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
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (output_draft_tokens == nullptr) {
        return Status::invalid_argument("dflash2 select output is null", __FILE__, __LINE__);
    }
    Status st = dflash2_forward_backbone_stateless(
        executor, noise_embedding, target_feature, context_rows, context_position_start,
        block_position_start, block_rows, executor.backbone_output.data<bf16_t>(), stream);
    if (!st.ok()) return st;
    return dflash2_select_draft_tokens(executor, executor.backbone_output.data<bf16_t>(),
                                       block_rows, anchor_token, output_draft_tokens, stream);
}

namespace {

Status validate_cached_context(
    const DFlash2Executor& executor,
    const DFlash2ContextState& context,
    uint32_t block_position_start,
    uint32_t block_rows) {
    if (!executor.initialized || executor.config == nullptr || executor.weights == nullptr) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state("dflash2 context state not initialized", __FILE__,
                                     __LINE__);
    }
    const DFlash2Config& config = *executor.config;
    if (context.num_layers != config.num_hidden_layers ||
        context.kv_features != config.attention_kv_rows() ||
        context.capacity != config.sliding_window) {
        return Status::invalid_argument("dflash2 context contract mismatch", __FILE__,
                                        __LINE__);
    }
    if (context.length == 0u || context.length > context.capacity) {
        return Status::invalid_argument("dflash2 context length out of range", __FILE__,
                                        __LINE__);
    }
    if (block_position_start != context.next_position) {
        return Status::invalid_argument(
            "dflash2 cached block must start at context next_position", __FILE__, __LINE__);
    }
    if (block_rows == 0u || block_rows > executor.max_rows ||
        block_rows > config.block_size) {
        return Status::invalid_argument("dflash2 cached block rows out of range", __FILE__,
                                        __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Status dflash2_forward_layer_cached(
    DFlash2Executor& executor,
    const DFlash2ContextState& context,
    uint32_t layer_index,
    const bf16_t* hidden_in,
    uint32_t block_position_start,
    uint32_t block_rows,
    bf16_t* output,
    hipStream_t stream) {
    if (hidden_in == nullptr || output == nullptr) {
        return Status::invalid_argument("dflash2 cached layer buffers are null", __FILE__,
                                        __LINE__);
    }
    Status st = validate_cached_context(executor, context, block_position_start, block_rows);
    if (!st.ok()) return st;
    if (layer_index >= executor.weights->layers.size()) {
        return Status::invalid_argument("dflash2 layer index out of range", __FILE__, __LINE__);
    }

    const DFlash2Config& config = *executor.config;
    const DFlash2LayerWeights& layer = executor.weights->layers[layer_index];
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t q_features = static_cast<uint32_t>(config.attention_q_rows());
    const uint32_t kv_features = static_cast<uint32_t>(config.attention_kv_rows());
    const uint32_t head_dim = static_cast<uint32_t>(config.head_dim);
    const uint32_t q_heads = static_cast<uint32_t>(config.num_attention_heads);
    const uint32_t kv_heads = static_cast<uint32_t>(config.num_key_value_heads);
    const float eps = config.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const uint32_t context_rows = context.length;
    const uint32_t context_start = context.next_position - context.length;
    const std::size_t layer_stride =
        static_cast<std::size_t>(context.capacity) * context.kv_features;
    const bf16_t* k_ring = context.k_ring.data<bf16_t>() +
                           static_cast<std::size_t>(layer_index) * layer_stride;
    const bf16_t* v_ring = context.v_ring.data<bf16_t>() +
                           static_cast<std::size_t>(layer_index) * layer_stride;

    st = layer_rmsnorm_direct(
        hidden_in, layer.input_layernorm_weight, executor.layer_input_norm.data<bf16_t>(),
        block_rows, hidden, hidden, eps, "input norm", stream);
    if (!st.ok()) return st;

    st = dflash2_grouped_conv_prepare(executor, executor.layer_input_norm.data<bf16_t>(),
                                      layer, true, block_rows,
                                      executor.attention_prepared.data<bf16_t>(), stream);
    if (!st.ok()) return st;

    const bf16_t* attention_prepared = executor.attention_prepared.data<bf16_t>();

        st = executor_linear(executor, layer.attn_q_proj, attention_prepared, hidden,
                             executor.q_raw.data<bf16_t>(), q_features, block_rows, "q_proj",
                             stream);
        if (!st.ok()) return st;
        st = executor_linear(executor, layer.attn_k_proj, attention_prepared, hidden,
                             executor.k_noise_raw.data<bf16_t>(), kv_features, block_rows,
                             "k_noise_proj", stream);
        if (!st.ok()) return st;
        st = executor_linear(executor, layer.attn_v_proj, attention_prepared, hidden,
                             executor.v_noise.data<bf16_t>(), kv_features, block_rows,
                             "v_noise_proj", stream);
        if (!st.ok()) return st;
    

    hipError_t err = hipSuccess;
    st = layer_rmsnorm_direct(executor.q_raw.data<bf16_t>(), layer.attn_q_norm_weight,
                              executor.q_norm.data<bf16_t>(), block_rows, q_features,
                              head_dim, eps, "q_norm", stream);
    if (!st.ok()) return st;
    st = layer_rmsnorm_direct(executor.k_noise_raw.data<bf16_t>(),
                              layer.attn_k_norm_weight,
                              executor.k_noise_norm.data<bf16_t>(), block_rows,
                              kv_features, head_dim, eps, "k_noise_norm", stream);
    if (!st.ok()) return st;

    err = shelf::kernel::launch_dflash2_rope_bf16(
        executor.q_norm.data<bf16_t>(), q_features, executor.q_rope.data<bf16_t>(),
        q_features, block_rows, q_features, head_dim, block_position_start,
        config.rope_theta, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 q rope launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    err = shelf::kernel::launch_dflash2_rope_bf16(
        executor.k_noise_norm.data<bf16_t>(), kv_features,
        executor.k_noise_rope.data<bf16_t>(), kv_features, block_rows, kv_features,
        head_dim, block_position_start, config.rope_theta, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 k_noise rope launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    if (!attention_reference_enabled()) {
        err = shelf::kernel::launch_dflash2_attention_ring_gqa_bf16(
            executor.q_rope.data<bf16_t>(), q_features, k_ring, v_ring, context.capacity,
            context_rows, context_start, kv_features, executor.k_noise_rope.data<bf16_t>(),
            executor.v_noise.data<bf16_t>(), block_rows, kv_features,
            executor.attention_context.data<bf16_t>(), q_features, q_heads, kv_heads, head_dim,
            block_position_start, static_cast<uint32_t>(config.sliding_window), scale, stream);
    } else {
        err = hipErrorNotSupported;
    }
    if (err == hipErrorNotSupported) {
        err = shelf::kernel::launch_dflash2_attention_ring_bf16(
            executor.q_rope.data<bf16_t>(), q_features, k_ring, v_ring, context.capacity,
            context_rows, context_start, kv_features, executor.k_noise_rope.data<bf16_t>(),
            executor.v_noise.data<bf16_t>(), block_rows, kv_features,
            executor.attention_context.data<bf16_t>(), q_features, q_heads, kv_heads, head_dim,
            block_position_start, static_cast<uint32_t>(config.sliding_window), scale, stream);
    }
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 ring attention launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    return layer_attention_tail(executor, layer, hidden_in, block_rows, nullptr, output,
                                stream);
}

Status dflash2_forward_backbone_cached(
    DFlash2Executor& executor,
    const DFlash2ContextState& context,
    const bf16_t* noise_embedding,
    uint32_t block_position_start,
    uint32_t block_rows,
    bf16_t* output,
    hipStream_t stream) {
    if (noise_embedding == nullptr || output == nullptr) {
        return Status::invalid_argument("dflash2 cached backbone buffers are null", __FILE__,
                                        __LINE__);
    }
    Status st = validate_cached_context(executor, context, block_position_start, block_rows);
    if (!st.ok()) return st;
    const uint32_t layers = static_cast<uint32_t>(executor.weights->layers.size());
    if (layers == 0u || layers != executor.config->num_hidden_layers) {
        return Status::invalid_state("dflash2 backbone layer count mismatch", __FILE__,
                                     __LINE__);
    }

    bf16_t* hidden_a = executor.backbone_hidden_a.data<bf16_t>();
    bf16_t* hidden_b = executor.backbone_hidden_b.data<bf16_t>();
    const bf16_t* src = noise_embedding;
    for (uint32_t li = 0; li < layers; ++li) {
        bf16_t* dst = (li % 2u == 0u) ? hidden_a : hidden_b;
        st = dflash2_forward_layer_cached(executor, context, li, src, block_position_start,
                                          block_rows, dst, stream);
        if (!st.ok()) return st;
        src = dst;
    }

    const uint32_t hidden = static_cast<uint32_t>(executor.config->hidden_size);
    const hipError_t err = shelf::kernel::launch_dflash2_rmsnorm_direct_bf16(
        src, executor.weights->final_norm_weight.data<bf16_t>(), output, block_rows, hidden,
        hidden, executor.config->rms_norm_eps, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 final norm launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    return Status::make_ok();
}

Status dflash2_forward_and_select_cached(
    DFlash2Executor& executor,
    const DFlash2ContextState& context,
    const bf16_t* noise_embedding,
    uint32_t block_position_start,
    uint32_t block_rows,
    int32_t anchor_token,
    int32_t* output_draft_tokens,
    hipStream_t stream) {
    if (output_draft_tokens == nullptr) {
        return Status::invalid_argument("dflash2 cached select output is null", __FILE__,
                                        __LINE__);
    }
    Status st = dflash2_forward_backbone_cached(
        executor, context, noise_embedding, block_position_start, block_rows,
        executor.backbone_output.data<bf16_t>(), stream);
    if (!st.ok()) return st;
    return dflash2_select_draft_tokens(executor, executor.backbone_output.data<bf16_t>(),
                                       block_rows, anchor_token, output_draft_tokens, stream);
}

Status dflash2_prepare_noise_embedding(
    DFlash2Executor& executor,
    int32_t anchor_token,
    uint32_t block_rows,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (executor.config == nullptr || executor.target_embed_tokens == nullptr) {
        return Status::invalid_state("dflash2 executor has no target embed_tokens", __FILE__,
                                     __LINE__);
    }
    if (executor.noise_token_ids.ndim() == 0u || executor.noise_embedding.ndim() == 0u) {
        return Status::invalid_state("dflash2 noise buffers are not allocated", __FILE__,
                                     __LINE__);
    }
    if (block_rows == 0u || block_rows > executor.config->block_size ||
        block_rows > executor.noise_embedding.dim(0)) {
        return Status::invalid_argument("dflash2 noise block rows out of range", __FILE__,
                                        __LINE__);
    }

    const shelf::weights::MatrixWeight& embed = *executor.target_embed_tokens;
    const uint32_t hidden = static_cast<uint32_t>(executor.config->hidden_size);
    const uint32_t vocab = static_cast<uint32_t>(executor.config->vocab_size);
    const int32_t mask_token = static_cast<int32_t>(executor.config->mask_token_id);

    hipError_t err = shelf::kernel::launch_dflash2_noise_token_ids(
        anchor_token, mask_token, executor.noise_token_ids.data<int32_t>(), block_rows, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 noise token launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }

    const int32_t* token_ids = executor.noise_token_ids.data<int32_t>();
    bf16_t* output = executor.noise_embedding.data<bf16_t>();
    if (embed.encoding == shelf::weights::MatrixEncoding::Bf16) {
        err = shelf::kernel::launch_embedding_bf16(
            embed.bf16.data<bf16_t>(), token_ids, output, block_rows, vocab, hidden, hidden,
            nullptr, stream);
    } else {
        return Status::unsupported("dflash2 target embed_tokens encoding is not supported",
                                   __FILE__, __LINE__);
    }
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 noise embedding launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status dflash2_propose_cached(
    DFlash2Executor& executor,
    const DFlash2ContextState& context,
    int32_t anchor_token,
    uint32_t num_drafts,
    int32_t* output_drafts,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state("dflash2 context state not initialized", __FILE__,
                                     __LINE__);
    }
    if (output_drafts == nullptr) {
        return Status::invalid_argument("dflash2 proposal output is null", __FILE__, __LINE__);
    }
    if (num_drafts == 0u || num_drafts > executor.draft_rows) {
        return Status::invalid_argument("dflash2 proposal num_drafts out of range", __FILE__,
                                        __LINE__);
    }
    const uint32_t block_rows = num_drafts + 1u;
    Status st = dflash2_prepare_noise_embedding(executor, anchor_token, block_rows, stream);
    if (!st.ok()) return st;
    return dflash2_forward_and_select_cached(
        executor, context, executor.noise_embedding.data<bf16_t>(), context.next_position,
        block_rows, anchor_token, output_drafts, stream);
}

namespace {

struct DFlash2BatchStage {
    uint32_t request_count = 0u;
    uint32_t total_rows = 0u;
    uint32_t max_block_rows = 0u;
    uint32_t max_keys = 0u;
};

Status dflash2_forward_layer_cached_batch(
    DFlash2Executor& executor,
    const shelf::kernel::DFlash2RequestDescriptor* requests,
    const DFlash2BatchStage& stage,
    uint32_t layer_index,
    const bf16_t* hidden_in,
    bf16_t* output,
    hipStream_t stream) {
    const DFlash2Config& config = *executor.config;
    const DFlash2LayerWeights& layer = executor.weights->layers[layer_index];
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t q_features = static_cast<uint32_t>(config.attention_q_rows());
    const uint32_t kv_features = static_cast<uint32_t>(config.attention_kv_rows());
    const uint32_t head_dim = static_cast<uint32_t>(config.head_dim);
    const uint32_t q_heads = static_cast<uint32_t>(config.num_attention_heads);
    const uint32_t kv_heads = static_cast<uint32_t>(config.num_key_value_heads);
    const uint32_t capacity = static_cast<uint32_t>(config.sliding_window);
    const uint32_t total_rows = stage.total_rows;
    const float eps = config.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const uint32_t* row_has_prev = executor.conv_row_has_prev.data<uint32_t>();
    const uint64_t ring_layer_offset =
        static_cast<uint64_t>(layer_index) * capacity * kv_features;

    Status st = layer_rmsnorm_direct(
        hidden_in, layer.input_layernorm_weight, executor.layer_input_norm.data<bf16_t>(),
        total_rows, hidden, hidden, eps, "input norm", stream);
    if (!st.ok()) return st;

    st = grouped_conv_prepare_rows(executor, executor.layer_input_norm.data<bf16_t>(), layer,
                                   true, total_rows,
                                   executor.attention_prepared.data<bf16_t>(), row_has_prev,
                                   stream);
    if (!st.ok()) return st;

    const bf16_t* attention_prepared = executor.attention_prepared.data<bf16_t>();
    st = executor_linear(executor, layer.attn_q_proj, attention_prepared, hidden,
                         executor.q_raw.data<bf16_t>(), q_features, total_rows, "q_proj",
                         stream);
    if (!st.ok()) return st;
    st = executor_linear(executor, layer.attn_k_proj, attention_prepared, hidden,
                         executor.k_noise_raw.data<bf16_t>(), kv_features, total_rows,
                         "k_noise_proj", stream);
    if (!st.ok()) return st;
    st = executor_linear(executor, layer.attn_v_proj, attention_prepared, hidden,
                         executor.v_noise.data<bf16_t>(), kv_features, total_rows,
                         "v_noise_proj", stream);
    if (!st.ok()) return st;

    st = layer_rmsnorm_direct(executor.q_raw.data<bf16_t>(), layer.attn_q_norm_weight,
                              executor.q_norm.data<bf16_t>(), total_rows, q_features,
                              head_dim, eps, "q_norm", stream);
    if (!st.ok()) return st;
    st = layer_rmsnorm_direct(executor.k_noise_raw.data<bf16_t>(),
                              layer.attn_k_norm_weight,
                              executor.k_noise_norm.data<bf16_t>(), total_rows, kv_features,
                              head_dim, eps, "k_noise_norm", stream);
    if (!st.ok()) return st;

    hipError_t err = shelf::kernel::launch_dflash2_rope_positions_bf16(
        executor.q_norm.data<bf16_t>(), q_features, executor.q_rope.data<bf16_t>(), q_features,
        requests, stage.request_count, stage.max_block_rows, q_features, head_dim,
        config.rope_theta, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 q rope launch", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    err = shelf::kernel::launch_dflash2_rope_positions_bf16(
        executor.k_noise_norm.data<bf16_t>(), kv_features,
        executor.k_noise_rope.data<bf16_t>(), kv_features, requests, stage.request_count,
        stage.max_block_rows, kv_features, head_dim, config.rope_theta, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 k_noise rope launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    if (!attention_reference_enabled()) {
        err = shelf::kernel::launch_dflash2_attention_ring_gqa_requests_bf16(
            executor.q_rope.data<bf16_t>(), q_features, requests, stage.request_count,
            stage.max_block_rows, stage.max_keys, capacity, ring_layer_offset,
            executor.k_noise_rope.data<bf16_t>(), executor.v_noise.data<bf16_t>(),
            kv_features, executor.attention_context.data<bf16_t>(), q_features, q_heads,
            kv_heads, head_dim, static_cast<uint32_t>(config.sliding_window), scale, stream);
    } else {
        err = hipErrorNotSupported;
    }
    if (err == hipErrorNotSupported) {
        err = shelf::kernel::launch_dflash2_attention_ring_requests_bf16(
            executor.q_rope.data<bf16_t>(), q_features, requests, stage.request_count,
            stage.max_block_rows, stage.max_keys, capacity, ring_layer_offset,
            executor.k_noise_rope.data<bf16_t>(), executor.v_noise.data<bf16_t>(),
            kv_features, executor.attention_context.data<bf16_t>(), q_features, q_heads,
            kv_heads, head_dim, static_cast<uint32_t>(config.sliding_window), scale, stream);
    }
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 batch ring attention launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    return layer_attention_tail(executor, layer, hidden_in, total_rows, row_has_prev, output,
                               stream);
}

Status dflash2_forward_backbone_cached_batch(
    DFlash2Executor& executor,
    const shelf::kernel::DFlash2RequestDescriptor* requests,
    const DFlash2BatchStage& stage,
    const bf16_t* noise_embedding,
    bf16_t* output,
    hipStream_t stream) {
    const uint32_t layers = static_cast<uint32_t>(executor.weights->layers.size());
    if (layers == 0u || layers != executor.config->num_hidden_layers) {
        return Status::invalid_state("dflash2 batch backbone layer count mismatch", __FILE__,
                                     __LINE__);
    }

    bf16_t* hidden_a = executor.backbone_hidden_a.data<bf16_t>();
    bf16_t* hidden_b = executor.backbone_hidden_b.data<bf16_t>();
    const bf16_t* src = noise_embedding;
    for (uint32_t li = 0; li < layers; ++li) {
        bf16_t* dst = (li % 2u == 0u) ? hidden_a : hidden_b;
        const Status st = dflash2_forward_layer_cached_batch(executor, requests, stage, li, src,
                                                             dst, stream);
        if (!st.ok()) return st;
        src = dst;
    }

    const uint32_t hidden = static_cast<uint32_t>(executor.config->hidden_size);
    const hipError_t err = shelf::kernel::launch_dflash2_rmsnorm_direct_bf16(
        src, executor.weights->final_norm_weight.data<bf16_t>(), output, stage.total_rows,
        hidden, hidden, executor.config->rms_norm_eps, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 batch final norm launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status dflash2_select_draft_tokens_batch(
    DFlash2Executor& executor,
    const shelf::kernel::DFlash2RequestDescriptor* requests,
    const DFlash2BatchStage& stage,
    const bf16_t* final_hidden,
    hipStream_t stream) {
    const DFlash2Config& config = *executor.config;
    const uint32_t total_rows = stage.total_rows;
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t rank = static_cast<uint32_t>(config.selector_rank);
    const uint32_t vocab = static_cast<uint32_t>(config.vocab_size);
    const uint32_t top_k = static_cast<uint32_t>(config.selector_top_k);
    const shelf::weights::MatrixWeight& projection = executor.weights->selector.hidden_projection;

    if (executor.target_lm_head == nullptr) {
        return Status::invalid_state("dflash2 executor has no target lm_head", __FILE__,
                                     __LINE__);
    }

    const auto gemm = select_bf16_gemm(projection, total_rows, hidden,
                                       projection.bf16.data<void>(), final_hidden);
    if (!gemm) {
        return Status::unsupported(
            "dflash2 selector hidden projection has no exact bf16 gemm geometry", __FILE__,
            __LINE__);
    }
    hipError_t err = shelf::kernel::launch_gemm_bf16(
        *gemm, projection.bf16.data<bf16_t>(), final_hidden,
        executor.selector_hidden_proj.data<bf16_t>(), shelf::kernel::Bf16GemmOutputDType::BF16,
        total_rows, projection.rows, projection.cols, hidden, rank, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 batch selector hidden projection launch",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }

    Status st = lm_head_logits(executor, *executor.target_lm_head, final_hidden, total_rows,
                               executor.proposal_logits.data<float>(), vocab, stream);
    if (!st.ok()) return st;

    const bool use_reference =
        topk_reference_enabled() || top_k != shelf::kernel::kDflash2TopKMaxK;
    if (use_reference) {
        err = shelf::kernel::launch_dflash2_topk_f32(
            executor.proposal_logits.data<float>(), total_rows, vocab, vocab, top_k,
            executor.proposal_topk_ids.data<int32_t>(),
            executor.proposal_topk_logits.data<float>(),
            executor.topk_scratch_ids.data<int32_t>(),
            executor.topk_scratch_logits.data<float>(), stream);
    } else {
        err = shelf::kernel::launch_dflash2_topk_f32_optimized(
            executor.proposal_logits.data<float>(), total_rows, vocab, vocab, top_k,
            executor.proposal_topk_ids.data<int32_t>(),
            executor.proposal_topk_logits.data<float>(),
            executor.topk_scratch_ids.data<int32_t>(),
            executor.topk_scratch_logits.data<float>(), topk_plan(), stream);
    }
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 batch topk launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    shelf::kernel::DFlash2CandidateSelectorArgs args;
    args.hidden_proj = executor.selector_hidden_proj.data<bf16_t>();
    args.topk_ids = executor.proposal_topk_ids.data<int32_t>();
    args.topk_logits = executor.proposal_topk_logits.data<float>();
    args.predecessor_codebook =
        executor.weights->selector.predecessor_codebook.data<bf16_t>();
    args.successor_codebook = executor.weights->selector.successor_codebook.data<bf16_t>();
    args.rank = rank;
    args.top_k = top_k;
    args.vocab_size = vocab;
    args.requests = requests;
    args.request_count = stage.request_count;
    err = shelf::kernel::launch_dflash2_candidate_selector(args, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 batch candidate selector launch",
                                 hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

}  // namespace

Status dflash2_propose_batch(
    DFlash2Executor& executor,
    const DFlash2ProposeRequest* requests,
    int32_t* const* output_drafts,
    uint32_t request_count,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (requests == nullptr || output_drafts == nullptr) {
        return Status::invalid_argument("dflash2 batch proposal requests are null", __FILE__,
                                        __LINE__);
    }
    if (request_count == 0u) {
        return Status::invalid_argument("dflash2 batch proposal request count is zero",
                                        __FILE__, __LINE__);
    }
    if (executor.config == nullptr || executor.weights == nullptr) {
        return Status::invalid_state("dflash2 executor has no weights", __FILE__, __LINE__);
    }
    if (request_count == 1u) {
        if (requests[0].context == nullptr || output_drafts[0] == nullptr) {
            return Status::invalid_argument("dflash2 batch proposal request is null", __FILE__,
                                            __LINE__);
        }
        return dflash2_propose_cached(executor, *requests[0].context, requests[0].anchor_token,
                                      requests[0].num_drafts, output_drafts[0], stream);
    }
    if (executor.target_embed_tokens == nullptr ||
        executor.noise_embedding.ndim() == 0u || executor.noise_token_ids.ndim() == 0u) {
        return Status::invalid_state("dflash2 noise buffers are not allocated", __FILE__,
                                     __LINE__);
    }

    const DFlash2Config& config = *executor.config;
    const uint32_t hidden = static_cast<uint32_t>(config.hidden_size);
    const uint32_t vocab = static_cast<uint32_t>(config.vocab_size);
    const uint32_t capacity = static_cast<uint32_t>(config.sliding_window);
    const uint32_t kv_features = static_cast<uint32_t>(config.attention_kv_rows());
    const int32_t mask_token = static_cast<int32_t>(config.mask_token_id);

    if (request_count > executor.max_rows) {
        return Status::invalid_argument("dflash2 batch proposal request count exceeds max_rows",
                                        __FILE__, __LINE__);
    }

    std::vector<shelf::kernel::DFlash2RequestDescriptor> descriptors(request_count);
    DFlash2BatchStage stage;
    stage.request_count = request_count;
    uint32_t row_cursor = 0u;
    uint32_t max_block_rows = 0u;
    uint32_t max_keys = 0u;
    for (uint32_t i = 0; i < request_count; ++i) {
        const DFlash2ContextState* context = requests[i].context;
        if (context == nullptr || output_drafts[i] == nullptr) {
            return Status::invalid_argument("dflash2 batch proposal request is null", __FILE__,
                                            __LINE__);
        }
        if (!context->initialized) {
            return Status::invalid_state("dflash2 batch proposal context is not initialized",
                                         __FILE__, __LINE__);
        }
        if (context->num_layers != config.num_hidden_layers ||
            context->kv_features != kv_features || context->capacity != capacity) {
            return Status::invalid_argument("dflash2 batch proposal context contract mismatch",
                                            __FILE__, __LINE__);
        }
        if (context->length == 0u || context->length > context->capacity) {
            return Status::invalid_argument("dflash2 batch proposal context length out of range",
                                            __FILE__, __LINE__);
        }
        if (requests[i].num_drafts == 0u || requests[i].num_drafts > executor.draft_rows) {
            return Status::invalid_argument("dflash2 batch proposal num_drafts out of range",
                                            __FILE__, __LINE__);
        }
        const uint32_t block_rows = requests[i].num_drafts + 1u;
        shelf::kernel::DFlash2RequestDescriptor& descriptor = descriptors[i];
        descriptor.ring_k = context->k_ring.data<bf16_t>();
        descriptor.ring_v = context->v_ring.data<bf16_t>();
        descriptor.output_drafts = output_drafts[i];
        descriptor.context_rows = context->length;
        descriptor.context_start_position = context->next_position - context->length;
        descriptor.block_position_start = context->next_position;
        descriptor.block_rows = block_rows;
        descriptor.row_begin = row_cursor;
        descriptor.num_drafts = requests[i].num_drafts;
        descriptor.anchor_token = requests[i].anchor_token;
        row_cursor += block_rows;
        max_block_rows = std::max(max_block_rows, block_rows);
        max_keys = std::max(max_keys, context->length + block_rows);
    }
    stage.total_rows = row_cursor;
    stage.max_block_rows = max_block_rows;
    stage.max_keys = max_keys;
    if (stage.total_rows > executor.max_rows) {
        return Status::invalid_argument("dflash2 batch proposal rows exceed max_rows", __FILE__,
                                        __LINE__);
    }

    void* descriptor_device = executor.batch_descriptors.data<uint8_t>();
    const hipError_t copy_err = hipMemcpyAsync(
        descriptor_device, descriptors.data(),
        static_cast<std::size_t>(request_count) * sizeof(shelf::kernel::DFlash2RequestDescriptor),
        hipMemcpyHostToDevice, stream);
    if (copy_err != hipSuccess) {
        return Status::hip_error("dflash2 batch descriptor upload", hipGetErrorString(copy_err),
                                 __FILE__, __LINE__);
    }
    const shelf::kernel::DFlash2RequestDescriptor* descriptor_ptr =
        static_cast<const shelf::kernel::DFlash2RequestDescriptor*>(descriptor_device);

    hipError_t err = shelf::kernel::launch_dflash2_noise_token_ids_batch(
        descriptor_ptr, request_count, max_block_rows, mask_token,
        executor.noise_token_ids.data<int32_t>(), executor.conv_row_has_prev.data<uint32_t>(),
        stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 batch noise token launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    const shelf::weights::MatrixWeight& embed = *executor.target_embed_tokens;
    if (embed.encoding != shelf::weights::MatrixEncoding::Bf16) {
        return Status::unsupported("dflash2 target embed_tokens encoding is not supported",
                                   __FILE__, __LINE__);
    }
    err = shelf::kernel::launch_embedding_bf16(
        embed.bf16.data<bf16_t>(), executor.noise_token_ids.data<int32_t>(),
        executor.noise_embedding.data<bf16_t>(), stage.total_rows, vocab, hidden, hidden,
        nullptr, stream);
    if (err != hipSuccess) {
        return Status::hip_error("dflash2 batch noise embedding launch", hipGetErrorString(err),
                                 __FILE__, __LINE__);
    }

    Status st = dflash2_forward_backbone_cached_batch(
        executor, descriptor_ptr, stage, executor.noise_embedding.data<bf16_t>(),
        executor.backbone_output.data<bf16_t>(), stream);
    if (!st.ok()) return st;

    return dflash2_select_draft_tokens_batch(executor, descriptor_ptr, stage,
                                             executor.backbone_output.data<bf16_t>(), stream);
}

Status dflash2_append_target_taps(
    DFlash2Executor& executor,
    DFlash2ContextState& context,
    const bf16_t* const* target_taps,
    uint32_t tap_count,
    uint32_t rows,
    uint32_t position_start,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("dflash2 executor not initialized", __FILE__, __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state("dflash2 context state not initialized", __FILE__,
                                     __LINE__);
    }
    if (target_taps == nullptr) {
        return Status::invalid_argument("dflash2 target taps are null", __FILE__, __LINE__);
    }
    if (tap_count != shelf::kernel::kDFlash2TargetTaps) {
        return Status::invalid_argument("dflash2 target tap count mismatch", __FILE__, __LINE__);
    }
    if (rows == 0u) {
        return Status::make_ok();
    }
    if (position_start != context.next_position) {
        return Status::invalid_argument("dflash2 tap append is not contiguous", __FILE__,
                                        __LINE__);
    }
    if (executor.config == nullptr) {
        return Status::invalid_state("dflash2 executor has no config", __FILE__, __LINE__);
    }
    const uint32_t hidden = static_cast<uint32_t>(executor.config->hidden_size);
    std::array<const bf16_t*, shelf::kernel::kDFlash2TargetTaps> ptrs{};
    uint32_t offset = 0u;
    while (offset < rows) {
        const uint32_t n = std::min(executor.max_rows, rows - offset);
        for (uint32_t t = 0; t < tap_count; ++t) {
            if (target_taps[t] == nullptr) {
                return Status::invalid_argument("dflash2 target tap is null", __FILE__,
                                                __LINE__);
            }
            ptrs[t] = target_taps[t] + static_cast<std::size_t>(offset) * hidden;
        }
        Status st = dflash2_project_target_features(executor, ptrs.data(), tap_count, n, stream);
        if (!st.ok()) return st;
        st = dflash2_context_append(executor, context,
                                    executor.target_feature.data<bf16_t>(), n,
                                    position_start + offset, stream);
        if (!st.ok()) return st;
        offset += n;
    }
    return Status::make_ok();
}

}
