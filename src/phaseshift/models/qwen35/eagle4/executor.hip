#include <phaseshift/models/qwen35/eagle4/context_state.h>
#include <phaseshift/models/qwen35/eagle4/executor.h>
#include <phaseshift/models/qwen35/dflash2/executor.h>
#include <phaseshift/models/qwen35/kernels/dflash2/kv_ring.h>
#include <phaseshift/kernels/optimized/radix_topn.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rmsnorm.h>
#include <phaseshift/models/qwen35/kernels/dflash2/rope.h>
#include <phaseshift/models/qwen35/kernels/dflash2/swiglu.h>
#include <phaseshift/models/qwen35/kernels/eagle4/attention.h>
#include <phaseshift/models/qwen35/kernels/eagle4/candidate_ops.h>
#include <phaseshift/models/qwen35/kernels/eagle4/ops.h>
#include <phaseshift/kernels/optimized/embedding.h>
#include <phaseshift/models/qwen35/runtime/eagle4_tree.h>
#include <phaseshift/models/qwen35/runtime/executor.h>
#include <phaseshift/models/qwen35/runtime/tp_batch_hook.h>
#include <phaseshift/core/gpu/scoped_device.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace ps::qwen35::eagle4 {

namespace {

Status tp_sum_bf16(Eagle4Executor& executor, void* ptr, uint32_t rows,
                   uint32_t row_stride, uint32_t features, hipStream_t stream) {
    if (executor.tp_size <= 1u || !executor.tp_sum) {
        return Status::make_ok();
    }
    ps::runtime::TpSumTarget target;
    target.ptr = ptr;
    target.rows = rows;
    target.row_stride = row_stride;
    target.feature_count = features;
    target.dtype = ps::runtime::ValueDType::BF16;
    return executor.tp_sum(target, stream, executor.tp_rank);
}

constexpr uint32_t kEagle4RoundCapacity = runtime::kEagle4TreeMaxNodes;

template <typename T>
Status allocate_contiguous(
    gpu::GpuArena& arena,
    const std::vector<std::size_t>& shape,
    gpu::Tensor& out) {
    if (shape.empty()) {
        return Status::invalid_argument("eagle4 tensor shape is empty", __FILE__, __LINE__);
    }
    std::size_t count = 1;
    for (std::size_t d : shape) {
        if (d == 0) {
            return Status::invalid_argument("eagle4 tensor shape has a zero dim", __FILE__,
                                            __LINE__);
        }
        if (std::numeric_limits<std::size_t>::max() / count < d) {
            return Status::invalid_argument("eagle4 tensor count overflow", __FILE__,
                                            __LINE__);
        }
        count *= d;
    }
    auto view = arena.allocate_aligned(count * sizeof(T), 256);
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

Status allocate_bf16(
    gpu::GpuArena& arena,
    std::initializer_list<std::size_t> shape,
    gpu::Tensor& out) {
    return allocate_contiguous<bf16_t>(arena, std::vector<std::size_t>(shape), out);
}

Status allocate_f32(
    gpu::GpuArena& arena,
    std::initializer_list<std::size_t> shape,
    gpu::Tensor& out) {
    return allocate_contiguous<float>(arena, std::vector<std::size_t>(shape), out);
}

Status allocate_i32(
    gpu::GpuArena& arena,
    std::initializer_list<std::size_t> shape,
    gpu::Tensor& out) {
    return allocate_contiguous<int32_t>(arena, std::vector<std::size_t>(shape), out);
}

Status check_status(const hipError_t err, const char* tag) {
    if (err != hipSuccess) {
        return Status::hip_error(tag, hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

Status gemm(
    const ps::weights::MatrixWeight& weight,
    const bf16_t* input,
    uint32_t input_row_stride,
    bf16_t* output,
    uint32_t output_row_stride,
    uint32_t rows,
    const char* tag,
    hipStream_t stream) {
    Status st = ps::qwen35::dflash2::dflash2_linear(weight, input, input_row_stride, output,
                                                    output_row_stride, rows, tag, stream);
    if (st.ok() || st.code() != Status::Code::unsupported) return st;
    return check_status(
        ps::kernel::launch_eagle4_gemm_fallback_bf16(
            weight.bf16.data<bf16_t>(), weight.rows, weight.cols, input, input_row_stride,
            output, output_row_stride, rows, stream),
        "eagle4 fallback gemm launch");
}

Status rmsnorm(
    const bf16_t* input,
    const bf16_t* weight,
    bf16_t* output,
    uint32_t rows,
    uint32_t features,
    float eps,
    hipStream_t stream) {
    return check_status(
        ps::kernel::launch_dflash2_rmsnorm_direct_bf16(input, weight, output, rows, features,
                                                      features, eps, stream),
        "eagle4 rmsnorm launch");
}

Status rope(
    const bf16_t* input,
    bf16_t* output,
    uint32_t rows,
    uint32_t features,
    uint32_t head_dim,
    uint32_t position_start,
    float theta,
    hipStream_t stream) {
    return check_status(
        ps::kernel::launch_dflash2_rope_bf16(input, features, output, features, rows, features,
                                             head_dim, position_start, theta, stream),
        "eagle4 rope launch");
}

Status candidate_head(
    Eagle4Executor& executor,
    const bf16_t* hidden,
    uint32_t hidden_row_stride,
    uint32_t rows,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    float* out_probs,
    hipStream_t stream);

struct TreeStepMeta {
    uint32_t position = 0u;
    uint32_t committed_end = 0u;
    const int32_t* row_nodes = nullptr;
    const int32_t* ancestor_offsets = nullptr;
    const int32_t* ancestor_nodes = nullptr;
    bf16_t* round_k = nullptr;
    bf16_t* round_v = nullptr;
    uint32_t round_capacity = 0u;
};

Status forward_rows(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const int32_t* token_ids,
    const bf16_t* block_hidden_in,
    uint32_t rows,
    uint32_t position_start,
    bf16_t* out_summed,
    bf16_t* out_normed,
    hipStream_t stream,
    const TreeStepMeta* tree = nullptr) {
    const Eagle4Config& config = *executor.config;
    const Eagle4Weights& weights = *executor.weights;
    const Eagle4LayerWeights& layer = weights.layers[0];
    const uint32_t hidden_size = static_cast<uint32_t>(config.hidden_size);
    const uint32_t intermediate = static_cast<uint32_t>(config.intermediate_size);
    const uint32_t head_dim = static_cast<uint32_t>(config.head_dim);
    const uint32_t heads = static_cast<uint32_t>(config.num_attention_heads);
    const uint32_t two_hidden = hidden_size * 2u;
    const bool tp_active = executor.tp_size > 1u;
    const uint32_t local_heads = tp_active ? heads / executor.tp_size : heads;
    const uint32_t local_hidden = local_heads * head_dim;
    const uint32_t local_intermediate =
        tp_active ? intermediate / executor.tp_size : intermediate;
    const float eps = config.rms_norm_eps;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const uint32_t embed_local_vocab =
        static_cast<uint32_t>(executor.embed_tokens->rows);
    const uint32_t embed_vocab_offset = tp_active
                                            ? executor.tp_rank * embed_local_vocab
                                            : 0u;
    const uint32_t embed_partial = tp_active ? 1u : 0u;
    Status st = check_status(
        ps::kernel::launch_embedding_bf16(
            executor.embed_tokens->bf16.data<bf16_t>(), token_ids,
            executor.embed_out.data<bf16_t>(), rows, embed_local_vocab, hidden_size,
            hidden_size, executor.error_word.data<uint32_t>(), stream,
            embed_vocab_offset, embed_partial),
        "eagle4 embedding launch");
    if (!st.ok()) return st;
    st = tp_sum_bf16(executor, executor.embed_out.data<bf16_t>(), rows, hidden_size,
                     hidden_size, stream);
    if (!st.ok()) return st;
    st = check_status(
        ps::kernel::launch_eagle4_layer_input_norm_bf16(
            executor.embed_out.data<bf16_t>(), block_hidden_in,
            layer.input_layernorm_weight.data<bf16_t>(),
            layer.hidden_norm_weight.data<bf16_t>(), hidden_size, rows,
            executor.layer_input.data<bf16_t>(), eps, stream),
        "eagle4 layer input norm launch");
    if (!st.ok()) return st;

    bf16_t* q_raw = executor.q_raw.data<bf16_t>();
    bf16_t* k_raw = executor.k_raw.data<bf16_t>();
    bf16_t* v_raw = executor.v_raw.data<bf16_t>();
    st = gemm(layer.attn_q_proj, executor.layer_input.data<bf16_t>(), two_hidden, q_raw,
              local_hidden, rows, "eagle4 q_proj", stream);
    if (!st.ok()) return st;
    st = gemm(layer.attn_k_proj, executor.layer_input.data<bf16_t>(), two_hidden, k_raw,
              local_hidden, rows, "eagle4 k_proj", stream);
    if (!st.ok()) return st;
    st = gemm(layer.attn_v_proj, executor.layer_input.data<bf16_t>(), two_hidden, v_raw,
              local_hidden, rows, "eagle4 v_proj", stream);
    if (!st.ok()) return st;

    if (tree == nullptr) {
        st = rope(q_raw, executor.q_rope.data<bf16_t>(), rows, local_hidden, head_dim,
                  position_start, config.rope_theta, stream);
        if (!st.ok()) return st;
        st = rope(k_raw, executor.k_rope.data<bf16_t>(), rows, local_hidden, head_dim,
                  position_start, config.rope_theta, stream);
        if (!st.ok()) return st;
    } else {
        st = check_status(
            ps::kernel::launch_eagle4_rope_uniform_bf16(
                q_raw, local_hidden, executor.q_rope.data<bf16_t>(), local_hidden, rows,
                head_dim, tree->position, config.rope_theta, stream),
            "eagle4 tree rope q launch");
        if (!st.ok()) return st;
        st = check_status(
            ps::kernel::launch_eagle4_rope_uniform_bf16(
                k_raw, local_hidden, executor.k_rope.data<bf16_t>(), local_hidden, rows,
                head_dim, tree->position, config.rope_theta, stream),
            "eagle4 tree rope k launch");
        if (!st.ok()) return st;
    }

    if (tree == nullptr) {
        st = check_status(
            ps::kernel::launch_dflash2_kv_ring_append_bf16(
                executor.k_rope.data<bf16_t>(), v_raw, context.k_ring.data<bf16_t>(),
                context.v_ring.data<bf16_t>(), rows, context.kv_features, context.capacity,
                position_start, stream),
            "eagle4 kv ring append launch");
        if (!st.ok()) return st;
    } else {
        st = check_status(
            ps::kernel::launch_eagle4_round_kv_append_bf16(
                executor.k_rope.data<bf16_t>(), v_raw, tree->round_k, tree->round_v, rows,
                context.kv_features, tree->round_capacity, tree->row_nodes, stream),
            "eagle4 round kv append launch");
        if (!st.ok()) return st;
    }

    if (tree == nullptr) {
        st = check_status(
            ps::kernel::launch_eagle4_ring_attention_bf16(
                executor.q_rope.data<bf16_t>(), local_hidden, context.k_ring.data<bf16_t>(),
                context.v_ring.data<bf16_t>(), context.capacity, context.kv_features,
                position_start, rows, local_heads, head_dim,
                static_cast<uint32_t>(config.sliding_window), scale,
                executor.attn_out.data<bf16_t>(), local_hidden, stream),
            "eagle4 ring attention launch");
        if (!st.ok()) return st;
    } else {
        st = check_status(
            ps::kernel::launch_eagle4_tree_attention_bf16(
                executor.q_rope.data<bf16_t>(), local_hidden, context.k_ring.data<bf16_t>(),
                context.v_ring.data<bf16_t>(), context.capacity, context.kv_features,
                tree->round_k, tree->round_v, tree->round_capacity, tree->committed_end,
                tree->position, rows, local_heads, head_dim,
                static_cast<uint32_t>(config.sliding_window), scale,
                tree->ancestor_offsets, tree->ancestor_nodes,
                executor.attn_out.data<bf16_t>(), local_hidden, stream),
            "eagle4 tree attention launch");
        if (!st.ok()) return st;
    }

    st = gemm(layer.attn_o_proj, executor.attn_out.data<bf16_t>(), local_hidden,
              executor.attn_o.data<bf16_t>(), hidden_size, rows, "eagle4 o_proj", stream);
    if (!st.ok()) return st;
    st = tp_sum_bf16(executor, executor.attn_o.data<bf16_t>(), rows, hidden_size,
                     hidden_size, stream);
    if (!st.ok()) return st;

    bf16_t* residual = executor.residual.data<bf16_t>();
    st = check_status(
        ps::kernel::launch_eagle4_add_bf16(block_hidden_in, executor.attn_o.data<bf16_t>(),
                                           residual,
                                           static_cast<std::size_t>(rows) * hidden_size,
                                           stream),
        "eagle4 attention residual launch");
    if (!st.ok()) return st;

    st = rmsnorm(residual, layer.post_attention_layernorm_weight.data<bf16_t>(),
                 executor.post_norm.data<bf16_t>(), rows, hidden_size, eps, stream);
    if (!st.ok()) return st;

    bf16_t* gate = executor.mlp_gate.data<bf16_t>();
    bf16_t* up = executor.mlp_up.data<bf16_t>();
    st = gemm(layer.mlp_gate_proj, executor.post_norm.data<bf16_t>(), hidden_size, gate,
              local_intermediate, rows, "eagle4 mlp gate", stream);
    if (!st.ok()) return st;
    st = gemm(layer.mlp_up_proj, executor.post_norm.data<bf16_t>(), hidden_size, up,
              local_intermediate, rows, "eagle4 mlp up", stream);
    if (!st.ok()) return st;
    st = check_status(
        ps::kernel::launch_dflash2_swiglu_bf16(gate, local_intermediate, up,
                                               local_intermediate,
                                               executor.mlp_swiglu.data<bf16_t>(),
                                               local_intermediate, rows,
                                               local_intermediate, stream),
        "eagle4 swiglu launch");
    if (!st.ok()) return st;
    st = gemm(layer.mlp_down_proj, executor.mlp_swiglu.data<bf16_t>(), local_intermediate,
              executor.mlp_down.data<bf16_t>(), hidden_size, rows, "eagle4 mlp down",
              stream);
    if (!st.ok()) return st;
    st = tp_sum_bf16(executor, executor.mlp_down.data<bf16_t>(), rows, hidden_size,
                     hidden_size, stream);
    if (!st.ok()) return st;

    st = check_status(
        ps::kernel::launch_eagle4_add_bf16(residual, executor.mlp_down.data<bf16_t>(),
                                           out_summed,
                                           static_cast<std::size_t>(rows) * hidden_size,
                                           stream),
        "eagle4 mlp residual launch");
    if (!st.ok()) return st;
    return rmsnorm(out_summed, weights.final_norm_weight.data<bf16_t>(), out_normed, rows,
                   hidden_size, eps, stream);
}

Status candidate_head(
    Eagle4Executor& executor,
    const bf16_t* hidden,
    uint32_t hidden_row_stride,
    uint32_t rows,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    float* out_probs,
    hipStream_t stream) {
    const Eagle4Config& config = *executor.config;
    const uint32_t local_vocab = static_cast<uint32_t>(executor.lm_head->rows);
    if (rows == 0u || rows > executor.max_candidate_rows) {
        return Status::invalid_argument("eagle4 candidate head rows out of range", __FILE__,
                                        __LINE__);
    }
    if (!executor.use_lowrank_candidate_head) {
        if (k != 1u) {
            return Status::unsupported(
                "eagle4 gemv candidate head supports only one draft per row", __FILE__,
                __LINE__);
        }
        Status st = gemm(*executor.lm_head, hidden, hidden_row_stride,
                         executor.gemv_logits.data<bf16_t>(), local_vocab, rows,
                         "eagle4 lm_head", stream);
        if (!st.ok()) return st;
        st = check_status(
            ps::kernel::launch_eagle4_argmax_bf16(executor.gemv_logits.data<bf16_t>(), rows,
                                                  local_vocab,
                                                  executor.gemv_ids.data<int32_t>(), stream),
            "eagle4 argmax launch");
        if (!st.ok()) return st;
        const hipError_t err = hipMemcpyAsync(out_ids, executor.gemv_ids.data<int32_t>(),
                                              static_cast<std::size_t>(rows) * sizeof(int32_t),
                                              hipMemcpyDeviceToDevice, stream);
        return check_status(err, "eagle4 argmax relay");
    }

    const uint32_t vocab = local_vocab;
    const uint32_t rank = static_cast<uint32_t>(config.candidate_rank);
    const uint32_t candidates = executor.num_candidates;
    Status st = check_status(
        ps::kernel::launch_eagle4_rank_project_bf16(
            hidden, hidden_row_stride,
            executor.weights->candidate.efficient_lm_head_down_proj.bf16.data<bf16_t>(),
            executor.projected.data<bf16_t>(), rows, rank, hidden_row_stride, stream),
        "eagle4 rank project launch");
    if (!st.ok()) return st;
    st = check_status(
        ps::kernel::launch_eagle4_rough_scores_bf16(
            executor.projected.data<bf16_t>(), rank,
            executor.weights->candidate.efficient_lm_head.bf16.data<bf16_t>(),
            executor.rough_scores.data<float>(), rows, rank, vocab, stream),
        "eagle4 rough scores launch");
    if (!st.ok()) return st;

    st = check_status(
        ps::kernel::launch_dflash2_radix_topn(
            executor.rough_scores.data<float>(), rows, vocab, vocab, candidates,
            executor.candidate_ids.data<int32_t>(), executor.candidate_scratch_logits.data<float>(),
            ps::kernel::dflash2_radix_default_partitions(vocab),
            executor.radix_scratch.data<uint8_t>(), executor.radix_scratch.allocation_bytes(), stream),
        "eagle4 candidate selection launch");
    if (!st.ok()) return st;

    st = check_status(
        ps::kernel::launch_eagle4_candidate_logits(
            executor.lm_head->bf16.data<bf16_t>(),
            static_cast<uint32_t>(config.hidden_size), hidden, hidden_row_stride,
            executor.candidate_ids.data<int32_t>(), rows, candidates,
            static_cast<uint32_t>(config.hidden_size),
            executor.candidate_logits.data<float>(), stream),
        "eagle4 candidate logits launch");
    if (!st.ok()) return st;
    return check_status(
        ps::kernel::launch_eagle4_candidate_topk(
            executor.candidate_logits.data<float>(), executor.candidate_ids.data<int32_t>(),
            rows, candidates, k, out_ids, out_logits, out_probs, stream),
        "eagle4 candidate topk launch");
}

Status copy_row(
    const bf16_t* source,
    bf16_t* destination,
    uint32_t features,
    hipStream_t stream) {
    const hipError_t err = hipMemcpyAsync(destination, source,
                                          static_cast<std::size_t>(features) * sizeof(bf16_t),
                                          hipMemcpyDeviceToDevice, stream);
    return check_status(err, "eagle4 row copy");
}

}  // namespace

Result<Eagle4Executor> create_eagle4_executor(
    const Eagle4Config& config,
    const Eagle4Weights& weights,
    gpu::GpuArena& arena,
    const Eagle4ExecutorConfig& exec_config) {
    if (!weights.present) {
        return Status::invalid_argument("eagle4 weights are not loaded", __FILE__, __LINE__);
    }
    if (weights.layers.size() != config.num_hidden_layers) {
        return Status::invalid_argument("eagle4 layer count mismatch", __FILE__, __LINE__);
    }
    if (config.num_aux_layers != kEagle4AuxLayers) {
        return Status::invalid_argument("eagle4 requires three auxiliary hidden states",
                                        __FILE__, __LINE__);
    }
    if (config.head_dim != ps::kernel::kEagle4AttentionHeadDim) {
        return Status::unsupported("eagle4 draft head_dim must be 128", __FILE__, __LINE__);
    }
    if (config.num_attention_heads != config.num_key_value_heads) {
        return Status::unsupported("eagle4 draft attention must be MHA", __FILE__, __LINE__);
    }
    if (config.sliding_window == 0 || config.sliding_window > 4096u) {
        return Status::invalid_argument("eagle4 sliding_window out of range", __FILE__,
                                        __LINE__);
    }
    const uint32_t max_rows =
        exec_config.max_rows == 0u ? kEagle4ExtendRowsMax : exec_config.max_rows;
    if (max_rows > kEagle4ExtendRowsMax) {
        return Status::invalid_argument("eagle4 max_rows exceeds the supported bound",
                                        __FILE__, __LINE__);
    }
    const uint32_t candidate_rows = exec_config.max_candidate_rows == 0u
                                        ? kEagle4CandidateRowsMax
                                        : exec_config.max_candidate_rows;
    if (candidate_rows > kEagle4CandidateRowsMax || candidate_rows > max_rows) {
        return Status::invalid_argument("eagle4 max_candidate_rows out of range", __FILE__,
                                        __LINE__);
    }
    const uint32_t num_candidates = static_cast<uint32_t>(config.num_candidates);
    if (num_candidates == 0u || num_candidates > ps::kernel::kDflash2RadixTopnMaxPool) {
        return Status::invalid_argument("eagle4 num_candidates out of range", __FILE__,
                                        __LINE__);
    }

    Eagle4Executor ex;
    ex.config = &config;
    ex.weights = &weights;
    ex.max_rows = max_rows;
    ex.max_candidate_rows = candidate_rows;
    ex.num_candidates = num_candidates;
    const ps::weights::MatrixWeight* embed_tokens =
        exec_config.target_embed_tokens != nullptr ? exec_config.target_embed_tokens
                                                   : &weights.embed_tokens;
    const ps::weights::MatrixWeight* lm_head =
        exec_config.target_lm_head != nullptr ? exec_config.target_lm_head
                                              : &weights.candidate.lm_head;
    const std::size_t local_vocab =
        exec_config.tp_size > 1u ? config.vocab_size / exec_config.tp_size
                                 : config.vocab_size;
    if ((embed_tokens->rows != local_vocab &&
         embed_tokens->rows != config.vocab_size) ||
        embed_tokens->cols != config.hidden_size) {
        return Status::invalid_argument("eagle4 embed_tokens shape mismatch", __FILE__,
                                        __LINE__);
    }
    if ((lm_head->rows != local_vocab && lm_head->rows != config.vocab_size) ||
        lm_head->cols != config.hidden_size) {
        return Status::invalid_argument("eagle4 lm_head shape mismatch", __FILE__,
                                        __LINE__);
    }
    if (embed_tokens->encoding != ps::weights::MatrixEncoding::Bf16 ||
        lm_head->encoding != ps::weights::MatrixEncoding::Bf16) {
        return Status::unsupported("eagle4 requires BF16 embed_tokens and lm_head",
                                   __FILE__, __LINE__);
    }
    ex.embed_tokens = embed_tokens;
    ex.lm_head = lm_head;
    ex.use_lowrank_candidate_head = exec_config.use_lowrank_candidate_head;
    ex.max_root_candidates = num_candidates;
    ex.tp_rank = exec_config.tp_rank;
    ex.tp_size = exec_config.tp_size;
    ex.tp_sum = exec_config.tp_sum;
    {
        int current_device = -1;
        Status device_status =
            check_status(hipGetDevice(&current_device), "eagle4 executor hipGetDevice");
        if (!device_status.ok()) return device_status;
        ex.device = current_device;
    }

    const std::size_t rows = max_rows;
    const std::size_t aux_rows = candidate_rows;
    const std::size_t h = config.hidden_size;
    const std::size_t i = config.intermediate_size;
    const std::size_t v = config.vocab_size;
    const std::size_t rank = config.candidate_rank;
    const std::size_t taps = config.tap_feature_size();

    enum class Kind : uint8_t { Bf16, F32, I32 };
    struct Allocation {
        gpu::Tensor* target;
        std::size_t rows;
        std::size_t features;
        Kind kind;
    };
    Allocation bfs[] = {
        {&ex.tap_concat, rows, taps, Kind::Bf16},
        {&ex.fc_out, rows, h, Kind::Bf16},
        {&ex.embed_out, rows, h, Kind::Bf16},
        {&ex.embed_norm, rows, h, Kind::Bf16},
        {&ex.hidden_norm, rows, h, Kind::Bf16},
        {&ex.layer_input, rows, h * 2u, Kind::Bf16},
        {&ex.q_raw, rows, h, Kind::Bf16},
        {&ex.q_rope, rows, h, Kind::Bf16},
        {&ex.k_raw, rows, h, Kind::Bf16},
        {&ex.k_rope, rows, h, Kind::Bf16},
        {&ex.v_raw, rows, h, Kind::Bf16},
        {&ex.attn_out, rows, h, Kind::Bf16},
        {&ex.attn_o, rows, h, Kind::Bf16},
        {&ex.residual, rows, h, Kind::Bf16},
        {&ex.post_norm, rows, h, Kind::Bf16},
        {&ex.mlp_gate, rows, i, Kind::Bf16},
        {&ex.mlp_up, rows, i, Kind::Bf16},
        {&ex.mlp_swiglu, rows, i, Kind::Bf16},
        {&ex.mlp_down, rows, h, Kind::Bf16},
        {&ex.summed, rows, h, Kind::Bf16},
        {&ex.normed, rows, h, Kind::Bf16},
        {&ex.tree_hidden_in, rows, h, Kind::Bf16},
        {&ex.root_hidden, 1u, h, Kind::Bf16},
        {&ex.step_hidden_a, 1u, h, Kind::Bf16},
        {&ex.step_hidden_b, 1u, h, Kind::Bf16},
        {&ex.projected, aux_rows, rank, Kind::Bf16},
        {&ex.candidate_ids, aux_rows, num_candidates, Kind::I32},
        {&ex.candidate_logits, aux_rows, num_candidates, Kind::F32},
        {&ex.candidate_scratch_ids, aux_rows, num_candidates, Kind::I32},
        {&ex.candidate_scratch_logits, aux_rows, num_candidates, Kind::F32},
        {&ex.topk_ids, aux_rows, ps::kernel::kEagle4CandidateMaxK, Kind::I32},
        {&ex.topk_logits, aux_rows, ps::kernel::kEagle4CandidateMaxK, Kind::F32},
        {&ex.topk_probs, aux_rows, ps::kernel::kEagle4CandidateMaxK, Kind::F32},
    };
    for (const Allocation& a : bfs) {
        Status st = a.kind == Kind::F32
                        ? allocate_f32(arena, {a.rows, a.features}, *a.target)
                        : (a.kind == Kind::I32
                               ? allocate_i32(arena, {a.rows, a.features}, *a.target)
                               : allocate_bf16(arena, {a.rows, a.features}, *a.target));
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
    }
    {
        const std::size_t kv_features = config.attention_kv_rows();
        Status st = allocate_bf16(arena, {kEagle4RoundCapacity, kv_features}, ex.round_k);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_bf16(arena, {kEagle4RoundCapacity, kv_features}, ex.round_v);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
    }
    {
        Status st = allocate_f32(arena, {aux_rows, v}, ex.rough_scores);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        const std::size_t scratch = ps::kernel::dflash2_radix_topn_scratch_bytes(
            static_cast<uint32_t>(aux_rows),
            ps::kernel::dflash2_radix_default_partitions(static_cast<uint32_t>(v)),
            static_cast<uint32_t>(v));
        auto view = arena.allocate_aligned(scratch, 256u);
        if (!view.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return view.status();
        }
        auto scratch_tensor = gpu::Tensor::create<uint8_t>(view.release(), {scratch}, {1});
        if (!scratch_tensor.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return scratch_tensor.status();
        }
        ex.radix_scratch = scratch_tensor.release();
        st = allocate_bf16(arena, {aux_rows, v}, ex.gemv_logits);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {aux_rows, 1u}, ex.gemv_ids);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {rows, 1u}, ex.token_ids);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {rows, 1u}, ex.draft_tokens);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {aux_rows, 1u}, ex.tree_tokens);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {aux_rows, 1u}, ex.tree_row_nodes);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {aux_rows, 1u}, ex.tree_src_rows);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {aux_rows + 1u, 1u}, ex.tree_ancestor_offsets);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(
            arena,
            {aux_rows * static_cast<std::size_t>(kEagle4RoundCapacity), 1u},
            ex.tree_ancestor_nodes);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {aux_rows, 1u}, ex.tap_rows);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_bf16(arena,
                           {kEagle4AuxLayers * static_cast<std::size_t>(aux_rows),
                            config.target_hidden_size},
                           ex.gathered_taps);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
        st = allocate_i32(arena, {1u, 1u}, ex.error_word);
        if (!st.ok()) {
            (void)eagle4_executor_shutdown(ex);
            return st;
        }
    }

    const char* head_env = std::getenv("PHASESHIFT_EAGLE4_CANDIDATE_HEAD");
    if (head_env != nullptr && std::strcmp(head_env, "gemv") == 0) {
        ex.use_lowrank_candidate_head = false;
    }

    ex.initialized = true;
    return ex;
}

Status eagle4_executor_shutdown(Eagle4Executor& executor) noexcept {
    gpu::Tensor* tensors[] = {
        &executor.token_ids,   &executor.tap_concat,  &executor.fc_out,
        &executor.embed_out,   &executor.embed_norm,  &executor.hidden_norm,
        &executor.layer_input, &executor.q_raw,       &executor.q_rope,
        &executor.k_raw,       &executor.k_rope,      &executor.v_raw,
        &executor.attn_out,    &executor.attn_o,      &executor.residual,
        &executor.post_norm,   &executor.mlp_gate,    &executor.mlp_up,
        &executor.mlp_swiglu,  &executor.mlp_down,    &executor.summed,
        &executor.normed,      &executor.root_hidden, &executor.step_hidden_a,
        &executor.step_hidden_b, &executor.tree_hidden_in, &executor.round_k,
        &executor.round_v,     &executor.projected, &executor.rough_scores,
        &executor.candidate_ids, &executor.candidate_logits,
        &executor.candidate_scratch_ids, &executor.candidate_scratch_logits,
        &executor.radix_scratch, &executor.topk_ids, &executor.topk_logits,
        &executor.topk_probs,  &executor.gemv_logits, &executor.gemv_ids,
        &executor.error_word,  &executor.draft_tokens, &executor.tree_tokens,
        &executor.tree_row_nodes, &executor.tree_src_rows,
        &executor.tree_ancestor_offsets, &executor.tree_ancestor_nodes,
        &executor.tap_rows,    &executor.gathered_taps,
    };
    for (gpu::Tensor* t : tensors) *t = gpu::Tensor{};
    executor.embed_tokens = nullptr;
    executor.lm_head = nullptr;
    executor.config = nullptr;
    executor.weights = nullptr;
    executor.max_rows = 0;
    executor.max_candidate_rows = 0;
    executor.num_candidates = 0;
    executor.device = -1;
    executor.initialized = false;
    return Status::make_ok();
}

Status eagle4_extend_rows(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const bf16_t* const* taps,
    uint32_t tap_count,
    const int32_t* token_ids,
    uint32_t rows,
    uint32_t position_start,
    uint32_t num_root_candidates,
    int32_t* root_tokens,
    hipStream_t stream) {
    if (!executor.initialized) {
        return Status::invalid_state("eagle4 extend: executor not initialized", __FILE__,
                                     __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state("eagle4 extend: context not initialized", __FILE__,
                                     __LINE__);
    }
    if (rows == 0u || rows > executor.max_rows) {
        return Status::invalid_argument("eagle4 extend rows out of range", __FILE__,
                                        __LINE__);
    }
    if (tap_count != executor.config->num_aux_layers || taps == nullptr) {
        return Status::invalid_argument("eagle4 extend tap count mismatch", __FILE__,
                                        __LINE__);
    }
    if (num_root_candidates == 0u || num_root_candidates > executor.num_candidates) {
        return Status::invalid_argument("eagle4 extend root candidates out of range",
                                        __FILE__, __LINE__);
    }
    if (root_tokens == nullptr) {
        return Status::invalid_argument("eagle4 extend root token buffer is null", __FILE__,
                                        __LINE__);
    }

    const uint32_t hidden_size = static_cast<uint32_t>(executor.config->hidden_size);
    const uint32_t tap_features = static_cast<uint32_t>(executor.config->target_hidden_size);
    Status st = check_status(
        ps::kernel::launch_eagle4_tap_norm_concat_bf16(
            taps[0], taps[1], taps[2],
            executor.weights->fc_norm_weight[0].data<bf16_t>(),
            executor.weights->fc_norm_weight[1].data<bf16_t>(),
            executor.weights->fc_norm_weight[2].data<bf16_t>(), tap_features, rows,
            executor.tap_concat.data<bf16_t>(), executor.config->rms_norm_eps, stream),
        "eagle4 tap norm concat launch");
    if (!st.ok()) return st;
    st = gemm(executor.weights->fc, executor.tap_concat.data<bf16_t>(),
              static_cast<uint32_t>(executor.config->tap_feature_size()),
              executor.fc_out.data<bf16_t>(), hidden_size, rows, "eagle4 fc", stream);
    if (!st.ok()) return st;

    st = forward_rows(executor, context, token_ids, executor.fc_out.data<bf16_t>(), rows,
                      position_start, executor.summed.data<bf16_t>(),
                      executor.normed.data<bf16_t>(), stream);
    if (!st.ok()) return st;

    const bf16_t* next_hidden = executor.config->norm_output
                                    ? executor.normed.data<bf16_t>()
                                    : executor.summed.data<bf16_t>();
    st = copy_row(next_hidden + static_cast<std::size_t>(rows - 1u) * hidden_size,
                  executor.root_hidden.data<bf16_t>(), hidden_size, stream);
    if (!st.ok()) return st;

    st = candidate_head(executor,
                        executor.normed.data<bf16_t>() +
                            static_cast<std::size_t>(rows - 1u) * hidden_size,
                        hidden_size, 1u, num_root_candidates, root_tokens,
                        executor.topk_logits.data<float>(), executor.topk_probs.data<float>(),
                        stream);
    if (!st.ok()) return st;

    context.next_position = position_start + rows;

    return Status::make_ok();
}

Status eagle4_draft_steps(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    uint32_t first_position,
    uint32_t num_drafts,
    int32_t* drafts,
    hipStream_t stream,
    int32_t* step_out) {
    if (!executor.initialized || !context.initialized) {
        return Status::invalid_state("eagle4 draft steps: not initialized", __FILE__,
                                     __LINE__);
    }
    if (num_drafts == 0u) {
        return Status::invalid_argument("eagle4 draft steps: no drafts", __FILE__, __LINE__);
    }
    if (drafts == nullptr) {
        return Status::invalid_argument("eagle4 draft steps: draft buffer is null", __FILE__,
                                        __LINE__);
    }
    const uint32_t hidden_size = static_cast<uint32_t>(executor.config->hidden_size);

    for (uint32_t step = 0; step + 1u < num_drafts; ++step) {
        const uint32_t position = first_position + step;
        bf16_t* hidden_out = (step % 2u) == 0u ? executor.step_hidden_a.data<bf16_t>()
                                               : executor.step_hidden_b.data<bf16_t>();
        const bf16_t* hidden_in = step == 0u ? executor.root_hidden.data<bf16_t>()
                                             : ((step % 2u) == 1u
                                                    ? executor.step_hidden_a.data<bf16_t>()
                                                    : executor.step_hidden_b.data<bf16_t>());
        Status st = forward_rows(executor, context, drafts + step, hidden_in, 1u, position,
                                 hidden_out, executor.normed.data<bf16_t>(), stream);
        if (!st.ok()) return st;
        int32_t* out_ptr = step_out != nullptr ? step_out + step : drafts + step + 1u;
        st = candidate_head(executor, executor.normed.data<bf16_t>(), hidden_size, 1u, 1u,
                            out_ptr, executor.topk_logits.data<float>(),
                            executor.topk_probs.data<float>(), stream);
        if (!st.ok()) return st;
        if (executor.config->norm_output) {
            st = copy_row(executor.normed.data<bf16_t>(), hidden_out, hidden_size, stream);
            if (!st.ok()) return st;
        }
    }
    return Status::make_ok();
}

Status eagle4_tree_draft_step(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const int32_t* tokens_device,
    const int32_t* row_nodes_device,
    const int32_t* src_rows_device,
    bool src_is_root,
    const int32_t* ancestor_offsets_device,
    const int32_t* ancestor_nodes_device,
    uint32_t rows,
    uint32_t position,
    uint32_t committed_end,
    uint32_t candidate_k,
    hipStream_t stream) {
    if (!executor.initialized || !context.initialized) {
        return Status::invalid_state("eagle4 tree draft step: not initialized", __FILE__,
                                     __LINE__);
    }
    if (tokens_device == nullptr || row_nodes_device == nullptr ||
        src_rows_device == nullptr || ancestor_offsets_device == nullptr ||
        ancestor_nodes_device == nullptr) {
        return Status::invalid_argument("eagle4 tree draft step: input buffer is null",
                                        __FILE__, __LINE__);
    }
    if (rows == 0u || rows > executor.max_candidate_rows || rows > executor.max_rows) {
        return Status::invalid_argument("eagle4 tree draft step: rows out of range", __FILE__,
                                        __LINE__);
    }
    if (candidate_k == 0u || candidate_k > ps::kernel::kEagle4CandidateMaxK) {
        return Status::invalid_argument("eagle4 tree draft step: candidate k out of range",
                                        __FILE__, __LINE__);
    }
    if (position < committed_end) {
        return Status::invalid_argument("eagle4 tree draft step: position before committed end",
                                        __FILE__, __LINE__);
    }
    const uint32_t hidden_size = static_cast<uint32_t>(executor.config->hidden_size);
    const bf16_t* src = nullptr;
    if (src_is_root) {
        src = executor.root_hidden.data<bf16_t>();
    } else if (executor.config->norm_output) {
        src = executor.normed.data<bf16_t>();
    } else {
        src = executor.summed.data<bf16_t>();
    }
    Status st = check_status(
        ps::kernel::launch_eagle4_gather_rows_bf16(
            src, hidden_size, executor.tree_hidden_in.data<bf16_t>(), hidden_size,
            src_rows_device, rows, hidden_size, stream),
        "eagle4 tree hidden gather launch");
    if (!st.ok()) return st;

    TreeStepMeta tree;
    tree.position = position;
    tree.committed_end = committed_end;
    tree.row_nodes = row_nodes_device;
    tree.ancestor_offsets = ancestor_offsets_device;
    tree.ancestor_nodes = ancestor_nodes_device;
    tree.round_k = executor.round_k.data<bf16_t>();
    tree.round_v = executor.round_v.data<bf16_t>();
    tree.round_capacity = kEagle4RoundCapacity;

    st = forward_rows(executor, context, tokens_device,
                      executor.tree_hidden_in.data<bf16_t>(), rows, position,
                      executor.summed.data<bf16_t>(), executor.normed.data<bf16_t>(), stream,
                      &tree);
    if (!st.ok()) return st;

    return candidate_head(executor, executor.normed.data<bf16_t>(), hidden_size, rows,
                          candidate_k, executor.topk_ids.data<int32_t>(),
                          executor.topk_logits.data<float>(),
                          executor.topk_probs.data<float>(), stream);
}

Status eagle4_forward_rows(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const int32_t* token_ids,
    const bf16_t* const* taps,
    const bf16_t* hidden_in,
    uint32_t rows,
    uint32_t position_start,
    bf16_t* normed_out,
    bf16_t* summed_out,
    hipStream_t stream) {
    if (!executor.initialized || !context.initialized) {
        return Status::invalid_state("eagle4 forward rows: not initialized", __FILE__,
                                     __LINE__);
    }
    if (rows == 0u || rows > executor.max_rows) {
        return Status::invalid_argument("eagle4 forward rows out of range", __FILE__,
                                        __LINE__);
    }
    const uint32_t hidden_size = static_cast<uint32_t>(executor.config->hidden_size);
    const uint32_t tap_features = static_cast<uint32_t>(executor.config->target_hidden_size);
    if (hidden_in == nullptr) {
        Status st = check_status(
            ps::kernel::launch_eagle4_tap_norm_concat_bf16(
                taps[0], taps[1], taps[2],
                executor.weights->fc_norm_weight[0].data<bf16_t>(),
                executor.weights->fc_norm_weight[1].data<bf16_t>(),
                executor.weights->fc_norm_weight[2].data<bf16_t>(), tap_features, rows,
                executor.tap_concat.data<bf16_t>(), executor.config->rms_norm_eps, stream),
            "eagle4 tap norm concat launch");
        if (!st.ok()) return st;
        st = gemm(executor.weights->fc, executor.tap_concat.data<bf16_t>(),
                  static_cast<uint32_t>(executor.config->tap_feature_size()),
                  executor.fc_out.data<bf16_t>(), hidden_size, rows, "eagle4 fc", stream);
        if (!st.ok()) return st;
        hidden_in = executor.fc_out.data<bf16_t>();
    }
    return forward_rows(executor, context, token_ids, hidden_in, rows, position_start,
                        summed_out, normed_out, stream);
}

namespace {

Status eagle4_d2h(void* dst, const void* src, std::size_t bytes, const char* what) {
    const hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        return Status::hip_error(what, hipGetErrorString(err), __FILE__, __LINE__);
    }
    return Status::make_ok();
}

std::size_t tensor_element_count(const gpu::Tensor& t) {
    std::size_t count = 1;
    for (std::size_t i = 0; i < t.ndim(); ++i) count *= t.dim(i);
    return count;
}

Status upload_host_i32(gpu::Tensor& dst, const int32_t* host, uint32_t count,
                       hipStream_t stream, const char* what) {
    if (count == 0u) return Status::make_ok();
    if (host == nullptr) {
        return Status::invalid_argument("eagle4 rank staging source is null", __FILE__,
                                        __LINE__);
    }
    if (dst.ndim() == 0u || tensor_element_count(dst) < count) {
        return Status::invalid_state("eagle4 rank staging buffer is too small", __FILE__,
                                     __LINE__);
    }
    return check_status(
        hipMemcpyAsync(dst.data<int32_t>(), host,
                       static_cast<std::size_t>(count) * sizeof(std::int32_t),
                       hipMemcpyHostToDevice, stream),
        what);
}

Status merge_candidates(const std::vector<Eagle4RankDraft>& ranks, uint32_t rows, uint32_t k,
                        int32_t* out_ids, float* out_probs, int32_t* out_ids_host,
                        hipStream_t stream) {
    if (rows == 0u || k == 0u || out_ids == nullptr) {
        return Status::make_ok();
    }
    if (ranks.size() != 2u) {
        return Status::invalid_state("eagle4 candidate merge requires two ranks", __FILE__,
                                     __LINE__);
    }
    const std::size_t count = static_cast<std::size_t>(rows) * k;
    const Eagle4Executor& r0 = *ranks[0].executor;
    const Eagle4Executor& r1 = *ranks[1].executor;
    if (tensor_element_count(r0.topk_ids) < count ||
        tensor_element_count(r1.topk_ids) < count) {
        return Status::invalid_argument("eagle4 candidate merge rows out of range", __FILE__,
                                        __LINE__);
    }
    std::vector<std::int32_t> ids0(count);
    std::vector<std::int32_t> ids1(count);
    std::vector<float> log0(count);
    std::vector<float> log1(count);
    std::vector<float> probs0(out_probs != nullptr ? count : 0u);
    std::vector<float> probs1(out_probs != nullptr ? count : 0u);
    const std::int32_t local_vocab =
        r0.lm_head != nullptr ? static_cast<std::int32_t>(r0.lm_head->rows) : 0;
    for (std::size_t r = 0; r < 2u; ++r) {
        const Eagle4Executor& source = r == 0u ? r0 : r1;
        std::int32_t* ids = r == 0u ? ids0.data() : ids1.data();
        float* logits = r == 0u ? log0.data() : log1.data();
        float* probs = r == 0u ? probs0.data() : probs1.data();
        auto scope = ps::gpu::ScopedDevice::create(ranks[r].device);
        if (!scope.ok()) return scope.status();
        Status st = check_status(hipStreamSynchronize(ranks[r].stream),
                                 "eagle4 merge rank sync");
        if (!st.ok()) return st;
        st = eagle4_d2h(ids, source.topk_ids.data<std::int32_t>(),
                        count * sizeof(std::int32_t), "eagle4 merge ids");
        if (!st.ok()) return st;
        st = eagle4_d2h(logits, source.topk_logits.data<float>(), count * sizeof(float),
                        "eagle4 merge logits");
        if (!st.ok()) return st;
        if (out_probs != nullptr) {
            st = eagle4_d2h(probs, source.topk_probs.data<float>(), count * sizeof(float),
                            "eagle4 merge probs");
            if (!st.ok()) return st;
        }
    }

    struct Candidate {
        float logit;
        std::int32_t id;
        float prob;
    };
    std::vector<std::int32_t> out(count);
    std::vector<float> out_probs_h(out_probs != nullptr ? count : 0u);
    for (uint32_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * k;
        std::vector<Candidate> merged;
        merged.reserve(2u * k);
        for (uint32_t i = 0; i < k; ++i) {
            merged.push_back({log0[base + i], ids0[base + i],
                              out_probs != nullptr ? probs0[base + i] : 0.0f});
            merged.push_back({log1[base + i], ids1[base + i] + local_vocab,
                              out_probs != nullptr ? probs1[base + i] : 0.0f});
        }
        const std::size_t keep = std::min<std::size_t>(k, merged.size());
        std::partial_sort(merged.begin(), merged.begin() + keep, merged.end(),
                          [](const Candidate& a, const Candidate& b) {
                              return a.logit > b.logit;
                          });
        for (std::size_t i = 0; i < keep; ++i) {
            out[base + i] = merged[i].id;
            if (out_probs != nullptr) out_probs_h[base + i] = merged[i].prob;
        }
    }
    if (out_ids_host != nullptr) {
        for (std::size_t i = 0; i < count; ++i) out_ids_host[i] = out[i];
    }
    const hipError_t err = hipMemcpyAsync(out_ids, out.data(), count * sizeof(std::int32_t),
                                          hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        return Status::hip_error("eagle4 merge upload", hipGetErrorString(err), __FILE__,
                                 __LINE__);
    }
    if (out_probs != nullptr) {
        const hipError_t perr =
            hipMemcpyAsync(out_probs, out_probs_h.data(), count * sizeof(float),
                           hipMemcpyHostToDevice, stream);
        if (perr != hipSuccess) {
            return Status::hip_error("eagle4 merge prob upload",
                                     hipGetErrorString(perr), __FILE__, __LINE__);
        }
    }
    return Status::make_ok();
}

Status resolve_rank_taps(const Eagle4DraftJob& job, const Eagle4RankDraft& rank,
                         Eagle4Executor& executor, std::vector<const bf16_t*>& taps,
                         hipStream_t stream) {
    if (job.tap_count == 0u || job.tap_count > kEagle4AuxLayers) {
        return Status::invalid_argument("eagle4 rank job tap count out of range", __FILE__,
                                        __LINE__);
    }
    const std::size_t tap_features = executor.config->target_hidden_size;
    taps.assign(job.tap_count, nullptr);
    if (job.tap_gather_rows_host != nullptr) {
        if (rank.target == nullptr) {
            return Status::invalid_state("eagle4 rank has no target executor for tap gather",
                                         __FILE__, __LINE__);
        }
        if (job.tap_gather_count == 0u ||
            job.tap_gather_count > executor.max_candidate_rows ||
            job.tap_gather_count > kEagle4CandidateRowsMax) {
            return Status::invalid_argument("eagle4 rank tap gather rows out of range",
                                            __FILE__, __LINE__);
        }
        Status st = upload_host_i32(executor.tap_rows, job.tap_gather_rows_host,
                                    job.tap_gather_count, stream,
                                    "eagle4 rank tap rows upload");
        if (!st.ok()) return st;
        const std::size_t stage_rows = executor.max_candidate_rows;
        for (uint32_t t = 0; t < job.tap_count; ++t) {
            bf16_t* dst = executor.gathered_taps.data<bf16_t>() +
                          static_cast<std::size_t>(t) * stage_rows * tap_features;
            st = check_status(
                ps::kernel::launch_eagle4_gather_rows_bf16(
                    rank.target->dflash_target_hidden[t].data<bf16_t>(),
                    static_cast<uint32_t>(tap_features), dst,
                    static_cast<uint32_t>(tap_features),
                    executor.tap_rows.data<int32_t>(), job.tap_gather_count,
                    static_cast<uint32_t>(tap_features), stream),
                "eagle4 rank tap gather launch");
            if (!st.ok()) return st;
            taps[t] = dst;
        }
        return Status::make_ok();
    }
    if (rank.taps != nullptr) {
        if (rank.tap_count < job.tap_count) {
            return Status::invalid_argument("eagle4 rank tap count mismatch", __FILE__,
                                            __LINE__);
        }
        for (uint32_t t = 0; t < job.tap_count; ++t) taps[t] = rank.taps[t];
        return Status::make_ok();
    }
    if (rank.target != nullptr) {
        const std::size_t offset =
            static_cast<std::size_t>(job.tap_row_offset) * tap_features;
        for (uint32_t t = 0; t < job.tap_count; ++t) {
            taps[t] = rank.target->dflash_target_hidden[t].data<bf16_t>() + offset;
        }
        return Status::make_ok();
    }
    if (job.taps != nullptr) {
        const std::size_t offset =
            static_cast<std::size_t>(job.tap_row_offset) * tap_features;
        for (uint32_t t = 0; t < job.tap_count; ++t) taps[t] = job.taps[t] + offset;
        return Status::make_ok();
    }
    return Status::invalid_state("eagle4 rank has no rank-local tap source", __FILE__,
                                 __LINE__);
}

bool tp_dispatch(const Eagle4Executor& executor, const std::vector<Eagle4RankDraft>* ranks) {
    return executor.tp_size > 1u && executor.tp_hook != nullptr && ranks != nullptr &&
           ranks->size() > 1u;
}

}  // namespace

Status run_eagle4_rank_job(const Eagle4DraftJob& job, std::size_t rank) {
    if (rank >= job.ranks.size()) {
        return Status::invalid_argument("eagle4 rank job out of range", __FILE__, __LINE__);
    }
    const Eagle4RankDraft& step = job.ranks[rank];
    if (step.executor == nullptr || step.context == nullptr) {
        return Status::invalid_state("eagle4 rank job slot missing", __FILE__, __LINE__);
    }
    auto scope = ps::gpu::ScopedDevice::create(step.device);
    if (!scope.ok()) return scope.status();
    Eagle4Executor& executor = *step.executor;
    switch (job.op) {
        case Eagle4DraftJob::Op::TreeStep: {
            Status st = upload_host_i32(executor.tree_tokens, job.tokens_host, job.rows,
                                        step.stream, "eagle4 rank tree token upload");
            if (!st.ok()) return st;
            st = upload_host_i32(executor.tree_row_nodes, job.row_nodes_host, job.rows,
                                 step.stream, "eagle4 rank tree node upload");
            if (!st.ok()) return st;
            st = upload_host_i32(executor.tree_src_rows, job.src_rows_host, job.rows,
                                 step.stream, "eagle4 rank tree source row upload");
            if (!st.ok()) return st;
            st = upload_host_i32(executor.tree_ancestor_offsets, job.ancestor_offsets_host,
                                 job.ancestor_offset_count, step.stream,
                                 "eagle4 rank tree ancestor offset upload");
            if (!st.ok()) return st;
            st = upload_host_i32(executor.tree_ancestor_nodes, job.ancestor_nodes_host,
                                 job.ancestor_node_count, step.stream,
                                 "eagle4 rank tree ancestor node upload");
            if (!st.ok()) return st;
            return eagle4_tree_draft_step(
                executor, *step.context, executor.tree_tokens.data<int32_t>(),
                executor.tree_row_nodes.data<int32_t>(),
                executor.tree_src_rows.data<int32_t>(), job.src_is_root,
                executor.tree_ancestor_offsets.data<int32_t>(),
                executor.tree_ancestor_nodes.data<int32_t>(), job.rows, job.position,
                job.committed_end, job.candidate_k, step.stream);
        }
        case Eagle4DraftJob::Op::ExtendRows: {
            Status st = upload_host_i32(executor.token_ids, job.tokens_host, job.extend_rows,
                                        step.stream, "eagle4 rank token upload");
            if (!st.ok()) return st;
            std::vector<const bf16_t*> tap_ptrs;
            st = resolve_rank_taps(job, step, executor, tap_ptrs, step.stream);
            if (!st.ok()) return st;
            std::int32_t* root_out =
                job.ranks.size() > 1u || job.root_tokens == nullptr
                    ? executor.topk_ids.data<int32_t>()
                    : job.root_tokens;
            return eagle4_extend_rows(executor, *step.context, tap_ptrs.data(),
                                      job.tap_count, executor.token_ids.data<int32_t>(),
                                      job.extend_rows, job.position,
                                      job.num_root_candidates, root_out, step.stream);
        }
        case Eagle4DraftJob::Op::DraftSteps: {
            Status st = upload_host_i32(executor.draft_tokens, job.tokens_host, 1u,
                                        step.stream, "eagle4 rank draft token upload");
            if (!st.ok()) return st;
            return eagle4_draft_steps(executor, *step.context, job.first_position,
                                      job.num_drafts, executor.draft_tokens.data<int32_t>(),
                                      step.stream, executor.topk_ids.data<int32_t>());
        }
    }
    return Status::invalid_argument("eagle4 rank job op unknown", __FILE__, __LINE__);
}

Status eagle4_extend_rows_tp(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const std::vector<Eagle4RankDraft>* ranks,
    const bf16_t* const* taps,
    uint32_t tap_count,
    uint32_t tap_row_offset,
    const int32_t* token_ids_host,
    uint32_t rows,
    uint32_t position_start,
    uint32_t num_root_candidates,
    int32_t* root_tokens,
    hipStream_t stream,
    const int32_t* tap_gather_rows_host,
    uint32_t tap_gather_count) {
    if (ranks == nullptr || ranks->empty()) {
        return Status::invalid_argument("eagle4 extend rows requires rank slots", __FILE__,
                                        __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state("eagle4 extend rows requires an initialized context",
                                     __FILE__, __LINE__);
    }
    if ((*ranks)[0].executor != &executor) {
        return Status::invalid_state("eagle4 extend rows rank0 executor mismatch", __FILE__,
                                     __LINE__);
    }
    Eagle4DraftJob job;
    job.op = Eagle4DraftJob::Op::ExtendRows;
    job.ranks = *ranks;
    job.taps = taps;
    job.tap_count = tap_count;
    job.tap_row_offset = tap_row_offset;
    job.tap_gather_rows_host = tap_gather_rows_host;
    job.tap_gather_count = tap_gather_count;
    job.extend_rows = rows;
    job.position = position_start;
    job.num_root_candidates = num_root_candidates;
    job.tokens_host = token_ids_host;
    job.root_tokens = root_tokens;
    if (!tp_dispatch(executor, ranks)) {
        return run_eagle4_rank_job(job, 0u);
    }
    Status st = executor.tp_hook->on_eagle4_draft(job);
    if (!st.ok()) return st;
    return merge_candidates(job.ranks, 1u, num_root_candidates, root_tokens, nullptr, nullptr,
                            stream);
}

Status eagle4_draft_steps_tp(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const std::vector<Eagle4RankDraft>* ranks,
    uint32_t first_position,
    uint32_t num_drafts,
    int32_t* drafts,
    hipStream_t stream) {
    if (ranks == nullptr || ranks->empty()) {
        return Status::invalid_argument("eagle4 draft steps requires rank slots", __FILE__,
                                        __LINE__);
    }
    if ((*ranks)[0].executor != &executor) {
        return Status::invalid_state("eagle4 draft steps rank0 executor mismatch", __FILE__,
                                     __LINE__);
    }
    if (!tp_dispatch(executor, ranks)) {
        return eagle4_draft_steps(executor, context, first_position, num_drafts, drafts,
                                  stream);
    }
    if (drafts == nullptr) {
        return Status::invalid_argument("eagle4 draft steps output buffer is null", __FILE__,
                                        __LINE__);
    }
    std::int32_t next = 0;
    {
        auto scope = ps::gpu::ScopedDevice::create((*ranks)[0].device);
        if (!scope.ok()) return scope.status();
        Status st = check_status(hipStreamSynchronize((*ranks)[0].stream),
                                 "eagle4 draft steps seed sync");
        if (!st.ok()) return st;
        st = eagle4_d2h(&next, drafts, sizeof(next), "eagle4 draft steps seed");
        if (!st.ok()) return st;
    }
    for (uint32_t step = 0; step + 1u < num_drafts; ++step) {
        Eagle4DraftJob job;
        job.op = Eagle4DraftJob::Op::DraftSteps;
        job.ranks = *ranks;
        job.first_position = first_position + step;
        job.num_drafts = 2u;
        job.tokens_host = &next;
        job.rows = 1u;
        Status st = executor.tp_hook->on_eagle4_draft(job);
        if (!st.ok()) return st;
        st = merge_candidates(job.ranks, 1u, 1u, drafts + step + 1u, nullptr, &next, stream);
        if (!st.ok()) return st;
    }
    return Status::make_ok();
}

Status eagle4_tree_draft_step_tp(
    Eagle4Executor& executor,
    Eagle4ContextState& context,
    const std::vector<Eagle4RankDraft>* ranks,
    const int32_t* tokens_host,
    const int32_t* row_nodes_host,
    const int32_t* src_rows_host,
    bool src_is_root,
    const int32_t* ancestor_offsets_host,
    const int32_t* ancestor_nodes_host,
    uint32_t ancestor_offset_count,
    uint32_t ancestor_node_count,
    uint32_t rows,
    uint32_t position,
    uint32_t committed_end,
    uint32_t candidate_k,
    hipStream_t stream) {
    if (ranks == nullptr || ranks->empty()) {
        return Status::invalid_argument("eagle4 tree draft step requires rank slots",
                                        __FILE__, __LINE__);
    }
    if (!context.initialized) {
        return Status::invalid_state(
            "eagle4 tree draft step requires an initialized context", __FILE__, __LINE__);
    }
    if ((*ranks)[0].executor != &executor) {
        return Status::invalid_state("eagle4 tree draft step rank0 executor mismatch",
                                     __FILE__, __LINE__);
    }
    Eagle4DraftJob job;
    job.op = Eagle4DraftJob::Op::TreeStep;
    job.ranks = *ranks;
    job.tokens_host = tokens_host;
    job.row_nodes_host = row_nodes_host;
    job.src_rows_host = src_rows_host;
    job.src_is_root = src_is_root;
    job.ancestor_offsets_host = ancestor_offsets_host;
    job.ancestor_nodes_host = ancestor_nodes_host;
    job.ancestor_offset_count = ancestor_offset_count;
    job.ancestor_node_count = ancestor_node_count;
    job.rows = rows;
    job.position = position;
    job.committed_end = committed_end;
    job.candidate_k = candidate_k;
    if (!tp_dispatch(executor, ranks)) {
        return run_eagle4_rank_job(job, 0u);
    }
    Status st = executor.tp_hook->on_eagle4_draft(job);
    if (!st.ok()) return st;
    return merge_candidates(job.ranks, rows, candidate_k,
                            job.ranks[0].executor->topk_ids.data<int32_t>(),
                            job.ranks[0].executor->topk_probs.data<float>(), nullptr, stream);
}

}  // namespace ps::qwen35::eagle4
