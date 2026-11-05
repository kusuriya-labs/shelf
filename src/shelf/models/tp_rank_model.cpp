#include <shelf/models/tp_rank_model.h>

#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35moe/model/lower_to_primitives.h>
#include <shelf/models/qwen35moe/model/qwen35moe_config.h>
#include <shelf/models/qwen35moe/model/tensor_parallel_context.h>
#include <shelf/models/qwen35moe/runtime/executor_footprint.h>
#include <shelf/models/qwen35moe/weights/moe_tensor_parallel_plan.h>

#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace shelf::models {

namespace {

namespace q35 = shelf::qwen35;
namespace q35r = shelf::qwen35::runtime;
namespace qhy = shelf::qwen_hybrid;
namespace qmoe = shelf::qwen35moe;

class MoeTpRankModel final : public TpRankModel {
public:
    MoeTpRankModel(qmoe::Qwen35MoeModel&& model,
                   qhy::TensorParallelContext&& context)
        : model_(std::move(model)), context_(context) {}

    const qhy::QwenHybridConfig& hybrid_config() const override { return model_.text_config(); }
    const q35::Qwen35TensorParallelContext& context() const override { return context_; }

    std::uint64_t weight_bytes() const override {
        const qmoe::Qwen35MoeModelWeights& w = model_.weights();
        std::uint64_t total = 0;
        add_matrix_bytes(total, w.embed_tokens);
        if (!w.lm_head_tied) add_matrix_bytes(total, w.lm_head);
        for (const qmoe::Qwen35MoeLayerWeights& layer : w.layers) {
            add_matrix_bytes(total, layer.ffn.router);
            add_matrix_bytes(total, layer.ffn.shared_gate);
            add_matrix_bytes(total, layer.ffn.shared_gate_proj);
            add_matrix_bytes(total, layer.ffn.shared_up_proj);
            add_matrix_bytes(total, layer.ffn.shared_down_proj);
            add_matrix_bytes(total, layer.ffn.expert_w13);
            add_matrix_bytes(total, layer.ffn.expert_w2);
            if (layer.is_gdn) {
                add_matrix_bytes(total, layer.attn_in_proj_qkv);
                add_matrix_bytes(total, layer.attn_in_proj_z);
                add_matrix_bytes(total, layer.attn_in_proj_a);
                add_matrix_bytes(total, layer.attn_in_proj_b);
                add_matrix_bytes(total, layer.attn_out_proj);
            } else {
                add_matrix_bytes(total, layer.attn_q_proj);
                add_matrix_bytes(total, layer.attn_k_proj);
                add_matrix_bytes(total, layer.attn_v_proj);
                add_matrix_bytes(total, layer.attn_o_proj);
            }
        }
        return total;
    }

    // A model with grouped experts has no token-row FFN of the routed kind: what still sits
    // at token rows is its shared expert, and that is already this rank's local width.
    std::size_t token_row_ffn_size() const override {
        return context_.local_shared_expert_intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t max_batch_rows) const override {
        return qmoe::moe_executor_grouped_bytes(model_.text_config(), max_batch_rows,
                                                context_.tp_size);
    }

    bool uses_exl3_linear() const override { return qmoe::uses_exl3_linear(model_.weights()); }

    shelf::Result<q35::Executor> create_executor(
        q35::SequenceSlotPool& sequence_pool, q35::GdnStatePool& gdn_state_pool,
        q35::PagedKVPool& kv_pool, gpu::GpuArena& arena,
        const q35::ExecutorConfig& config) const override {
        return create_moe_model_executor(model_, sequence_pool, gdn_state_pool, kv_pool, arena,
                                         config);
    }

    const qmoe::Qwen35MoeModel* grouped_model() const override { return &model_; }

private:
    static void add_matrix_bytes(std::uint64_t& total, const weights::MatrixWeight& w) {
        if (w.encoding == weights::MatrixEncoding::Bf16) {
            total += static_cast<std::uint64_t>(w.rows) * w.cols * 2u;
            return;
        }
        const auto add = [&total](const gpu::Tensor& t) {
            if (t.ndim() == 0) return;
            std::size_t bytes = 0;
            if (!t.required_span_bytes(bytes).ok()) return;
            total += bytes;
        };
        add(w.codes);
        add(w.scales);
        add(w.compute_codes);
        add(w.compute_scales_bf16);
    }

    qmoe::Qwen35MoeModel model_;
    qhy::TensorParallelContext context_;
};

}  // namespace

Result<q35::Executor> create_moe_model_executor(
    const qmoe::Qwen35MoeModel& model,
    q35::SequenceSlotPool& sequence_pool,
    q35::GdnStatePool& gdn_state_pool,
    q35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const q35::ExecutorConfig& config) {
    std::uint32_t tp_size = 1u;
    std::uint32_t tp_rank = 0u;
    if (config.tp != nullptr) {
        tp_size = config.tp->tp_size;
        tp_rank = config.tp->tp_rank;
    }
    if (tp_size > 1u && config.tp->global_num_experts == 0u) {
        return Status::unsupported(
            "a model with grouped experts needs a tensor parallel context that states its "
            "expert range: the expert planes are not part of the matrix partition contract",
            __FILE__, __LINE__);
    }

    std::vector<q35::LayerOverrideSource> overrides =
        q35::read_layer_override_sources(static_cast<std::uint32_t>(model.text_config().hidden_size));
    std::vector<std::uint32_t> override_layers;
    override_layers.reserve(overrides.size());
    for (const auto& ov : overrides) override_layers.push_back(ov.layer);

    qmoe::Qwen35MoeLowerOptions lower_options;
    lower_options.hidden_taps = std::span<const std::uint32_t>(
        config.target_hidden_taps.data(), config.target_hidden_tap_count);
    lower_options.override_layers =
        std::span<const std::uint32_t>(override_layers.data(), override_layers.size());
    lower_options.rope_attention_factor = static_cast<float>(config.rope_profile.attention_factor);
    lower_options.emit_normed_token_hidden = config.emit_normed_token_hidden;
    lower_options.tp_size = tp_size;
    lower_options.tp_rank = tp_rank;

    auto lowered = qmoe::lower_qwen35_moe_to_primitives(model.text_config(), model.weights(),
                                                        lower_options);
    if (!lowered.ok()) return lowered.status();

    q35::ExecutorModelInput input;
    input.model.config = &model.text_config();
    input.model.embed_tokens = &model.weights().embed_tokens;
    input.lowered = lowered.release();
    input.layer_overrides = std::move(overrides);
    return q35::create_executor(std::move(input), sequence_pool, gdn_state_pool, kv_pool, arena,
                                config);
}

Result<std::unique_ptr<TpRankModel>> build_moe_tp_rank_model(
    const q35r::TpRankRuntimeConfig& config, gpu::GpuArena& arena, hipStream_t stream) {
    if (config.load_options.stage.enabled) {
        return Status::unsupported(
            "a model with grouped experts has no pipeline stage weight load", __FILE__,
            __LINE__);
    }
    if (config.load_options.load_mtp_layers && config.tp_size > 1u) {
        return Status::unsupported(
            "an MTP block is not partitioned across tensor parallel ranks", __FILE__, __LINE__);
    }

    auto config_result = qmoe::read_qwen35_moe_text_config(config.model_dir);
    if (!config_result.ok()) return config_result.status();
    const qmoe::Qwen35MoeTextConfig& text_config = config_result.value();

    // The routed experts split along the expert axis, which a matrix partition plan cannot
    // state, so the plan carries the backbone and the shared expert only. The MoE lm_head
    // stays replicated: the vocab-parallel head is implemented for the dense family.
    qmoe::Qwen35MoeLoadOptions load_options;
    static_cast<qhy::HybridLoadOptions&>(load_options) =
        static_cast<const qhy::HybridLoadOptions&>(config.load_options);
    load_options.load_mtp_layers = config.load_options.load_mtp_layers;
    // The plan outlives the load: the loader keeps a pointer to it, so it cannot be a
    // temporary inside the branch.
    std::optional<weights::TensorPartitionPlan> plan;
    if (config.tp_size > 1u) {
        auto plan_result = qmoe::build_qwen35moe_tensor_partition_plan(
            text_config, config.tp_size, config.tp_rank, false);
        if (!plan_result.ok()) return plan_result.status();
        plan.emplace(plan_result.release());
        load_options.weights.partition_plan = &plan.value();
    }

    auto model_result = qmoe::Qwen35MoeModel::load_from_safetensors(config.model_dir, arena,
                                                                    stream, load_options);
    if (!model_result.ok()) return model_result.status();

    auto context_result = qmoe::make_qwen35moe_tensor_parallel_context(text_config,
                                                                       config.tp_size,
                                                                       config.tp_rank, false);
    if (!context_result.ok()) return context_result.status();

    return std::unique_ptr<TpRankModel>(
        std::make_unique<MoeTpRankModel>(model_result.release(), context_result.release()));
}

Status select_tp_rank_model(q35r::TpCoordinatorConfig& config) {
    if (config.build_rank_model != nullptr) {
        return Status::invalid_argument("tp coordinator already has a rank model builder",
                                        __FILE__, __LINE__);
    }
    auto family = probe_model_dir(config.model_dir);
    if (!family.ok()) return family.status();
    config.build_rank_model = family.value() == ModelFamily::Moe
        ? &build_moe_tp_rank_model
        : &q35r::build_dense_tp_rank_model;
    return Status::make_ok();
}

Result<TpGroup> TpGroup::create(const q35r::TpCoordinatorConfig& config_in) {
    q35r::TpCoordinatorConfig config = config_in;
    Status selected = select_tp_rank_model(config);
    if (!selected.ok()) return selected;

    auto coordinator_result = q35r::TpCoordinator::create(config);
    if (!coordinator_result.ok()) return coordinator_result.status();

    TpGroup group;
    group.coordinator_ = coordinator_result.release();
    const q35r::TpRankRuntime& rank0 = group.coordinator_->rank(0);
    const TpRankModel* model = rank0.rank_model();
    if (model == nullptr) {
        return Status::invalid_state("tp rank 0 built no model", __FILE__, __LINE__);
    }
    if (const auto* grouped = model->grouped_model()) {
        group.model_.emplace(LoadedModel::borrow(*grouped));
    } else if (const auto* dense = model->dense_model()) {
        group.model_.emplace(LoadedModel::borrow(*dense));
    } else {
        return Status::invalid_state("tp rank 0's model names no family", __FILE__, __LINE__);
    }
    return group;
}

}  // namespace shelf::models
