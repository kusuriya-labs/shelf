#include <shelf/models/pipeline_stage_model.h>

#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35/runtime/pipeline_parallel.h>
#include <shelf/models/qwen35moe/model/qwen35moe_model.h>
#include <shelf/models/qwen35moe/runtime/executor_footprint.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>
#include <shelf/models/qwen4exp/model/qwen4exp_model.h>
#include <shelf/models/qwen4exp/runtime/ple_input.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35qsa/model/qwen35qsa_model.h>
#include <shelf/models/tp_rank_model.h>
#include <shelf/runtime/execution/row_bucket.h>
#include <shelf/runtime/graph/grouped_row_shape.h>
#include <shelf/runtime/program/moe_plan_scratch.h>
#include <shelf/weights/weight_bytes.h>

#include <memory>
#include <utility>

namespace shelf::models {

namespace {

namespace q35 = shelf::qwen35;
namespace q35r = shelf::qwen35::runtime;
namespace qhy = shelf::qwen_hybrid;
namespace qmoe = shelf::qwen35moe;

using ::shelf::weights::add_matrix_bytes;

class MoePipelineStageModel final : public PipelineStageModel {
public:
    MoePipelineStageModel(qmoe::Qwen35MoeModel&& model) : model_(std::move(model)) {}

    const qhy::QwenHybridConfig& hybrid_config() const override { return model_.text_config(); }

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

    std::uint64_t host_pinned_bytes() const override {
        return model_.weights().pinned_host.used();
    }

    // A model with grouped experts has no token-row FFN of the routed kind: what still sits
    // at token rows is its shared expert.
    std::size_t token_row_ffn_size() const override {
        return model_.text_config().ffn.shared_expert_intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t max_batch_rows) const override {
        return qmoe::moe_executor_grouped_bytes(model_.text_config(), max_batch_rows, 1u);
    }

    bool uses_exl3_linear() const override { return qmoe::uses_exl3_linear(model_.weights()); }

    Result<q35::Executor> create_executor(q35::SequenceSlotPool& sequence_pool,
                                          q35::GdnStatePool& gdn_state_pool,
                                          q35::PagedKVPool& kv_pool, gpu::GpuArena& arena,
                                          const q35::ExecutorConfig& config) const override {
        return create_moe_model_executor(model_, sequence_pool, gdn_state_pool, kv_pool, arena,
                                         config);
    }

    const qmoe::Qwen35MoeModel* grouped_model() const override { return &model_; }

private:
    qmoe::Qwen35MoeModel model_;
};

namespace q4 = shelf::qwen4exp;
namespace q35qsa = shelf::qwen35qsa;

class QsaPipelineStageModel final : public PipelineStageModel {
public:
    explicit QsaPipelineStageModel(q35qsa::Qwen35QsaModel&& model)
        : model_(std::move(model)) {}

    const qhy::QwenHybridConfig& hybrid_config() const override {
        return model_.text_config();
    }

    std::uint64_t weight_bytes() const override {
        const auto& w = model_.weights().backbone;
        std::uint64_t total = 0;
        add_matrix_bytes(total, w.embed_tokens);
        if (!w.lm_head_tied) add_matrix_bytes(total, w.lm_head);
        for (const auto& layer : w.layers) {
            add_matrix_bytes(total, layer.ffn.gate);
            add_matrix_bytes(total, layer.ffn.up);
            add_matrix_bytes(total, layer.ffn.down);
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
        for (const auto& indexer : model_.weights().indexer) {
            add_matrix_bytes(total, indexer.q_proj);
            add_matrix_bytes(total, indexer.k_proj);
        }
        return total;
    }

    std::size_t token_row_ffn_size() const override {
        return model_.text_config().ffn.intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t) const override { return 0u; }

    bool uses_exl3_linear() const override {
        return q35::uses_exl3_linear(model_.weights().backbone);
    }

    Result<q35::Executor> create_executor(q35::SequenceSlotPool& sequence_pool,
                                          q35::GdnStatePool& gdn_state_pool,
                                          q35::PagedKVPool& kv_pool, gpu::GpuArena& arena,
                                          const q35::ExecutorConfig& config) const override {
        return create_qwen35qsa_model_executor(
            model_, sequence_pool, gdn_state_pool, kv_pool, arena, config);
    }

    const q35qsa::Qwen35QsaModel* qsa_model() const override { return &model_; }

private:
    q35qsa::Qwen35QsaModel model_;
};

class Qwen4ExpPipelineStageModel final : public PipelineStageModel {
public:
    explicit Qwen4ExpPipelineStageModel(q4::Qwen4ExpModel&& model)
        : model_(std::move(model)) {}

    const qhy::QwenHybridConfig& hybrid_config() const override { return model_.text_config(); }

    std::uint64_t weight_bytes() const override {
        const q4::Qwen4ExpRankWeights& w = model_.weights();
        std::uint64_t total = 0;
        auto add_hyper = [&](const q4::Qwen4ExpHyperConnectionWeights& h) {
            add_matrix_bytes(total, h.input_mix_down);
            add_matrix_bytes(total, h.input_mix_up);
            add_matrix_bytes(total, h.block_inject);
        };
        add_matrix_bytes(total, w.embed_tokens);
        add_matrix_bytes(total, w.lm_head);
        add_hyper(w.mixer);
        for (const auto& layer : w.layers) {
            add_hyper(layer.attn_hyper_connection);
            add_hyper(layer.mlp_hyper_connection);
            if (layer.attention.is_gdn) {
                add_matrix_bytes(total, layer.attention.in_proj_qkv);
                add_matrix_bytes(total, layer.attention.in_proj_z);
                add_matrix_bytes(total, layer.attention.in_proj_b);
                add_matrix_bytes(total, layer.attention.in_proj_a);
                add_matrix_bytes(total, layer.attention.out_proj);
            } else {
                add_matrix_bytes(total, layer.attention.q_proj);
                add_matrix_bytes(total, layer.attention.k_proj);
                add_matrix_bytes(total, layer.attention.v_proj);
                add_matrix_bytes(total, layer.attention.o_proj);
                add_matrix_bytes(total, layer.attention.index_qk_proj);
            }
            add_matrix_bytes(total, layer.mlp.router);
            add_matrix_bytes(total, layer.mlp.shared_gate);
            add_matrix_bytes(total, layer.mlp.shared_gate_proj);
            add_matrix_bytes(total, layer.mlp.shared_up_proj);
            add_matrix_bytes(total, layer.mlp.shared_down_proj);
            add_matrix_bytes(total, layer.mlp.gate_experts);
            add_matrix_bytes(total, layer.mlp.up_experts);
            add_matrix_bytes(total, layer.mlp.down_experts);
            if (layer.ple.present) {
                add_matrix_bytes(total, layer.ple.key_proj);
                add_matrix_bytes(total, layer.ple.value_proj);
            }
        }
        return total;
    }

    std::size_t token_row_ffn_size() const override {
        return model_.text_config().ffn.shared_expert_intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t max_batch_rows) const override {
        const auto& cfg = model_.text_config();
        const std::uint32_t local_experts =
            static_cast<std::uint32_t>(cfg.ffn.num_experts);
        const std::uint32_t top_k =
            static_cast<std::uint32_t>(cfg.ffn.num_experts_per_tok);
        const std::uint32_t tokens = static_cast<std::uint32_t>(max_batch_rows);
        const std::uint32_t hidden = static_cast<std::uint32_t>(cfg.hidden_size);
        const std::uint32_t inter =
            static_cast<std::uint32_t>(cfg.ffn.moe_intermediate_size);
        const std::uint32_t group_rows = ::shelf::runtime::grouped_row_capacity(
            tokens, ::shelf::runtime::GroupedRowShape{local_experts, top_k});
        std::size_t total = static_cast<std::size_t>(group_rows) * inter * 2u * 2u;
        total += static_cast<std::size_t>(group_rows) * hidden * 2u;
        total += ::shelf::runtime::MoePlanScratch::make(tokens, local_experts, top_k).bytes();
        return total;
    }

    bool uses_exl3_linear() const override { return true; }

    Result<q35::Executor> create_executor(q35::SequenceSlotPool& sequence_pool,
                                          q35::GdnStatePool& gdn_state_pool,
                                          q35::PagedKVPool& kv_pool, gpu::GpuArena& arena,
                                          const q35::ExecutorConfig& config) const override {
        return create_qwen4exp_model_executor(
            model_, sequence_pool, gdn_state_pool, kv_pool, arena, config);
    }

    const q4::Qwen4ExpModel* qwen4exp_model() const override { return &model_; }

    std::uint32_t pipeline_hidden_width() const override {
        return static_cast<std::uint32_t>(model_.text_config().hc_count *
                                          model_.text_config().hidden_size);
    }

private:
    q4::Qwen4ExpModel model_;
};

}  // namespace

Result<std::unique_ptr<PipelineStageModel>> build_qsa_pipeline_stage_model(
    const std::string& model_dir, const qhy::PipelineStage& stage, gpu::GpuArena& arena,
    hipStream_t stream, const qhy::HybridLoadOptions& load_options) {
    q35qsa::Qwen35QsaLoadOptions options(load_options);
    options.stage = stage;
    auto model =
        q35qsa::Qwen35QsaModel::load_from_safetensors(model_dir, arena, stream, options);
    if (!model.ok()) return model.status();
    return std::unique_ptr<PipelineStageModel>(
        std::make_unique<QsaPipelineStageModel>(model.release()));
}

Result<std::unique_ptr<PipelineStageModel>> build_qwen4exp_pipeline_stage_model(
    const std::string& model_dir, const qhy::PipelineStage& stage, gpu::GpuArena& arena,
    hipStream_t stream, const qhy::HybridLoadOptions& load_options) {
    q4::Qwen4ExpLoadOptions options;
    options.stage = stage;
    options.verify_quantized_payload_crc = load_options.verify_quantized_payload_crc;
    auto model = q4::Qwen4ExpModel::load_from_safetensors(model_dir, 1u, 0u, arena, stream,
                                                          options);
    if (!model.ok()) return model.status();
    return std::unique_ptr<PipelineStageModel>(
        std::make_unique<Qwen4ExpPipelineStageModel>(model.release()));
}

Result<std::unique_ptr<PipelineStageModel>> build_moe_pipeline_stage_model(
    const std::string& model_dir, const qhy::PipelineStage& stage, gpu::GpuArena& arena,
    hipStream_t stream, const qhy::HybridLoadOptions& load_options) {
    qmoe::Qwen35MoeLoadOptions moe_options;
    static_cast<qhy::HybridLoadOptions&>(moe_options) = load_options;
    moe_options.stage = stage;
    auto model_result =
        qmoe::Qwen35MoeModel::load_from_safetensors(model_dir, arena, stream, moe_options);
    if (!model_result.ok()) return model_result.status();
    return std::unique_ptr<PipelineStageModel>(
        std::make_unique<MoePipelineStageModel>(std::move(model_result.value())));
}

Result<PipelineStageModelBuilder> select_pipeline_stage_model(const std::string& model_dir) {
    const auto family = probe_model_dir(model_dir);
    if (!family.ok()) return family.status();
    if (family.value() == ModelFamily::Plamo3)
        return Status::unsupported("plamo3: runtime path is not connected", __FILE__, __LINE__);
    if (family.value() == ModelFamily::Qsa) return &build_qsa_pipeline_stage_model;
    if (family.value() == ModelFamily::Qwen4Exp)
        return &build_qwen4exp_pipeline_stage_model;
    if (family.value() == ModelFamily::Moe) return &build_moe_pipeline_stage_model;
    return &q35r::build_dense_pipeline_stage_model;
}

}  // namespace shelf::models
