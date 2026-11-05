#include <shelf/models/pipeline_stage_model.h>

#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35/runtime/pipeline_parallel.h>
#include <shelf/models/qwen35moe/model/qwen35moe_model.h>
#include <shelf/models/qwen35moe/runtime/executor_footprint.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>
#include <shelf/models/tp_rank_model.h>
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

}  // namespace

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
    if (family.value() == ModelFamily::Moe) return &build_moe_pipeline_stage_model;
    return &q35r::build_dense_pipeline_stage_model;
}

}  // namespace shelf::models
