#include <shelf/models/qwen35/runtime/pipeline_parallel.h>

#include <shelf/models/pipeline_stage_model.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/weights/model_weights.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/weights/weight_bytes.h>

namespace shelf {
namespace qwen35 {
namespace runtime {

namespace {

using ::shelf::weights::add_matrix_bytes;

class DensePipelineStageModel final : public shelf::models::PipelineStageModel {
public:
    DensePipelineStageModel(Qwen35Model&& model) : model_(std::move(model)) {}

    const shelf::qwen_hybrid::QwenHybridConfig& hybrid_config() const override {
        return model_.text_config();
    }

    std::uint64_t weight_bytes() const override {
        const Qwen35ModelWeights& w = model_.weights();
        std::uint64_t total = 0;
        add_matrix_bytes(total, w.embed_tokens);
        if (!w.lm_head_tied) add_matrix_bytes(total, w.lm_head);
        for (const Qwen35LayerWeights& layer : w.layers) {
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
        return total;
    }

    std::uint64_t host_pinned_bytes() const override {
        return model_.weights().pinned_host.used();
    }

    std::size_t token_row_ffn_size() const override {
        return model_.text_config().ffn.intermediate_size;
    }
    std::size_t ffn_workspace_reserve_bytes(std::size_t) const override { return 0u; }
    bool uses_exl3_linear() const override {
        return ::shelf::qwen35::uses_exl3_linear(model_.weights());
    }

    Result<Executor> create_executor(SequenceSlotPool& sequence_pool,
                                     GdnStatePool& gdn_state_pool, PagedKVPool& kv_pool,
                                     gpu::GpuArena& arena,
                                     const ExecutorConfig& config) const override {
        return create_model_executor(model_, sequence_pool, gdn_state_pool, kv_pool, arena,
                                     config);
    }

    const Qwen35Model* dense_model() const override { return &model_; }

private:
    Qwen35Model model_;
};

}  // namespace

Result<std::unique_ptr<shelf::models::PipelineStageModel>> build_dense_pipeline_stage_model(
    const std::string& model_dir, const PipelineStage& stage, gpu::GpuArena& arena,
    hipStream_t stream, const shelf::qwen_hybrid::HybridLoadOptions& load_options) {
    Qwen35LoadOptions dense_options;
    static_cast<shelf::qwen_hybrid::HybridLoadOptions&>(dense_options) = load_options;
    dense_options.stage = stage;
    auto model_result =
        Qwen35Model::load_from_safetensors(model_dir, arena, stream, dense_options);
    if (!model_result.ok()) return model_result.status();
    return std::unique_ptr<shelf::models::PipelineStageModel>(
        std::make_unique<DensePipelineStageModel>(std::move(model_result.value())));
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
