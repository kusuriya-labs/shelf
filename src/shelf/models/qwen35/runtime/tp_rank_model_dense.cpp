#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/models/tp_rank_model.h>

#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/weights/weight_bytes.h>

namespace shelf {
namespace qwen35 {
namespace runtime {

using ::shelf::weights::add_matrix_bytes;

namespace {

class DenseTpRankModel final : public shelf::models::TpRankModel {
public:
    DenseTpRankModel(Qwen35Model&& model, Qwen35TensorParallelContext&& context)
        : model_(std::move(model)), context_(context) {}

    const shelf::qwen_hybrid::QwenHybridConfig& hybrid_config() const override {
        return model_.text_config();
    }
    const Qwen35TensorParallelContext& context() const override { return context_; }

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
                add_matrix_bytes(total, layer.attn_out_proj);
            }
        }
        return total;
    }

    std::size_t token_row_ffn_size() const override {
        return model_.text_config().ffn.intermediate_size;
    }

    std::size_t ffn_workspace_reserve_bytes(std::size_t) const override { return 0u; }

    bool uses_exl3_linear() const override {
        return ::shelf::qwen35::uses_exl3_linear(model_.weights());
    }

    shelf::weights::Exl3LinearShape exl3_linear_shape() const override {
        return ::shelf::qwen35::exl3_linear_shape(model_.weights());
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
    Qwen35TensorParallelContext context_;
};

}  // namespace

Result<std::unique_ptr<shelf::models::TpRankModel>> build_dense_tp_rank_model(
    const TpRankRuntimeConfig& config, gpu::GpuArena& arena, hipStream_t stream) {
    auto model_result = Qwen35Model::load_tensor_parallel_from_safetensors(
        config.model_dir, config.tp_size, config.tp_rank, arena, stream, config.load_options);
    if (!model_result.ok()) return model_result.status();

    auto context_result = make_qwen35_tensor_parallel_context(
        model_result.value().text_config(), config.tp_size, config.tp_rank,
        config.load_options.vocab_parallel_lm_head);
    if (!context_result.ok()) return context_result.status();

    return std::unique_ptr<shelf::models::TpRankModel>(std::make_unique<DenseTpRankModel>(
        model_result.release(), context_result.release()));
}

}  // namespace runtime
}  // namespace qwen35
}  // namespace shelf
