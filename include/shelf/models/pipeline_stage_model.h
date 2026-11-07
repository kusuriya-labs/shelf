#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/config.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/models/common/qwen_hybrid/weight_loader.h>
#include <shelf/models/qwen35/runtime/executor.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace shelf {
namespace qwen35 {
class Qwen35Model;
}
namespace qwen35moe {
class Qwen35MoeModel;
}
}  // namespace shelf

namespace shelf::models {

// One pipeline stage's weights and the path from them to an executor. A stage holds only its
// own layer range, and the runtime that drives the stages does not know which family a
// checkpoint is, so a caller supplies this. shelf::models implements it for both families and
// picks one (select_pipeline_stage_model below); the dense family's builder is
// shelf::qwen35::runtime::build_dense_pipeline_stage_model.
class PipelineStageModel {
public:
    PipelineStageModel(const PipelineStageModel&) = delete;
    PipelineStageModel& operator=(const PipelineStageModel&) = delete;
    virtual ~PipelineStageModel() = default;

    // The geometry the backbone, the state pools and the footprint are sized by. It does not
    // depend on what a layer's FFN is.
    virtual const shelf::qwen_hybrid::QwenHybridConfig& hybrid_config() const = 0;
    // Resident GPU bytes of the weights this stage holds, for the memory plan.
    virtual std::uint64_t weight_bytes() const = 0;
    // The FFN width whose activations still sit at token rows: a dense model's intermediate
    // size, or a grouped-expert model's shared expert.
    virtual std::size_t token_row_ffn_size() const = 0;
    // Program workspace the FFN needs on top of the token-row term. Zero for a dense model,
    // whose estimate already covers every FFN activation.
    virtual std::size_t ffn_workspace_reserve_bytes(std::size_t max_batch_rows) const = 0;
    // True when this stage's linears are carried in the rotated-domain trellis form, whose
    // execution workspace is large and only reserved when a linear actually uses it.
    virtual bool uses_exl3_linear() const = 0;

    virtual shelf::Result<shelf::qwen35::Executor> create_executor(
        shelf::qwen35::SequenceSlotPool& sequence_pool,
        shelf::qwen35::GdnStatePool& gdn_state_pool,
        shelf::qwen35::PagedKVPool& kv_pool,
        gpu::GpuArena& arena,
        const shelf::qwen35::ExecutorConfig& config) const = 0;

    // The family's model, for a caller that needs its weights or its config's stop tokens.
    // Exactly one of the two is non-null.
    virtual const shelf::qwen35::Qwen35Model* dense_model() const { return nullptr; }
    virtual const shelf::qwen35moe::Qwen35MoeModel* grouped_model() const { return nullptr; }

protected:
    PipelineStageModel() = default;
};

// Loads one stage's weights in the arena the stage runtime created. `stage` states the layer
// range; the embedding, the LM head and the final norm follow the first / last rule.
using PipelineStageModelBuilder =
    shelf::Result<std::unique_ptr<PipelineStageModel>> (*)(
        const std::string& model_dir,
        const shelf::qwen_hybrid::PipelineStage& stage,
        gpu::GpuArena& arena,
        hipStream_t stream,
        const shelf::qwen_hybrid::HybridLoadOptions& load_options);

// The grouped-expert family's builder.
shelf::Result<std::unique_ptr<PipelineStageModel>> build_moe_pipeline_stage_model(
    const std::string& model_dir,
    const shelf::qwen_hybrid::PipelineStage& stage,
    gpu::GpuArena& arena,
    hipStream_t stream,
    const shelf::qwen_hybrid::HybridLoadOptions& load_options);

// The builder for the checkpoint in `model_dir`, chosen by its declared architecture. The
// stage runtime takes it as a value: the choice between the families lives in this layer.
shelf::Result<PipelineStageModelBuilder> select_pipeline_stage_model(
    const std::string& model_dir);

}  // namespace shelf::models
