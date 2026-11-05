#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/common/qwen_hybrid/stage_range.h>
#include <shelf/models/qwen35moe/weights/qwen35moe_model_load.h>

#include <hip/hip_runtime.h>

#include <string>

namespace shelf::qwen35moe {

// A loaded MoE model: its config and one MoE layer per transformer layer, resident in the
// caller's arena. There is no dense view of these weights. The checkpoint's MTP block is
// loaded when the options ask for it and is the speculative decode drafter.
class Qwen35MoeModel final {
public:
    Qwen35MoeModel(const Qwen35MoeModel&) = delete;
    Qwen35MoeModel& operator=(const Qwen35MoeModel&) = delete;

    Qwen35MoeModel(Qwen35MoeModel&&) noexcept = default;
    Qwen35MoeModel& operator=(Qwen35MoeModel&&) noexcept = delete;

    ~Qwen35MoeModel() = default;

    static Result<Qwen35MoeModel> load_from_safetensors(
        const std::string& model_dir,
        gpu::GpuArena& arena,
        hipStream_t stream,
        const Qwen35MoeLoadOptions& options = {});

    static Result<Qwen35MoeModel> adopt(
        Qwen35MoeModelWeights&& weights,
        Qwen35MoeTextConfig text_config,
        const shelf::qwen_hybrid::PipelineStage& stage = {});

    // The layer range this model holds. A full-model load reports a disabled stage.
    [[nodiscard]]
    const shelf::qwen_hybrid::PipelineStage&
    stage() const noexcept {
        return stage_;
    }

    [[nodiscard]]
    const Qwen35MoeModelWeights&
    weights() const noexcept {
        return weights_;
    }

    [[nodiscard]]
    const Qwen35MoeTextConfig&
    text_config() const noexcept {
        return text_config_;
    }

private:
    Qwen35MoeModel(Qwen35MoeModelWeights&& weights, Qwen35MoeTextConfig text_config,
                   shelf::qwen_hybrid::PipelineStage stage)
        : weights_(std::move(weights)), text_config_(std::move(text_config)), stage_(stage) {}

    Qwen35MoeModelWeights weights_;
    Qwen35MoeTextConfig text_config_;
    shelf::qwen_hybrid::PipelineStage stage_;
};

}  // namespace shelf::qwen35moe
