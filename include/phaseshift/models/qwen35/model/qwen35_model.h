#pragma once

#include <phaseshift/models/qwen35/model/qwen35_config.h>
#include <phaseshift/models/qwen35/weights/model_weights.h>
#include <phaseshift/core/status.h>
#include <phaseshift/core/memory/arena.h>

#include <hip/hip_runtime.h>
#include <string>

namespace ps::qwen35 {

class Qwen35Model final {
public:
    Qwen35Model(const Qwen35Model&) = delete;
    Qwen35Model& operator=(const Qwen35Model&) = delete;

    Qwen35Model(Qwen35Model&&) noexcept = default;
    Qwen35Model& operator=(Qwen35Model&&) noexcept = delete;

    ~Qwen35Model() = default;

    static Result<Qwen35Model> load_from_safetensors(
        const std::string& model_dir,
        gpu::GpuArena& arena,
        hipStream_t stream,
        const Qwen35LoadOptions& options = {});

    static Result<Qwen35Model> load_tensor_parallel_from_safetensors(
        const std::string& model_dir,
        std::uint32_t tp_size,
        std::uint32_t tp_rank,
        gpu::GpuArena& arena,
        hipStream_t stream,
        const Qwen35LoadOptions& options = {});

    static Result<Qwen35Model> adopt(
        Qwen35ModelWeights&& weights,
        Qwen35TextConfig text_config);

    [[nodiscard]]
    const Qwen35ModelWeights&
    weights() const noexcept {
        return weights_;
    }

    [[nodiscard]]
    const Qwen35TextConfig&
    text_config() const noexcept {
        return text_config_;
    }

private:
    Qwen35Model(
        Qwen35ModelWeights&& weights,
        Qwen35TextConfig text_config);

    Qwen35ModelWeights weights_;
    Qwen35TextConfig text_config_;
};

} // namespace ps::qwen35
