#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35qsa/model/qwen35qsa_config.h>
#include <shelf/models/qwen35qsa/weights/model_weights.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>

namespace shelf {
namespace qwen35qsa {

class Qwen35QsaModel final {
public:
    Qwen35QsaModel(const Qwen35QsaModel&) = delete;
    Qwen35QsaModel& operator=(const Qwen35QsaModel&) = delete;
    Qwen35QsaModel(Qwen35QsaModel&&) noexcept = default;
    Qwen35QsaModel& operator=(Qwen35QsaModel&&) noexcept = delete;
    ~Qwen35QsaModel() = default;

    static Result<Qwen35QsaModel> load_from_safetensors(
        const std::string& model_dir,
        gpu::GpuArena& arena,
        hipStream_t stream,
        const Qwen35QsaLoadOptions& options = {});

    static Result<Qwen35QsaModel> adopt(
        Qwen35QsaModelWeights&& weights,
        Qwen35QsaTextConfig text_config);

    [[nodiscard]]
    const Qwen35QsaModelWeights& weights() const noexcept {
        return weights_;
    }

    [[nodiscard]]
    const Qwen35QsaTextConfig& text_config() const noexcept {
        return text_config_;
    }

private:
    Qwen35QsaModel(Qwen35QsaModelWeights&& weights, Qwen35QsaTextConfig text_config);

    Qwen35QsaModelWeights weights_;
    Qwen35QsaTextConfig text_config_;
};

// Lowers the sparse backbone and hands it to the shared executor with the QSA state pool
// sized from the same block count the graph was lowered with.
Result<shelf::qwen35::Executor> create_qwen35qsa_model_executor(
    const Qwen35QsaModel& model,
    shelf::qwen35::SequenceSlotPool& sequence_pool,
    shelf::qwen35::GdnStatePool& gdn_state_pool,
    shelf::qwen35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const shelf::qwen35::ExecutorConfig& config);

// Workspace the executor keeps besides the KV pool: the shared backbone footprint plus the
// indexer select workspace and the per-sequence QSA key state.
std::size_t qwen35qsa_executor_reserve_bytes(
    const Qwen35QsaTextConfig& config,
    std::size_t ffn_intermediate_size,
    const shelf::qwen35::ExecutorConfig& exec_config,
    std::size_t tp_size,
    bool exl3_linear_workspace,
    std::size_t max_sequences);

}
}
