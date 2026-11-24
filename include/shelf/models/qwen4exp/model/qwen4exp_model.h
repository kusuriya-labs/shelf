#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/qwen4exp/model/qwen4exp_config.h>
#include <shelf/models/qwen4exp/weights/qwen4exp_weights.h>

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace shelf {
namespace qwen4exp {

class PleTable;

class Qwen4ExpModel final {
 public:
    Qwen4ExpModel(const Qwen4ExpModel&) = delete;
    Qwen4ExpModel& operator=(const Qwen4ExpModel&) = delete;
    Qwen4ExpModel(Qwen4ExpModel&&) noexcept = default;
    Qwen4ExpModel& operator=(Qwen4ExpModel&&) noexcept = default;
    ~Qwen4ExpModel() = default;

    static Result<Qwen4ExpModel> load_from_safetensors(
        const std::string& model_dir, std::uint32_t tp_size, std::uint32_t tp_rank,
        gpu::GpuArena& arena, hipStream_t stream,
        const Qwen4ExpLoadOptions& options = {},
        std::uint32_t ep_size = 0u, std::uint32_t ep_rank = 0u);

    static Result<Qwen4ExpModel> adopt(Qwen4ExpRankWeights&& weights,
                                       Qwen4ExpTextConfig config,
                                       const std::string& model_dir = {});

    [[nodiscard]] const Qwen4ExpTextConfig& text_config() const noexcept { return config_; }
    [[nodiscard]] const Qwen4ExpRankWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const std::shared_ptr<const PleTable>& ple_table() const noexcept { return ple_table_; }

 private:
    Qwen4ExpModel(Qwen4ExpRankWeights&& weights, Qwen4ExpTextConfig config);

    Qwen4ExpRankWeights weights_;
    Qwen4ExpTextConfig config_;
    std::shared_ptr<const PleTable> ple_table_;
};

}  // namespace qwen4exp
}  // namespace shelf
