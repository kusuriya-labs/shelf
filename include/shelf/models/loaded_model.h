#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/model_weights_factory.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/mtp_drafter.h>
#include <shelf/models/qwen35moe/model/qwen35moe_model.h>
#include <shelf/models/qwen4exp/model/qwen4exp_model.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <memory>
#include <string>

namespace shelf::models {

// A loaded model and the one runtime entry that runs it. The family is decided once, when
// the directory is read; from there the model is lowered by its own family and handed to the
// executor as the shared lowered form, so the executor and the scheduler name no family.
//
// A borrowed instance (a tensor-parallel rank's model, which the coordinator owns) does not
// keep the weights alive: the caller must outlive it.
class LoadedModel final {
public:
    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;
    LoadedModel(LoadedModel&&) noexcept = default;
    LoadedModel& operator=(LoadedModel&&) noexcept = default;
    ~LoadedModel() = default;

    // Reads config.json, loads with that family's loader, and validates it.
    static Result<LoadedModel> load(
        const std::string& model_dir,
        gpu::GpuArena& arena,
        hipStream_t stream,
        bool verify_payload_crc = false,
        bool load_mtp = false);

    static LoadedModel borrow(const shelf::qwen35::Qwen35Model& model);
    static LoadedModel borrow(const shelf::qwen35moe::Qwen35MoeModel& model);
    static LoadedModel borrow(const shelf::qwen4exp::Qwen4ExpModel& model);

    [[nodiscard]]
    ModelFamily family() const noexcept {
        if (qwen4exp_ != nullptr) return ModelFamily::Qwen4Exp;
        return dense_ != nullptr ? ModelFamily::Dense : ModelFamily::Moe;
    }

    // True when the model's FFN runs on grouped expert rows, which is the case the executor
    // workspace estimate cannot take from a dense intermediate width.
    [[nodiscard]]
    bool runs_grouped_experts() const noexcept {
        return moe_ != nullptr || qwen4exp_ != nullptr;
    }

    [[nodiscard]]
    const shelf::qwen_hybrid::QwenHybridConfig& hybrid_config() const noexcept;

    [[nodiscard]]
    const shelf::weights::MatrixWeight& embed_tokens() const noexcept;

    [[nodiscard]]
    const shelf::weights::MatrixWeight& lm_head() const noexcept;

    // Null unless the model is the dense family. Tensor parallel and the drafter's target
    // glue use it; a model with grouped experts has no dense model to reach.
    [[nodiscard]]
    const shelf::qwen35::Qwen35Model* dense_model() const noexcept {
        return dense_;
    }

    // Null unless the model's FFN runs on grouped expert rows.
    [[nodiscard]]
    const shelf::qwen35moe::Qwen35MoeModel* moe_model() const noexcept {
        return moe_;
    }

    [[nodiscard]]
    const shelf::qwen4exp::Qwen4ExpModel* qwen4exp_model() const noexcept {
        return qwen4exp_;
    }

    // The arena bytes the executor needs: the shared part plus, for grouped experts, the
    // group-row planes and the routing plan.
    [[nodiscard]]
    std::size_t executor_reserve_bytes(std::size_t max_batch_rows,
                                       std::size_t max_scheduled_requests,
                                       std::size_t max_scheduled_output_rows,
                                       std::size_t target_hidden_tap_count,
                                       std::size_t tp_size) const;

    Result<shelf::qwen35::Executor> create_executor(
        shelf::qwen35::SequenceSlotPool& sequence_pool,
        shelf::qwen35::GdnStatePool& gdn_state_pool,
        shelf::qwen35::PagedKVPool& kv_pool,
        gpu::GpuArena& arena,
        const shelf::qwen35::ExecutorConfig& config) const;

    // True when the checkpoint's own MTP block was loaded. That block is the drafter for
    // speculative decode; a model that was loaded without it has none.
    [[nodiscard]]
    bool has_mtp_block() const noexcept;

    // The drafter input: this model's geometry and embedding, and its MTP block lowered by
    // the family that loaded it. Fails when there is no block, so a caller cannot ask for
    // speculative decode on a checkpoint that does not carry one.
    Result<shelf::qwen35::runtime::MtpDrafterInput> mtp_drafter_input(
        const shelf::qwen35::runtime::MtpDrafterConfig& options) const;

    // Arena bytes the drafter needs, with the executor terms this model's FFN adds at the
    // drafter's row count.
    [[nodiscard]]
    std::size_t mtp_drafter_reserve_bytes(
        const shelf::qwen35::runtime::MtpDrafterConfig& options) const;

private:
    LoadedModel() = default;

    std::unique_ptr<shelf::qwen35::Qwen35Model> owns_dense_;
    std::unique_ptr<shelf::qwen35moe::Qwen35MoeModel> owns_moe_;
    std::unique_ptr<shelf::qwen4exp::Qwen4ExpModel> owns_qwen4exp_;
    const shelf::qwen35::Qwen35Model* dense_ = nullptr;
    const shelf::qwen35moe::Qwen35MoeModel* moe_ = nullptr;
    const shelf::qwen4exp::Qwen4ExpModel* qwen4exp_ = nullptr;
};

}  // namespace shelf::models
