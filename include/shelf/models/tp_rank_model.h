#pragma once
#include <shelf/core/memory/arena.h>
#include <shelf/core/status.h>
#include <shelf/models/loaded_model.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/runtime/mtp/mtp_drafter.h>
#include <shelf/models/qwen35/runtime/tensor_parallel.h>
#include <shelf/runtime/state_types.h>
#include <shelf/models/qwen35moe/model/qwen35moe_model.h>
#include <shelf/models/qwen35qsa/model/qwen35qsa_model.h>
#include <shelf/models/qwen4exp/model/qwen4exp_model.h>
#include <shelf/models/qwen4exp/weights/ple_table.h>
#include <shelf/weights/matrix_weight.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace shelf::models {

// The part of a tensor parallel rank that the rank runtime cannot decide by itself: which
// model family its checkpoint is, what local geometry that family's split produces, and how
// that family's model reaches the executor. The rank runtime lives in the dense family's
// namespace and must not name the grouped-expert family, so it reaches a rank's model through
// this interface. Everything it does happens once, at startup.
//
// The dense implementation is in models/qwen35/runtime, where naming the dense family is
// allowed. The grouped-expert implementation, and the choice between the two, are in
// models/tp_rank_model.cpp: this layer is the only one that may name both families.
class TpRankModel {
public:
    TpRankModel(const TpRankModel&) = delete;
    TpRankModel& operator=(const TpRankModel&) = delete;
    virtual ~TpRankModel() = default;

    // The geometry the backbone, the state pools and the footprint are sized by. It does not
    // depend on what a layer's FFN is.
    virtual const shelf::qwen_hybrid::QwenHybridConfig& hybrid_config() const = 0;
    // This rank's local widths: the backbone heads, the vocab split, and for a grouped-expert
    // model the expert range and the shared expert's local width.
    virtual const shelf::qwen35::Qwen35TensorParallelContext& context() const = 0;

    // Resident GPU bytes of the weights this rank holds, for the memory plan.
    virtual std::uint64_t weight_bytes() const = 0;
    // Page-locked host bytes this rank holds outside its arena: the embedding table today.
    virtual std::uint64_t host_pinned_bytes() const { return 0u; }
    // Cold routed experts kept in host RAM outside the arena.
    virtual std::uint64_t host_offload_expert_bytes() const { return 0u; }
    // Process-shared PLE table. The memory plan reports it once, not per rank.
    virtual std::uint64_t host_ple_table_bytes() const { return 0u; }
    // The FFN width whose activations still sit at token rows: a dense model's intermediate
    // size, or a grouped-expert model's shared expert.
    virtual std::size_t token_row_ffn_size() const = 0;
    // Program workspace the FFN needs on top of the token-row term. Zero for a dense model,
    // whose estimate already covers every FFN activation.
    virtual std::size_t ffn_workspace_reserve_bytes(std::size_t max_batch_rows) const = 0;
    // True when this rank's linears are carried in the rotated-domain trellis form. The
    // executor's staging planes for that form are large, and the memory plan only reserves
    // them when a program will actually run such a linear.
    virtual bool uses_exl3_linear() const = 0;
    // Largest rotated-domain linear this rank runs, which sizes those staging planes. An
    // empty shape keeps the model-independent upper bound.
    virtual shelf::weights::Exl3LinearShape exl3_linear_shape() const { return {}; }
    // The same geometry for the MTP draft block, which runs in its own executor.
    virtual shelf::weights::Exl3LinearShape mtp_exl3_linear_shape() const { return {}; }
    // Lowers this rank's MTP block into the drafter graph. A rank without the block
    // returns unsupported.
    virtual shelf::Result<shelf::runtime::mtp::MtpDrafterInput> mtp_drafter_input(
        const shelf::runtime::mtp::MtpDrafterConfig& options) const {
        (void)options;
        return Status::unsupported("this rank has no MTP block", __FILE__, __LINE__);
    }
    // State pools the executor owns and allocates after the memory plan has sized the KV
    // pool: the QSA pooled-key pool and the PLE conv pool. Zero for a family whose pools
    // the rank runtime already holds.
    virtual std::size_t executor_state_reserve_bytes(std::size_t) const {
        return 0u;
    }
    // True when this rank's program runs QSA key selection and needs the select workspace
    // and the pooled-key state pool. The dense QSA family and Qwen4Exp set it.
    virtual bool uses_qsa_state() const { return false; }
    // Group-row workspace the family's MTP block needs on top of the shared executor
    // estimate. Zero for a family whose draft block has no grouped experts.
    virtual std::size_t mtp_grouped_reserve_bytes(std::size_t, std::uint32_t) const {
        return 0u;
    }
    // Width of the hidden the family exposes for MTP. Zero means the model's hidden size.
    // Qwen4Exp exposes the pre-mixer wide residual (hc * hidden).
    virtual std::uint32_t normed_token_hidden_width() const { return 0u; }

    virtual shelf::Result<shelf::qwen35::Executor> create_executor(
        shelf::qwen35::SequenceSlotPool& sequence_pool,
        shelf::qwen35::GdnStatePool& gdn_state_pool,
        shelf::qwen35::PagedKVPool& kv_pool,
        gpu::GpuArena& arena,
        const shelf::qwen35::ExecutorConfig& config) const = 0;

    // The rank's own target head. It is this rank's vocabulary shard when the load split
    // the vocabulary, and the full head otherwise. Null for a family without an untied head.
    virtual const shelf::weights::MatrixWeight* lm_head() const { return nullptr; }
    // The rank's own embedding table. It is this rank's vocabulary shard when the load
    // split the vocabulary, and the full table otherwise.
    virtual const shelf::weights::MatrixWeight* embed_tokens() const { return nullptr; }
    // A family that keeps a complete replicated head / embedding on every rank returns
    // it here. Null means the caller uses the rank's private target load instead.
    virtual const shelf::weights::MatrixWeight* replicated_lm_head() const { return nullptr; }
    virtual const shelf::weights::MatrixWeight* replicated_embed_tokens() const {
        return nullptr;
    }
    // True when MTP drafter inputs are lowered per rank instead of from rank 0's model.
    virtual bool mtp_uses_per_rank_input() const { return false; }
    // The dense family's model, for callers that need its weights. Null for a model whose FFN
    // runs on grouped expert rows, which has no dense model to reach.
    virtual const shelf::qwen35::Qwen35Model* dense_model() const { return nullptr; }
    // The grouped-expert family's model, for the same reason. Null for a dense model. The
    // rank runtime cannot see this: the interface declares it, and only this layer's
    // implementations override it.
    virtual const shelf::qwen35moe::Qwen35MoeModel* grouped_model() const { return nullptr; }
    // The Qwen4Exp model behind this rank. Null for the other families.
    virtual const shelf::qwen4exp::Qwen4ExpModel* qwen4exp_model() const { return nullptr; }
    // The dense QSA model behind this rank. Null for the other families.
    virtual const shelf::qwen35qsa::Qwen35QsaModel* qsa_model() const { return nullptr; }

protected:
    TpRankModel() = default;
};

// Builds one rank's model in the arena the rank runtime created. `config` states the model
// directory, the rank and the load options.
using TpRankModelBuilder = shelf::runtime::TpRankModelBuilder;

// The grouped-expert family's builder. The dense family's is
// shelf::qwen35::runtime::build_dense_tp_rank_model.
shelf::Result<std::unique_ptr<TpRankModel>> build_qwen4exp_tp_rank_model(
    const shelf::runtime::TpRankRuntimeConfig& config, gpu::GpuArena& arena,
    hipStream_t stream);

shelf::Result<std::unique_ptr<TpRankModel>> build_moe_tp_rank_model(
    const shelf::runtime::TpRankRuntimeConfig& config, gpu::GpuArena& arena,
    hipStream_t stream);

void prefetch_qwen4exp_ple_table(const std::string& model_dir);
void release_prefetched_qwen4exp_ple_table();

// Sets config.build_rank_model to the family config.model_dir declares, read from
// num_experts before any tensor is opened.
shelf::Status select_tp_rank_model(shelf::runtime::TpCoordinatorConfig& config);

// Lowers a grouped-expert model with this rank's expert range and hands the shared lowered
// form to the executor. LoadedModel and a tensor parallel rank both reach the executor this
// way, so the lowering options are stated once.
shelf::Result<shelf::qwen35::Executor> create_qwen4exp_model_executor(
    const shelf::qwen4exp::Qwen4ExpModel& model,
    shelf::qwen35::SequenceSlotPool& sequence_pool,
    shelf::qwen35::GdnStatePool& gdn_state_pool,
    shelf::qwen35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const shelf::qwen35::ExecutorConfig& config);

shelf::Result<shelf::qwen35::Executor> create_moe_model_executor(
    const shelf::qwen35moe::Qwen35MoeModel& model,
    shelf::qwen35::SequenceSlotPool& sequence_pool,
    shelf::qwen35::GdnStatePool& gdn_state_pool,
    shelf::qwen35::PagedKVPool& kv_pool,
    gpu::GpuArena& arena,
    const shelf::qwen35::ExecutorConfig& config);

// A tensor parallel group plus the loaded-model handle of its rank 0. The coordinator owns
// the rank's weights, so the handle is borrowed and must not outlive the group. This is the
// entry an application uses; TpCoordinator::create on its own requires a builder, and picking
// one is this layer's job.
class TpGroup final {
public:
    TpGroup(const TpGroup&) = delete;
    TpGroup& operator=(const TpGroup&) = delete;
    TpGroup(TpGroup&&) noexcept = default;
    TpGroup& operator=(TpGroup&&) noexcept = default;
    ~TpGroup() = default;

    static shelf::Result<TpGroup> create(
        const shelf::runtime::TpCoordinatorConfig& config);

    shelf::runtime::TpCoordinator& coordinator() noexcept { return *coordinator_; }
    const shelf::runtime::TpCoordinator& coordinator() const noexcept {
        return *coordinator_;
    }
    LoadedModel& model() noexcept { return model_.value(); }
    const LoadedModel& model() const noexcept { return model_.value(); }

private:
    TpGroup() = default;

    std::unique_ptr<shelf::runtime::TpCoordinator> coordinator_;
    std::optional<LoadedModel> model_;
};

}  // namespace shelf::models
