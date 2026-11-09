#pragma once

#include <shelf/core/memory/arena.h>
#include <shelf/models/qwen35/dflash2/config.h>
#include <shelf/models/qwen35/model/qwen35_model.h>
#include <shelf/models/qwen35/model/tensor_parallel_context.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/tp_batch_hook.h>
#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/runtime/tp/tp_barrier_group.h>
#include <shelf/runtime/tp/tp_transports.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <condition_variable>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace shelf {
namespace models {
class TpRankModel;
}
}

namespace shelf {
namespace qwen35 {
namespace runtime {

using PrivateWeightLoader = std::function<Status(
    gpu::GpuArena&, hipStream_t, weights::MatrixWeight&, weights::MatrixWeight&)>;

// Builds one rank's model: its weights, the local geometry of the split, and the path from
// that model to the executor. The rank runtime does not know which model family a checkpoint
// is, so a caller supplies this. shelf::models implements it for both model families and picks
// one (shelf::models::select_tp_rank_model); build_dense_tp_rank_model below is the dense
// family's, usable by a caller that already knows its checkpoint is dense.
using TpRankModelBuilder =
    ::shelf::Result<std::unique_ptr<shelf::models::TpRankModel>> (*)(
        const struct TpRankRuntimeConfig&, gpu::GpuArena&, hipStream_t);

// The batched MTP serve's arena, added to the rank's working-set reserve after the target
// executor estimate. The rank runtime computes it from the rank model it just loaded, so
// the app does not have to know the model's MTP geometry before the group exists.
struct MtpBatchReserveConfig {
    bool enabled = false;
    std::uint32_t num_drafts = 0;
    std::uint32_t max_seq_len = 0;
    std::uint32_t page_tokens = 0;
    std::uint32_t max_batch_rows = 0;
    KVCacheDType kv_dtype = KVCacheDType::BF16;
    std::uint32_t max_requests = 0;
};

struct TpRankRuntimeConfig {
    std::string model_dir;
    TpRankModelBuilder build_rank_model = nullptr;
    std::uint32_t tp_size = 1;
    std::uint32_t tp_rank = 0;
    int device_id = 0;
    std::uint32_t max_batch_rows = 0;
    std::uint32_t max_scheduled_requests = 0;
    std::uint32_t max_scheduled_output_rows = 0;
    std::uint32_t max_seq_len = 16;
    std::uint32_t page_tokens = 16;
    KVCacheDType kv_cache_dtype = KVCacheDType::BF16;
    ::shelf::kernel::OctaveKVCacheFormat octave_kv_format{};
    std::size_t arena_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    std::size_t working_set_reserve_bytes = 0;
    const ::shelf::qwen35::dflash2::DFlash2Config* drafter_config = nullptr;
    std::size_t drafter_weights_bytes = 0;
    std::uint32_t drafter_drafts = 0;
    std::uint32_t drafter_sessions = 0;
    std::size_t drafter_lm_head_k = 0;
    bool prefix_cache = false;
    std::uint32_t prefix_concurrency = 4;
    std::uint32_t prefix_spec_ring_bytes = 0;
    PrivateWeightLoader load_private_weights;
    std::array<std::uint32_t, kMaxTargetHiddenTaps> target_hidden_taps{};
    std::uint32_t target_hidden_tap_count = 0;
    std::uint32_t tp_combine_f32 = 0;
    bool emit_normed_token_hidden = false;
    EffectiveRopeProfile rope_profile;
    Qwen35LoadOptions load_options;
    MtpBatchReserveConfig mtp_batch;
};

class TpRankRuntime {
public:
    static Result<std::unique_ptr<TpRankRuntime>> create(
        const TpRankRuntimeConfig& config);
    ~TpRankRuntime() noexcept;

    TpRankRuntime(const TpRankRuntime&) = delete;
    TpRankRuntime& operator=(const TpRankRuntime&) = delete;

    int device_id() const noexcept { return device_id_; }
    hipStream_t stream() const noexcept { return stream_; }
    gpu::GpuArena& arena() { return arena_.value(); }
    // The rank's model, which owns the weights and states the rank's local geometry.
    const shelf::models::TpRankModel* rank_model() const;
    // The dense family's model. Null for a model whose FFN runs on grouped expert rows.
    const Qwen35Model* dense_model() const;
    // The geometry the backbone, the state pools and the footprint are sized by. It is the
    // same whatever the FFN of a layer is.
    const shelf::qwen_hybrid::QwenHybridConfig& hybrid_config() const;
    const Qwen35TensorParallelContext& context() const;
    SequenceSlotPool& sequence_pool() { return seq_pool_.value(); }
    GdnStatePool& gdn_pool() { return gdn_pool_.value(); }
    PagedKVPool& kv_pool() { return kv_pool_.value(); }
    Executor& executor() { return executor_; }

    weights::MatrixWeight& private_lm_head() { return private_lm_head_; }
    weights::MatrixWeight& private_embed_tokens() { return private_embed_tokens_; }

    std::uint64_t weight_bytes() const;
    std::uint64_t kv_bytes() const;
    std::uint64_t gdn_state_bytes() const;

private:
    TpRankRuntime() = default;

    int device_id_ = -1;
    hipStream_t stream_ = nullptr;
    std::optional<gpu::GpuArena> arena_;
    std::unique_ptr<shelf::models::TpRankModel> model_;
    std::optional<SequenceSlotPool> seq_pool_;
    std::optional<GdnStatePool> gdn_pool_;
    std::optional<PagedKVPool> kv_pool_;
    Executor executor_;
    bool executor_ready_ = false;
    weights::MatrixWeight private_lm_head_;
    weights::MatrixWeight private_embed_tokens_;
};

// The dense family's rank model builder: loads one rank through the dense tensor parallel
// weight load and states the rank's local widths from the dense config.
Result<std::unique_ptr<shelf::models::TpRankModel>> build_dense_tp_rank_model(
    const TpRankRuntimeConfig& config, gpu::GpuArena& arena, hipStream_t stream);

using TpRankModelPrefetch = void (*)(const std::string&);
using TpRankModelRelease = void (*)();

struct TpCoordinatorConfig {
    std::string model_dir;
    TpRankModelBuilder build_rank_model = nullptr;
    TpRankModelPrefetch prefetch_rank_model = nullptr;
    TpRankModelRelease release_rank_model = nullptr;
    std::vector<int> devices;
    std::uint32_t max_batch_rows = 0;
    std::uint32_t max_scheduled_requests = 0;
    std::uint32_t max_scheduled_output_rows = 0;
    std::uint32_t max_seq_len = 16;
    std::uint32_t page_tokens = 16;
    KVCacheDType kv_cache_dtype = KVCacheDType::BF16;
    ::shelf::kernel::OctaveKVCacheFormat octave_kv_format{};
    std::size_t arena_bytes = 16ull * 1024ull * 1024ull * 1024ull;
    std::size_t working_set_reserve_bytes = 0;
    const ::shelf::qwen35::dflash2::DFlash2Config* drafter_config = nullptr;
    std::size_t drafter_weights_bytes = 0;
    std::uint32_t drafter_drafts = 0;
    std::uint32_t drafter_sessions = 0;
    std::size_t drafter_lm_head_k = 0;
    bool prefix_cache = false;
    std::uint32_t prefix_concurrency = 4;
    std::uint32_t prefix_spec_ring_bytes = 0;
    PrivateWeightLoader load_private_weights;
    std::array<std::uint32_t, kMaxTargetHiddenTaps> target_hidden_taps{};
    std::uint32_t target_hidden_tap_count = 0;
    std::uint32_t tp_combine_f32 = 0;
    bool emit_normed_token_hidden = false;
    EffectiveRopeProfile rope_profile;
    Qwen35LoadOptions load_options;
    MtpBatchReserveConfig mtp_batch;
};

class TpCoordinator final : public TpBatchHook {
public:
    static Result<std::unique_ptr<TpCoordinator>> create(
        const TpCoordinatorConfig& config);
    ~TpCoordinator() noexcept;

    TpCoordinator(const TpCoordinator&) = delete;
    TpCoordinator& operator=(const TpCoordinator&) = delete;

    int tp_size() const noexcept { return static_cast<int>(ranks_.size()); }
    TpRankRuntime& rank(std::size_t index) { return *ranks_[index]; }
    const char* transport_name() const noexcept;

    uint32_t on_prefix_rank_count() override;
    PagedSequenceState* on_prefix_rank_sequence(
        const PagedSequenceState& source, uint32_t rank) override;
    Status on_sequence_created(const PagedSequenceState& source) override;
    Status on_sequence_released(const PagedSequenceState& source) override;
    Result<BatchExecutionOutput> on_execute(const ScheduledBatch& batch) override;
    Result<BatchExecutionOutput> on_execute(const ScheduledBatch& batch,
                                            const ExecuteBatchOptions* per_rank) override;

    Status draft_sum(const shelf::runtime::TpSumTarget& target, hipStream_t stream, int rank);

    void abort(const Status& reason) noexcept;
    std::string debug_report() const;

private:
    struct Worker {
        std::mutex mutex;
        std::condition_variable cv;
        const ScheduledBatch* job = nullptr;
        const ExecuteBatchOptions* options = nullptr;
        bool job_pending = false;
        bool stop = false;
        bool done = true;
        Status status{Status::make_ok()};
        BatchExecutionOutput output;
        std::thread thread;

        Worker() : status(Status::make_ok()) {}
    };

    struct MirrorSequences {
        std::vector<PagedSequenceState> states;
    };

    TpCoordinator() = default;

    Status initialize(const TpCoordinatorConfig& config);
    void worker_loop(std::size_t rank);
    Status sync_mirror_state(const ScheduledBatch& batch);
    Status run_rank_step(std::size_t rank, const ScheduledBatch& batch,
                         BatchExecutionOutput& output,
                         const ExecuteBatchOptions* options);

    std::vector<int> devices_;
    std::vector<std::unique_ptr<TpRankRuntime>> ranks_;
    std::unique_ptr<::shelf::runtime::TpTransport> transport_;
    std::unique_ptr<::shelf::runtime::TpBarrierGroup> barrier_group_;
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unordered_map<std::uint64_t, MirrorSequences> mirrors_;
};

}
}
}
