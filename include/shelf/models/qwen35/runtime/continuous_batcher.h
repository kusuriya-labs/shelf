#pragma once

#include <shelf/models/qwen35/runtime/runtime_request.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/runtime/dflash2_session_pool.h>
#include <shelf/models/qwen35/runtime/token_budget_scheduler.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/tp_batch_hook.h>
#include <shelf/models/qwen35/stop_tokens.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace ps {
namespace qwen35 {

struct Executor;

namespace runtime {

struct ContinuousBatcherConfig {
    uint32_t max_num_batched_tokens = 0;
    uint32_t max_scheduled_requests = 0;
    uint32_t max_seq_len = 0;
    uint32_t page_tokens = 0;
    StopTokens eos_token_ids;
    KVAdmissionPolicy admission_policy = KVAdmissionPolicy::BankerSafe;
};

class PrefixCache;

constexpr uint32_t kDFlash2PositionBuckets = 32u;
constexpr uint32_t kDFlash2PositionBucketTokens = 256u;

struct DFlash2PositionStats {
    uint32_t rounds = 0;
    uint32_t accepted_drafts = 0;
    uint32_t generated_tokens = 0;
    double round_ms = 0.0;
};

struct DFlash2BatcherStats {
    uint32_t rounds = 0;
    uint32_t full_accepts = 0;
    uint32_t partial_accepts = 0;
    uint32_t accepted_drafts = 0;
    uint32_t generated_tokens = 0;
    uint64_t verify_rows = 0;
    double round_ms = 0.0;
    double draft_gpu_ms = 0.0;
    double verify_gpu_ms = 0.0;
    double draft_ms = 0.0;
    double draft_d2h_ms = 0.0;
    double proposal_wait_ms = 0.0;
    double proposal_copy_ms = 0.0;
    double gdn_snapshot_ms = 0.0;
    double verify_ms = 0.0;
    double decision_wait_ms = 0.0;
    double decision_copy_ms = 0.0;
    double gdn_restore_ms = 0.0;
    double rerun_ms = 0.0;
    double commit_ms = 0.0;
    uint64_t gdn_history_bytes = 0;
    uint64_t gdn_snapshot_bytes = 0;
    uint64_t gdn_compact_bytes = 0;
    DFlash2PositionStats position[kDFlash2PositionBuckets];
};

struct DFlash2BatcherBinding {
    const dflash2::DFlash2Config* config = nullptr;
    dflash2::DFlash2Executor* executor = nullptr;
    DFlash2SpecBatchContext* batch = nullptr;
    gpu::GpuArena* arena = nullptr;
    DFlash2SessionPool* sessions = nullptr;
    uint32_t num_drafts = 7u;
    DFlash2BatcherStats* stats = nullptr;
    const std::vector<DFlash2GdnRankBinding>* rank_bindings = nullptr;
};

class ContinuousBatcher {
 public:
    ContinuousBatcher(
        Executor& executor,
        SequenceSlotPool& seq_pool,
        GdnStatePool& gdn_pool,
        PagedKVPool& kv_pool,
        ContinuousBatcherConfig config,
        hipStream_t stream);

    ContinuousBatcher(const ContinuousBatcher&) = delete;
    ContinuousBatcher& operator=(const ContinuousBatcher&) = delete;

    Result<uint64_t> submit(std::vector<int32_t> input_tokens, uint32_t max_new_tokens);

    Result<uint64_t> submit(
        std::vector<int32_t> input_tokens,
        uint32_t max_new_tokens,
        const SamplingConfig& sampling);

    Result<uint64_t> submit(
        std::vector<int32_t> input_tokens,
        uint32_t max_new_tokens,
        const SamplingConfig& sampling,
        bool speculative_decode);

    Result<uint64_t> submit(
        std::vector<int32_t> input_tokens,
        uint32_t max_new_tokens,
        const SamplingConfig& sampling,
        const ThinkingBudget& thinking);

    Result<uint64_t> submit(
        std::vector<int32_t> input_tokens,
        uint32_t max_new_tokens,
        const SamplingConfig& sampling,
        bool speculative_decode,
        const ThinkingBudget& thinking);

    Status cancel(uint64_t id);

    Status retire(uint64_t id);

    Result<StepResult> step();

    Status cancel_all();

    bool has_pending() const noexcept;

    const RuntimeRequest* request(uint64_t id) const;

    std::size_t queued_count() const noexcept;
    std::size_t active_count() const noexcept;
    std::size_t finished_count() const noexcept;

    KVAdmissionPolicy admission_policy() const noexcept {
        return config_.admission_policy;
    }

    void set_tp_batch_hook(TpBatchHook* hook) noexcept {
        tp_batch_hook_ = hook;
    }

    void set_prefix_cache(PrefixCache* cache) noexcept {
        prefix_cache_ = cache;
    }

    Status set_dflash2(const DFlash2BatcherBinding& binding);

    bool dflash2_available() const noexcept { return dflash_.batch != nullptr; }

    KVCapacitySnapshot kv_capacity_snapshot() const;

 private:
    Status finish_request(RuntimeRequest& request, FinishReason reason, hipStream_t stream);

    void prefix_save_prefill(RuntimeRequest& request);

    void prefix_begin_capture(RuntimeRequest& request);

    void prefix_capture_after_step(RuntimeRequest& request);

    void prefix_finish_capture(RuntimeRequest& request, FinishReason reason,
                               hipStream_t stream);

    Status open_dflash2_context(RuntimeRequest& request);

    Status open_dflash2_session(RuntimeRequest& request);

    Status run_dflash2_round(std::vector<RuntimeRequest*>& active, bool& executed,
                             uint32_t& tokens, uint32_t& requests);

    KVBankerState build_kv_banker_state() const;

    Executor& executor_;
    SequenceSlotPool& seq_pool_;
    GdnStatePool& gdn_pool_;
    PagedKVPool& kv_pool_;
    ContinuousBatcherConfig config_;
    hipStream_t stream_;

    KVCapacityManager capacity_;
    TpBatchHook* tp_batch_hook_ = nullptr;
    PrefixCache* prefix_cache_ = nullptr;
    DFlash2BatcherBinding dflash_{};

    uint64_t next_id_ = 0;

    using RequestList = std::list<RuntimeRequest>;
    RequestList requests_;
    std::unordered_map<uint64_t, RequestList::iterator> request_index_;
};

}
}
}
