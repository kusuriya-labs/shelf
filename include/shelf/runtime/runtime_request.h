#pragma once

#include <shelf/runtime/state_types.h>
#include <shelf/runtime/scheduling_types.h>
#include <shelf/runtime/dflash2_spec_decoder.h>
#include <shelf/runtime/dflash2/dflash2_adaptive_state.h>
#include <shelf/models/common/qwen_hybrid/stop_tokens.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace shelf {
namespace runtime {

enum class RequestState : uint8_t {
    Queued,
    Active,
    Finished,
};

enum class FinishReason : uint8_t {
    None,
    Eos,
    MaxNewTokens,
    Cancelled,
    Error,
};

enum class DraftMaskState : uint8_t { Idle, PendingMask, Ready };

struct DFlash2RequestSession {
    dflash2::DFlash2ContextState* context = nullptr;
    std::unique_ptr<DFlash2SpecDecoder> decoder;
    uint32_t pool_index = kDFlash2NoSessionSlot;
    DFlash2SpecTiming timing{};
    uint32_t adaptive_rows = 0;
    DFlash2AdaptiveController adaptive;
    int32_t pending_token = -1;
    bool ready = false;
};

struct ThinkingBudget {
    uint32_t tokens = 0;
    int32_t close_token = -1;
};

struct RuntimeRequest {
    uint64_t id = 0;
    int64_t external_request_id = 0;

    std::vector<int32_t> input_tokens;
    uint32_t max_new_tokens = 0;
    ThinkingBudget thinking{};
    bool thinking_closed = false;

    SamplingConfig sampling{};

    // The caller owns the device mask until the request is retired.  A null
    // pointer preserves the ordinary, unconstrained sampling path.
    const uint32_t* token_bitmask = nullptr;
    bool waiting_for_token_bitmask = false;
    bool structured_dflash = false;
    bool grammar_delayed = false;
    int32_t grammar_close_token = -1;
    DraftMaskState draft_mask_state = DraftMaskState::Idle;
    uint64_t draft_round_id = 0;
    std::vector<int32_t> pending_drafts;
    const uint32_t* draft_bitmask = nullptr;
    uint32_t draft_mask_stride = 0;
    uint32_t draft_valid_prefix = 0;

    uint32_t max_kv_tokens = 0;
    uint32_t max_kv_pages = 0;

    RequestState state = RequestState::Queued;
    FinishReason finish_reason = FinishReason::None;

    PagedSequenceState sequence;
    int32_t pending_decode_token = -1;

    uint32_t generated_tokens = 0;
    std::vector<int32_t> generated;

    uint32_t restored_tokens = 0;
    uint32_t prefix_capture = 0;
    uint32_t prefix_boundary_pos = 0;
    uint32_t prefix_slot = 0xFFFFFFFFu;
    uint32_t prefix_save_position = 0;
    uint32_t prefix_junction_position = 0;
    uint32_t prefix_last_save_position = 0;
    double prefill_ms = 0.0;
    double prefill_log_ms = 0.0;
    uint32_t prefill_log_tokens = 0;

    bool speculative_decode = false;
    bool dflash_used = false;
    bool ignore_eos = false;
    std::unique_ptr<DFlash2RequestSession> dflash;

    std::string error_message;
};

Status admit_request(
    RuntimeRequest& request,
    SequenceSlotPool& seq_pool,
    GdnStatePool& gdn_pool,
    PagedKVPool& kv_pool,
    hipStream_t stream);

struct StepResult {
    bool executed = false;
    uint32_t num_tokens = 0;
    uint32_t num_requests = 0;
};

enum class CommitAction : uint8_t {
    Continue,
    Finished,
};

CommitAction commit_sampled_token(
    RuntimeRequest& request,
    int32_t token,
    const qwen_hybrid::StopTokens& stop_tokens);

bool thinking_budget_enabled(const RuntimeRequest& request) noexcept;

uint32_t thinking_allowance(const RuntimeRequest& request) noexcept;

void note_generated_token(RuntimeRequest& request, int32_t token) noexcept;

int32_t enforce_thinking_budget(RuntimeRequest& request, int32_t token) noexcept;

CommitAction commit_final_prefill_without_sample(RuntimeRequest& request);

Status record_sampled_token(RuntimeRequest& request, int32_t token);

}
}
