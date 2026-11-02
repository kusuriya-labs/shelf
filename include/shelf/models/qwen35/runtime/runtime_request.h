#pragma once

#include <shelf/models/qwen35/state/state_types.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/runtime/dflash2_spec_decoder.h>
#include <shelf/models/qwen35/stop_tokens.h>
#include <shelf/core/status.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <memory>
#include <vector>

namespace ps {
namespace qwen35 {

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

struct DFlash2RequestSession {
    dflash2::DFlash2ContextState* context = nullptr;
    std::unique_ptr<DFlash2SpecDecoder> decoder;
    uint32_t pool_index = kDFlash2NoSessionSlot;
    DFlash2SpecTiming timing{};
    int32_t pending_token = -1;
    bool ready = false;
};

struct RuntimeRequest {
    uint64_t id = 0;

    std::vector<int32_t> input_tokens;
    uint32_t max_new_tokens = 0;

    SamplingConfig sampling{};

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

    bool speculative_decode = false;
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
    const StopTokens& stop_tokens);

CommitAction commit_final_prefill_without_sample(RuntimeRequest& request);

Status record_sampled_token(RuntimeRequest& request, int32_t token);

}
}
}
