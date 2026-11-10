#pragma once

#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/scheduling/sampling_params.h>
#include <shelf/runtime/execution/execution_types.h>
#include <shelf/runtime/request_handle.h>
#include <shelf/core/status.h>
#include <cstdint>
#include <span>

namespace shelf {
namespace scheduling {


struct ScheduledRequest {
    ::shelf::state::PagedSequenceState* sequence = nullptr;

    ::shelf::runtime::RequestHandle handle{};

    ::shelf::runtime::ExecutionClass execution_class = ::shelf::runtime::ExecutionClass::DECODE;

    uint32_t token_begin = 0;
    uint32_t num_tokens = 0;

    uint32_t prefix_tokens = 0;

    bool compute_logits = false;
    bool sample = false;

    ::shelf::scheduling::SamplingConfig sampling{};
    uint64_t sampling_index = 0;
    const uint32_t* token_bitmask = nullptr;

    uint32_t num_output_rows = 1;
};

enum class TokenIdsLocation : uint8_t {
    Host,
    Device,
};

struct ScheduledBatch {
    const int32_t* token_ids = nullptr;
    TokenIdsLocation token_ids_location = TokenIdsLocation::Host;

    std::span<const ScheduledRequest> requests;

    uint32_t num_tokens = 0;
    uint32_t num_requests = 0;

    uint32_t num_decode_requests = 0;
    uint32_t num_verify_requests = 0;
    uint32_t num_prefill_requests = 0;

    bool speculative_verify = false;
};

Status validate_scheduled_batch(
    const ScheduledBatch& batch,
    uint32_t max_tokens,
    uint32_t max_requests);

}
}
