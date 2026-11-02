#pragma once

#include <shelf/inference/scheduling/scheduled_batch.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/core/status.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ps {
namespace scheduling {

struct SchedulePlan {
    std::vector<int32_t> token_ids;
    std::vector<ScheduledRequest> scheduled_requests;

    ScheduledBatch batch;

    bool empty() const noexcept { return batch.num_requests == 0; }

    std::span<const ScheduledRequest> requests() const noexcept {
        return batch.requests;
    }

    Status append_decode(
        ::ps::state::PagedSequenceState* sequence,
        int32_t token,
        uint32_t prefix_tokens,
        bool compute_logits,
        bool sample,
        const ::ps::scheduling::SamplingConfig& sampling,
        uint64_t sampling_index) {

        if (sequence == nullptr) {
            return Status::invalid_argument(
                "SchedulePlan decode sequence is null", __FILE__, __LINE__);
        }

        if (verify_started_ || prefill_started_) {
            return Status::invalid_state(
                "SchedulePlan cannot append DECODE after VERIFY or PREFILL",
                __FILE__,
                __LINE__);
        }

        const uint32_t token_begin = static_cast<uint32_t>(token_ids.size());

        token_ids.push_back(token);

        ScheduledRequest scheduled;
        scheduled.sequence = sequence;
        scheduled.handle = sequence->request_handle();
        scheduled.execution_class = ::ps::runtime::ExecutionClass::DECODE;
        scheduled.token_begin = token_begin;
        scheduled.num_tokens = 1;
        scheduled.prefix_tokens = prefix_tokens;
        scheduled.compute_logits = compute_logits;
        scheduled.sample = sample;
        scheduled.sampling = sampling;
        scheduled.sampling_index = sampling_index;

        scheduled_requests.push_back(scheduled);

        ++decode_count_;

        return Status::make_ok();
    }

    Status append_prefill(
        ::ps::state::PagedSequenceState* sequence,
        std::span<const int32_t> tokens,
        uint32_t prefix_tokens,
        bool compute_logits,
        bool sample,
        const ::ps::scheduling::SamplingConfig& sampling,
        uint64_t sampling_index) {

        if (sequence == nullptr) {
            return Status::invalid_argument(
                "SchedulePlan prefill sequence is null", __FILE__, __LINE__);
        }

        if (tokens.empty()) {
            return Status::invalid_argument(
                "SchedulePlan PREFILL tokens are empty", __FILE__, __LINE__);
        }

        prefill_started_ = true;

        const uint32_t token_begin = static_cast<uint32_t>(token_ids.size());

        token_ids.insert(token_ids.end(), tokens.begin(), tokens.end());

        ScheduledRequest scheduled;
        scheduled.sequence = sequence;
        scheduled.handle = sequence->request_handle();
        scheduled.execution_class = ::ps::runtime::ExecutionClass::PREFILL;
        scheduled.token_begin = token_begin;
        scheduled.num_tokens = static_cast<uint32_t>(tokens.size());
        scheduled.prefix_tokens = prefix_tokens;
        scheduled.compute_logits = compute_logits;
        scheduled.sample = sample;
        scheduled.sampling = sampling;
        scheduled.sampling_index = sampling_index;

        scheduled_requests.push_back(scheduled);

        ++prefill_count_;

        return Status::make_ok();
    }

    Status append_verify(
        ::ps::state::PagedSequenceState* sequence,
        std::span<const int32_t> tokens,
        uint32_t prefix_tokens,
        uint32_t num_output_rows,
        const ::ps::scheduling::SamplingConfig& sampling,
        uint64_t sampling_index) {

        if (sequence == nullptr) {
            return Status::invalid_argument(
                "SchedulePlan verify sequence is null", __FILE__, __LINE__);
        }

        if (tokens.empty()) {
            return Status::invalid_argument(
                "SchedulePlan VERIFY tokens are empty", __FILE__, __LINE__);
        }

        if (num_output_rows < tokens.size()) {
            return Status::invalid_argument(
                "SchedulePlan VERIFY needs one output row per candidate",
                __FILE__,
                __LINE__);
        }

        verify_started_ = true;

        const uint32_t token_begin = static_cast<uint32_t>(token_ids.size());

        token_ids.insert(token_ids.end(), tokens.begin(), tokens.end());

        ScheduledRequest scheduled;
        scheduled.sequence = sequence;
        scheduled.handle = sequence->request_handle();
        scheduled.execution_class = ::ps::runtime::ExecutionClass::SPEC_VERIFY;
        scheduled.token_begin = token_begin;
        scheduled.num_tokens = static_cast<uint32_t>(tokens.size());
        scheduled.prefix_tokens = prefix_tokens;
        scheduled.compute_logits = true;
        scheduled.sample = true;
        scheduled.sampling = sampling;
        scheduled.sampling_index = sampling_index;
        scheduled.num_output_rows = num_output_rows;

        scheduled_requests.push_back(scheduled);

        ++verify_count_;

        return Status::make_ok();
    }

    void finalize() noexcept {
        batch.token_ids =
            token_ids.empty() ? nullptr : token_ids.data();

        batch.requests = std::span<const ScheduledRequest>(
            scheduled_requests.data(),
            scheduled_requests.size());

        batch.num_tokens = static_cast<uint32_t>(token_ids.size());
        batch.num_requests =
            static_cast<uint32_t>(scheduled_requests.size());

        batch.num_decode_requests = decode_count_;
        batch.num_verify_requests = verify_count_;
        batch.num_prefill_requests = prefill_count_;
        batch.speculative_verify = verify_count_ != 0;
    }

 private:
    uint32_t decode_count_ = 0;
    uint32_t verify_count_ = 0;
    uint32_t prefill_count_ = 0;
    bool verify_started_ = false;
    bool prefill_started_ = false;
};

}
}
