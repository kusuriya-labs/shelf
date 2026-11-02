#include <phaseshift/inference/scheduling/scheduled_batch.h>
#include <phaseshift/inference/state/paged_sequence_state.h>

namespace ps {
namespace scheduling {

namespace {
inline constexpr uint32_t kScheduledVerifyCandidatesMax = 32u;
}

Status validate_scheduled_batch(
    const ScheduledBatch& batch,
    uint32_t max_tokens,
    uint32_t max_requests) {

    if (batch.num_tokens == 0) {
        return Status::invalid_argument("num_tokens must be > 0", __FILE__, __LINE__);
    }
    if (batch.num_requests == 0) {
        return Status::invalid_argument("num_requests must be > 0", __FILE__, __LINE__);
    }
    if (max_tokens != 0 && batch.num_tokens > max_tokens) {
        return Status::invalid_argument(
            "num_tokens exceeds max_scheduled_tokens", __FILE__, __LINE__);
    }
    if (max_requests != 0 && batch.num_requests > max_requests) {
        return Status::invalid_argument(
            "num_requests exceeds max_scheduled_requests", __FILE__, __LINE__);
    }
    if (batch.token_ids == nullptr) {
        return Status::invalid_argument("token_ids must be non-null", __FILE__, __LINE__);
    }
    if (batch.requests.size() != batch.num_requests) {
        return Status::invalid_argument(
            "requests span size != num_requests", __FILE__, __LINE__);
    }

    if (batch.num_decode_requests +
        batch.num_verify_requests +
        batch.num_prefill_requests !=
        batch.num_requests) {
        return Status::invalid_argument(
            "decode + verify + prefill request counts != num_requests",
            __FILE__,
            __LINE__);
    }

    uint32_t decode_count = 0;
    uint32_t verify_count = 0;
    uint32_t prefill_count = 0;
    uint32_t cursor = 0;

    enum class BatchRegion : uint8_t {
        Decode,
        Verify,
        Prefill,
    };
    BatchRegion region = BatchRegion::Decode;

    for (uint32_t i = 0; i < batch.num_requests; ++i) {
        const ScheduledRequest& req = batch.requests[i];

        if (req.sequence == nullptr) {
            return Status::invalid_argument("request sequence is null", __FILE__, __LINE__);
        }
        if (!req.sequence->is_usable()) {
            return Status::invalid_argument("request sequence is not usable", __FILE__, __LINE__);
        }
        if (!::ps::runtime::request_handle_valid(req.handle)) {
            return Status::invalid_argument("request handle is invalid", __FILE__, __LINE__);
        }
        if (!::ps::runtime::request_handle_equal(req.handle, req.sequence->request_handle())) {
            return Status::invalid_argument(
                "scheduled request handle is stale", __FILE__, __LINE__);
        }
        if (req.num_tokens == 0) {
            return Status::invalid_argument("request num_tokens must be > 0", __FILE__, __LINE__);
        }

        if (req.token_begin != cursor) {
            return Status::invalid_argument(
                "token range gap/overlap detected", __FILE__, __LINE__);
        }

        for (uint32_t j = 0; j < i; ++j) {
            if (batch.requests[j].handle.slot == req.handle.slot) {
                return Status::invalid_argument(
                    "duplicate sequence slot in batch", __FILE__, __LINE__);
            }
        }

        if (req.prefix_tokens != req.sequence->position) {
            return Status::invalid_argument(
                "prefix_tokens != sequence.position", __FILE__, __LINE__);
        }
        if (req.prefix_tokens + req.num_tokens > req.sequence->max_seq_len) {
            return Status::invalid_argument(
                "prefix + num_tokens exceeds max_seq_len", __FILE__, __LINE__);
        }

        switch (req.execution_class) {
            case ::ps::runtime::ExecutionClass::DECODE:
                if (req.num_tokens != 1) {
                    return Status::invalid_argument(
                        "DECODE request must have num_tokens == 1", __FILE__, __LINE__);
                }
                if (region != BatchRegion::Decode) {
                    return Status::invalid_argument(
                        "DECODE request violates target batch order", __FILE__, __LINE__);
                }
                ++decode_count;
                break;
            case ::ps::runtime::ExecutionClass::SPEC_VERIFY:
                if (region == BatchRegion::Prefill) {
                    return Status::invalid_argument(
                        "SPEC_VERIFY after PREFILL violates target batch order",
                        __FILE__,
                        __LINE__);
                }
                if (req.num_tokens > kScheduledVerifyCandidatesMax) {
                    return Status::invalid_argument(
                        "SPEC_VERIFY candidate count exceeds the cap", __FILE__,
                        __LINE__);
                }
                if (req.num_output_rows < req.num_tokens) {
                    return Status::invalid_argument(
                        "SPEC_VERIFY needs one output row per candidate",
                        __FILE__, __LINE__);
                }
                if (!req.compute_logits || !req.sample) {
                    return Status::invalid_argument(
                        "SPEC_VERIFY requires logits and sampling", __FILE__,
                        __LINE__);
                }
                region = BatchRegion::Verify;
                ++verify_count;
                break;
            case ::ps::runtime::ExecutionClass::PREFILL:
                region = BatchRegion::Prefill;
                ++prefill_count;
                break;
            case ::ps::runtime::ExecutionClass::SPEC_DRAFT:
                return Status::invalid_argument(
                    "SPEC_DRAFT must not appear in Target ScheduledBatch",
                    __FILE__,
                    __LINE__);
        }

        if (req.sample && !req.compute_logits) {
            return Status::invalid_argument(
                "sample requires compute_logits", __FILE__, __LINE__);
        }
        if (req.sample) {
            const Status sampling_status = ::ps::scheduling::validate_sampling_config(
                req.sampling);
            if (!sampling_status.ok()) return sampling_status;
        }
        if (!req.compute_logits && req.num_output_rows != 1u) {
            return Status::invalid_argument(
                "num_output_rows requires compute_logits", __FILE__, __LINE__);
        }
        if (req.compute_logits &&
            (req.num_output_rows == 0u || req.num_output_rows > req.num_tokens)) {
            return Status::invalid_argument(
                "num_output_rows must be within [1, num_tokens]", __FILE__, __LINE__);
        }

        cursor += req.num_tokens;
    }

    if (cursor != batch.num_tokens) {
        return Status::invalid_argument(
            "flattened token count != num_tokens", __FILE__, __LINE__);
    }
    if (decode_count != batch.num_decode_requests) {
        return Status::invalid_argument(
            "decode request count mismatch", __FILE__, __LINE__);
    }
    if (verify_count != batch.num_verify_requests) {
        return Status::invalid_argument(
            "verify request count mismatch", __FILE__, __LINE__);
    }
    if (prefill_count != batch.num_prefill_requests) {
        return Status::invalid_argument(
            "prefill request count mismatch", __FILE__, __LINE__);
    }

    return Status::make_ok();
}

}
}
