#pragma once

#include <shelf/core/status.h>
#include <shelf/models/qwen35/runtime/executor.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/models/qwen35/state/state_types.h>

namespace shelf {
namespace qwen35 {
namespace runtime {

class TpBatchHook {
public:
    virtual ~TpBatchHook() = default;

    virtual Status on_sequence_created(const PagedSequenceState& source) = 0;

    virtual Status on_sequence_restored(const PagedSequenceState& source) {
        (void)source;
        return Status::make_ok();
    }

    virtual uint32_t on_prefix_rank_count() { return 1u; }

    virtual PagedSequenceState* on_prefix_rank_sequence(
        const PagedSequenceState& source, uint32_t rank) {
        (void)source;
        (void)rank;
        return nullptr;
    }

    virtual Status on_sequence_released(const PagedSequenceState& source) = 0;

    virtual Result<BatchExecutionOutput> on_execute(const ScheduledBatch& batch) = 0;

    virtual Result<BatchExecutionOutput> on_execute(
        const ScheduledBatch& batch, const ExecuteBatchOptions* per_rank) = 0;
};

}
}
}
