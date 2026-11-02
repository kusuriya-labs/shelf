#pragma once

#include <shelf/models/qwen35/runtime/runtime_request.h>
#include <shelf/models/qwen35/runtime/scheduling_types.h>
#include <shelf/core/status.h>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ps {
namespace qwen35 {

namespace runtime {

Result<SchedulePlan> schedule_requests(
    std::span<const RuntimeRequest*> active,
    uint32_t max_scheduled_tokens,
    uint32_t max_scheduled_requests,
    uint32_t page_tokens,
    KVBankerState* kv);

}
}
}
