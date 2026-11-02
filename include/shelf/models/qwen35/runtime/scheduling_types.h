#pragma once

#include <shelf/inference/scheduling/sampling_params.h>
#include <shelf/inference/scheduling/scheduled_batch.h>
#include <shelf/inference/scheduling/schedule_plan.h>
#include <shelf/inference/scheduling/kv_capacity_manager.h>
#include <shelf/inference/scheduling/kv_banker.h>

namespace shelf {
namespace qwen35 {

using ::shelf::scheduling::ScheduledRequest;
using ::shelf::scheduling::ScheduledBatch;
using ::shelf::scheduling::TokenIdsLocation;
using ::shelf::scheduling::validate_scheduled_batch;
using ::shelf::scheduling::SchedulePlan;

namespace runtime {

using ::shelf::scheduling::SamplingConfig;
using ::shelf::scheduling::SamplingMode;
using ::shelf::scheduling::SamplingRequest;
using ::shelf::scheduling::validate_sampling_config;
using ::shelf::scheduling::make_sampling_request;
using ::shelf::scheduling::sampling_config_is_default;

using ::shelf::scheduling::KVBankerProcess;
using ::shelf::scheduling::KVBankerState;
using ::shelf::scheduling::validate_banker_state;
using ::shelf::scheduling::is_safe;
using ::shelf::scheduling::can_add_process;
using ::shelf::scheduling::can_grant_pages;
using ::shelf::scheduling::max_safe_grant_pages;

using ::shelf::scheduling::KVAdmissionPolicy;
using ::shelf::scheduling::KVCapacitySnapshot;
using ::shelf::scheduling::KVCapacityManager;
using ::shelf::scheduling::compute_max_kv_tokens;
using ::shelf::scheduling::kv_pages_for_tokens;
using ::shelf::scheduling::additional_kv_pages;

}

}
}
