#pragma once

#include <phaseshift/inference/scheduling/sampling_params.h>
#include <phaseshift/inference/scheduling/scheduled_batch.h>
#include <phaseshift/inference/scheduling/schedule_plan.h>
#include <phaseshift/inference/scheduling/kv_capacity_manager.h>
#include <phaseshift/inference/scheduling/kv_banker.h>

namespace ps {
namespace qwen35 {

using ::ps::scheduling::ScheduledRequest;
using ::ps::scheduling::ScheduledBatch;
using ::ps::scheduling::TokenIdsLocation;
using ::ps::scheduling::validate_scheduled_batch;
using ::ps::scheduling::SchedulePlan;

namespace runtime {

using ::ps::scheduling::SamplingConfig;
using ::ps::scheduling::SamplingMode;
using ::ps::scheduling::SamplingRequest;
using ::ps::scheduling::validate_sampling_config;
using ::ps::scheduling::make_sampling_request;
using ::ps::scheduling::sampling_config_is_default;

using ::ps::scheduling::KVBankerProcess;
using ::ps::scheduling::KVBankerState;
using ::ps::scheduling::validate_banker_state;
using ::ps::scheduling::is_safe;
using ::ps::scheduling::can_add_process;
using ::ps::scheduling::can_grant_pages;
using ::ps::scheduling::max_safe_grant_pages;

using ::ps::scheduling::KVAdmissionPolicy;
using ::ps::scheduling::KVCapacitySnapshot;
using ::ps::scheduling::KVCapacityManager;
using ::ps::scheduling::compute_max_kv_tokens;
using ::ps::scheduling::kv_pages_for_tokens;
using ::ps::scheduling::additional_kv_pages;

}

}
}
