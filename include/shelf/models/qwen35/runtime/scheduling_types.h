#pragma once

#include <shelf/runtime/scheduling_types.h>

namespace shelf::qwen35::runtime {

using namespace ::shelf::runtime;

using ::shelf::scheduling::ScheduledRequest;
using ::shelf::scheduling::ScheduledBatch;
using ::shelf::scheduling::TokenIdsLocation;
using ::shelf::scheduling::validate_scheduled_batch;
using ::shelf::scheduling::SchedulePlan;
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

}  // namespace shelf::qwen35::runtime

namespace shelf::qwen35 {

using ::shelf::runtime::ScheduledBatch;
using ::shelf::runtime::ScheduledRequest;
using ::shelf::runtime::SchedulePlan;
using ::shelf::runtime::TokenIdsLocation;
using ::shelf::runtime::validate_scheduled_batch;

}  // namespace shelf::qwen35
