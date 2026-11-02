#pragma once

#include <phaseshift/inference/state/paged_types.h>
#include <phaseshift/inference/state/paged_kv_pool.h>
#include <phaseshift/inference/state/sequence_slot_pool.h>
#include <phaseshift/inference/state/gdn_state.h>
#include <phaseshift/inference/state/gdn_state_pool.h>
#include <phaseshift/inference/state/paged_sequence_state.h>
#include <phaseshift/kernels/kv_types.h>

namespace ps {
namespace qwen35 {

using ::ps::state::PageId;
using ::ps::state::SequenceSlotId;
using ::ps::state::kInvalidPageId;
using ::ps::state::kInvalidSlot;

using ::ps::state::SequenceBlockTableDeviceView;
using ::ps::state::SequenceSlotPool;

using ::ps::state::PagedKVBatchDeviceView;
using ::ps::state::PagedKVBatchFP8DeviceView;
using ::ps::state::PagedKVPoolBF16View;
using ::ps::state::PagedKVPoolFP8View;
using ::ps::state::PagedKVPool;
using ::ps::state::make_paged_kv_bf16_view;
using ::ps::state::make_paged_kv_fp8_view;

using ::ps::state::GDNState;
using ::ps::state::GdnStatePoolDeviceView;
using ::ps::state::GdnStatePoolLayout;
using ::ps::state::GdnStatePool;

using ::ps::state::PagedSequenceState;
using ::ps::state::create_paged_sequence_state;
using ::ps::state::release_paged_sequence_state;
using ::ps::state::reserve_sequence_append;
using ::ps::state::rollback_sequence_append;
using ::ps::state::commit_sequence_append;
using ::ps::state::validate_sequence_append_commit;
using ::ps::state::runtime_install_sequence_state;

using ::ps::kernel::KVCacheDType;
using ::ps::kernel::fp8e4m3_storage_t;

}
}
