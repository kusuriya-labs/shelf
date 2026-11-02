#pragma once

#include <shelf/inference/state/paged_types.h>
#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/inference/state/gdn_state.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/kernels/kv_types.h>

namespace shelf {
namespace qwen35 {

using ::shelf::state::PageId;
using ::shelf::state::SequenceSlotId;
using ::shelf::state::kInvalidPageId;
using ::shelf::state::kInvalidSlot;

using ::shelf::state::SequenceBlockTableDeviceView;
using ::shelf::state::SequenceSlotPool;

using ::shelf::state::PagedKVBatchDeviceView;
using ::shelf::state::PagedKVBatchFP8DeviceView;
using ::shelf::state::PagedKVPoolBF16View;
using ::shelf::state::PagedKVPoolFP8View;
using ::shelf::state::PagedKVPool;
using ::shelf::state::make_paged_kv_bf16_view;
using ::shelf::state::make_paged_kv_fp8_view;

using ::shelf::state::GDNState;
using ::shelf::state::GdnStatePoolDeviceView;
using ::shelf::state::GdnStatePoolLayout;
using ::shelf::state::GdnStatePool;

using ::shelf::state::PagedSequenceState;
using ::shelf::state::create_paged_sequence_state;
using ::shelf::state::release_paged_sequence_state;
using ::shelf::state::reserve_sequence_append;
using ::shelf::state::rollback_sequence_append;
using ::shelf::state::commit_sequence_append;
using ::shelf::state::validate_sequence_append_commit;
using ::shelf::state::runtime_install_sequence_state;

using ::shelf::kernel::KVCacheDType;
using ::shelf::kernel::fp8e4m3_storage_t;

}
}
