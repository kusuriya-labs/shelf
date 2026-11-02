#pragma once

#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/inference/state/gdn_state_pool.h>

namespace shelf {
namespace kernel {

struct ModelDispatchStateView {
    ::shelf::state::SequenceBlockTableDeviceView block_table{};
    ::shelf::state::GdnStatePoolDeviceView gdn{};
    ::shelf::state::PagedKVPoolBF16View kv_bf16{};
    ::shelf::state::PagedKVPoolFP8View kv_fp8{};
    uint32_t kv_dtype = 0;
};

}
}
