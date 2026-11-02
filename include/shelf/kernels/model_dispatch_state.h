#pragma once

#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/inference/state/gdn_state_pool.h>

namespace ps {
namespace kernel {

struct ModelDispatchStateView {
    ::ps::state::SequenceBlockTableDeviceView block_table{};
    ::ps::state::GdnStatePoolDeviceView gdn{};
    ::ps::state::PagedKVPoolBF16View kv_bf16{};
    ::ps::state::PagedKVPoolFP8View kv_fp8{};
    uint32_t kv_dtype = 0;
};

}
}
