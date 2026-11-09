#pragma once

#include <shelf/inference/state/paged_kv_pool.h>
#include <shelf/inference/state/paged_sequence_state.h>
#include <shelf/inference/state/sequence_slot_pool.h>
#include <shelf/inference/state/gdn_state_pool.h>
#include <shelf/inference/state/qsa_state_pool.h>

namespace shelf {
namespace kernel {

struct PleConvStateView {
    shelf::bf16_t* base = nullptr;
    uint32_t max_sequences = 0;
    uint32_t channels = 0;
    uint64_t slot_stride = 0;
    uint64_t layer_stride = 0;
    uint64_t history_stride = 0;
};

struct ModelDispatchStateView {
    ::shelf::state::SequenceBlockTableDeviceView block_table{};
    ::shelf::state::GdnStatePoolDeviceView gdn{};
    ::shelf::state::PagedKVPoolBF16View kv_bf16{};
    ::shelf::state::PagedKVPoolFP8View kv_fp8{};
    ::shelf::state::PagedKVPoolOctaveView kv_octave{};
    PleConvStateView ple_conv{};
    ::shelf::state::QsaKeyStatePoolDeviceView qsa{};
    uint32_t kv_dtype = 0;
};

}
}
