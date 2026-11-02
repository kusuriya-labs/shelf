#pragma once

#include <phaseshift/core/memory/types.h>
#include <cstdint>

namespace ps {
namespace execution {

struct GdnSpecHistoryDeviceView {
    bf16_t* conv_history = nullptr;
    fp16_t* recurrent_history = nullptr;

    uint32_t capture_rows = 0u;

    uint64_t conv_row_stride = 0u;
    uint64_t recurrent_row_stride = 0u;
};

struct GdnCompactLogDeviceView {
    float* delta = nullptr;
    float* k = nullptr;
    float* a = nullptr;

    uint64_t delta_layer_stride = 0u;
    uint64_t k_layer_stride = 0u;
    uint64_t a_layer_stride = 0u;
};

struct SpecDecodeContext {
    const uint32_t* tree_anc_offsets = nullptr;
    const uint32_t* tree_anc_positions = nullptr;
    uint32_t tree_committed_end = 0;
    const uint32_t* tree_committed_end_rows = nullptr;
    const int32_t* tree_row_parent = nullptr;

    uint32_t min_visible_tokens = 0;
    uint32_t max_visible_tokens = 0;

    GdnSpecHistoryDeviceView gdn_spec_history{};
    GdnCompactLogDeviceView gdn_compact{};
};

}
}
