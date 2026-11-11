#pragma once

#include <shelf/core/memory/types.h>
#include <cstdint>

namespace shelf {
namespace execution {

struct GdnSpecHistoryDeviceView {
    bf16_t* conv_history = nullptr;
    fp16_t* recurrent_history = nullptr;
    // Optional tree-only mask: save recurrent state only at marked rows.
    const uint32_t* recurrent_capture_mask = nullptr;

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

    // Verify leaves the live recurrent state untouched; the accepted prefix is
    // replayed from this log afterwards (commit_gdn_compact_log).
    uint32_t defer_state_store = 0u;
};

struct QsaSpecHistoryDeviceView {
    bf16_t* partial = nullptr;
    uint32_t* counts = nullptr;

    uint32_t rows = 0u;

    uint64_t partial_state_stride = 0u;
    uint64_t count_state_stride = 0u;
};

struct PleConvSpecHistoryDeviceView {
    bf16_t* store = nullptr;

    uint32_t rows = 0u;

    uint64_t state_stride = 0u;
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

    QsaSpecHistoryDeviceView qsa_spec_history{};
    PleConvSpecHistoryDeviceView ple_conv_history{};
};

}
}
