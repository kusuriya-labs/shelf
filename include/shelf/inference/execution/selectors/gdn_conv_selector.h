#pragma once

#include <shelf/runtime/graph/value_type.h>
#include <cstdint>

namespace ps::execution {

enum class GdnConvImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct GdnConvSelectorInput {
    ::ps::runtime::ValueDType input_dtype;
    ::ps::runtime::ValueDType output_dtype;
    uint32_t conv_dim;
    uint32_t history;
    uint32_t rows;
    uint32_t num_requests;
    uint32_t max_request_rows;
};

GdnConvImplementation select_gdn_conv_implementation(
    const GdnConvSelectorInput& in);

}  // namespace ps::execution
