#pragma once
#include <cstdint>

namespace shelf::execution {

enum class GdnRecurrenceImplementation : uint8_t {
    Correctness,
    Optimized,
};

struct GdnRecurrenceSelectorInput {
    uint32_t rows = 0;
    uint32_t num_requests = 0;
    uint32_t max_request_rows = 0;
    uint32_t key_heads = 0;
    uint32_t num_v_heads = 0;
    uint32_t head_k = 0;
    uint32_t head_v = 0;
};

GdnRecurrenceImplementation
select_gdn_recurrence_implementation(const GdnRecurrenceSelectorInput& input);

}  // namespace shelf::execution
