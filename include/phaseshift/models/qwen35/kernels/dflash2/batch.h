#pragma once

#include <phaseshift/core/memory/types.h>
#include <cstdint>

namespace ps::kernel {

struct DFlash2RequestDescriptor {
    const bf16_t* ring_k;
    const bf16_t* ring_v;
    int32_t* output_drafts;
    uint32_t context_rows;
    uint32_t context_start_position;
    uint32_t block_position_start;
    uint32_t block_rows;
    uint32_t row_begin;
    uint32_t num_drafts;
    int32_t anchor_token;
};

}
