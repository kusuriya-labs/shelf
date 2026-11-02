#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

hipError_t launch_dflash2_kv_ring_append_bf16(
    const bf16_t* k_src,
    const bf16_t* v_src,
    bf16_t* k_ring,
    bf16_t* v_ring,
    uint32_t rows,
    uint32_t kv_features,
    uint32_t capacity,
    uint32_t position_start,
    hipStream_t stream);

}  // namespace ps::kernel
