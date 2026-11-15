#pragma once

#include <shelf/kernels/kv_types.h>
#include <shelf/kernels/optimized/kv_append.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {

hipError_t launch_kv_append_octave(
    const KvAppendCommonArgs& common,
    uint8_t* k_pool,
    uint8_t* v_pool,
    const float* levels,
    OctaveKVCacheFormat format,
    hipStream_t stream);

}  // namespace shelf::kernel
