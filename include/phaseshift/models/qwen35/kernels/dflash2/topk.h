#pragma once

#include <phaseshift/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kDflash2TopKThreads = 256u;
inline constexpr uint32_t kDflash2TopKMaxK = 16u;
inline constexpr uint32_t kDflash2TopKMaxPartitions = 64u;
inline constexpr uint32_t kDflash2TopKScratchPerRow =
    kDflash2TopKThreads * kDflash2TopKMaxK;

struct DFlash2TopKPlan {
    uint32_t partitions = 32u;
    uint32_t threads = 256u;
};

DFlash2TopKPlan dflash2_topk_default_plan();

hipError_t launch_dflash2_topk_f32(
    const float* logits,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t row_stride,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    int32_t* scratch_ids,
    float* scratch_logits,
    hipStream_t stream);

hipError_t launch_dflash2_topk_f32_optimized(
    const float* logits,
    uint32_t rows,
    uint32_t vocab_size,
    uint32_t row_stride,
    uint32_t k,
    int32_t* out_ids,
    float* out_logits,
    int32_t* scratch_ids,
    float* scratch_logits,
    const DFlash2TopKPlan& plan,
    hipStream_t stream);

}  // namespace ps::kernel
