#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ps::kernel {

inline constexpr uint32_t kDflash2RadixTopnThreads = 256u;
inline constexpr uint32_t kDflash2RadixSortThreads = 128u;
inline constexpr uint32_t kDflash2RadixTopnBins = 256u;
inline constexpr uint32_t kDflash2RadixTopnPasses = 4u;
inline constexpr uint32_t kDflash2RadixMaxPartitions = 128u;
inline constexpr uint32_t kDflash2RadixTopnCrossoverPool = 64u;
inline constexpr uint32_t kDflash2RadixTopnMaxPool = 128u;
inline constexpr uint32_t kDflash2RadixTopnPerPartition = 512u * 16u;

inline uint32_t dflash2_radix_default_partitions(uint32_t vocab) noexcept {
    const uint32_t per_partition = kDflash2RadixTopnPerPartition;
    uint32_t partitions = (vocab + per_partition - 1u) / per_partition;
    if (partitions == 0u)
        partitions = 1u;
    if (partitions > kDflash2RadixMaxPartitions)
        partitions = kDflash2RadixMaxPartitions;
    return partitions;
}

std::size_t dflash2_radix_topn_scratch_bytes(uint32_t rows, uint32_t partitions,
                                             uint32_t vocab);

bool dflash2_radix_topn_preferred(uint32_t pool);

hipError_t launch_dflash2_radix_topn(
    const float* logits,
    uint32_t rows,
    uint32_t vocab,
    uint32_t row_stride,
    uint32_t pool,
    int32_t* out_ids,
    float* out_logits,
    uint32_t partitions,
    void* scratch,
    std::size_t scratch_bytes,
    hipStream_t stream);

}  // namespace ps::kernel
