#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

struct RowGatherItem {
    std::uint64_t src_offset = 0;
    std::uint64_t dst_offset = 0;
    std::uint32_t bytes = 0;
};

hipError_t launch_row_gather(const void* src, const RowGatherItem* device_items,
                             std::uint32_t count, void* dst, hipStream_t stream);

}  // namespace ps::kernel
