#include <shelf/kernels/optimized/row_gather.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

namespace {

constexpr std::uint32_t kThreads = 256;

__global__ __launch_bounds__(kThreads)
void row_gather_kernel(const char* __restrict__ src,
                       const RowGatherItem* __restrict__ items,
                       std::uint32_t count, char* __restrict__ dst) {
    const std::uint32_t i = blockIdx.x;
    if (i >= count) return;
    const RowGatherItem it = items[i];
    const char* s = src + it.src_offset;
    char* d = dst + it.dst_offset;
    for (std::uint32_t c = threadIdx.x; c < it.bytes; c += kThreads) {
        d[c] = s[c];
    }
}

}  // namespace

hipError_t launch_row_gather(const void* src, const RowGatherItem* device_items,
                             std::uint32_t count, void* dst, hipStream_t stream) {
    if (src == nullptr || device_items == nullptr || dst == nullptr || count == 0u) {
        return hipErrorInvalidValue;
    }
    row_gather_kernel<<<count, kThreads, 0, stream>>>(
        static_cast<const char*>(src), device_items, count, static_cast<char*>(dst));
    return hipGetLastError();
}

}  // namespace ps::kernel
