#include <shelf/models/qwen35/kernels/dflash2/kv_ring.h>
#include <shelf/kernels/optimized/detail/vector_io.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace ps::kernel {

using namespace ::ps::kernel::detail;

constexpr uint32_t kKvRingThreads = 256u;

namespace {

__global__ void dflash2_kv_ring_append_kernel(
    const bf16* k_src,
    const bf16* v_src,
    bf16* k_ring,
    bf16* v_ring,
    uint32_t rows,
    uint32_t kv_features,
    uint32_t capacity,
    uint32_t position_start) {
    const std::size_t total = static_cast<std::size_t>(rows) * kv_features;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const uint32_t row = static_cast<uint32_t>(i / kv_features);
        const uint32_t feature = static_cast<uint32_t>(i % kv_features);
        const uint32_t slot = (position_start + row) % capacity;
        const std::size_t dst =
            static_cast<std::size_t>(slot) * kv_features + feature;
        k_ring[dst] = k_src[i];
        v_ring[dst] = v_src[i];
    }
}

}  // namespace

hipError_t launch_dflash2_kv_ring_append_bf16(
    const bf16_t* k_src,
    const bf16_t* v_src,
    bf16_t* k_ring,
    bf16_t* v_ring,
    uint32_t rows,
    uint32_t kv_features,
    uint32_t capacity,
    uint32_t position_start,
    hipStream_t stream) {
    if (k_src == nullptr || v_src == nullptr || k_ring == nullptr || v_ring == nullptr) {
        return hipErrorInvalidValue;
    }
    if (rows == 0u) {
        return hipSuccess;
    }
    if (kv_features == 0u || capacity == 0u) {
        return hipErrorInvalidValue;
    }
    const std::size_t total = static_cast<std::size_t>(rows) * kv_features;
    const uint32_t blocks =
        static_cast<uint32_t>((total + kKvRingThreads - 1u) / kKvRingThreads);
    dflash2_kv_ring_append_kernel<<<blocks, kKvRingThreads, 0, stream>>>(
        reinterpret_cast<const bf16*>(k_src), reinterpret_cast<const bf16*>(v_src),
        reinterpret_cast<bf16*>(k_ring), reinterpret_cast<bf16*>(v_ring), rows,
        kv_features, capacity, position_start);
    return hipGetLastError();
}

}  // namespace ps::kernel
