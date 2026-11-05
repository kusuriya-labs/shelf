#include <shelf/kernels/optimized/hyper_connection.h>

#include <shelf/core/memory/types.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {
namespace {

constexpr uint32_t kBlock = 256u;

__device__ __forceinline__ float bf16_load(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<std::uint32_t>(p->data) << 16);
}

__device__ __forceinline__ shelf::bf16_t bf16_store(float value) {
    return shelf::bf16_t{static_cast<std::uint16_t>(__bfloat16_as_ushort(
        __float2bfloat16(value)))};
}

__global__ __launch_bounds__(kBlock) void hyper_apply_kernel(HyperApplyArgs args) {
    const uint32_t row = blockIdx.x;
    const shelf::bf16_t* raw = static_cast<const shelf::bf16_t*>(args.raw) +
                               static_cast<std::uint64_t>(row) * args.raw_stride;
    const shelf::bf16_t* y = static_cast<const shelf::bf16_t*>(args.y) +
                             static_cast<std::uint64_t>(row) * args.y_stride;
    const shelf::bf16_t* injection = static_cast<const shelf::bf16_t*>(args.injection) +
                                     static_cast<std::uint64_t>(row) * args.injection_stride;
    shelf::bf16_t* output = static_cast<shelf::bf16_t*>(args.output) +
                            static_cast<std::uint64_t>(row) * args.output_stride;
    const uint32_t width = args.hc * args.hidden;
    for (uint32_t index = threadIdx.x; index < width; index += blockDim.x) {
        const uint32_t h = index / args.hidden;
        const uint32_t i = index % args.hidden;
        const float value = bf16_load(raw + index) +
                            bf16_load(y + i) * bf16_load(injection + h);
        output[index] = bf16_store(value);
    }
}

}  // namespace

hipError_t launch_hyper_apply(const HyperApplyArgs& args, hipStream_t stream) {
    if (args.raw == nullptr || args.y == nullptr || args.injection == nullptr ||
        args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.hc == 0u || args.hidden == 0u) return hipErrorInvalidValue;
    if (args.raw_stride < args.hc * args.hidden || args.y_stride < args.hidden ||
        args.injection_stride < args.hc ||
        args.output_stride < args.hc * args.hidden) {
        return hipErrorInvalidValue;
    }
    hyper_apply_kernel<<<args.rows, kBlock, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
