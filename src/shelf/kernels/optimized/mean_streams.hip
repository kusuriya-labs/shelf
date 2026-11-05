#include <shelf/kernels/optimized/mean_streams.h>

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

__global__ __launch_bounds__(kBlock) void mean_streams_kernel(MeanStreamsArgs args) {
    const uint32_t row = blockIdx.x;
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input) +
                                 static_cast<std::uint64_t>(row) * args.input_stride;
    shelf::bf16_t* output = static_cast<shelf::bf16_t*>(args.output) +
                            static_cast<std::uint64_t>(row) * args.output_stride;
    for (uint32_t i = threadIdx.x; i < args.hidden; i += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < args.hc; ++h)
            acc += bf16_load(input + static_cast<std::uint64_t>(h) * args.hidden + i);
        output[i] = bf16_store(acc / static_cast<float>(args.hc));
    }
}

}  // namespace

hipError_t launch_mean_streams_bf16(const MeanStreamsArgs& args, hipStream_t stream) {
    if (args.input == nullptr || args.output == nullptr) return hipErrorInvalidValue;
    if (args.rows == 0u || args.hc == 0u || args.hidden == 0u) return hipErrorInvalidValue;
    if (args.input_stride < static_cast<std::uint64_t>(args.hc) * args.hidden ||
        args.output_stride < args.hidden)
        return hipErrorInvalidValue;
    mean_streams_kernel<<<args.rows, kBlock, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
