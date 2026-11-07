#include <shelf/kernels/optimized/repeat_streams.h>

#include <hip/hip_runtime.h>

namespace shelf::kernel {
namespace {

constexpr uint32_t kBlock = 256u;

__global__ __launch_bounds__(kBlock) void repeat_streams_kernel(
    RepeatStreamsArgs args) {
    const uint32_t row = blockIdx.x;
    const shelf::bf16_t* input =
        args.input + static_cast<std::uint64_t>(row) * args.input_stride;
    shelf::bf16_t* output =
        args.output + static_cast<std::uint64_t>(row) * args.output_stride;
    for (uint32_t i = threadIdx.x; i < args.hidden; i += blockDim.x) {
        const shelf::bf16_t value = input[i];
        for (uint32_t h = 0; h < args.hc; ++h) output[h * args.hidden + i] = value;
    }
}

}  // namespace

hipError_t launch_repeat_streams(const RepeatStreamsArgs& args, hipStream_t stream) {
    if (args.input == nullptr || args.output == nullptr) return hipErrorInvalidValue;
    if (args.rows == 0u || args.hc == 0u || args.hidden == 0u) return hipErrorInvalidValue;
    if (args.input_stride < args.hidden ||
        args.output_stride < args.hc * args.hidden) {
        return hipErrorInvalidValue;
    }
    repeat_streams_kernel<<<args.rows, kBlock, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
