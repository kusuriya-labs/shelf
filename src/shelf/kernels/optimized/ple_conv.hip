#include <shelf/kernels/optimized/ple_conv.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::kernel {
namespace {

constexpr uint32_t kThreads = kPleConvThreads;

__device__ __forceinline__ float bf16_value(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<std::uint32_t>(p->data) << 16);
}

__device__ __forceinline__ shelf::bf16_t bf16_from(float value) {
    return shelf::bf16_t{
        static_cast<std::uint16_t>(__bfloat16_as_ushort(__float2bfloat16(value)))};
}

__global__ __launch_bounds__(kThreads) void ple_conv_kernel(PleConvArgs args) {
    const std::uint32_t ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= args.channels) return;
    const ::shelf::runtime::DeviceRequestDescriptor request = args.requests[blockIdx.y];
    const std::uint32_t slot = request.request_handle.slot;
    if (slot >= args.max_sequences) return;

    shelf::bf16_t* state = args.state +
        static_cast<std::uint64_t>(slot) * args.state_slot_stride +
        static_cast<std::uint64_t>(args.state_index) * args.state_layer_stride;
    float history[kPleConvHistory];
    for (std::uint32_t i = 0; i < kPleConvHistory; ++i) {
        history[i] = bf16_value(state + static_cast<std::uint64_t>(i) *
                                             args.state_history_stride + ch);
    }

    const shelf::bf16_t* weight = args.weight + static_cast<std::uint64_t>(ch) * 4u;
    const float w0 = bf16_value(weight + 0);
    const float w1 = bf16_value(weight + 1);
    const float w2 = bf16_value(weight + 2);
    const float w3 = bf16_value(weight + 3);

    for (std::uint32_t t = 0; t < request.row_count; ++t) {
        const std::uint32_t row = request.row_begin + t;
        if (row >= args.actual_rows) break;
        if (args.state_history_store != nullptr && row < args.history_rows) {
            shelf::bf16_t* hp = args.state_history_store +
                static_cast<std::uint64_t>(args.state_index) * args.history_state_stride +
                static_cast<std::uint64_t>(row) * args.state_history_stride *
                    kPleConvHistory;
            for (std::uint32_t i = 0; i < kPleConvHistory; ++i) {
                hp[static_cast<std::uint64_t>(i) * args.state_history_stride + ch] =
                    bf16_from(history[i]);
            }
        }
        const float x = bf16_value(args.input +
                                   static_cast<std::uint64_t>(row) *
                                       args.input_row_stride + ch);
        const float y = w0 * history[0] + w1 * history[3] + w2 * history[6] + w3 * x;
        const float silu = y / (1.0f + expf(-y));
        args.output[static_cast<std::uint64_t>(row) * args.output_row_stride + ch] =
            bf16_from(silu);
        for (std::uint32_t i = 0; i + 1u < kPleConvHistory; ++i) history[i] = history[i + 1u];
        history[kPleConvHistory - 1u] = x;
    }

    for (std::uint32_t i = 0; i < kPleConvHistory; ++i) {
        state[static_cast<std::uint64_t>(i) * args.state_history_stride + ch] =
            bf16_from(history[i]);
    }
}

}  // namespace

hipError_t launch_ple_conv(const PleConvArgs& args, hipStream_t stream) {
    if (args.input == nullptr || args.output == nullptr || args.weight == nullptr ||
        args.state == nullptr || args.requests == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.num_requests == 0u || args.actual_rows == 0u || args.channels == 0u ||
        args.max_sequences == 0u) {
        return hipErrorInvalidValue;
    }
    if (args.input_row_stride < args.channels ||
        args.output_row_stride < args.channels ||
        args.state_slot_stride < args.state_layer_stride ||
        args.state_layer_stride < kPleConvHistory * args.state_history_stride ||
        args.state_history_stride < args.channels) {
        return hipErrorInvalidValue;
    }
    ple_conv_kernel<<<dim3(ple_conv_blocks(args.channels), args.num_requests), kThreads, 0,
                      stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
