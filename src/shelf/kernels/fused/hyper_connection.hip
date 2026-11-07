#include <shelf/kernels/fused/hyper_connection.h>

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

// Sigmoid is fused into the stream reduction so the [rows, hc * hidden] gate buffer is
// written once and read once.
__global__ __launch_bounds__(kBlock) void hyper_mix_kernel(HyperMixArgs args) {
    const uint32_t row = blockIdx.x;
    const shelf::bf16_t* gate = static_cast<const shelf::bf16_t*>(args.gate) +
                                static_cast<std::uint64_t>(row) * args.gate_stride;
    const shelf::bf16_t* normed = static_cast<const shelf::bf16_t*>(args.normed) +
                                  static_cast<std::uint64_t>(row) * args.normed_stride;
    shelf::bf16_t* output = static_cast<shelf::bf16_t*>(args.output) +
                            static_cast<std::uint64_t>(row) * args.output_stride;
    for (uint32_t i = threadIdx.x; i < args.hidden; i += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < args.hc; ++h) {
            const std::uint32_t offset = h * args.hidden + i;
            float g = bf16_load(gate + offset);
            if (args.sigmoid_gate != 0u) g = 1.0f / (1.0f + __expf(-g));
            acc += g * bf16_load(normed + offset);
        }
        output[i] = bf16_store(acc / static_cast<float>(args.hc));
    }
}

// Scale(1/hc) -> Sigmoid -> Scale(2) in one pass.
__global__ __launch_bounds__(kBlock) void hyper_inject_kernel(HyperInjectArgs args) {
    const uint32_t row = blockIdx.x;
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input) +
                                 static_cast<std::uint64_t>(row) * args.input_stride;
    shelf::bf16_t* output = static_cast<shelf::bf16_t*>(args.output) +
                            static_cast<std::uint64_t>(row) * args.output_stride;
    for (uint32_t h = threadIdx.x; h < args.hc; h += blockDim.x) {
        const float x = bf16_load(input + h) * args.inverse_hc;
        output[h] = bf16_store(args.scale / (1.0f + __expf(-x)));
    }
}

}  // namespace

hipError_t launch_hyper_mix(const HyperMixArgs& args, hipStream_t stream) {
    if (args.gate == nullptr || args.normed == nullptr || args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.hc == 0u || args.hidden == 0u) return hipErrorInvalidValue;
    if (args.gate_stride < args.hc * args.hidden ||
        args.normed_stride < args.hc * args.hidden ||
        args.output_stride < args.hidden) {
        return hipErrorInvalidValue;
    }
    hyper_mix_kernel<<<args.rows, kBlock, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_hyper_inject(const HyperInjectArgs& args, hipStream_t stream) {
    if (args.input == nullptr || args.output == nullptr) return hipErrorInvalidValue;
    if (args.rows == 0u || args.hc == 0u || args.inverse_hc == 0.0f)
        return hipErrorInvalidValue;
    if (args.input_stride < args.hc || args.output_stride < args.hc)
        return hipErrorInvalidValue;
    hyper_inject_kernel<<<args.rows, 64, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
