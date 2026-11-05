#include <shelf/kernels/optimized/ple.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace shelf::kernel {
namespace {

constexpr uint32_t kBlock = 256u;

__device__ __forceinline__ float bf16_value(const shelf::bf16_t* p) {
    return __uint_as_float(static_cast<std::uint32_t>(p->data) << 16);
}

__device__ __forceinline__ shelf::bf16_t bf16_from(float v) {
    return shelf::bf16_t{
        static_cast<std::uint16_t>(__bfloat16_as_ushort(__float2bfloat16(v)))};
}

__device__ __forceinline__ float block_sum(float* scratch, float value) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (uint32_t stride = kBlock / 2u; stride > 0u; stride >>= 1u) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

__global__ __launch_bounds__(kBlock) void ple_stream_dot_kernel(
    PleStreamDotArgs args) {
    const uint32_t task = blockIdx.x;
    const uint32_t row = task / args.hc;
    const uint32_t stream = task % args.hc;
    __shared__ float scratch[kBlock];
    const shelf::bf16_t* key =
        args.key + static_cast<std::uint64_t>(row) * args.key_stride +
        static_cast<std::uint64_t>(stream) * args.hidden;
    const shelf::bf16_t* query =
        args.query + static_cast<std::uint64_t>(row) * args.query_stride +
        static_cast<std::uint64_t>(stream) * args.hidden;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < args.hidden; i += blockDim.x)
        sum += bf16_value(key + i) * bf16_value(query + i);
    const float total = block_sum(scratch, sum);
    if (threadIdx.x == 0u) {
        args.output[static_cast<std::uint64_t>(row) * args.output_stride + stream] =
            bf16_from(total);
    }
}

__global__ __launch_bounds__(kBlock) void ple_gate_kernel(PleGateArgs args) {
    const uint32_t row = blockIdx.x;
    const shelf::bf16_t* input = args.input + static_cast<std::uint64_t>(row) * args.input_stride;
    shelf::bf16_t* output = args.output + static_cast<std::uint64_t>(row) * args.output_stride;
    for (uint32_t i = threadIdx.x; i < args.hidden; i += blockDim.x) {
        const float x = bf16_value(input + i);
        const float magnitude = fabsf(x);
        const float root = sqrtf(magnitude > 1.0e-6f ? magnitude : 1.0e-6f);
        const float sign = x < 0.0f ? -1.0f : (x > 0.0f ? 1.0f : 0.0f);
        output[i] = bf16_from(sign * root);
    }
}

__global__ __launch_bounds__(kBlock) void ple_broadcast_mul_kernel(
    PleBroadcastMulArgs args) {
    const uint32_t row = blockIdx.x;
    const shelf::bf16_t* gate = args.gate + static_cast<std::uint64_t>(row) * args.gate_stride;
    const shelf::bf16_t* value = args.value + static_cast<std::uint64_t>(row) * args.value_stride;
    shelf::bf16_t* output = args.output + static_cast<std::uint64_t>(row) * args.output_stride;
    for (uint32_t i = threadIdx.x; i < args.hidden; i += blockDim.x) {
        const float v = bf16_value(value + i);
        for (uint32_t h = 0; h < args.hc; ++h) {
            const float sigmoid = 1.0f / (1.0f + expf(-bf16_value(gate + h)));
            output[h * args.hidden + i] = bf16_from(v * sigmoid);
        }
    }
}

}  // namespace

hipError_t launch_ple_stream_dot(const PleStreamDotArgs& args, hipStream_t stream) {
    if (args.key == nullptr || args.query == nullptr || args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.hc == 0u || args.hidden == 0u) return hipErrorInvalidValue;
    if (args.key_stride < args.hc * args.hidden ||
        args.query_stride < args.hc * args.hidden ||
        args.output_stride < args.hc) {
        return hipErrorInvalidValue;
    }
    ple_stream_dot_kernel<<<args.rows * args.hc, kBlock, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_ple_gate(const PleGateArgs& args, hipStream_t stream) {
    if (args.input == nullptr || args.output == nullptr) return hipErrorInvalidValue;
    if (args.rows == 0u || args.hidden == 0u) return hipErrorInvalidValue;
    if (args.input_stride < args.hidden || args.output_stride < args.hidden) {
        return hipErrorInvalidValue;
    }
    ple_gate_kernel<<<args.rows, kBlock, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_ple_broadcast_mul(const PleBroadcastMulArgs& args, hipStream_t stream) {
    if (args.gate == nullptr || args.value == nullptr || args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.hc == 0u || args.hidden == 0u) return hipErrorInvalidValue;
    if (args.gate_stride < args.hc || args.value_stride < args.hidden ||
        args.output_stride < args.hc * args.hidden) {
        return hipErrorInvalidValue;
    }
    ple_broadcast_mul_kernel<<<args.rows, kBlock, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
