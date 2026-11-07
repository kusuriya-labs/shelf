#include <shelf/kernels/fused/tp_combine.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {
namespace {

constexpr uint32_t kThreads = 256u;
constexpr uint32_t kMaxBlocks = 1024u;

__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
    return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        if ((u & 0x007FFFFFu) != 0u) return 0x7FC0u;
        return static_cast<uint16_t>(u >> 16);
    }
    const uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    u += bias;
    return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ void load8(const uint16_t* p, float out[8]) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        out[j] = bf16_to_f32(
            static_cast<uint16_t>((w[j >> 1u] >> (16u * (j & 1u))) & 0xFFFFu));
    }
}

__device__ __forceinline__ void add8(const uint16_t* p, float acc[8]) {
    const uint4 v = *reinterpret_cast<const uint4*>(p);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        acc[j] += bf16_to_f32(
            static_cast<uint16_t>((w[j >> 1u] >> (16u * (j & 1u))) & 0xFFFFu));
    }
}

__device__ __forceinline__ void store8(uint16_t* p, const float v[8]) {
    uint4 out;
    uint32_t* w = reinterpret_cast<uint32_t*>(&out);
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        w[j] = static_cast<uint32_t>(f32_to_bf16(v[2u * j])) |
               (static_cast<uint32_t>(f32_to_bf16(v[2u * j + 1u])) << 16u);
    }
    *reinterpret_cast<uint4*>(p) = out;
}

__device__ __forceinline__ void sum8(const uint16_t* own,
                                     const uint16_t* const* peers, uint32_t peer_count,
                                     uint64_t offset, float acc[8]) {
    load8(own + offset, acc);
    for (uint32_t p = 0; p < peer_count; ++p) add8(peers[p] + offset, acc);
}

__global__ __launch_bounds__(kThreads) void tp_sum_residual_bf16_kernel(
    TpSumResidualArgs args) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const bool vectorized = (args.features & 7u) == 0u && (args.combine_stride & 7u) == 0u &&
                            (args.residual_stride & 7u) == 0u &&
                            (args.output_stride & 7u) == 0u;
    if (vectorized) {
        const uint64_t per_row = args.features >> 3u;
        const uint64_t vectors = args.rows * per_row;
        for (uint64_t v = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             v < vectors; v += stride) {
            const uint64_t row = v / per_row;
            const uint64_t col = (v - row * per_row) << 3u;
            float acc[8];
            const uint64_t offset = row * args.combine_stride + col;
            sum8(static_cast<const uint16_t*>(args.own),
                 reinterpret_cast<const uint16_t* const*>(args.peers), args.peer_count,
                 offset, acc);
            add8(static_cast<const uint16_t*>(args.residual) +
                     row * args.residual_stride + col,
                 acc);
            store8(static_cast<uint16_t*>(args.out) + row * args.output_stride + col, acc);
        }
        return;
    }
    const uint64_t total = args.rows * args.features;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const uint64_t row = i / args.features;
        const uint64_t col = i - row * args.features;
        float value = bf16_to_f32(
            static_cast<const uint16_t*>(args.own)[row * args.combine_stride + col]);
        for (uint32_t p = 0; p < args.peer_count; ++p) {
            value += bf16_to_f32(reinterpret_cast<const uint16_t* const*>(args.peers)[p]
                                     [row * args.combine_stride + col]);
        }
        value += bf16_to_f32(
            static_cast<const uint16_t*>(args.residual)[row * args.residual_stride + col]);
        static_cast<uint16_t*>(args.out)[row * args.output_stride + col] =
            f32_to_bf16(value);
    }
}

__global__ __launch_bounds__(kThreads) void tp_sum_hyper_apply_bf16_kernel(
    TpSumHyperApplyArgs args) {
    const uint64_t stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    const bool vectorized = (args.hidden & 7u) == 0u && (args.combine_stride & 7u) == 0u &&
                            (args.raw_stride & 7u) == 0u &&
                            (args.output_stride & 7u) == 0u;
    if (vectorized) {
        const uint64_t per_row = args.hidden >> 3u;
        const uint64_t vectors = args.rows * per_row;
        for (uint64_t v = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             v < vectors; v += stride) {
            const uint64_t row = v / per_row;
            const uint64_t col = (v - row * per_row) << 3u;
            float y[8];
            const uint64_t combine_offset = row * args.combine_stride + col;
            sum8(static_cast<const uint16_t*>(args.own),
                 reinterpret_cast<const uint16_t* const*>(args.peers), args.peer_count,
                 combine_offset, y);
            const uint16_t* injection = static_cast<const uint16_t*>(args.injection) +
                                        row * args.injection_stride;
            for (uint32_t h = 0; h < args.hc; ++h) {
                const float gate = bf16_to_f32(injection[h]);
                const uint64_t offset = row * args.raw_stride +
                                        static_cast<uint64_t>(h) * args.hidden + col;
                float out[8];
                load8(static_cast<const uint16_t*>(args.raw) + offset, out);
#pragma unroll
                for (uint32_t j = 0; j < 8u; ++j) out[j] += y[j] * gate;
                store8(static_cast<uint16_t*>(args.out) +
                           row * args.output_stride +
                           static_cast<uint64_t>(h) * args.hidden + col,
                       out);
            }
        }
        return;
    }
    const uint64_t width = static_cast<uint64_t>(args.hc) * args.hidden;
    const uint64_t total = args.rows * width;
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const uint64_t row = i / width;
        const uint64_t rest = i - row * width;
        const uint32_t h = static_cast<uint32_t>(rest / args.hidden);
        const uint64_t col = rest - static_cast<uint64_t>(h) * args.hidden;
        float value = bf16_to_f32(
            static_cast<const uint16_t*>(args.own)[row * args.combine_stride + col]);
        for (uint32_t p = 0; p < args.peer_count; ++p) {
            value += bf16_to_f32(reinterpret_cast<const uint16_t* const*>(args.peers)[p]
                                     [row * args.combine_stride + col]);
        }
        const float gate = bf16_to_f32(
            static_cast<const uint16_t*>(args.injection)[row * args.injection_stride + h]);
        const uint64_t offset = row * args.raw_stride +
                                static_cast<uint64_t>(h) * args.hidden + col;
        const float raw = bf16_to_f32(static_cast<const uint16_t*>(args.raw)[offset]);
        static_cast<uint16_t*>(args.out)[row * args.output_stride +
                                         static_cast<uint64_t>(h) * args.hidden + col] =
            f32_to_bf16(raw + value * gate);
    }
}

uint32_t grid_for(uint64_t work) {
    const uint64_t blocks = (work + kThreads - 1u) / kThreads;
    if (blocks == 0u) return 1u;
    return static_cast<uint32_t>(blocks > kMaxBlocks ? kMaxBlocks : blocks);
}

}  // namespace

hipError_t launch_tp_sum_residual_bf16(const TpSumResidualArgs& args, hipStream_t stream) {
    if (args.out == nullptr || args.own == nullptr || args.residual == nullptr ||
        args.peers == nullptr || args.peer_count == 0u) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.features == 0u) return hipErrorInvalidValue;
    tp_sum_residual_bf16_kernel<<<grid_for(args.rows * args.features), kThreads, 0, stream>>>(
        args);
    return hipGetLastError();
}

hipError_t launch_tp_sum_hyper_apply_bf16(const TpSumHyperApplyArgs& args,
                                          hipStream_t stream) {
    if (args.out == nullptr || args.own == nullptr || args.raw == nullptr ||
        args.injection == nullptr || args.peers == nullptr || args.peer_count == 0u) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.hidden == 0u || args.hc == 0u) return hipErrorInvalidValue;
    tp_sum_hyper_apply_bf16_kernel<<<grid_for(args.rows * args.hidden), kThreads, 0,
                                     stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
