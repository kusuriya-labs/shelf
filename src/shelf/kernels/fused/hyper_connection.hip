#include <shelf/kernels/fused/hyper_connection.h>

#include <shelf/core/memory/types.h>

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>

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

constexpr uint32_t kUpMixWarps = 8u;
constexpr uint32_t kUpMixThreads = kUpMixWarps * 32u;
constexpr uint32_t kUpMixMaxHc = kHyperUpMixMaxHc;

__device__ __forceinline__ float bf16_round_f32(float value) {
    return __uint_as_float(static_cast<std::uint32_t>(
                               __bfloat16_as_ushort(__float2bfloat16(value)))
                           << 16);
}

__global__ __launch_bounds__(kUpMixThreads) void hyper_up_mix_kernel(HyperUpMixArgs args) {
    const uint32_t row = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t i = blockIdx.x * kUpMixWarps + warp;
    if (row >= args.rows || i >= args.hidden) return;
    const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input) +
                                 static_cast<std::uint64_t>(row) * args.input_stride;
    const shelf::bf16_t* up = static_cast<const shelf::bf16_t*>(args.up_weight);
    float gate_sum[kUpMixMaxHc];
    #pragma unroll
    for (uint32_t h = 0u; h < kUpMixMaxHc; ++h) gate_sum[h] = 0.0f;
    for (uint32_t kk = lane; kk < args.k; kk += 32u) {
        const float x = bf16_load(input + kk);
        #pragma unroll
        for (uint32_t h = 0u; h < kUpMixMaxHc; ++h) {
            if (h < args.hc)
                gate_sum[h] += bf16_load(up +
                    static_cast<std::uint64_t>(h * args.hidden + i) * args.k + kk) * x;
        }
    }
    #pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u) {
        #pragma unroll
        for (uint32_t h = 0u; h < kUpMixMaxHc; ++h)
            gate_sum[h] += __shfl_down(gate_sum[h], off, 32);
    }
    if (lane != 0u) return;
    const shelf::bf16_t* normed = static_cast<const shelf::bf16_t*>(args.normed) +
                                  static_cast<std::uint64_t>(row) * args.normed_stride;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t h = 0u; h < kUpMixMaxHc; ++h) {
        if (h >= args.hc) break;
        const float g = bf16_round_f32(gate_sum[h]);
        const float gate = (args.sigmoid_gate != 0u) ? (1.0f / (1.0f + __expf(-g))) : g;
        total += gate * bf16_load(normed + h * args.hidden + i);
    }
    shelf::bf16_t* out = static_cast<shelf::bf16_t*>(args.output) +
                         static_cast<std::uint64_t>(row) * args.output_stride;
    out[i] = bf16_store(total / static_cast<float>(args.hc));
}

// Decode-sized row batches: one warp keeps every row of the tile for one hidden index,
// so the [hc, hidden, k] up weight is loaded once for all rows instead of once per row.
template <uint32_t kRowsTile>
__global__ __launch_bounds__(kUpMixThreads) void hyper_up_mix_rows_kernel(
    HyperUpMixArgs args) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t i = blockIdx.x * kUpMixWarps + warp;
    const uint32_t row0 = blockIdx.y * kRowsTile;
    if (i >= args.hidden || row0 >= args.rows) return;
    const uint32_t rows = min(kRowsTile, args.rows - row0);
    const shelf::bf16_t* up = static_cast<const shelf::bf16_t*>(args.up_weight);
    float gate_sum[kRowsTile][kUpMixMaxHc];
    #pragma unroll
    for (uint32_t r = 0u; r < kRowsTile; ++r)
        #pragma unroll
        for (uint32_t h = 0u; h < kUpMixMaxHc; ++h) gate_sum[r][h] = 0.0f;
    for (uint32_t kk = lane; kk < args.k; kk += 32u) {
        float w[kUpMixMaxHc];
        #pragma unroll
        for (uint32_t h = 0u; h < kUpMixMaxHc; ++h)
            w[h] = h < args.hc
                ? bf16_load(up + (static_cast<std::uint64_t>(h * args.hidden + i) * args.k) + kk)
                : 0.0f;
        #pragma unroll
        for (uint32_t r = 0u; r < kRowsTile; ++r) {
            if (r >= rows) break;
            const shelf::bf16_t* input = static_cast<const shelf::bf16_t*>(args.input) +
                static_cast<std::uint64_t>(row0 + r) * args.input_stride;
            const float x = bf16_load(input + kk);
            #pragma unroll
            for (uint32_t h = 0u; h < kUpMixMaxHc; ++h)
                if (h < args.hc) gate_sum[r][h] += w[h] * x;
        }
    }
    #pragma unroll
    for (uint32_t r = 0u; r < kRowsTile; ++r) {
        if (r >= rows) break;
        #pragma unroll
        for (uint32_t h = 0u; h < kUpMixMaxHc; ++h) {
            if (h >= args.hc) break;
            #pragma unroll
            for (uint32_t off = 16u; off > 0u; off >>= 1u)
                gate_sum[r][h] += __shfl_down(gate_sum[r][h], off, 32);
        }
    }
    if (lane != 0u) return;
    #pragma unroll
    for (uint32_t r = 0u; r < kRowsTile; ++r) {
        if (r >= rows) break;
        const uint32_t row = row0 + r;
        const shelf::bf16_t* normed = static_cast<const shelf::bf16_t*>(args.normed) +
                                      static_cast<std::uint64_t>(row) * args.normed_stride;
        float total = 0.0f;
        #pragma unroll
        for (uint32_t h = 0u; h < kUpMixMaxHc; ++h) {
            if (h >= args.hc) break;
            const float g = bf16_round_f32(gate_sum[r][h]);
            const float gate = (args.sigmoid_gate != 0u) ? (1.0f / (1.0f + __expf(-g))) : g;
            total += gate * bf16_load(normed + h * args.hidden + i);
        }
        shelf::bf16_t* out = static_cast<shelf::bf16_t*>(args.output) +
                             static_cast<std::uint64_t>(row) * args.output_stride;
        out[i] = bf16_store(total / static_cast<float>(args.hc));
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

hipError_t launch_hyper_up_mix(const HyperUpMixArgs& args, hipStream_t stream) {
    if (args.up_weight == nullptr || args.input == nullptr || args.normed == nullptr ||
        args.output == nullptr) {
        return hipErrorInvalidValue;
    }
    if (args.rows == 0u || args.hc == 0u || args.hidden == 0u || args.k == 0u) {
        return hipErrorInvalidValue;
    }
    if (args.hc > kUpMixMaxHc) return hipErrorInvalidValue;
    if (args.input_stride < args.k || args.normed_stride < args.hc * args.hidden ||
        args.output_stride < args.hidden) {
        return hipErrorInvalidValue;
    }
    static const bool rows_tile_enabled = [] {
        const char* e = std::getenv("SHELF_HYPER_UP_MIX_ROWS");
        return e == nullptr || e[0] == '\0' || e[0] != '0';
    }();
    if (rows_tile_enabled && args.rows <= 8u) {
        constexpr uint32_t kRowsTile = 8u;
        const dim3 grid((args.hidden + kUpMixWarps - 1u) / kUpMixWarps,
                        (args.rows + kRowsTile - 1u) / kRowsTile);
        hyper_up_mix_rows_kernel<kRowsTile><<<grid, kUpMixThreads, 0, stream>>>(args);
        return hipGetLastError();
    }
    const dim3 grid((args.hidden + kUpMixWarps - 1u) / kUpMixWarps, args.rows);
    hyper_up_mix_kernel<<<grid, kUpMixThreads, 0, stream>>>(args);
    return hipGetLastError();
}

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
