#include <shelf/kernels/fused/exl3/group_gemm.h>
#include <shelf/kernels/fused/exl3/group_body.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>

#include <algorithm>
#include <cstdint>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;

__global__ void __launch_bounds__(128) group_out(PlamoExl3ProjectionGroup args) {
    const auto p = args.p[blockIdx.z];
    const auto yprime = p.yprime;
    const auto svh = p.svh;
    const auto out = p.out;
    const uint32_t n = p.n, rows = args.rows, splits = p.splits, out_row_stride = p.out_stride;
    constexpr bool BF16 = true;
    constexpr int WAVES = 4;

    const uint32_t m = blockIdx.x;
    if (m >= rows)
        return;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t cb = blockIdx.y * WAVES + (threadIdx.x >> 5u);
         cb < min(n / kBlock, (blockIdx.y + 1u) * WAVES); cb += WAVES) {
        const uint32_t col = cb * kBlock + lane * 4u;
        float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
        for (uint32_t z = 0u; z < splits; ++z) {
            const float4 y =
                *reinterpret_cast<const float4 *>(yprime + (static_cast<size_t>(z) * rows + m) * n + col);
            v0 += y.x;
            v1 += y.y;
            v2 += y.z;
            v3 += y.w;
        }
        exl3_detail::fwht128(v0, v1, v2, v3, lane);
        v0 = (v0 * kInvSqrt128) * f16_bits_to_f32(svh[col + 0u]);
        v1 = (v1 * kInvSqrt128) * f16_bits_to_f32(svh[col + 1u]);
        v2 = (v2 * kInvSqrt128) * f16_bits_to_f32(svh[col + 2u]);
        v3 = (v3 * kInvSqrt128) * f16_bits_to_f32(svh[col + 3u]);
        const size_t index = static_cast<size_t>(m) * out_row_stride + col;
        if constexpr (BF16) {
            uint2 packed;
            packed.x = static_cast<uint32_t>(f32_to_bf16_rne_device(v0)) |
                       (static_cast<uint32_t>(f32_to_bf16_rne_device(v1)) << 16u);
            packed.y = static_cast<uint32_t>(f32_to_bf16_rne_device(v2)) |
                       (static_cast<uint32_t>(f32_to_bf16_rne_device(v3)) << 16u);
            *reinterpret_cast<uint2 *>(reinterpret_cast<::shelf::bf16_t *>(out) + index) = packed;
        } else {
            *reinterpret_cast<float4 *>(reinterpret_cast<float *>(out) + index) = float4{v0, v1, v2, v3};
        }
    }
}
template <int BITS> __global__ void __launch_bounds__(256) group_gemm_out(PlamoExl3ProjectionGroup args) {
    uint32_t tile = blockIdx.x;
    for (uint32_t i = 0u; i < args.count; ++i) {
        const auto p = args.p[i];
        if (tile >= p.n / 128u) {
            tile -= p.n / 128u;
            continue;
        }
        group_body<BITS, 1, true, 8, true>(p.codes, p.scales, args.k, args.scale_stride_bytes / 4u,
                                           reinterpret_cast<const uint32_t *>(p.trellis), p.yprime,
                                           args.rows, nullptr, p.n, args.k, args.k / 128u, 1u, tile, 0u,
                                           &p);
        return;
    }
}
template <int BITS> __global__ void __launch_bounds__(128) group_gemm_fused(PlamoExl3ProjectionGroup args) {
    uint32_t tile = blockIdx.x;
    for (uint32_t i = 0u; i < args.count; ++i) {
        const auto p = args.p[i];
        const uint32_t blocks = p.n / 64u;
        if (tile >= blocks * p.splits) {
            tile -= blocks * p.splits;
            continue;
        }
        extern __shared__ uint8_t scratch[];
        auto codes = reinterpret_cast<int8_t *>(scratch);
        auto scales = reinterpret_cast<float *>(scratch + args.k * args.rows);
        const auto input = args.input;
        const auto suh = p.suh;
        const uint32_t input_row_stride = args.input_stride, code_row_stride = args.k;
        const uint32_t scale_row_stride_floats = args.k / 128u, groups = args.k / 128u;
        for (uint32_t m = 0u; m < args.rows; ++m) {
            const uint32_t wave = threadIdx.x >> 5u;
            const uint32_t lane = threadIdx.x & 31u;
            for (uint32_t g = wave; g < groups; g += 4u) {
                const uint32_t col = g * kBlock + lane * 4u;
                const uint2 raw = *reinterpret_cast<const uint2 *>(
                    input + static_cast<size_t>(m) * input_row_stride + col);
                float v0 = bf16_bits_to_f32(static_cast<uint16_t>(raw.x & 0xFFFFu)) *
                           f16_bits_to_f32(suh[col + 0u]);
                float v1 =
                    bf16_bits_to_f32(static_cast<uint16_t>(raw.x >> 16u)) * f16_bits_to_f32(suh[col + 1u]);
                float v2 = bf16_bits_to_f32(static_cast<uint16_t>(raw.y & 0xFFFFu)) *
                           f16_bits_to_f32(suh[col + 2u]);
                float v3 =
                    bf16_bits_to_f32(static_cast<uint16_t>(raw.y >> 16u)) * f16_bits_to_f32(suh[col + 3u]);
                exl3_detail::fwht128(v0, v1, v2, v3, lane);
                v0 *= kInvSqrt128;
                v1 *= kInvSqrt128;
                v2 *= kInvSqrt128;
                v3 *= kInvSqrt128;
                float peak = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
                for (uint32_t off = 16u; off > 0u; off >>= 1u)
                    peak = fmaxf(peak, __shfl_xor(peak, off, 32));
                const float scale = peak / 127.0f;
                float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
                if (scale > 0.0f) {
                    q0 = fminf(fmaxf(rintf(v0 / scale), -128.0f), 127.0f);
                    q1 = fminf(fmaxf(rintf(v1 / scale), -128.0f), 127.0f);
                    q2 = fminf(fmaxf(rintf(v2 / scale), -128.0f), 127.0f);
                    q3 = fminf(fmaxf(rintf(v3 / scale), -128.0f), 127.0f);
                }
                const uint32_t packed = (static_cast<uint32_t>(static_cast<int32_t>(q0)) & 0xFFu) |
                                        ((static_cast<uint32_t>(static_cast<int32_t>(q1)) & 0xFFu) << 8u) |
                                        ((static_cast<uint32_t>(static_cast<int32_t>(q2)) & 0xFFu) << 16u) |
                                        ((static_cast<uint32_t>(static_cast<int32_t>(q3)) & 0xFFu) << 24u);
                *reinterpret_cast<uint32_t *>(codes + static_cast<size_t>(m) * code_row_stride + col) =
                    packed;
                if (lane == 0u)
                    scales[static_cast<size_t>(m) * scale_row_stride_floats + g] = scale;
            }
        }
        __syncthreads();
        group_body<BITS, 1, BITS != 5>(codes, scales, args.k, args.k / 128u,
                                       reinterpret_cast<const uint32_t *>(p.trellis), p.yprime, args.rows,
                                       nullptr, p.n, args.k, (args.k / 128u + p.splits - 1u) / p.splits, 1u,
                                       tile % blocks, tile / blocks);
        return;
    }
}
}  // namespace

hipError_t launch_group_out(const PlamoExl3ProjectionGroup &args, hipStream_t stream) {
    uint32_t n = 0u;
    for (uint32_t i = 0u; i < args.count; ++i) n = std::max(n, args.p[i].n);
    group_out<<<dim3(args.rows, (n / 128u + 3u) / 4u, args.count), 128, 0, stream>>>(args);
    return hipGetLastError();
}

hipError_t launch_fused_output(const PlamoExl3ProjectionGroup &args, uint32_t bits, hipStream_t stream) {
    uint32_t blocks = 0u;
    for (uint32_t i = 0u; i < args.count; ++i)
        blocks += args.p[i].n / 128u;
    if (bits == 4u)
        group_gemm_out<4><<<blocks, 256, 0, stream>>>(args);
    else if (bits == 5u)
        group_gemm_out<5><<<blocks, 256, 0, stream>>>(args);
    else
        group_gemm_out<8><<<blocks, 256, 0, stream>>>(args);
    return hipGetLastError();
}
hipError_t launch_fused_input(const PlamoExl3ProjectionGroup &args, hipStream_t stream) {
    uint32_t blocks = 0u, n = 0u;
    for (uint32_t i = 0u; i < args.count; ++i) {
        blocks += args.p[i].n / 64u * args.p[i].splits;
        n = std::max(n, args.p[i].n);
    }
    const uint32_t shared = (args.k + args.k / 128u * 4u) * args.rows;
    group_gemm_fused<4><<<blocks, 128, shared, stream>>>(args);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        return err;
    group_out<<<dim3(args.rows, (n / 128u + 3u) / 4u, args.count), 128, 0, stream>>>(args);
    return hipGetLastError();
}

}  // namespace shelf::kernel
