#include <shelf/kernels/fused/exl3/group_body.h>
#include <shelf/kernels/fused/exl3/group_gemm.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/projection.h>
#include <shelf/kernels/optimized/linear/exl3_family/mul24/decode.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <algorithm>
#include <cstddef>

namespace shelf::kernel {
namespace {
using namespace exl3_detail;
using namespace exl3_detail::mul24;
__global__ void __launch_bounds__(128) group_rotate(PlamoExl3ProjectionGroup args) {
    const auto p = args.p[blockIdx.z];
    const auto input = args.input;
    const auto suh = p.suh;
    const auto codes = p.codes;
    const auto scales = p.scales;
    const uint32_t input_row_stride = args.input_stride, code_row_stride = args.k,
                   scale_row_stride_floats = args.scale_stride_bytes / 4;

    __shared__ float smem[kBlock];
    __shared__ float wave_peak[kBlock / 32u];
    const uint32_t t = threadIdx.x;
    const uint32_t m = blockIdx.x;
    const uint32_t g = blockIdx.y;
    const uint32_t col = g * kBlock + t;
    float v =
        bf16_to_f32(input[static_cast<size_t>(m) * input_row_stride + col]) * f16_bits_to_f32(suh[col]);
    v = fwht128_smem(v, smem, t) * kInvSqrt128;

    float peak = fabsf(v);
#pragma unroll
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        peak = fmaxf(peak, __shfl_xor(peak, off));
    if ((t & 31u) == 0u)
        wave_peak[t >> 5u] = peak;
    __syncthreads();
    peak = fmaxf(fmaxf(wave_peak[0], wave_peak[1]), fmaxf(wave_peak[2], wave_peak[3]));

    const float scale = peak / 127.0f;
    float q = 0.0f;
    if (scale > 0.0f)
        q = fminf(fmaxf(rintf(v / scale), -128.0f), 127.0f);
    codes[static_cast<size_t>(m) * code_row_stride + col] = static_cast<int8_t>(static_cast<int>(q));
    if (t == 0u)
        scales[static_cast<size_t>(m) * scale_row_stride_floats + g] = scale;
}
__global__ void __launch_bounds__(256) group_rotate_f4(PlamoExl3ProjectionGroup args) {
    const auto p = args.p[blockIdx.z];
    const auto input = args.input;
    const auto suh = p.suh;
    const auto codes = p.codes;
    const auto scales = p.scales;
    const uint32_t input_row_stride = args.input_stride, code_row_stride = args.k,
                   scale_row_stride_floats = args.scale_stride_bytes / 4, groups = args.k / 128u;

    const uint32_t m = blockIdx.x;
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    for (uint32_t g = wave; g < groups; g += 8u) {
        const uint32_t col = g * kBlock + lane * 4u;
        const uint2 raw =
            *reinterpret_cast<const uint2 *>(input + static_cast<size_t>(m) * input_row_stride + col);
        float v0 =
            bf16_bits_to_f32(static_cast<uint16_t>(raw.x & 0xFFFFu)) * f16_bits_to_f32(suh[col + 0u]);
        float v1 = bf16_bits_to_f32(static_cast<uint16_t>(raw.x >> 16u)) * f16_bits_to_f32(suh[col + 1u]);
        float v2 =
            bf16_bits_to_f32(static_cast<uint16_t>(raw.y & 0xFFFFu)) * f16_bits_to_f32(suh[col + 2u]);
        float v3 = bf16_bits_to_f32(static_cast<uint16_t>(raw.y >> 16u)) * f16_bits_to_f32(suh[col + 3u]);
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
        *reinterpret_cast<uint32_t *>(codes + static_cast<size_t>(m) * code_row_stride + col) = packed;
        if (lane == 0u)
            scales[static_cast<size_t>(m) * scale_row_stride_floats + g] = scale;
    }
}

template <int B, int MT, bool PIPE>
__global__ void __launch_bounds__(128) __attribute__((amdgpu_waves_per_eu((B == 4 && MT >= 3) ? 8 : 1)))
group_gemm(PlamoExl3ProjectionGroup args) {
    uint32_t tile = blockIdx.x;
    const uint32_t rb = (args.rows + 16u * MT - 1u) / (16u * MT);
    for (uint32_t i = 0; i < args.count; ++i) {
        const auto p = args.p[i];
        const uint32_t blocks = p.n / 64u * rb, total = blocks * p.splits;
        if (tile < total) {
            group_body<B, MT, PIPE>(p.codes, p.scales, args.k, args.scale_stride_bytes / 4u,
                                    reinterpret_cast<const uint32_t *>(p.trellis), p.yprime, args.rows,
                                    nullptr, p.n, args.k, (args.k / 128u + p.splits - 1u) / p.splits, rb,
                                    tile % blocks, tile / blocks);
            return;
        }
        tile -= total;
    }
}
template <bool BF16>
__global__ void __launch_bounds__(128)
    plamo_out_i32(const int32_t *__restrict__ acc, const float *__restrict__ rowscale,
                  const int32_t *__restrict__ slots, const float *__restrict__ yprime2,
                  const uint16_t *__restrict__ svh, void *__restrict__ out, const uint32_t n,
                  const uint32_t out_row_stride, const uint32_t rows) {
    const uint32_t m = blockIdx.x;
    if (m >= rows)
        return;
    const uint32_t lane = threadIdx.x & 31u;
    const int32_t slot = slots[m];
    const float rs = rowscale[m];
    for (uint32_t cb = blockIdx.y * 4u + (threadIdx.x >> 5u); cb < min(n / kBlock, (blockIdx.y + 1u) * 4u);
         cb += 4u) {
        const uint32_t col = cb * kBlock + lane * 4u;
        float v0, v1, v2, v3;
        if (slot >= 0) {
            const float4 y =
                *reinterpret_cast<const float4 *>(yprime2 + static_cast<size_t>(slot) * n + col);
            v0 = y.x;
            v1 = y.y;
            v2 = y.z;
            v3 = y.w;
        } else {
            const int4 a = *reinterpret_cast<const int4 *>(acc + static_cast<size_t>(m) * n + col);
            v0 = (static_cast<float>(a.x) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v1 = (static_cast<float>(a.y) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v2 = (static_cast<float>(a.z) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
            v3 = (static_cast<float>(a.w) * rs) * ::shelf::quantization::exl3::kExl3Int8GridStep;
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
template <int BITS, int MT, bool PIPE>
__global__ void __launch_bounds__(128)
    plamo_flag_gemm(const int8_t *codes, const float *scales, uint32_t code_stride, uint32_t scale_stride,
                    const uint32_t *trellis, float *yprime, uint32_t rows, const uint32_t *active_rows,
                    uint32_t n, uint32_t k, uint32_t rb) {
    group_body<BITS, MT, PIPE>(codes, scales, code_stride, scale_stride, trellis, yprime, rows, active_rows,
                               n, k, k / 128u, rb, blockIdx.x, 0u);
}
template <int BITS, int MT, bool PIPE>
hipError_t launch_flag(const int8_t *codes, const float *scales, uint32_t code_stride,
                       uint32_t scale_stride, const uint16_t *trellis, float *yprime, uint32_t rows,
                       const uint32_t *active_rows, uint32_t n, uint32_t k, hipStream_t stream) {
    const uint32_t rb = (rows + 16u * MT - 1u) / (16u * MT);
    plamo_flag_gemm<BITS, MT, PIPE><<<n / 64u * rb, 128, 0, stream>>>(
        codes, scales, code_stride, scale_stride / 4u, reinterpret_cast<const uint32_t *>(trellis), yprime,
        rows, active_rows, n, k, rb);
    return hipGetLastError();
}
template <int B, int MT, bool PIPE>
hipError_t launch_group_bits(const PlamoExl3ProjectionGroup &args, hipStream_t stream) {
    uint32_t blocks = 0;
    const uint32_t rb = (args.rows + 16u * MT - 1u) / (16u * MT);
    for (uint32_t i = 0; i < args.count; ++i)
        blocks += args.p[i].n / 64u * rb * args.p[i].splits;
    group_gemm<B, MT, PIPE><<<blocks, 128, 0, stream>>>(args);
    return hipGetLastError();
}
}
hipError_t launch_plamo_exl3_projection_rotate(const PlamoExl3ProjectionGroup &args,
                                               hipStream_t stream) {
    if (!args.input || !args.rows || !args.k || args.k % 128u || !args.count || args.count > 3u ||
        args.input_stride < args.k || args.scale_stride_bytes % 4u ||
        args.scale_stride_bytes < args.k / 128u * 4u)
        return hipErrorInvalidValue;
    for (uint32_t i = 0u; i < args.count; ++i)
        if (!args.p[i].suh || !args.p[i].codes || !args.p[i].scales)
            return hipErrorInvalidValue;
    if (args.rows >= 32u && args.input_stride % 4u == 0u)
        group_rotate_f4<<<dim3(args.rows, 1, args.count), 256, 0, stream>>>(args);
    else
        group_rotate<<<dim3(args.rows, args.k / 128u, args.count), 128, 0, stream>>>(args);
    return hipGetLastError();
}
hipError_t launch_plamo_exl3_projection(const PlamoExl3ProjectionGroup &args, uint32_t bits,
                                        hipStream_t stream, uint32_t *launch_count,
                                        bool allow_shape_fusion, bool input_rotated) {
    if (launch_count)
        *launch_count = 0u;
    const bool grouped_3bit = bits == 3u && args.rows <= 16u && !allow_shape_fusion;
    if (args.count == 0u || args.count > 3u || args.rows == 0u || args.rows >= 256u ||
        args.k % 128u != 0u || args.scale_stride_bytes % 4u != 0u || !args.input ||
        args.input_stride < args.k || args.scale_stride_bytes < args.k / 128u * 4u ||
        (!grouped_3bit && bits != 4u && bits != 5u && bits != 8u))
        return hipErrorInvalidValue;
    uint32_t n = 0u;
    for (uint32_t i = 0; i < args.count; ++i) {
        const auto &p = args.p[i];
        if (!p.trellis || !p.suh || !p.svh || !p.codes || !p.scales || !p.yprime || !p.out || p.n == 0u ||
            p.n % 128u != 0u || p.splits == 0u || p.splits > args.k / 128u || p.out_stride < p.n ||
            p.out_stride % 4u != 0u)
            return hipErrorInvalidValue;
        n = std::max(n, p.n);
    }
    if (launch_count)
        *launch_count = 3u;
    if (!input_rotated && allow_shape_fusion && args.rows == 1u && bits == 4u &&
        ((args.k == 2560u && args.p[0].n == 3072u) || (args.k == 3072u && args.p[0].n == 2560u))) {
        if (launch_count)
            *launch_count = 2u;
        return launch_fused_input(args, stream);
    }
    if (!input_rotated) {
        if (args.rows >= 32u && args.input_stride % 4u == 0u)
            group_rotate_f4<<<dim3(args.rows, 1, args.count), 256, 0, stream>>>(args);
        else
            group_rotate<<<dim3(args.rows, args.k / 128u, args.count), 128, 0, stream>>>(args);
    } else if (launch_count) {
        *launch_count = 2u;
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        return err;
    bool fused_output = allow_shape_fusion && args.rows <= 16u && args.p[0].n >= 10240u;
    for (uint32_t i = 0u; i < args.count; ++i)
        fused_output = fused_output && args.p[i].splits == 1u;
    if (fused_output) {
        if (launch_count)
            *launch_count = input_rotated ? 1u : 2u;
        return launch_fused_output(args, bits, stream);
    }
    if (bits == 3u)
        err = launch_group_bits<3, 1, true>(args, stream);
    else if (bits == 4u)
        err = args.rows <= 16u    ? launch_group_bits<4, 1, true>(args, stream)
              : args.rows <= 32u ? launch_group_bits<4, 2, true>(args, stream)
              : args.rows <= 48u ? launch_group_bits<4, 3, true>(args, stream)
              : args.rows <= 64u ? launch_group_bits<4, 4, true>(args, stream)
              : args.rows <= 128u ? launch_group_bits<4, 2, true>(args, stream)
                                  : launch_group_bits<4, 4, true>(args, stream);
    else if (bits == 5u)
        err = args.rows <= 16u ? launch_group_bits<5, 1, true>(args, stream)
                               : launch_group_bits<5, 4, false>(args, stream);
    else
        err = args.rows <= 16u   ? launch_group_bits<8, 1, true>(args, stream)
              : args.rows <= 64u ? launch_group_bits<8, 2, true>(args, stream)
                                 : launch_group_bits<8, 4, true>(args, stream);
    if (err != hipSuccess)
        return err;
    const hipError_t out_error = launch_group_out(args, stream);
    if (out_error != hipSuccess) return out_error;
    return hipGetLastError();
}
hipError_t launch_plamo_exl3_flagged_gemm(const int8_t *codes, const float *scales, uint32_t code_stride,
                                          uint32_t scale_stride_bytes, const uint16_t *trellis,
                                          uint32_t bits, float *yprime, uint32_t rows,
                                          const uint32_t *active_rows, uint32_t n, uint32_t k,
                                          hipStream_t stream) {
    if (!codes || !scales || !trellis || !yprime || !active_rows || rows == 0u || k % 128u || n % 128u ||
        code_stride < k || scale_stride_bytes % 4u || scale_stride_bytes < k / 128u * 4u)
        return hipErrorInvalidValue;
    if (bits == 4u)
        return launch_flag<4, 2, true>(codes, scales, code_stride, scale_stride_bytes, trellis, yprime,
                                       rows, active_rows, n, k, stream);
    if (bits == 5u)
        return launch_flag<5, 4, false>(codes, scales, code_stride, scale_stride_bytes, trellis, yprime,
                                        rows, active_rows, n, k, stream);
    if (bits == 8u)
        return launch_flag<8, 4, true>(codes, scales, code_stride, scale_stride_bytes, trellis, yprime,
                                       rows, active_rows, n, k, stream);
    return hipErrorInvalidValue;
}
hipError_t launch_plamo_exl3_rotate_out_i32(const int32_t *acc, const float *rowscale, const int32_t *slots,
                                            const float *yprime, const uint16_t *svh, void *out,
                                            bool out_bf16, uint32_t rows, uint32_t n, uint32_t out_stride,
                                            hipStream_t stream) {
    if (!acc || !rowscale || !slots || !yprime || !svh || !out || rows == 0u || n == 0u || n % 128u ||
        out_stride < n || out_stride % 4u)
        return hipErrorInvalidValue;
    const dim3 grid(rows, (n / 128u + 3u) / 4u);
    if (out_bf16)
        plamo_out_i32<true>
            <<<grid, 128, 0, stream>>>(acc, rowscale, slots, yprime, svh, out, n, out_stride, rows);
    else
        plamo_out_i32<false>
            <<<grid, 128, 0, stream>>>(acc, rowscale, slots, yprime, svh, out, n, out_stride, rows);
    return hipGetLastError();
}
}
