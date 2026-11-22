#include <shelf/kernels/fused/exl3/group_body.h>
#include <shelf/models/hy4_flash/linear.h>

namespace {
template <int Bits, int Tiles>
__global__ void batch(const int8_t *codes, const float *scales, const uint32_t *weight, float *partial,
                      int rows, int k, int n, int blocks) {
    shelf::kernel::exl3_detail::mul24::group_body<Bits, Tiles, false, 4, false, true>(
        codes, scales, k, k / 128, weight, partial, rows, nullptr, n, k, (k / 128 + gridDim.z - 1) / gridDim.z,
        blocks, blockIdx.x, blockIdx.z);
}
template <int Bits>
void launch(const int8_t *codes, const float *scales, const uint32_t *weight, float *partial, int rows, int k,
            int n, hipStream_t stream, int splits) {
    if (rows >= 32) {
        int blocks = (rows + 31) / 32;
        batch<Bits, 2><<<dim3(blocks * (n / 64), 1, splits), 128, 0, stream>>>(codes, scales, weight, partial,
                                                                               rows, k, n, blocks);
    } else {
        int blocks = (rows + 15) / 16;
        batch<Bits, 1><<<dim3(blocks * (n / 64), 1, splits), 128, 0, stream>>>(codes, scales, weight, partial,
                                                                               rows, k, n, blocks);
    }
}
}
namespace {
template <int Bits, bool OutFused>
__global__ void __launch_bounds__(256u) gemm_fused(
    const uint16_t *input, const uint16_t *suh, const uint16_t *svh, uint16_t *output, const uint32_t *trellis,
    float *partial, int rows, int input_width, int output_width, int padded_input, int padded_output, int row_chunk,
    int splits) {
    using namespace shelf::kernel::exl3_detail;
    using namespace shelf::kernel::exl3_detail::mul24;
    extern __shared__ uint8_t shared[];
    auto codes = reinterpret_cast<int8_t *>(shared);
    const uint32_t groups = padded_input / 128u;
    const uint32_t gps = (groups + splits - 1u) / splits;
    auto scales = reinterpret_cast<float *>(shared + size_t(row_chunk) * padded_input);
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
    constexpr uint32_t waves = 8u;
    const uint32_t tiles = padded_output / 128u;
    const uint32_t tile = blockIdx.x % tiles, split = blockIdx.x / tiles;
    const uint32_t g0 = split * gps, g1 = groups < g0 + gps ? groups : g0 + gps;
    const int row0 = int(blockIdx.y) * row_chunk;
    const int cr = rows - row0 < row_chunk ? rows - row0 : row_chunk;
    for (int m = 0; m < cr; ++m) {
        for (uint32_t g = g0 + wave; g < g1; g += waves) {
            const uint32_t col = g * 128u + lane * 4u;
            float v[4];
            for (int i = 0; i < 4; ++i) {
                int c = int(g * 128u) + int(lane * 4u) + i;
                v[i] = c < input_width ? bf16_bits_to_f32(input[size_t(row0 + m) * input_width + c]) *
                                             f16_bits_to_f32(suh[c])
                                       : 0.0f;
            }
            fwht128(v[0], v[1], v[2], v[3], lane);
            float peak = 0;
            for (float &x : v) {
                x *= kInvSqrt128;
                peak = fmaxf(peak, fabsf(x));
            }
            for (int mask = 16; mask; mask /= 2)
                peak = fmaxf(peak, __shfl_xor(peak, mask, 32));
            const float scale = peak / 127.0f;
            uint32_t packed = 0;
            for (int i = 0; i < 4; ++i) {
                int8_t q = scale ? int8_t(rintf(v[i] / scale)) : 0;
                packed |= uint32_t(uint8_t(q)) << (8 * i);
            }
            *reinterpret_cast<uint32_t *>(codes + size_t(m) * padded_input + col) = packed;
            if (!lane)
                scales[size_t(m) * groups + g] = scale;
        }
    }
    __syncthreads();
    shelf::kernel::PlamoExl3Projection ep{};
    if constexpr (OutFused) {
        ep.svh = svh;
        ep.out = reinterpret_cast<shelf::bf16_t *>(output) + size_t(row0) * output_width;
        ep.out_stride = output_width;
    }
    float *yprime = partial + (size_t(row0) + size_t(split) * (rows - cr)) * padded_output;
    shelf::kernel::exl3_detail::mul24::group_body<Bits, 1, OutFused, waves, OutFused, true>(
        codes, scales, padded_input, groups, trellis, yprime, cr, nullptr, padded_output, padded_input, gps, 1u,
        tile, split, &ep);
}
}
hipError_t shelf_hy4_linear_fused_gemm(const uint16_t *input, const uint16_t *suh, const uint16_t *svh,
                                       const uint32_t *trellis, float *partial, uint16_t *output, int rows,
                                       int input_width, int output_width, int padded_input, int padded_output,
                                       int bits, int splits, hipStream_t stream) {
    if (!input || !suh || !trellis || !partial || rows < 1 || rows > 16 || input_width < 1 || output_width < 1 ||
        padded_input < input_width || padded_output < output_width || padded_input % 128 || padded_output % 128 ||
        splits < 1 || splits > padded_input / 128 || (bits != 4 && bits != 6))
        return hipErrorInvalidValue;
    const bool out_fused = svh && output && output_width == padded_output && splits == 1;
    const uint32_t groups = padded_input / 128u;
    const uint32_t gps = (groups + splits - 1u) / splits;
    const size_t per_row = size_t(padded_input) + size_t(groups) * 4u;
    const size_t budget = out_fused ? 56u * 1024u : 64u * 1024u;
    int row_chunk = rows < 16 ? rows : 16;
    while (row_chunk > 1 && size_t(row_chunk) * per_row > budget)
        row_chunk /= 2;
    const size_t shared = size_t(row_chunk) * per_row;
    const uint32_t tiles = padded_output / 128u;
    const dim3 grid(tiles * splits, (rows + row_chunk - 1) / row_chunk);
    if (out_fused) {
        if (bits == 4)
            gemm_fused<4, true><<<grid, 256, shared, stream>>>(
                input, suh, svh, output, trellis, partial, rows, input_width, output_width, padded_input,
                padded_output, row_chunk, splits);
        else
            gemm_fused<6, true><<<grid, 256, shared, stream>>>(
                input, suh, svh, output, trellis, partial, rows, input_width, output_width, padded_input,
                padded_output, row_chunk, splits);
    } else {
        if (bits == 4)
            gemm_fused<4, false><<<grid, 256, shared, stream>>>(
                input, suh, svh, output, trellis, partial, rows, input_width, output_width, padded_input,
                padded_output, row_chunk, splits);
        else
            gemm_fused<6, false><<<grid, 256, shared, stream>>>(
                input, suh, svh, output, trellis, partial, rows, input_width, output_width, padded_input,
                padded_output, row_chunk, splits);
    }
    return hipGetLastError();
}
hipError_t shelf_hy4_wmma(const int8_t *codes, const float *scales, const uint32_t *weight, float *partial,
                          int rows, int k, int n, int bits, hipStream_t stream, int splits) {
    if (bits == 4)
        launch<4>(codes, scales, weight, partial, rows, k, n, stream, splits);
    else if (bits == 6)
        launch<6>(codes, scales, weight, partial, rows, k, n, stream, splits);
    else
        return hipErrorInvalidValue;
    return hipGetLastError();
}

namespace {
template <bool Mixed>
__global__ void expert_batch(const int8_t *codes, const float *scales, const Hy4ExpertProjection *matrices,
                             const Hy4Route *routes, const Hy4ExpertBatch *batches, float *partial, int k,
                             int n) {
    auto batch = batches[blockIdx.y];
    size_t first = batch.first;
    auto matrix = matrices[3 * routes[first].expert];
    auto weight = matrix.trellis;
    if (Mixed && matrix.bits == 3)
        shelf::kernel::exl3_detail::mul24::group_body<3, 1, false, 4, false, true>(
            codes + first * k, scales + first * (k / 128), k, k / 128, weight, partial + first * n, batch.count,
            nullptr, n, k, k / 128, 1, blockIdx.x, 0);
    else
        shelf::kernel::exl3_detail::mul24::group_body<4, 1, false, 4, false, true>(
            codes + first * k, scales + first * (k / 128), k, k / 128, weight, partial + first * n, batch.count,
            nullptr, n, k, k / 128, 1, blockIdx.x, 0);
}
}
hipError_t shelf_hy4_wmma_experts(const int8_t *codes, const float *scales, const Hy4ExpertProjection *matrices,
                                  const Hy4Route *routes, const Hy4ExpertBatch *batches, int count,
                                  float *partial, bool mixed, int k, int n, hipStream_t stream) {
    if (n % 64)
        return hipErrorInvalidValue;
    if (mixed)
        expert_batch<true><<<dim3(n / 64, count), 128, 0, stream>>>(codes, scales, matrices, routes, batches, partial, k, n);
    else
        expert_batch<false><<<dim3(n / 64, count), 128, 0, stream>>>(codes, scales, matrices, routes, batches, partial, k, n);
    return hipGetLastError();
}
