#include <shelf/kernels/fused/exl3/group_body.h>
#include <shelf/models/hy4_flash/linear.h>

namespace {
template <int Bits, int Tiles, bool Pipe = false, bool ShufflePrev = false>
__global__ void batch(const int8_t *codes, const float *scales, const uint32_t *weight, float *partial,
                      int rows, int k, int n, int blocks) {
    shelf::kernel::exl3_detail::mul24::group_body<Bits, Tiles, Pipe, 4, false, true, ShufflePrev>(
        codes, scales, k, k / 128, weight, partial, rows, nullptr, n, k, (k / 128 + gridDim.z - 1) / gridDim.z,
        blocks, blockIdx.x, blockIdx.z);
}
template <int Bits>
void launch(const int8_t *codes, const float *scales, const uint32_t *weight, float *partial, int rows, int k,
            int n, hipStream_t stream, int splits) {
    if (Bits == 6 && rows > 48 && rows <= 96) {
        int blocks = (rows + 47) / 48;
        batch<Bits, 3, false, Bits == 4><<<dim3(blocks * (n / 64), 1, splits), 128, 0, stream>>>(
            codes, scales, weight, partial, rows, k, n, blocks);
    } else if (rows >= 32) {
        int blocks = (rows + 31) / 32;
        batch<Bits, 2, false, Bits == 4><<<dim3(blocks * (n / 64), 1, splits), 128, 0, stream>>>(
            codes, scales, weight, partial, rows, k, n, blocks);
    } else {
        int blocks = (rows + 15) / 16;
        batch<Bits, 1, false, Bits == 4><<<dim3(blocks * (n / 64), 1, splits), 128, 0, stream>>>(
            codes, scales, weight, partial, rows, k, n, blocks);
    }
}
}
namespace {
template <int Bits>
__global__ void __launch_bounds__(128u) gemm_fused_4w(
    const uint16_t *input, const uint16_t *suh, const uint32_t *trellis, float *partial,
    int rows, int input_width, int padded_input, int padded_output, int splits) {
    using namespace shelf::kernel::exl3_detail;
    using namespace shelf::kernel::exl3_detail::mul24;
    extern __shared__ uint8_t shared[];
    auto codes = reinterpret_cast<int8_t *>(shared);
    const uint32_t groups = padded_input / 128u;
    const uint32_t gps = (groups + splits - 1u) / splits;
    const uint32_t code_stride = gps * 128u;
    auto scales = reinterpret_cast<float *>(shared + size_t(rows) * code_stride);
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
    const uint32_t tiles = padded_output / 64u;
    const uint32_t tile = blockIdx.x % tiles, split = blockIdx.x / tiles;
    const uint32_t g0 = split * gps, g1 = groups < g0 + gps ? groups : g0 + gps;
    const uint32_t active_groups = g1 > g0 ? g1 - g0 : 0u;
    for (uint32_t part = wave; part < uint32_t(rows) * active_groups; part += 4u) {
        const int m = int(part / active_groups);
        const uint32_t g = g0 + part % active_groups;
        const uint32_t local_group = g - g0;
        float v[4];
        const int base = int(g * 128u) + int(lane * 4u);
        if (base + 3 < input_width && (input_width & 3) == 0) {
            const uint2 raw = *reinterpret_cast<const uint2 *>(input + size_t(m) * input_width + base);
            const uint2 hs = *reinterpret_cast<const uint2 *>(suh + base);
            v[0] = bf16_bits_to_f32(uint16_t(raw.x & 0xffffu)) * f16_bits_to_f32(uint16_t(hs.x & 0xffffu));
            v[1] = bf16_bits_to_f32(uint16_t(raw.x >> 16)) * f16_bits_to_f32(uint16_t(hs.x >> 16));
            v[2] = bf16_bits_to_f32(uint16_t(raw.y & 0xffffu)) * f16_bits_to_f32(uint16_t(hs.y & 0xffffu));
            v[3] = bf16_bits_to_f32(uint16_t(raw.y >> 16)) * f16_bits_to_f32(uint16_t(hs.y >> 16));
        } else {
            for (int i = 0; i < 4; ++i)
                v[i] = base + i < input_width ? bf16_bits_to_f32(input[size_t(m) * input_width + base + i]) *
                                                    f16_bits_to_f32(suh[base + i])
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
        *reinterpret_cast<uint32_t *>(codes + size_t(m) * code_stride + local_group * 128u + lane * 4u) = packed;
        if (!lane)
            scales[size_t(m) * gps + local_group] = scale;
    }
    __syncthreads();
    trellis += size_t(g0) * 128u * padded_output * Bits / 32u;
    float *yprime = partial + size_t(split) * rows * padded_output;
    shelf::kernel::exl3_detail::mul24::group_body<Bits, 1, false, 4, false, true, Bits == 4>(
        codes, scales, code_stride, gps, trellis, yprime, rows, nullptr, padded_output, code_stride, active_groups, 1u,
        tile, 0u);
}

template <int Bits, bool OutFused, bool Pair = false, int CompactRows = 0>
__global__ void __launch_bounds__(256u) gemm_fused(
    const uint16_t *input, const uint16_t *suh, const uint16_t *svh, uint16_t *output, const uint32_t *trellis,
    float *partial, int rows, int input_width, int output_width, int padded_input, int padded_output, int row_chunk,
    int splits, const uint16_t *up_suh = nullptr, const uint32_t *up_trellis = nullptr) {
    if constexpr (CompactRows > 0) {
        rows = row_chunk = CompactRows;
        input_width = padded_input = 3072;
        output_width = padded_output = 1536;
        splits = 8;
    }
    if constexpr (Pair) {
        if (blockIdx.z) {
            suh = up_suh;
            trellis = up_trellis;
            partial += size_t(rows) * padded_output * splits;
        }
    }
    using namespace shelf::kernel::exl3_detail;
    using namespace shelf::kernel::exl3_detail::mul24;
    extern __shared__ uint8_t shared[];
    auto codes = reinterpret_cast<int8_t *>(shared);
    const uint32_t groups = padded_input / 128u;
    const uint32_t gps = (groups + splits - 1u) / splits;
    const uint32_t local_groups = CompactRows > 0 ? gps : groups;
    const uint32_t code_stride = local_groups * 128;
    auto scales = reinterpret_cast<float *>(shared + size_t(row_chunk) * code_stride);
    const uint32_t lane = threadIdx.x & 31u, wave = threadIdx.x >> 5u;
    constexpr uint32_t waves = 8u;
    const uint32_t tiles = padded_output / 128u;
    const uint32_t tile = blockIdx.x % tiles, split = blockIdx.x / tiles;
    const uint32_t g0 = split * gps, g1 = groups < g0 + gps ? groups : g0 + gps;
    const int row0 = CompactRows > 0 ? 0 : int(blockIdx.y) * row_chunk;
    const int cr = CompactRows > 0 ? CompactRows : min(rows - row0, row_chunk);
    auto rotate = [&](int m, uint32_t g) {
        const uint32_t local_group = CompactRows > 0 ? g - g0 : g;
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
        *reinterpret_cast<uint32_t *>(codes + size_t(m) * code_stride + local_group * 128 + lane * 4) = packed;
        if (!lane)
            scales[size_t(m) * local_groups + local_group] = scale;
    };
    if constexpr (CompactRows > 0) {
        for (uint32_t part = wave; part < uint32_t(CompactRows * 3); part += waves)
            rotate(part / 3, g0 + part % 3);
    } else {
        for (int m = 0; m < cr; ++m)
            for (uint32_t g = g0 + wave; g < g1; g += waves)
                rotate(m, g);
    }
    __syncthreads();
    shelf::kernel::PlamoExl3Projection ep{};
    if constexpr (OutFused) {
        ep.svh = svh;
        ep.out = reinterpret_cast<shelf::bf16_t *>(output) + size_t(row0) * output_width;
        ep.out_stride = output_width;
    }
    float *yprime = partial + (size_t(row0) + size_t(split) * (rows - cr)) * padded_output;
    if constexpr (CompactRows > 0) {
        trellis += size_t(g0) * 128 * padded_output * Bits / 32;
        yprime += size_t(split) * cr * padded_output;
    }
    shelf::kernel::exl3_detail::mul24::group_body<Bits, 1, OutFused, waves, OutFused, true, Bits == 4 && !OutFused>(
        codes, scales, code_stride, local_groups, trellis, yprime, cr, nullptr, padded_output, code_stride, gps, 1u,
        tile, CompactRows > 0 ? 0 : split, &ep);
}
__global__ void shared_swiglu(const float *partial, const uint16_t *gate_svh, const uint16_t *up_svh,
                              uint16_t *output) {
    using namespace shelf::kernel::exl3_detail;
    using namespace shelf::kernel::exl3_detail::mul24;
    int row = blockIdx.y, col = blockIdx.x * 128 + threadIdx.x * 4;
    size_t plane = size_t(gridDim.y) * 1536;
    auto gate = partial + row * 1536 + col;
    auto up = gate + 8 * plane;
    float g[4], u[4];
    for (int i = 0; i < 4; ++i) {
        g[i] = gate[i];
        u[i] = up[i];
        for (int split = 1; split < 8; ++split) {
            g[i] += gate[split * plane + i];
            u[i] += up[split * plane + i];
        }
    }
    fwht128(g[0], g[1], g[2], g[3], threadIdx.x);
    fwht128(u[0], u[1], u[2], u[3], threadIdx.x);
    for (int i = 0; i < 4; ++i) {
        float a = round_bf16(g[i] * kInvSqrt128 * f16_bits_to_f32(gate_svh[col + i]));
        float b = round_bf16(u[i] * kInvSqrt128 * f16_bits_to_f32(up_svh[col + i]));
        output[row * 1536 + col + i] = f32_to_bf16_rne_device(round_bf16(a / (1.f + expf(-a))) * b);
    }
}

template <int CompactRows>
void launch_shared_gate_up(const uint16_t *input, const Hy4ExpertProjection &gate, const Hy4ExpertProjection &up,
                           float *partial, int rows, hipStream_t stream) {
    gemm_fused<4, false, true, CompactRows><<<dim3(12 * 8, 1, 2), 256, CompactRows * 3 * 132, stream>>>(
        input, gate.suh, nullptr, nullptr, gate.trellis, partial, rows, 3072, 1536, 3072, 1536, rows, 8,
        up.suh, up.trellis);
}

}
hipError_t shelf_hy4_shared_gate_up(const uint16_t *input, Hy4ExpertProjection gate, Hy4ExpertProjection up,
                                    float *partial, uint16_t *output, int rows, hipStream_t stream) {
    auto supported = [](const Hy4ExpertProjection &matrix) {
        return matrix.suh && matrix.svh && matrix.trellis && matrix.bits == 4 && matrix.k == 3072 && matrix.n == 1536;
    };
    if (!input || !output || !partial || rows < 1 || rows > 16 || !supported(gate) || !supported(up))
        return hipErrorInvalidValue;
    switch (rows) {
    case 1:
        launch_shared_gate_up<1>(input, gate, up, partial, rows, stream);
        break;
    case 3:
        launch_shared_gate_up<3>(input, gate, up, partial, rows, stream);
        break;
    case 4:
        launch_shared_gate_up<4>(input, gate, up, partial, rows, stream);
        break;
    case 5:
        launch_shared_gate_up<5>(input, gate, up, partial, rows, stream);
        break;
    case 6:
        launch_shared_gate_up<6>(input, gate, up, partial, rows, stream);
        break;
    case 16:
        launch_shared_gate_up<16>(input, gate, up, partial, rows, stream);
        break;
    default:
        gemm_fused<4, false, true><<<dim3(12 * 8, 1, 2), 256, size_t(rows) * (3072 + 24 * sizeof(float)), stream>>>(
            input, gate.suh, nullptr, nullptr, gate.trellis, partial, rows, 3072, 1536, 3072, 1536, rows, 8,
            up.suh, up.trellis);
        break;
    }
    auto status = hipGetLastError();
    if (status != hipSuccess) return status;
    shared_swiglu<<<dim3(12, rows), 32, 0, stream>>>(partial, gate.svh, up.svh, output);
    return hipGetLastError();
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
    const size_t compact_shared = size_t(rows) * gps * 132u;
    if (!out_fused && !(bits == 4 && rows == 6 && splits == 8 && input_width == 3072 && output_width == 1536) &&
        compact_shared <= 48u * 1024u) {
        const dim3 grid4((padded_output / 64u) * splits, 1);
        if (bits == 4)
            gemm_fused_4w<4><<<grid4, 128, compact_shared, stream>>>(
                input, suh, trellis, partial, rows, input_width, padded_input, padded_output, splits);
        else
            gemm_fused_4w<6><<<grid4, 128, compact_shared, stream>>>(
                input, suh, trellis, partial, rows, input_width, padded_input, padded_output, splits);
        return hipGetLastError();
    }
    const size_t per_row = size_t(padded_input) + size_t(groups) * 4u;
    const size_t budget = out_fused ? 56u * 1024u : 64u * 1024u;
    int row_chunk = rows < 16 ? rows : 16;
    while (row_chunk > 1 && size_t(row_chunk) * per_row > budget)
        row_chunk /= 2;
    const size_t shared = size_t(row_chunk) * per_row;
    const uint32_t tiles = padded_output / 128u;
    const dim3 grid(tiles * splits, (rows + row_chunk - 1) / row_chunk);
    if (bits == 4 && rows == 6 && splits == 8 && input_width == 3072 && padded_input == 3072 &&
        output_width == 1536 && padded_output == 1536) {
        gemm_fused<4, false, false, 6><<<grid, 256, rows * gps * 132, stream>>>(
            input, suh, svh, output, trellis, partial, rows, input_width, output_width, padded_input,
            padded_output, row_chunk, splits);
    } else if (out_fused) {
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
template <bool Mixed, bool Decode = false, bool Prefill = false>
__global__ void expert_batch(const int8_t *codes, const float *scales, const Hy4ExpertProjection *matrices,
                             const Hy4Route *routes, const Hy4ExpertBatch *batches, float *partial, int k,
                             int n, const int *count = nullptr, int capacity = 0) {
    Hy4ExpertBatch batch;
    if constexpr (Decode) {
        if (int(blockIdx.y) >= *count)
            return;
        batch = batches ? batches[blockIdx.y] : Hy4ExpertBatch{int(blockIdx.y), 1};
        size_t plane = size_t(blockIdx.z) * capacity;
        codes += plane * k;
        scales += plane * (k / 128);
        partial += plane * n;
        matrices += blockIdx.z;
    } else {
        batch = batches[blockIdx.y];
    }
    size_t first = batch.first;
    auto matrix = matrices[3 * routes[first].expert];
    auto weight = matrix.trellis;
    if constexpr (Prefill) {
        if (matrix.bits == 4) {
            shelf::kernel::exl3_detail::mul24::group_body<4, 2, false, 4, false, true, true>(
                codes + first * k, scales + first * (k / 128), k, k / 128, weight, partial + first * n, batch.count,
                nullptr, n, k, k / 128, 1, blockIdx.x, 0);
        } else {
            shelf::kernel::exl3_detail::mul24::group_body<3, 2, false, 4, false, true>(
                codes + first * k, scales + first * (k / 128), k, k / 128, weight, partial + first * n, batch.count,
                nullptr, n, k, k / 128, 1, blockIdx.x, 0);
        }
        return;
    }
    if constexpr (Decode) {
        if (k == 1536 && n == 1536 && matrix.bits == 4) {
            if (batch.count == 1)
                shelf::kernel::exl3_detail::mul24::group_body<4, 1, false, 4, false, true>(
                    codes + first * 1536, scales + first * 12, 1536, 12, weight, partial + first * 1536, 1,
                    nullptr, 1536, 1536, 12, 1, blockIdx.x, 0);
            else
                shelf::kernel::exl3_detail::mul24::group_body<4, 1, false, 4, false, true, true>(
                    codes + first * 1536, scales + first * 12, 1536, 12, weight, partial + first * 1536, batch.count,
                    nullptr, 1536, 1536, 12, 1, blockIdx.x, 0);
            return;
        }
    }
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

hipError_t shelf_hy4_wmma_decode(const int8_t *codes, const float *scales, const Hy4ExpertProjection *matrices,
                                 const Hy4Route *routes, const int *count, float *partial, int capacity,
                                 int k, int n, int projections, hipStream_t stream, const Hy4ExpertBatch *batches,
                                 const int *num_batches, bool prefill) {
    if (prefill) {
        expert_batch<true, true, true><<<dim3(n / 64, std::min(capacity, 512 + capacity / 32), projections), 128, 0, stream>>>(
            codes, scales, matrices, routes, batches, partial, k, n, num_batches, capacity);
        return hipGetLastError();
    }
    const int groups = batches ? std::min(capacity, 512 + capacity / 16) : capacity;
    expert_batch<true, true><<<dim3(n / 64, groups, projections), 128, 0, stream>>>(
        codes, scales, matrices, routes, batches, partial, k, n, batches ? num_batches : count, capacity);
    return hipGetLastError();
}
