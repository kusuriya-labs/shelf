// vendored from PhaseShift (own work, CopenDeCamp); namespace/type adapted only.
#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstddef>

namespace shelf::hipk {

extern "C" __global__ __launch_bounds__(256)
void phaseshift_gemm_bf16_exact_rows_1(Bf16GemmExactRowsArgs args);

namespace {

using short8_t = short __attribute__((ext_vector_type(8)));
using float8_t = float __attribute__((ext_vector_type(8)));
using short2_t = short __attribute__((ext_vector_type(2)));

constexpr uint32_t kTileOut = 16u;
constexpr uint32_t kTileRow = 16u;
constexpr uint32_t kWideRowTiles = 4u;
constexpr uint32_t kThreadsPerBlock = 32u;

constexpr uint32_t kSplitWarps = 8u;
constexpr uint32_t kSplitThreadsPerBlock = 32u * kSplitWarps;

constexpr uint32_t kExactRowsThreads = 256u;
static_assert(kExactRowsThreads == kBf16GemmExactRowsThreads);
static_assert(kTileOut == kBf16WmmaOutTile);
static_assert(kTileRow == kBf16WmmaRowTile);
static_assert(kWideRowTiles == kBf16WmmaWideRowTiles);
static_assert(kThreadsPerBlock == kBf16WmmaThreads);
static_assert(kSplitThreadsPerBlock == kBf16SplitKWmmaThreads);

__device__ __forceinline__ uint64_t load_bf16x4_masked(
    const shelf::bf16_t* base, uint32_t start_k, uint32_t logical_k) {
    const shelf::bf16_t* p = base + start_k;
    if (start_k + 3u < logical_k && (reinterpret_cast<uintptr_t>(p) & 7u) == 0u) {
        return *reinterpret_cast<const uint64_t*>(p);
    }
    uint64_t out = 0;
    #pragma unroll
    for (uint32_t q = 0; q < 4u; ++q) {
        const uint32_t kk = start_k + q;
        const uint16_t bits = (kk < logical_k) ? p[q].data : 0u;
        out |= static_cast<uint64_t>(bits) << (16u * q);
    }
    return out;
}

__device__ __forceinline__ short8_t assemble_short8(uint64_t lo, uint64_t hi) {
    short8_t v = {};
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        v[q] = static_cast<short>((lo >> (16u * q)) & 0xFFFFu);
        v[q + 4] = static_cast<short>((hi >> (16u * q)) & 0xFFFFu);
    }
    return v;
}

__device__ __forceinline__ float bf16_f32(short s) {
    const uint32_t ui = static_cast<uint32_t>(static_cast<unsigned short>(s)) << 16u;
    union { uint32_t u; float f; } b;
    b.u = ui;
    return b.f;
}

__device__ __forceinline__ short8_t wide_load_a(
    const shelf::bf16_t* wrow, uint32_t ks, bool out_valid, uint32_t k, uint32_t k_half) {
    uint64_t a0 = 0, a1 = 0;
    if (out_valid) {
        a0 = load_bf16x4_masked(wrow, ks + k_half * 4u, k);
        a1 = load_bf16x4_masked(wrow, ks + 8u + k_half * 4u, k);
    }
    return assemble_short8(a0, a1);
}

__device__ __forceinline__ void wide_tile(
    uint32_t ks, const short8_t& a, const shelf::bf16_t* xrow, bool row_valid,
    float8_t& acc, uint32_t k, uint32_t k_half) {
    uint64_t b0 = 0, b1 = 0;
    if (row_valid) {
        b0 = load_bf16x4_masked(xrow, ks + k_half * 4u, k);
        b1 = load_bf16x4_masked(xrow, ks + 8u + k_half * 4u, k);
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, assemble_short8(b0, b1), acc);
}

template <uint32_t kRowTiles>
__device__ __forceinline__ void gemm_bf16_wmma_ktiles(
    const shelf::bf16_t* wrow,
    const shelf::bf16_t* const* xrow,
    const bool* row_valid,
    float8_t* acc,
    uint32_t k_tiles,
    uint32_t k,
    uint32_t k_half,
    bool out_valid) {
    for (uint32_t kb = 0; kb < k_tiles; ++kb) {
        const uint32_t ks = kb * 16u;
        const short8_t a = wide_load_a(wrow, ks, out_valid, k, k_half);
        #pragma unroll
        for (uint32_t rt = 0; rt < kRowTiles; ++rt)
            wide_tile(ks, a, xrow[rt], row_valid[rt], acc[rt], k, k_half);
    }
}

template <uint32_t kRowTiles>
__device__ __forceinline__ void bf16_wmma_body(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    static_assert(kRowTiles >= 1u, "bf16 wmma row tiles must be positive");

    const uint32_t lane = threadIdx.x;
    const uint32_t lane_low4 = lane & 15u;
    const uint32_t k_half = lane >> 4;

    const uint32_t out_start = blockIdx.x * kTileOut;
    const uint32_t row_start = blockIdx.y * (kRowTiles * kTileRow);

    if (out_start >= out_features || row_start >= rows)
        return;

    const uint32_t a_row = out_start + lane_low4;
    const bool out_valid = a_row < out_features;

    const shelf::bf16_t* wrow = weight + static_cast<size_t>(a_row) * k;

    float8_t acc[kRowTiles];
    #pragma unroll
    for (uint32_t rt = 0; rt < kRowTiles; ++rt)
        acc[rt] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const shelf::bf16_t* xrow[kRowTiles];
    bool row_valid[kRowTiles];
    #pragma unroll
    for (uint32_t rt = 0; rt < kRowTiles; ++rt) {
        const uint32_t b_col = row_start + rt * kTileRow + lane_low4;
        row_valid[rt] = b_col < rows;
        xrow[rt] = input + static_cast<size_t>(b_col) * input_row_stride;
    }

    const uint32_t k_tiles = (k + 15u) >> 4;
    if constexpr (kRowTiles == 1u) {
        gemm_bf16_wmma_ktiles<1u>(wrow, xrow, row_valid, acc, k_tiles, k, k_half, out_valid);
    } else {
        const uint32_t active = (rows - row_start + kTileRow - 1u) / kTileRow;
        switch (active) {
        case 1u:
            gemm_bf16_wmma_ktiles<1u>(wrow, xrow, row_valid, acc, k_tiles, k, k_half, out_valid);
            break;
        case 2u:
            if constexpr (kRowTiles >= 2u)
                gemm_bf16_wmma_ktiles<2u>(wrow, xrow, row_valid, acc, k_tiles, k, k_half, out_valid);
            break;
        case 3u:
            if constexpr (kRowTiles >= 3u)
                gemm_bf16_wmma_ktiles<3u>(wrow, xrow, row_valid, acc, k_tiles, k, k_half, out_valid);
            break;
        default:
            gemm_bf16_wmma_ktiles<kRowTiles>(wrow, xrow, row_valid, acc, k_tiles, k, k_half, out_valid);
            break;
        }
    }

    #pragma unroll
    for (uint32_t rt = 0; rt < kRowTiles; ++rt) {
        if (row_start + rt * kTileRow < rows) {
            const uint32_t r = row_start + rt * kTileRow + lane_low4;
            if (r < rows) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t o = out_start + 8u * (lane >> 4) + static_cast<uint32_t>(j);
                    if (o >= out_features)
                        continue;
                    const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o;
                    if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16)) {
                        static_cast<shelf::bf16_t*>(output)[out_idx] =
                            shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(acc[rt][j]))};
                    } else {
                        static_cast<float*>(output)[out_idx] = acc[rt][j];
                    }
                }
            }
        }
    }
}

template <uint32_t kRowTiles>
__global__ void __launch_bounds__(kThreadsPerBlock) gemm_bf16_wmma_kernel(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    bf16_wmma_body<kRowTiles>(weight, input, output, output_dtype, rows,
                              out_features, k, input_row_stride,
                              output_row_stride);
}

__device__ __forceinline__ void bf16_splitk_wmma_body(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane_low4 = lane & 15u;
    const uint32_t k_half = lane >> 4;

    const uint32_t out_start = blockIdx.x * kTileOut;
    const uint32_t row_start = blockIdx.y * kTileRow;

    if (out_start >= out_features || row_start >= rows)
        return;

    const uint32_t a_row = out_start + lane_low4;
    const uint32_t b_col = row_start + lane_low4;
    const bool out_valid = a_row < out_features;
    const bool row_valid = b_col < rows;

    const shelf::bf16_t* wrow = weight + static_cast<size_t>(a_row) * k;
    const shelf::bf16_t* xrow = input + static_cast<size_t>(b_col) * input_row_stride;

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const uint32_t k_tiles = (k + 15u) >> 4;
    const uint32_t kb_begin = (k_tiles * warp) / kSplitWarps;
    const uint32_t kb_end = (k_tiles * (warp + 1u)) / kSplitWarps;
    for (uint32_t kb = kb_begin; kb < kb_end; ++kb) {
        const uint32_t ks = kb * 16u;
        uint64_t a0 = 0, a1 = 0, b0 = 0, b1 = 0;
        if (out_valid) {
            a0 = load_bf16x4_masked(wrow, ks + k_half * 4u, k);
            a1 = load_bf16x4_masked(wrow, ks + 8u + k_half * 4u, k);
        }
        if (row_valid) {
            b0 = load_bf16x4_masked(xrow, ks + k_half * 4u, k);
            b1 = load_bf16x4_masked(xrow, ks + 8u + k_half * 4u, k);
        }
        const short8_t a = assemble_short8(a0, a1);
        const short8_t b = assemble_short8(b0, b1);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc);
    }

    __shared__ float red[kSplitWarps][8][32];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        red[warp][j][lane] = acc[j];
    __syncthreads();

    if (warp == 0u) {
        float s[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (uint32_t w = 0; w < kSplitWarps; ++w) {
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                s[j] += red[w][j][lane];
        }
        const uint32_t r = row_start + lane_low4;
        if (r < rows) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint32_t o = out_start + 8u * (lane >> 4) + static_cast<uint32_t>(j);
                if (o >= out_features)
                    continue;
                const size_t out_idx = static_cast<size_t>(r) * output_row_stride + o;
                if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16)) {
                    static_cast<shelf::bf16_t*>(output)[out_idx] =
                        shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(s[j]))};
                } else {
                    static_cast<float*>(output)[out_idx] = s[j];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(kSplitThreadsPerBlock) gemm_bf16_splitk_wmma_kernel(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t rows,
    const uint32_t out_features,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    bf16_splitk_wmma_body(weight, input, output, output_dtype, rows,
                          out_features, k, input_row_stride, output_row_stride);
}

template <uint32_t kRows, bool kDot2 = false>
__device__ __forceinline__ void gemm_bf16_exact_rows_body(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t n,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    const uint32_t wave = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t r = blockIdx.x * 8u + wave;
    if (r >= n)
        return;
    const shelf::bf16_t* wrow = weight + static_cast<size_t>(r) * k;
    float acc[kRows];
    #pragma unroll
    for (uint32_t m = 0; m < kRows; ++m)
        acc[m] = 0.0f;
    const uint32_t iters = k >> 8u;
    for (uint32_t i = 0; i < iters; ++i) {
        const uint32_t off = (i << 8) + (lane << 3);
        const short8_t w = *reinterpret_cast<const short8_t*>(wrow + off);
        if constexpr (kDot2) {
            #pragma unroll
            for (uint32_t m = 0; m < kRows; ++m) {
                const short8_t x = *reinterpret_cast<const short8_t*>(
                    input + static_cast<size_t>(m) * input_row_stride + off);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; j += 2u) {
                    const short2_t w2 = short2_t{w[j], w[j + 1u]};
                    const short2_t x2 = short2_t{x[j], x[j + 1u]};
                    acc[m] = __builtin_amdgcn_fdot2_f32_bf16(w2, x2, acc[m], false);
                }
            }
        } else {
            float wf[8];
            #pragma unroll
            for (uint32_t j = 0; j < 8u; ++j)
                wf[j] = bf16_f32(w[j]);
            #pragma unroll
            for (uint32_t m = 0; m < kRows; ++m) {
                const short8_t x = *reinterpret_cast<const short8_t*>(
                    input + static_cast<size_t>(m) * input_row_stride + off);
                #pragma unroll
                for (uint32_t j = 0; j < 8u; ++j)
                    acc[m] += wf[j] * bf16_f32(x[j]);
            }
        }
    }
    const uint32_t tail = k & 255u;
    if (tail != 0u) {
        const uint32_t base = iters << 8;
        #pragma unroll
        for (uint32_t m = 0; m < kRows; ++m) {
            const shelf::bf16_t* xrow = input + static_cast<size_t>(m) * input_row_stride;
            for (uint32_t j = lane; j < tail; j += 32u)
                acc[m] += bf16_f32(static_cast<short>(wrow[base + j].data))
                        * bf16_f32(static_cast<short>(xrow[base + j].data));
        }
    }
    #pragma unroll
    for (uint32_t m = 0; m < kRows; ++m) {
        float a = acc[m];
        #pragma unroll
        for (uint32_t s = 16u; s > 0u; s >>= 1)
            a += __shfl_down(a, s, 32);
        if (lane == 0u) {
            const size_t oidx = static_cast<size_t>(m) * output_row_stride + r;
            if (output_dtype == static_cast<uint32_t>(Bf16GemmOutputDType::BF16))
                static_cast<shelf::bf16_t*>(output)[oidx] =
                    shelf::bf16_t{__bfloat16_as_ushort(__float2bfloat16(a))};
            else
                static_cast<float*>(output)[oidx] = a;
        }
    }
}

template <uint32_t kRows>
__global__ void __launch_bounds__(kExactRowsThreads) gemm_bf16_exact_rows_kernel(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t n,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    gemm_bf16_exact_rows_body<kRows>(
        weight, input, output, output_dtype, n, k, input_row_stride,
        output_row_stride);
}

template <uint32_t kRows>
__global__ void __launch_bounds__(kExactRowsThreads) gemm_bf16_exact_rows_dot2_kernel(
    const shelf::bf16_t* __restrict__ weight,
    const shelf::bf16_t* __restrict__ input,
    void* __restrict__ output,
    const uint32_t output_dtype,
    const uint32_t n,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t output_row_stride) {
    gemm_bf16_exact_rows_body<kRows, true>(
        weight, input, output, output_dtype, n, k, input_row_stride,
        output_row_stride);
}

#define PS_LAUNCH_EXACT_ROWS(R)                                                    \
    case R:                                                                        \
        gemm_bf16_exact_rows_kernel<R><<<grid, kExactRowsThreads, 0, stream>>>(    \
            weight, input, output, static_cast<uint32_t>(output_dtype),            \
            out_features, k, input_row_stride, output_row_stride);                 \
        break;

hipError_t launch_gemm_bf16_exact_rows_dot2(
    const bf16_t* weight,
    const bf16_t* input,
    void* output,
    Bf16GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    hipStream_t stream);

hipError_t launch_gemm_bf16_exact_rows(
    const bf16_t* weight,
    const bf16_t* input,
    void* output,
    Bf16GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    hipStream_t stream) {
    const uint32_t grid = bf16_exact_rows_grid_x(out_features);
    if (rows > kBf16GemmExactRowsKernelMax) {
        const size_t output_elem =
            output_dtype == Bf16GemmOutputDType::BF16 ? sizeof(shelf::bf16_t) : sizeof(float);
        for (uint32_t base = 0u; base < rows; base += kBf16GemmExactRowsKernelMax) {
            const uint32_t chunk =
                (rows - base) < kBf16GemmExactRowsKernelMax ? (rows - base)
                                                            : kBf16GemmExactRowsKernelMax;
            const hipError_t err = launch_gemm_bf16_exact_rows(
                weight, input + static_cast<size_t>(base) * input_row_stride,
                static_cast<char*>(output) +
                    static_cast<size_t>(base) * output_row_stride * output_elem,
                output_dtype, chunk, out_features, k, input_row_stride, output_row_stride,
                stream);
            if (err != hipSuccess) return err;
        }
        return hipGetLastError();
    }
    switch (rows) {
        case 1u: {
            Bf16GemmExactRowsArgs args{};
            args.weight = weight;
            args.input = input;
            args.output = output;
            args.output_dtype = static_cast<uint32_t>(output_dtype);
            args.n = out_features;
            args.k = k;
            args.input_row_stride = input_row_stride;
            args.output_row_stride = output_row_stride;
            phaseshift_gemm_bf16_exact_rows_1<<<grid, kExactRowsThreads,
                                                       0, stream>>>(args);
            break;
        }
        PS_LAUNCH_EXACT_ROWS(2u)
        PS_LAUNCH_EXACT_ROWS(3u)
        PS_LAUNCH_EXACT_ROWS(4u)
        PS_LAUNCH_EXACT_ROWS(5u)
        PS_LAUNCH_EXACT_ROWS(6u)
        PS_LAUNCH_EXACT_ROWS(7u)
        PS_LAUNCH_EXACT_ROWS(8u)
        PS_LAUNCH_EXACT_ROWS(9u)
        PS_LAUNCH_EXACT_ROWS(10u)
        PS_LAUNCH_EXACT_ROWS(11u)
        PS_LAUNCH_EXACT_ROWS(12u)
        PS_LAUNCH_EXACT_ROWS(13u)
        PS_LAUNCH_EXACT_ROWS(14u)
        PS_LAUNCH_EXACT_ROWS(15u)
        PS_LAUNCH_EXACT_ROWS(16u)
        default:
            return hipErrorNotSupported;
    }
    return hipGetLastError();
}

#define PS_LAUNCH_EXACT_ROWS_DOT2(R)                                               \
    case R:                                                                        \
        gemm_bf16_exact_rows_dot2_kernel<R><<<grid, kExactRowsThreads, 0, stream>>>(\
            weight, input, output, static_cast<uint32_t>(output_dtype),            \
            out_features, k, input_row_stride, output_row_stride);                 \
        break;

hipError_t launch_gemm_bf16_exact_rows_dot2(
    const bf16_t* weight,
    const bf16_t* input,
    void* output,
    Bf16GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    hipStream_t stream) {
    const uint32_t grid = bf16_exact_rows_grid_x(out_features);
    if (rows > kBf16GemmExactRowsKernelMax) {
        const size_t output_elem =
            output_dtype == Bf16GemmOutputDType::BF16 ? sizeof(shelf::bf16_t) : sizeof(float);
        for (uint32_t base = 0u; base < rows; base += kBf16GemmExactRowsKernelMax) {
            const uint32_t chunk =
                (rows - base) < kBf16GemmExactRowsKernelMax ? (rows - base)
                                                            : kBf16GemmExactRowsKernelMax;
            const bf16_t* chunk_input =
                input + static_cast<size_t>(base) * input_row_stride;
            char* chunk_output =
                static_cast<char*>(output) +
                static_cast<size_t>(base) * output_row_stride * output_elem;
            if (chunk == 1u) {
                const hipError_t err = launch_gemm_bf16_exact_rows(
                    weight, chunk_input, chunk_output, output_dtype, chunk, out_features, k,
                    input_row_stride, output_row_stride, stream);
                if (err != hipSuccess) return err;
                continue;
            }
            const hipError_t err = launch_gemm_bf16_exact_rows_dot2(
                weight, chunk_input, chunk_output, output_dtype, chunk, out_features, k,
                input_row_stride, output_row_stride, stream);
            if (err != hipSuccess) return err;
        }
        return hipGetLastError();
    }
    switch (rows) {
        PS_LAUNCH_EXACT_ROWS_DOT2(2u)
        PS_LAUNCH_EXACT_ROWS_DOT2(3u)
        PS_LAUNCH_EXACT_ROWS_DOT2(4u)
        PS_LAUNCH_EXACT_ROWS_DOT2(5u)
        PS_LAUNCH_EXACT_ROWS_DOT2(6u)
        PS_LAUNCH_EXACT_ROWS_DOT2(7u)
        PS_LAUNCH_EXACT_ROWS_DOT2(8u)
        PS_LAUNCH_EXACT_ROWS_DOT2(9u)
        PS_LAUNCH_EXACT_ROWS_DOT2(10u)
        PS_LAUNCH_EXACT_ROWS_DOT2(11u)
        PS_LAUNCH_EXACT_ROWS_DOT2(12u)
        PS_LAUNCH_EXACT_ROWS_DOT2(13u)
        PS_LAUNCH_EXACT_ROWS_DOT2(14u)
        PS_LAUNCH_EXACT_ROWS_DOT2(15u)
        PS_LAUNCH_EXACT_ROWS_DOT2(16u)
        default:
            return hipErrorNotSupported;
    }
    return hipGetLastError();
}

#undef PS_LAUNCH_EXACT_ROWS_DOT2
#undef PS_LAUNCH_EXACT_ROWS

}  // namespace

extern "C" __global__ __launch_bounds__(kExactRowsThreads)
void phaseshift_gemm_bf16_exact_rows_1(Bf16GemmExactRowsArgs args) {
    gemm_bf16_exact_rows_body<1>(
        static_cast<const bf16_t*>(args.weight),
        static_cast<const bf16_t*>(args.input), args.output, args.output_dtype,
        args.n, args.k, args.input_row_stride, args.output_row_stride);
}

#define PS_BF16_EXACT_ROWS_ENTRY(R)                                            \
    extern "C" __global__ __launch_bounds__(kExactRowsThreads)                 \
    void phaseshift_gemm_bf16_exact_rows_##R(                           \
        Bf16GemmExactRowsArgs args) {                                          \
        gemm_bf16_exact_rows_body<R>(                                          \
            static_cast<const bf16_t*>(args.weight),                           \
            static_cast<const bf16_t*>(args.input), args.output,               \
            args.output_dtype, args.n, args.k, args.input_row_stride,          \
            args.output_row_stride);                                           \
    }

PS_BF16_EXACT_ROWS_ENTRY(2)
PS_BF16_EXACT_ROWS_ENTRY(3)
PS_BF16_EXACT_ROWS_ENTRY(4)
PS_BF16_EXACT_ROWS_ENTRY(5)
PS_BF16_EXACT_ROWS_ENTRY(6)
PS_BF16_EXACT_ROWS_ENTRY(7)
PS_BF16_EXACT_ROWS_ENTRY(8)
PS_BF16_EXACT_ROWS_ENTRY(9)
PS_BF16_EXACT_ROWS_ENTRY(10)
PS_BF16_EXACT_ROWS_ENTRY(11)
PS_BF16_EXACT_ROWS_ENTRY(12)
PS_BF16_EXACT_ROWS_ENTRY(13)
PS_BF16_EXACT_ROWS_ENTRY(14)
PS_BF16_EXACT_ROWS_ENTRY(15)
PS_BF16_EXACT_ROWS_ENTRY(16)

#undef PS_BF16_EXACT_ROWS_ENTRY

extern "C" __global__ __launch_bounds__(kThreadsPerBlock)
void phaseshift_gemm_bf16_wmma(Bf16GemmWmmaArgs args) {
    bf16_wmma_body<1u>(
        static_cast<const shelf::bf16_t*>(args.weight),
        static_cast<const shelf::bf16_t*>(args.input), args.output,
        args.output_dtype, args.rows, args.out_features, args.k,
        args.input_row_stride, args.output_row_stride);
}

extern "C" __global__ __launch_bounds__(kThreadsPerBlock)
void phaseshift_gemm_bf16_wmma_wide(Bf16GemmWmmaArgs args) {
    bf16_wmma_body<kWideRowTiles>(
        static_cast<const shelf::bf16_t*>(args.weight),
        static_cast<const shelf::bf16_t*>(args.input), args.output,
        args.output_dtype, args.rows, args.out_features, args.k,
        args.input_row_stride, args.output_row_stride);
}

extern "C" __global__ __launch_bounds__(kSplitThreadsPerBlock)
void phaseshift_gemm_bf16_splitk_wmma(Bf16GemmWmmaArgs args) {
    bf16_splitk_wmma_body(
        static_cast<const shelf::bf16_t*>(args.weight),
        static_cast<const shelf::bf16_t*>(args.input), args.output,
        args.output_dtype, args.rows, args.out_features, args.k,
        args.input_row_stride, args.output_row_stride);
}

hipError_t launch_gemm_bf16(
    const Bf16GemmConfig& config,
    const bf16_t* weight,
    const bf16_t* input,
    void* output,
    Bf16GemmOutputDType output_dtype,
    uint32_t rows,
    uint32_t out_features,
    uint32_t k,
    uint32_t input_row_stride,
    uint32_t output_row_stride,
    hipStream_t stream) {
    if (rows == 0 || out_features == 0 || k == 0)
        return hipSuccess;

    const uint32_t out_tiles = bf16_wmma_grid_x(out_features);

    switch (config.id) {
    case Bf16GemmConfigId::ExactRows: {
        if (config.exact_rows == 0u || config.exact_rows > kBf16GemmExactRowsMax ||
            config.exact_rows != rows)
            return hipErrorNotSupported;
        return launch_gemm_bf16_exact_rows(
            weight, input, output, output_dtype, config.exact_rows, out_features, k,
            input_row_stride, output_row_stride, stream);
    }
    case Bf16GemmConfigId::ExactRowsDot2: {
        if (config.exact_rows == 0u || config.exact_rows > kBf16GemmExactRowsMax ||
            config.exact_rows != rows)
            return hipErrorNotSupported;
        return launch_gemm_bf16_exact_rows_dot2(
            weight, input, output, output_dtype, config.exact_rows, out_features, k,
            input_row_stride, output_row_stride, stream);
    }
    case Bf16GemmConfigId::WmmaKPartition: {
        const dim3 grid(out_tiles, bf16_wmma_grid_y(rows, 1u), 1);
        gemm_bf16_splitk_wmma_kernel<<<grid, kSplitThreadsPerBlock, 0, stream>>>(
            weight, input, output,
            static_cast<uint32_t>(output_dtype), rows, out_features, k,
            input_row_stride, output_row_stride);
        return hipGetLastError();
    }
    case Bf16GemmConfigId::WmmaWide: {
        const dim3 grid(out_tiles, bf16_wmma_grid_y(rows, kWideRowTiles), 1);
        gemm_bf16_wmma_kernel<kWideRowTiles><<<grid, kThreadsPerBlock, 0, stream>>>(
            weight, input, output,
            static_cast<uint32_t>(output_dtype), rows, out_features, k,
            input_row_stride, output_row_stride);
        return hipGetLastError();
    }
    case Bf16GemmConfigId::Wmma:
    default: {
        const dim3 grid(out_tiles, bf16_wmma_grid_y(rows, 1u), 1);
        gemm_bf16_wmma_kernel<1u><<<grid, kThreadsPerBlock, 0, stream>>>(
            weight, input, output,
            static_cast<uint32_t>(output_dtype), rows, out_features, k,
            input_row_stride, output_row_stride);
        return hipGetLastError();
    }
    }
}

}  // namespace shelf::hipk
