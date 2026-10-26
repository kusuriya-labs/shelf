// vendored from PhaseShift (own work, CopenDeCamp); namespace/type adapted only.
#include <shelf/backend/hip/kernels/gemm/linear_bf16.h>
#include <shelf/backend/hip/kernels/gemm/linear_bf16_config.h>
#include <shelf/backend/hip/kernels/detail/tile/bf16_wmma_gfx12.h>
#include <shelf/backend/hip/kernels/gemm/detail/linear_bf16_device.h>
#include <shelf/foundation/numeric.hpp>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstddef>

namespace shelf::hipk {

// Device helpers live in the private headers above; entries/launchers stay here.
using namespace tile;
using namespace gemm_detail;

extern "C" __global__ __launch_bounds__(256)
void phaseshift_gemm_bf16_exact_rows_1(Bf16GemmExactRowsArgs args);

namespace {

constexpr uint32_t kExactRowsThreads = 256u;
static_assert(kExactRowsThreads == kBf16GemmExactRowsThreads);
static_assert(kTileOut == kBf16WmmaOutTile);
static_assert(kTileRow == kBf16WmmaRowTile);
static_assert(kWideRowTiles == kBf16WmmaWideRowTiles);
static_assert(kThreadsPerBlock == kBf16WmmaThreads);
static_assert(kSplitThreadsPerBlock == kBf16SplitKWmmaThreads);

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
