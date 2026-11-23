#include <shelf/kernels/fused/bf16_linear_pair.h>
#include <shelf/kernels/fused/detail/bf16_exact_rows_device.h>
#include <shelf/kernels/optimized/linear/config.h>

#include <hip/hip_runtime.h>

#include <cstdint>

namespace shelf::kernel {
namespace {

using detail::gemm_bf16_exact_rows_body;

template <uint32_t kRows>
__global__ void __launch_bounds__(detail::kExactRowsThreads) gemm_bf16_exact_rows_pair_kernel(
    const bf16_t* __restrict__ weight0,
    void* __restrict__ output0,
    const uint32_t output_dtype0,
    const uint32_t n0,
    const uint32_t output_row_stride0,
    const bf16_t* __restrict__ weight1,
    void* __restrict__ output1,
    const uint32_t output_dtype1,
    const uint32_t n1,
    const uint32_t output_row_stride1,
    const bf16_t* __restrict__ input,
    const uint32_t k,
    const uint32_t input_row_stride,
    const uint32_t blocks0) {
    const uint32_t block = blockIdx.x;
    if (block < blocks0) {
        gemm_bf16_exact_rows_body<kRows>(weight0, input, output0, output_dtype0, n0, k,
                                         input_row_stride, output_row_stride0, block);
    } else {
        gemm_bf16_exact_rows_body<kRows>(weight1, input, output1, output_dtype1, n1, k,
                                         input_row_stride, output_row_stride1,
                                         block - blocks0);
    }
}

}  // namespace

#define PS_LAUNCH_EXACT_ROWS_PAIR(R)                                               \
    case R: {                                                                      \
        const uint32_t blocks0 = bf16_exact_rows_grid_x(out_features0);             \
        const uint32_t blocks1 = bf16_exact_rows_grid_x(out_features1);             \
        gemm_bf16_exact_rows_pair_kernel<R>                                         \
            <<<blocks0 + blocks1, detail::kExactRowsThreads, 0, stream>>>(          \
                weight0, output0, static_cast<uint32_t>(output_dtype0), out_features0, \
                output_row_stride0, weight1, output1,                               \
                static_cast<uint32_t>(output_dtype1), out_features1,                \
                output_row_stride1, input, k, input_row_stride, blocks0);           \
        break;                                                                      \
    }

hipError_t launch_gemm_bf16_exact_rows_pair(
    const bf16_t* weight0,
    void* output0,
    Bf16GemmOutputDType output_dtype0,
    uint32_t out_features0,
    uint32_t output_row_stride0,
    const bf16_t* weight1,
    void* output1,
    Bf16GemmOutputDType output_dtype1,
    uint32_t out_features1,
    uint32_t output_row_stride1,
    const bf16_t* input,
    uint32_t input_row_stride,
    uint32_t rows,
    uint32_t k,
    bool dot2,
    hipStream_t stream) {
    if (rows == 0u || rows > kBf16GemmExactRowsKernelMax || dot2) {
        return hipErrorNotSupported;
    }
    switch (rows) {
        PS_LAUNCH_EXACT_ROWS_PAIR(1u)
        PS_LAUNCH_EXACT_ROWS_PAIR(2u)
        PS_LAUNCH_EXACT_ROWS_PAIR(3u)
        PS_LAUNCH_EXACT_ROWS_PAIR(4u)
        PS_LAUNCH_EXACT_ROWS_PAIR(5u)
        PS_LAUNCH_EXACT_ROWS_PAIR(6u)
        PS_LAUNCH_EXACT_ROWS_PAIR(7u)
        PS_LAUNCH_EXACT_ROWS_PAIR(8u)
        PS_LAUNCH_EXACT_ROWS_PAIR(9u)
        PS_LAUNCH_EXACT_ROWS_PAIR(10u)
        PS_LAUNCH_EXACT_ROWS_PAIR(11u)
        PS_LAUNCH_EXACT_ROWS_PAIR(12u)
        PS_LAUNCH_EXACT_ROWS_PAIR(13u)
        PS_LAUNCH_EXACT_ROWS_PAIR(14u)
        PS_LAUNCH_EXACT_ROWS_PAIR(15u)
        PS_LAUNCH_EXACT_ROWS_PAIR(16u)
        default:
            return hipErrorNotSupported;
    }
    return hipGetLastError();
}

#undef PS_LAUNCH_EXACT_ROWS_PAIR

}  // namespace shelf::kernel
