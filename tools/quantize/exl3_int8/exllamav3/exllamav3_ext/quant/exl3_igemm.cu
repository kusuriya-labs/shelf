// =============================================================================
// exl3_igemm.cu -- int8 -> int32 row-major GEMM for the EXL3 W8A8 prefill path
// =============================================================================
//
// Row-major C = A @ B with A/B int8 and C int32, fp32-style call convention
// mirroring hgemm_gemmex_impl (the cuBLAS call is column-major, so m/n/lda
// are swapped the same way). One function, no PTX, no architecture
// specialisation: it compiles for CUDA (cuBLAS) and, through the cuda_shim
// cublas->hipblas aliases, for ROCm (hipBLAS/rocBLAS) unchanged.
//
// Numerical contract (relied on by modules/quant/exl3.py):
//   - exact int32 accumulation (rocBLAS/cuBLAS int8 GEMM uses int32
//     accumulation; K * 127 * 127 < 2^31 for every EXL3 shape).
//   - C must be int32 and contiguous in its last dim; A/B int8 row-major.
// Per-row/per-column rescaling is the caller's job (the Python side folds
// the weight scales into svh and applies the activation scales after).
// =============================================================================

#include <cublas_v2.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "../util.h"
#include "../util.cuh"
#include "exl3_devctx.cuh"
#include <limits>

void igemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
)
// Batched row-major C[b] = A[b] @ W[b], A/W int8, C int32, one rocBLAS/
// cuBLAS strided-batched launch for the grouped INT8 MoE path (gate/up
// share one launch via an N-concatenated weight slab). All three slabs
// must be contiguous; batch/ M/K/N come from the tensor shapes.
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    TORCH_CHECK_DTYPE(a, kChar);
    TORCH_CHECK_DTYPE(w, kChar);
    TORCH_CHECK_DTYPE(c, kInt);
    TORCH_CHECK_DIM(a, 3);
    TORCH_CHECK_DIM(w, 3);
    TORCH_CHECK_DIM(c, 3);
    TORCH_CHECK(a.is_contiguous() && w.is_contiguous() && c.is_contiguous(),
                "igemm_batched: slabs must be contiguous");
    TORCH_CHECK_SHAPES(a, 0, w, 0, 1);
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK_SHAPES(a, 2, w, 1, 1);
    TORCH_CHECK_SHAPES(a, 1, c, 1, 1);
    TORCH_CHECK_SHAPES(w, 2, c, 2, 1);

    int batch = a.size(0);
    int size_m = a.size(1);
    int size_k = a.size(2);
    int size_n = w.size(2);
    if (!batch || !size_m || !size_n || !size_k) return;

    const int8_t* a_ptr = (const int8_t*) a.data_ptr();
    const int8_t* w_ptr = (const int8_t*) w.data_ptr();

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, at::cuda::getCurrentCUDAStream().stream());
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    int32_t alpha_ = 1;
    int32_t beta_ = 0;
    auto r = cublasGemmStridedBatchedEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, w_ptr, CUDA_R_8I, size_n, (long long) size_k * size_n,
                 a_ptr, CUDA_R_8I, size_k, (long long) size_m * size_k,
        &beta_,  c.data_ptr(), CUDA_R_32I, size_n, (long long) size_m * size_n,
        batch,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}

void igemm_recon
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
)
{
    const at::cuda::OptionalCUDAGuard device_guard(a.device());

    TORCH_CHECK_DTYPE(a, kChar);
    TORCH_CHECK_DTYPE(b, kChar);
    TORCH_CHECK_DTYPE(c, kInt);
    TORCH_CHECK_DIM(b, 2);
    TORCH_CHECK(c.dim() == 2, "igemm_recon: c must be rank 2");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "igemm_recon: a and b must be contiguous");
    TORCH_CHECK_SHAPES(a, -1, b, 0, 1);
    TORCH_CHECK_SHAPES(b, 1, c, -1, 1);
    TORCH_CHECK(c.stride(-1) == 1, "c must have contiguous columns");

    const int8_t* a_ptr = (const int8_t*) a.data_ptr();
    const int8_t* b_ptr = (const int8_t*) b.data_ptr();

    int size_k = a.size(-1);
    int size_m = a.numel() / size_k;
    int size_n = b.size(-1);
    int64_t c_stride_m = c.stride(-2);
    TORCH_CHECK(c_stride_m >= size_n, "c row stride is too small");
    TORCH_CHECK(c_stride_m <= std::numeric_limits<int>::max(), "c row stride is too large");

    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(cublas_handle, at::cuda::getCurrentCUDAStream().stream());
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    int device;
    cudaGetDevice(&device);
    void* ws = DevCtx::instance().get_ws(device);
    cublasSetWorkspace(cublas_handle, ws, WORKSPACE_SIZE);

    int32_t alpha_ = 1;
    int32_t beta_ = 0;
    auto r = cublasGemmEx
    (
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        size_n, size_m, size_k,
        &alpha_, b_ptr, CUDA_R_8I, size_n,
                 a_ptr, CUDA_R_8I, size_k,
        &beta_,  c.data_ptr(), CUDA_R_32I, (int) c_stride_m,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT
    );
    cublas_check(r);
    cuda_check(cudaPeekAtLastError());
}
