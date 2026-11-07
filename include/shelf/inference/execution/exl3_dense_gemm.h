#pragma once

#include <shelf/kernels/optimized/linear/exl3.h>
#include <hip/hip_runtime.h>

namespace shelf::execution {

hipError_t exl3_dense_gemm(const shelf::kernel::Exl3GemmArgs& args, const void* dense,
                           void* accum, hipStream_t stream);

// int8 x int8 -> int32: accum [rows][out_features] = codes [rows][k] . dense [out_features][k]^T.
hipError_t exl3_dense_gemm_int8(uint32_t rows, uint32_t out_features, uint32_t k, const void* dense,
                                const void* codes, void* accum, hipStream_t stream);

}  // namespace shelf::execution
