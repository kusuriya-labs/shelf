#pragma once

#include <ATen/Tensor.h>

// int8 -> int32 row-major GEMM for the EXL3 W8A8 prefill path
// (implemented in quant/exl3_igemm.cu, both backends).
void igemm_recon(at::Tensor a, at::Tensor b, at::Tensor c);
// Strided-batched variant for the grouped INT8 MoE path:
// C[b] = A[b] @ W[b] with [B,M,K]/[B,K,N] int8 slabs, [B,M,N] int32 out.
void igemm_batched(at::Tensor a, at::Tensor w, at::Tensor c);
