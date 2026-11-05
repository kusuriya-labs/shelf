#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace shelf::kernel {

enum class Exl3Int8OutputDType : uint8_t {
    BF16,
    F32,
};

// EXL3-INT8 linear (rocm_exl3 doc/exl3_int8.md): y = svh * H128( S_FIX * (Q . quant(H128(suh * x))) )
// split into three launches that share one workspace.
//
// 1. rotate_quantize: x (bf16 [rows][input_row_stride]) -> int8 codes [rows][code_row_stride]
//    and one F32 scale per (row, 128-K block): H128 on every K block of suh * x, then
//    scale = amax / 127, code = round-half-even(v / scale) in [-128, 127].
// 2. gemm: yprime (F32 [rows][n]) = S_FIX * sum_g scale[row][g] * (codes . q)[g], with the int8 WMMA
//    accumulating each 128-K group exactly in int32. q is decoded from the resident EXL3 trellis
//    (uint16 [N/16][K/16][16 * bits], tiles in n-major order) on every read; no int8 weight matrix exists in memory.
//    The activation codes are in natural K order (the tile order of the trellis is handled inside the kernel).
// 3. rotate_out: out = H128(sum over the K splits of yprime) * svh, per 128-wide output block.
hipError_t launch_exl3_int8_rotate_quantize(
    const ::shelf::bf16_t* input,
    uint32_t input_row_stride,
    const uint16_t* suh,
    int8_t* codes,
    float* scales,
    uint32_t rows,
    uint32_t k,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    hipStream_t stream);

hipError_t launch_exl3_int8_gemm(
    const int8_t* codes,
    const float* scales,
    uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes,
    const uint16_t* trellis,
    uint32_t bits,
    float* yprime,
    uint32_t rows,
    uint32_t n,
    uint32_t k,
    uint32_t splits,
    hipStream_t stream);

// K splits for the decode-sized gemm (rows <= 16): each split writes its own yprime slice, so yprime must hold
// splits * rows * n floats (scratch_rows is the capacity in rows). Chosen so the grid fills the wave slots.
uint32_t exl3_int8_gemm_splits(uint32_t rows, uint32_t n, uint32_t k, uint32_t scratch_rows);

hipError_t launch_exl3_int8_rotate_out(
    const float* yprime,
    const uint16_t* svh,
    void* out,
    Exl3Int8OutputDType out_dtype,
    uint32_t rows,
    uint32_t n,
    uint32_t out_row_stride,
    uint32_t splits,
    hipStream_t stream);

}
