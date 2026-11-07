#pragma once

#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

namespace shelf::kernel {

// Prefill rows that keep their per-group activation scales are at most this many per batch (about 1/8 of the rows).
inline constexpr std::size_t exl3_int8_flag_cap(std::size_t rows) noexcept {
    return (((rows + 7u) / 8u) + 15u) & ~std::size_t{15u};
}

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
    hipStream_t stream,
    const uint32_t* active_rows = nullptr);

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


// Prefill path (hipBLASLt int8 GEMM, same structure as the W16A16 dense path):
//   rotate_quantize -> rowscale -> recon (trellis -> int8 [n][k]) -> int8 GEMM (+ fused per-group GEMM for a few
//   rows) -> rotate_out_i32.
// rowscale re-rounds the per-group codes onto one scale per row (the max group scale, rowscale[row]). A row where one
// group holds more than `share` of the scale energy (max_g s_g^2 > share * sum_g s_g^2; a massive-activation token,
// whose other groups would lose most of their resolution) keeps its group codes: it gets a slot (slots[row] >= 0, at most `cap` of them, in row order), its codes and scales are copied to
// codes_f / scales_f (row stride = the same strides), and *count = the number of slots. slots[row] = -1 otherwise.
hipError_t launch_exl3_int8_rowscale(
    int8_t* codes, const float* scales, float* rowscale, int32_t* slots, uint32_t* count, int8_t* codes_f,
    float* scales_f, uint32_t rows, uint32_t cap, uint32_t k, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, float share, hipStream_t stream);

hipError_t launch_exl3_int8_recon(
    const uint16_t* trellis, uint32_t bits, int8_t* dense, uint32_t n, uint32_t k, hipStream_t stream);

// out = H128(v) * svh, with v = acc * rowscale[row] * S_FIX, or yprime2[slots[row]] for rows with a slot.
hipError_t launch_exl3_int8_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3Int8OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream);

}
