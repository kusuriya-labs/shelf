#pragma once

// mul24 (EXL3-INT8 weights) codebook entry points. Shared EXL3 types and contracts live
// in exl3_family/types.h; mul1 entry points live in exl3_mul1.h.

#include <shelf/kernels/optimized/linear/exl3_family/types.h>

namespace shelf::kernel {

// Prefill rows that keep their per-group activation scales are at most this many per batch (about 1/8 of the rows).
inline constexpr std::size_t exl3_mul24_flag_cap(std::size_t rows) noexcept {
    return (((rows + 7u) / 8u) + 15u) & ~std::size_t{15u};
}

// EXL3-MUL24 linear (rocm_exl3 doc/exl3_int8.md): y = svh * H128( S_FIX * (Q . quant(H128(suh * x))) )
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
hipError_t launch_exl3_mul24_rotate_quantize(
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

hipError_t launch_exl3_mul24_gemm(
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
uint32_t exl3_mul24_gemm_splits(uint32_t rows, uint32_t n, uint32_t k, uint32_t scratch_rows,
                                uint32_t bits = 0u);

hipError_t launch_exl3_mul24_rotate_out(
    const float* yprime,
    const uint16_t* svh,
    void* out,
    Exl3OutputDType out_dtype,
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
hipError_t launch_exl3_mul24_rowscale(
    int8_t* codes, const float* scales, float* rowscale, int32_t* slots, uint32_t* count, int8_t* codes_f,
    float* scales_f, uint32_t rows, uint32_t cap, uint32_t k, uint32_t code_row_stride_bytes,
    uint32_t scale_row_stride_bytes, float share, hipStream_t stream);

// B prefill: reconstructed dense INT8 weights in trellis n-major tile order,
// per-128-K activation scales, gfx12 iu8 WMMA into int32 followed by
// group-scale FP32 FMA.
// Writes F32 yprime [rows][n] including the EXL3 grid-step factor.
hipError_t launch_exl3_mul24_prefill_wmma(
    const int8_t* codes, const float* scales,
    uint32_t code_stride, uint32_t scale_stride_bytes,
    const int8_t* dense_weights, float* yprime,
    uint32_t rows, uint32_t n, uint32_t k, hipStream_t stream);

hipError_t launch_exl3_mul24_recon(
    const uint16_t* trellis, uint32_t bits, int8_t* dense, uint32_t n, uint32_t k, hipStream_t stream);

// Same decode as launch_exl3_mul24_recon, but the output keeps the trellis n-major tile order:
// one 256-byte 16x16 tile per (n tile, k tile), lane l holding its eight weights. A WMMA B
// fragment (32 lanes x 8 bytes) is then one contiguous 256-byte read.
hipError_t launch_exl3_mul24_recon_wmma(
    const uint16_t* trellis, uint32_t bits, int8_t* dense, uint32_t n, uint32_t k, hipStream_t stream);

// out = H128(v) * svh, with v = acc * rowscale[row] * S_FIX, or yprime2[slots[row]] for rows with a slot.
hipError_t launch_exl3_mul24_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream);

// Prefill-only output rotation. Uses the same math as launch_exl3_mul24_rotate_out_i32; the
// implementation is the opt-in 5-shuffle / 2-smem hybrid (or the float4 form when
// SHELF_EXL3_INT8_ROTATE_F4 is set).
hipError_t launch_exl3_mul24_prefill_rotate_out_i32(
    const int32_t* acc, const float* rowscale, const int32_t* slots, const float* yprime2, const uint16_t* svh,
    void* out, Exl3OutputDType out_dtype, uint32_t rows, uint32_t n, uint32_t out_row_stride, hipStream_t stream);

// Grouped mul24 GEMM (exl3_family/mul24/grouped.hip). `args` is the grouped plan as
// launch_gemm_exl3_w16a16_grouped builds it (rows = one group tile, groups = fused
// groups per block, 1 or 2; more is run as 2). int8_act selects INT8 activations with
// an iu8 WMMA; otherwise an FP16 WMMA on the exact integer weights. n_tiles is 1 or 2
// (128 output columns each).
hipError_t launch_exl3_mul24_grouped(const Exl3GemmArgs& args, bool int8_act, uint32_t n_tiles,
                                     uint32_t grid_x, uint32_t grid_z, hipStream_t stream);

}  // namespace shelf::kernel
