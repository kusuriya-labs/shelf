#pragma once

#include <shelf/kernels/optimized/linear/config.h>
#include <shelf/core/memory/types.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::kernel {

inline constexpr const char* kExl3W16A16Symbol = "shelf_gemm_exl3_w16a16";
inline constexpr uint32_t kExl3W16A16OutBlock = 128u;
inline constexpr uint32_t kExl3W16A16RowTile = 16u;
inline constexpr uint32_t kExl3W16A16Threads = 256u;
inline constexpr uint32_t kExl3GroupedFuse = 2u;

// Output element type shared by every EXL3 launch. It is independent of the codebook
// (mul1 / mul24) and of the activation precision (FP16 / INT8).
enum class Exl3OutputDType : uint8_t {
    BF16,
    F32,
};

// Codebook, execution phase and activation precision are independent axes. They are
// used at the dispatch boundary; the kernels keep their existing argument fields
// (ka, half, fast_decode, ...) until the phase-specific orchestration lands.
enum class Exl3Codebook : uint8_t { Mul1, Mul24 };
enum class Exl3Phase : uint8_t { Prefill, Decode, Verify };
enum class Exl3ActivationType : uint8_t { FP16, BF16, INT8 };

// ---- I/O contracts ----
//
// Activation / output rows are [rows][stride] with stride in elements of that plane's
// type (bf16 input, Exl3OutputDType output). `code_row_stride_bytes` and
// `scale_row_stride_bytes` are the exceptions and are named in bytes.
//
// The vectorized float4 rotation paths need the corresponding row stride to be a
// multiple of 4 elements; otherwise the launch falls back to the scalar form. The
// mul24 gemm needs n % 64 == 0 and k % 128 == 0; recon needs n % 16 == 0 and k % 128 == 0.
//
// In-place operation is not supported: input, output, codes, scales and workspace are
// distinct buffers. All device pointers must stay valid until the launch completes on
// `stream`; the launch does not synchronise.
//
// Split-K: `splitk_scratch` holds splits * rows * n f32 partials followed by
// splits * rows f32 row sums, and `splitk_floats` is its capacity in floats. The reduce
// pass sums splits in split index order, so the split count must be stable for a given
// request (see `splitk_tuning_rows`).

// Verify reproducibility: the split-K choice changes the K reduction order, so it must
// not depend on how many rows this particular call happens to run. `tuning_rows` is the
// fixed capacity the plan is tuned for; the actual active row count stays in
// Exl3GemmArgs::rows. 0 means "tune on the active rows" (decode / prefill).
struct Exl3ExecutionPolicy {
    Exl3Phase phase = Exl3Phase::Decode;
    uint32_t tuning_rows = 0;
};

struct Exl3GemmArgs {
    const void* trellis = nullptr;
    const void* suh = nullptr;
    const void* svh = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t output_dtype = 0;
    uint32_t ka = 0;
    uint32_t half = 0;
    float* splitk_scratch = nullptr;
    uint32_t splitk_floats = 0;
    uint32_t splitk_splits = 0;
    // Row count the split-K choice is tuned for. A speculative verify runs a different
    // number of rows from round to round (how many requests are active), and the split
    // count changes the K reduction order, so tuning on the row count would make the
    // same request produce a different token depending on its batch neighbours. The
    // executor passes the verify capacity here; 0 tunes on `rows`.
    uint32_t splitk_tuning_rows = 0;
    void* xrot = nullptr;
    uint32_t groups = 0;
    // Optional n-tile override for the grouped path. 0 keeps pick_n_tiles' shape choice.
    uint32_t n_tiles = 0;
    // Experimental ka=5 half=0 dequant path (dense 5 bpw checkpoints).
    uint32_t fast_decode = 0;

    // Optional routed-expert group plan. When group_expert is non-null, blockIdx.z is
    // the group, group_expert[g] selects the expert plane and group_rows[g] bounds the
    // rows this group owns. The trellis/suh/svh pointers are bases and the batch
    // strides select one expert. Input and output are the group-row planes; group g
    // owns rows [g * kExl3W16A16RowTile, ...).
    const uint32_t* group_expert = nullptr;
    const uint32_t* group_rows = nullptr;
    const uint32_t* group_count = nullptr;
    // Optional per-expert group range. When non-null, consecutive groups of one expert are
    // fused into one block: a block whose group index is the first of a kGroups stride inside
    // [expert_group_begin[e], expert_group_begin[e+1]) owns up to kGroups groups of expert e
    // and dequantizes its weight tiles once for all of them.
    const uint32_t* expert_group_begin = nullptr;
    // Optional token row for each group row. When non-null, the input plane is token rows
    // and the kernel gathers through this table; when null, the input plane is already in
    // group-row order.
    const int32_t* row_token = nullptr;
    uint64_t trellis_batch_stride = 0;
    uint64_t suh_batch_stride = 0;
    uint64_t svh_batch_stride = 0;
};

// One routed-expert projection over a grouped row plan. The expert planes are the
// packed EXL3 layout the Qwen4Exp loader produces:
//
//   trellis [experts, trellis_bytes]
//   suh     [experts, k]
//   svh     [experts, n]
//
// Group g owns activation and output rows [g * kExl3W16A16RowTile,
// (g + 1) * kExl3W16A16RowTile), the same row contract as the MXFP4 grouped linear.
// group_expert / group_rows / group_count are device pointers from the router plan.
struct Exl3GroupedArgs {
    const void* trellis = nullptr;
    const void* suh = nullptr;
    const void* svh = nullptr;
    const void* input = nullptr;
    void* output = nullptr;
    uint32_t out_features = 0;
    uint32_t k = 0;
    uint32_t input_row_stride = 0;
    uint32_t output_row_stride = 0;
    uint32_t output_dtype = 0;
    uint32_t ka = 0;
    uint32_t half = 0;
    // 0: mul1 codebook (W16A16). 1: mul24 codebook (EXL3-INT8 weights, q * kExl3Int8GridStep) run as an FP16 GEMM.
    // The trellis stays in the canonical [kt][nt] tile order in both cases. mul24 requires half == 0.
    uint32_t mul24 = 0;
    // With mul24: 1 quantizes the (rotated) activation per row and 128-K block to int8 and runs the iu8 WMMA with an
    // int32 accumulator per block (EXL3-INT8, as the dense decode path); 0 is the FP16 GEMM. Requires mul24.
    uint32_t int8_act = 0;
    std::uint64_t trellis_batch_stride = 0;
    std::uint64_t suh_batch_stride = 0;
    std::uint64_t svh_batch_stride = 0;
    const uint32_t* group_expert = nullptr;
    const uint32_t* group_rows = nullptr;
    const uint32_t* group_count = nullptr;
    // Optional per-expert group range. When non-null, consecutive groups of one expert are
    // fused into one block so the expert's weight tiles are dequantized once per fused chunk.
    const uint32_t* expert_group_begin = nullptr;
    const int32_t* row_token = nullptr;
    uint32_t group_capacity = 0;
};

// ---- mul1 (W16A16) ----

hipError_t launch_gemm_exl3_w16a16(const Exl3GemmArgs& args, hipStream_t stream);
hipError_t launch_rotate_exl3(const Exl3GemmArgs& args, void* xrot, hipStream_t stream);
hipError_t launch_rotate_exl3_rowsum(const Exl3GemmArgs& args, void* xrot, float* rowsum,
                                     hipStream_t stream);
hipError_t launch_recon_exl3_dense(const Exl3GemmArgs& args, void* dense, hipStream_t stream);
hipError_t launch_quantize_dense_w8(const Exl3GemmArgs& args, void* dense,
                                    hipStream_t stream);
hipError_t launch_quantize_xrot_x8(const Exl3GemmArgs& args, void* xrot, float* rowsum,
                                   bool residual, hipStream_t stream);
hipError_t launch_epilogue_exl3_dense(const Exl3GemmArgs& args, const void* rowsum,
                                      const void* accum, hipStream_t stream);
hipError_t launch_gemm_exl3_w16a16_pre(const Exl3GemmArgs& args, const void* xrot,
                                       hipStream_t stream);

// ---- grouped (MoE) ----

// Correctness-first grouped EXL3 linear. It dispatches the existing dense EXL3
// kernel once per group, so it preserves the packed expert layout but has a launch
// per group. A fused grouped kernel replaces this for throughput later.
hipError_t launch_gemm_exl3_w16a16_grouped(const Exl3GroupedArgs& args,
                                           hipStream_t stream);

// Grouped mul24 GEMM (exl3_mul24_grouped.hip). `args` is the grouped plan as launch_gemm_exl3_w16a16_grouped builds it
// (rows = one group tile, groups = fused groups per block, 1 or 2; more is run as 2). int8_act selects INT8 activations
// with an iu8 WMMA; otherwise an FP16 WMMA on the exact integer weights. n_tiles is 1 or 2 (128 output columns each).
hipError_t launch_exl3_mul24_grouped(const Exl3GemmArgs& args, bool int8_act, uint32_t n_tiles,
                                     uint32_t grid_x, uint32_t grid_z, hipStream_t stream);

// ---- mul24 (EXL3-INT8 weights) ----

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

hipError_t launch_exl3_mul24_recon(
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

}  // namespace shelf::kernel
