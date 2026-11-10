#pragma once

// Shared types and I/O contracts for the EXL3 family (mul1 and mul24). Codebook-specific
// launch declarations live in exl3_mul1.h / exl3_mul24.h.
//
// Unless stated otherwise, sizes are in elements (not bytes) and row strides are in
// elements. Byte strides appear only where the launch signature names them explicitly
// (code_row_stride_bytes, scale_row_stride_bytes).

#include "config.h"

#include <shelf/core/memory/types.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace shelf::kernel {

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

}  // namespace shelf::kernel

namespace shelf::kernel::exl3_detail {

// ---- alignment ----
// Vectorized float4 paths require the corresponding row strides to be multiples of 4.
// 16-byte-aligned arena views make every 4-element-aligned row start 16-byte aligned.
inline constexpr uint32_t kF4StrideElements = 4u;
inline constexpr uint32_t kReconNAlingment = 16u;
inline constexpr uint32_t kGemmNAlingment = 64u;
inline constexpr uint32_t kKBlock = 128u;

// ---- trellis ----
// EXL3 trellis words are uint16. A 16x16 tile holds 256 weights in WMMA B-fragment
// order. The resident (n-major) form is [n/16][k/16][8 * bits] uint32 words per tile,
// while the canonical form is [k/16][n/16]. `ka` is the integer bit rate; `half`
// adds 128 bits (ka=3 half=1 is 3.5 bpw).
struct TrellisContract {
    static constexpr uint32_t kTile = 16u;
    static constexpr uint32_t kWeightsPerTile = 256u;
    static constexpr uint32_t kWordBits = 16u;
    static constexpr uint32_t kKAlignment = kKBlock;
};

// ---- input / output ----
// The activation plane is [rows][input_row_stride] bf16; the output plane is
// [rows][output_row_stride] of Exl3OutputDType (bf16 or f32). output_row_stride is in
// elements of the output type. In-place operation is not supported.
struct ActivationContract {
    static constexpr uint32_t kInputDTypeBits = 16u;
    static constexpr uint32_t kCodeBlock = kKBlock;
};

// ---- rotation ----
// suh [k] and svh [n] are fp16 bits. H128 is a Sylvester Walsh-Hadamard transform over
// one 128-wide block, unnormalised; callers scale by 1/sqrt(128). rowsum is one f32 per
// row and is only produced by the rowsum rotate variant.
struct RotationContract {
    static constexpr uint32_t kBlock = kKBlock;
    static constexpr uint32_t kSuhLength = kKBlock;  // per block, repeated k/128 times
};

// ---- split-K ----
// splitk_scratch holds `splits * rows * n` f32 partials followed by `splits * rows`
// f32 row sums (the rowsum plane starts at splits * rows * n floats). The gemm writes
// its own split slice; the reduce pass sums splits in split index order. splitk_floats
// is the capacity in floats; a split is dropped until the capacity fits.
struct SplitKContract {
    static constexpr uint32_t kPartialsPerSplit = 1u;  // rows * n
    static constexpr uint32_t kRowSumsPerSplit = 1u;   // rows
};

// ---- grouped ----
// group_expert[g], group_rows[g], group_count[0] are device pointers that stay valid
// for the launch. row_token[t] maps a group row to an input token row (int32, < 0 skips
// the row). expert_group_begin[e]..[e+1] bounds the groups of expert e for the fused
// chunk. group_capacity is the blockIdx.z extent.
struct GroupedContract {
    static constexpr uint32_t kMaxFusedGroups = 2u;  // more is run as 2
};

// ---- mul24 flagged rows ----
// codes_f / scales_f are compact copies of the flagged rows, indexed by slots[row] >= 0.
// yprime2 is [cap][n] and holds the fused per-group result for the flagged rows. A row
// whose group scales differ too much keeps its group codes; at most `cap` rows do, in
// row order. Every other row is re-rounded in place onto rowscale[row].
struct FlaggedContract {
    static constexpr uint32_t kFlagStrideElements = 1u;
};

enum class Exl3Backend : uint8_t { Direct, ReconDense };

// Host-side description of one launch. It normalises the independent axes without
// changing the kernel argument ABI or adding a dynamic branch to any __global__.
struct Exl3KernelVariant {
    Exl3Phase phase = Exl3Phase::Decode;
    Exl3Codebook codebook = Exl3Codebook::Mul1;
    Exl3ActivationType activation_type = Exl3ActivationType::BF16;
    Exl3OutputDType output_type = Exl3OutputDType::BF16;
    Exl3Backend backend = Exl3Backend::Direct;
    bool grouped = false;
    bool pre_rotated = false;
};

// Rejects combinations the kernels do not implement. The external return codes and
// side effects of the launch functions stay unchanged; callers may use this before
// choosing a path.
bool exl3_variant_supported(const Exl3KernelVariant& variant);

}  // namespace shelf::kernel::exl3_detail
