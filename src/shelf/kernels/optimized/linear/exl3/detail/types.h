#pragma once

// Internal I/O contracts shared by the EXL3 kernels. Public callers only see exl3.h;
// this header is for the kernel implementation and tests.
//
// Unless stated otherwise, sizes are in elements (not bytes) and row strides are in
// elements. Byte strides appear only where the public launch signature names them
// explicitly (code_row_stride_bytes, scale_row_stride_bytes).

#include "config.h"
#include <shelf/kernels/optimized/linear/exl3.h>

#include <cstdint>

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
