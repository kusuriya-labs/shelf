#pragma once

#include <shelf/quantization/mxfp4/mxfp4.h>

#include <cstdint>
#include <vector>

namespace shelf::quantization::mxfp4 {

// GPU-native (WMMA fragment-order) payload for an MXFP4 weight matrix.
//
// Codes are tiled in 16-output x 32-K blocks. One block is 256 bytes: output
// row ol owns 16 contiguous bytes at ol * 16, and byte b of that run holds
// code[k = b] in the low nibble and code[k = 16 + b] in the high nibble. That
// pairing is what the gfx1201 FP8 WMMA consumer needs after the E2M1 -> E4M3
// v_perm expansion, so expansion writes the operand registers with no byte
// spread (see kernels/optimized/linear/mxfp4.hip).
//
// The E8M0 scales stay row-major [rows][padded_k / 32], one byte per scale.
// Padded output rows are zero-filled.
// Row reference exponent: max E8M0 byte of the row. The optimized kernels fold
// every E2M1 magnitude in the row by 2^(e8m0 - ref) into the E4M3 operand and
// apply 2^(ref - 127) once in the epilogue, which removes the per-32-K-block
// rescale from the inner loop. See kernels/optimized/linear/mxfp4.hip.
// Largest E8M0 exponent distance a row may span for the folded E4M3 operand to
// stay exact. Beyond it, 1.5 * 2^-d falls between two E4M3 subnormals.
constexpr uint32_t kMxfp4FoldExactMaxDistance = 8u;

struct Mxfp4NativeView {
    bool row_ref_exact = false;
    uint64_t batch = 1;
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t blocks = 0;
    uint64_t codes_row_stride_bytes = 0;
    uint64_t scale_row_stride_bytes = 0;
    const uint8_t* codes = nullptr;
    uint64_t codes_bytes = 0;
    const uint8_t* scales = nullptr;
    uint64_t scales_bytes = 0;
    const uint8_t* row_ref = nullptr;
    uint64_t row_ref_bytes = 0;

    bool validate() const;
};

struct Mxfp4NativeHost {
    uint64_t batch = 1;
    uint64_t rows = 0;
    uint64_t n = 0;
    uint64_t padded_k = 0;
    uint64_t blocks = 0;
    uint64_t codes_row_stride_bytes = 0;
    uint64_t scale_row_stride_bytes = 0;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> scales;
    std::vector<uint8_t> row_ref;
    // True when every row's E8M0 spread fits the exact fold table (max - min <= 8).
    // Outside that range the folded E4M3 operand rounds, so the kernels that fold
    // the reference must not be used for that payload.
    bool row_ref_exact = false;

    Mxfp4NativeView view() const;
};

// Byte offset inside one 16 x 32 block of the packed code byte holding the
// codes of output row `row` at K positions `k` and `k + 16`, both block-local.
constexpr uint32_t mxfp4_native_block_bytes = 256u;

constexpr uint32_t mxfp4_native_byte(uint32_t row, uint32_t k) noexcept {
    return row * 16u + (k & 15u);
}

// Input contract: the canonical view is the whole logical tensor owned by the
// caller. Under tensor parallelism that tensor is already rank-local; callers
// partition the global canonical payload before calling this function, and
// this function never sees tp_rank / tp_size.
bool preshuffle_mxfp4_native(const Mxfp4CanonicalView& v, Mxfp4NativeHost& out);

}  // namespace shelf::quantization::mxfp4
