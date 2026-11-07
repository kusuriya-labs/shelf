#pragma once

#include <cstdint>

namespace shelf::quantization::exl3 {

// EXL3-INT8 (mul24 codebook, fixed int8 grid, WMMA B-fragment tile order). Specification and the golden
// NumPy reference live in rocm_exl3 (doc/exl3_int8.md, research/codebook, exllamav3/util/exl3_int8_ref.py).
//
//   weight = q * kExl3Int8GridStep,   q = (B - 509) >> 2   in [-128, 127]
//   B      = sum of the four bytes of (state * 0xDCD12D)  (uint32 wraparound)
//   tile position of stream index t: row k = 8 * (t / 128) + t % 8, column n = (t / 8) % 16
constexpr float kExl3Int8GridStep = 4.0f * 887.0f / 131072.0f;
constexpr uint32_t kExl3TileDim = 16;
// The marker tensor of an EXL3 matrix holds the codebook multiplier, which names the format.
constexpr uint32_t kExl3Mul1Multiplier = 0x83DCD12Du;
constexpr uint32_t kExl3Mul24Multiplier = 0xDCD12Du;

// trellis: uint16[kt][nt][16 * bits], little endian, one 16x16 tile per [kt][nt].
// out:     int8[nt * 16][kt * 16], row = output n, column = input k (K contiguous).
// Returns false on an invalid argument (bits outside 1..8, null pointers).
bool exl3_decode_int8(
    const uint16_t* trellis,
    uint64_t kt,
    uint64_t nt,
    uint32_t bits,
    int8_t* out,
    unsigned threads);

// Resident layout of an INT8 trellis: whole 16x16 tiles reordered from the file's [kt][nt] to [nt][kt], so a wave
// that walks K reads one contiguous stream (a [kt][nt] walk strides by a whole tile row and ran about 20% slower
// on a weight set that does not fit the cache). src and dst are uint16[kt * nt * 16 * bits].
bool exl3_tiles_to_nmajor(
    const uint16_t* src,
    uint64_t kt,
    uint64_t nt,
    uint32_t bits,
    uint16_t* dst,
    unsigned threads);

}
